import torch
import logging
import time
import evaluate
import os
import tqdm
from typing import Dict, Union, Optional
from model.loader import T5ModelLoader, Model
import dataclasses

from model.config import ModelConfig
from train.config import TrainConfig, Device, TestTask
from train.utils import Averager

from torch.utils.data import DataLoader
from transformers import SpecialTokensMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutputWithPast
from accelerate import Accelerator
from datasets.iterable_dataset import IterableDataset

from train.optimizer import get_optimizer
from train.scheduler import get_lr_scheduler

Output = Union[Seq2SeqLMOutput, CausalLMOutputWithPast]
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class CurrentState:
    train_step: int
    train_epoch: int
    last_log: float

class Trainer:
    def __init__(self, train_config: TrainConfig):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.config = train_config
        # Averager for logging
        self.average_logger = Averager()
     
    def train(
        self,
        model: Model, 
        tokenizer: SpecialTokensMixin,
        train_dataloader: DataLoader, 
        test_dataloader: DataLoader, 
        eval_dataloader: DataLoader = None,
    ):
        accelerator = Accelerator(
            cpu = (self.config.device == Device.CPU.value),
        )
        
        logging.info(f"Using {accelerator.device}")
        
        optim_config = self.config.optim_config
        optimizer = get_optimizer(model, optim_config)
        
        lr_scheduler_config = optim_config.lr_scheduler_config
        lr_scheduler = get_lr_scheduler(
            optimizer, 
            optim_config.total_steps, 
            optim_config.base_lr,
            lr_scheduler_config
        )
        
        # Prepare distributed training, mixed precision, etc.
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader
        )
        
        if self.config.do_compile:
            torch.compile(model)
        
        # Set the initial state of the training
        self.current_state = CurrentState(
            train_step=1,
            train_epoch=1,
            last_log=time.time(),
        )
        
        self._train(
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=eval_dataloader,
            test_dataloader=test_dataloader,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tokenizer=tokenizer,
        )
        
        self.evaluate()

    def predict(self):
        pass
    
    def evaluate(
        self,
        dataloader: DataLoader,
        model: Model,
        tokenizer: SpecialTokensMixin,
        accelerator: Accelerator,
        ):
        #TODO(hyeontae): Implement evaluation
        pass
    
    def _train(
        self,
        model: Model,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader],
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        tokenizer: SpecialTokensMixin,
    ):
        # Set Model to train mode
        model.train()
        
        current_state = self.current_state
        optim_config = self.config.optim_config

        # Start training loop
        while current_state.train_step <= optim_config.total_steps:
            train_dataset = train_dataloader.dataset
            if isinstance(train_dataset, IterableDataset):
                train_dataset.set_epoch(current_state.train_epoch)

            # In case there is a remainder from previous epoch, we need to reset the optimizer
            optimizer.zero_grad(set_to_none=True)
            
            for batch_id, batch in enumerate(train_dataloader, start=1):
                if current_state.train_step > optim_config.total_steps:
                    break
                
                outputs: Output = model(**batch)
                loss = outputs.loss
                self.average_logger.update({'loss': loss.detach().float().item()})
                accelerator.backward(loss / self.config.grad_acc)

                if batch_id % self.config.grad_acc == 0:
                    self._update_metrics(model, batch, outputs)
                    
                    if self.config.optim_config.grad_clip > 0:
                        # clip grad norm
                        accelerator.clip_grad_norm_(
                            parameters=model.parameters(),
                            max_norm=self.config.optim_config.grad_clip,
                            norm_type=2,
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    # log hyperparameters
                    lr = optimizer.param_groups[0]['lr']
                    self.average_logger.update({'lr': lr})
                    
                    # reset gradients
                    optimizer.zero_grad(set_to_none=True)
                    self._maybe_log_metrics()
                    
                    # evaluate and save checkpoint
                    if accelerator.is_main_process:
                        if self.config.eval_config != None:
                            self._maybe_validate(
                                dataloader=validation_dataloader,
                                model=model,
                                tokenizer=tokenizer,
                                accelerator=accelerator,
                            )
                            
                        self._maybe_save_checkpoint(
                            accelerator,
                        )
                            
                    accelerator.wait_for_everyone()
                    current_state.train_step += 1
            current_state.train_epoch += 1

    def _update_metrics(self, model: Model, batch: Dict[str, torch.Tensor], outputs: Output):
        metrics = {}
        
        # TODO(hyeontae): Remove hard-coded metrics
        if self.config.logging_config.accuracy:
            correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
            accuracy = correct / batch["labels"].numel()
            metrics['accuracy'] = accuracy
            
        if self.config.logging_config.grad_l2:
            grad_l2 = (
                sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            )
            metrics['grad_l2'] = grad_l2
            
        if self.config.logging_config.weights_l2:
            weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            metrics['weights_l2'] = weights_l2
            
        self.average_logger.update(metrics)

    def _maybe_log_metrics(self):
        if self.current_state.train_step % self.config.logging_config.every_steps != 0:
            return 

        seconds_per_step = (time.time() - self.current_state.last_log) / self.config.logging_config.every_steps

        self.average_logger.update({"time_per_step": seconds_per_step})
        averaged_metrics = self.average_logger.average()

        msg_start = f'[train] Step {self.current_state.train_step} out of {self.config.optim_config.total_steps}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.6f}' for k, v in averaged_metrics.items()]) + ' | '

        msg = msg_start + dict_msg
        logger.info(msg)

        self.current_state.last_log = time.time()

    def _maybe_validate(self, **kwargs):
        if (
            self.current_state.train_step > self.config.optim_config.total_steps
            or self.current_state.train_step % self.config.predict_config.every_steps == 0
        ):
            self.evaluate(
                **kwargs
            )
            
    def _maybe_save_checkpoint(self, accelerator: Accelerator):
        if (
            self.current_state.train_step > self.config.optim_config.total_steps
            or self.current_state.train_step % self.config.checkpoint.every_steps == 0
        ):
            output_dir = f'checkpoint-{self.current_state.train_step}'
            accelerator.save_state(output_dir=output_dir)

def validate_config(config: TrainConfig):
        pass
    
def load_model(model_config: ModelConfig) -> Model:
    model_loader = T5ModelLoader(model_config)
    return model_loader.get_model()

def load_tokenizer(model_config: ModelConfig):
    model_loader = T5ModelLoader(model_config)
    return model_loader.get_tokenizer()

# def predict(
#     model: Model, 
#     dataloader: DataLoader,
#     config: TrainConfig,
#     tokenizer: SpecialTokensMixin,
#     accelerator: Accelerator,
#     prefix:str='test'
# ):
#     config.current_state.last_log = time.time()

#     if config.test_task == TestTask.MOL2TEXT:
#         metric = evaluate.load(os.path.join(__file__.split('biot5/utils')[0], 'biot5/metrics/translation_metrics'))
#     elif config.test_task == TestTask.TEXT2MOL:
#         metric = evaluate.load(os.path.join(__file__.split('biot5/utils')[0], 'biot5/metrics/save_only_metrics'))
#     elif config.test_task == TestTask.TEXT2FRAG:
#         metric = evaluate.load(os.path.join(__file__.split('biot5/utils')[0], 'biot5/metrics/save_only_metrics'))
#     elif config.test_task in [TestTask.DTI , TestTask.PEER, TestTask.MOLNET]:
#         metric = evaluate.load(os.path.join(__file__.split('biot5/utils')[0], 'biot5/metrics/dti_metrics'))
#     else:
#         raise ValueError("Invalid test task")
    
#     samples_seen = 0
#     selfies_invalid = 0

#     def decode(preds):
#         preds[preds == -100] = tokenizer.pad_token_id
#         preds = tokenizer.batch_decode(
#             preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
#         )
        
#         preds = [pred.strip() for pred in preds]
        
#         return preds

#     prediction_total = []
#     reference_total = []
#     input_total = []


#     for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#         batch = batch.to(accelerator.device)
#         if step == 100:
#             break
#         if config.test_task in [TestTask.DTI , TestTask.PEER, TestTask.MOLNET]:
#             generation_results = model.generate(
#                 input_ids=batch['input_ids'],
#                 attention_mask=batch['attention_mask'],
#                 max_length=args.data.max_target_len,
#                 generation_config=model.generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#             )
#             predictions, scores = generation_results.sequences, generation_results.scores
#         else:
#             if "galactica" in args.model.name:
#                 predictions = model.generate(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask'],
#                     max_length=args.data.max_target_len,
#                 )
#             else:
#                 predictions = model.generate(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask'],
#                     max_length=args.data.max_target_len,
#                     # generation_config=model.generation_config,
#                 )
#         predictions = decode(predictions)
#         references = decode(batch["labels"])
        
#         # print(batch["labels"])
#         # print(references)
#         # raise dd
        
#         inputs = decode(batch["input_ids"])
        
#         # if prefix == "test":
#         #     print(inputs[0])
#         #     print(batch['input_ids'][0])
#         #     print(batch['attention_mask'][0])
#         #     raise dd

        
#         if args.test_task == 'mol2text':
#             if args.representation == "selfies":
#                 # inputs = [sf.decoder(input_i.split('- Input: ')[-1].split(' Output:')[0]) for input_i in inputs]
#                 inputs = [input_i.split('- Input: ')[-1].split(' Output:')[0] for input_i in inputs]
            
#             elif args.representation == "smiles":
#                 inputs = [input_i.split('- Input: ')[-1].split(' Output:')[0] for input_i in inputs]

#             elif args.representation == "frag":
#                 inputs = [input_i.split('- Input: ')[-1].split(' Output:')[0] for input_i in inputs]
#                 # tmp_inputs = []
#                 # for input_i in inputs:
#                 #     input_i = input_i.split('- Input: ')[-1].split(' Output:')[0]
                    
#                 #     result_smiles = ""
#                 #     for smiles in input_i.split("[.]"):
#                 #         result_smiles += decode_linear(smiles).split(".")[0] + "."
#                 #     result_smiles = result_smiles[:-1]
                    
#                 #     tmp_inputs.append(result_smiles)
#                 # inputs = tmp_inputs
#             for input_i in inputs:
#                 input_total.append(input_i)
                    
                    
                    

#             references = [(references[i], inputs[i]) for i in range(len(references))]
#         elif args.test_task == 'text2mol':
            
#             inputs = [input_i.split('- Input: ')[-1].split(' Output:')[0] for input_i in inputs]

#             for i in range(len(predictions)):
#                 if args.representation == "selfies":
#                     try: 
#                         predictions[i] = Chem.MolToSmiles(Chem.MolFromSmiles(sf.decoder(predictions[i])), kekuleSmiles=True)
                        
#                     except:
#                         # predictions[i] = sf.decoder(filter_selfies(predictions[i]))
#                         selfies_invalid += 1
#                 elif args.representation == "smiles":
                    
#                     continue
#                 elif args.representation == "frag":
#                     frags = []
#                     opened = 0
#                     tmp_frag = ""
#                     try:
#                         result_smiles = ""
#                         for smiles in predictions[i].split("[.]"):
#                             result_smiles += decode_linear(smiles) + "."
#                         result_smiles = result_smiles[:-1]
#                         predictions[i] = result_smiles
#                     except:
#                         predictions[i] = "C"
                
                
#             references = [sf.decoder(ref_i) for ref_i in references]
#             references = [Chem.MolToSmiles(Chem.MolFromSmiles(ref), kekuleSmiles=True) for ref in references]
            
#             references = [(references[i], inputs[i]) for i in range(len(references))]
            
            
                
            

#         elif args.test_task == 'dti' or args.test_task == 'peer' or args.test_task == 'molnet':
#             # No: 465, Yes: 2163
#             predictions = [(scores[0][i][2163] / (scores[0][i][2163] + scores[0][i][465])).item() for i in range(len(predictions))]
#         else:
#             raise NotImplementedError

#         # If we are in a multiprocess environment, the last batch has duplicates
#         if step == len(dataloader) - 1:
#             predictions = predictions[: len(dataloader.dataset) - samples_seen]
#             references = references[: len(dataloader.dataset) - samples_seen]
#         else:
#             samples_seen += len(references)

        
#         for ref in references:
#             reference_total.append(ref)
#         for pred in predictions:
#             prediction_total.append(pred)
#         metric.add_batch(
#             predictions=predictions,
#             references=references,
#         )


#         # TODO for debug
#         # if step == 20:
#         #     break
        

#     eval_metric = metric.compute(tsv_path=os.path.join(args.working_dir, args.result_fn))

#     assert len(prediction_total) == len(reference_total)
#     if args.test_task == 'text2mol':
#         s_RDK_list = []
#         s_MACCS_list = []
#         s_Morgan_list = []
#         exact_canon_list = []
#         invalid = 0
#         method = args.data.data_dir.split("/")[-1].split()[0]
#         model_size = args.model.name.split("-")[-1].split()[0]
#         with open(f"/home/osikjs/BioT5/biot5/train_predictions/eval_fts_ref_{args.current_train_step}_{method}_{model_size}_{args.wandb_name}.txt", "wb") as g:
#             for i in range(len(prediction_total)):
#                 gen = Chem.MolFromSmiles(prediction_total[i])
#                 if gen == None:
#                     gen = Chem.MolFromSmiles("C")
#                 target_mol = Chem.MolFromSmiles(reference_total[i][0])
                
#                 # Chem.RemoveStereochemistry(gen)
#                 # Chem.RemoveStereochemistry(target_mol)

#                 gen_smiles = Chem.MolToSmiles(gen)
#                 target = Chem.MolToSmiles(target_mol)


#                 target_fp_RDK = FingerprintMols.GetRDKFingerprint(target_mol)
#                 gen_fp_RDK = FingerprintMols.GetRDKFingerprint(gen)

#                 target_fp_MACCS = MACCSkeys.GenMACCSKeys(target_mol)
#                 gen_fp_MACCS = MACCSkeys.GenMACCSKeys(gen)

#                 fpgen = AllChem.GetMorganGenerator(radius=2)

#                 target_fp_Morgan = fpgen.GetSparseCountFingerprint(target_mol)
#                 gen_fp_Morgan = fpgen.GetSparseCountFingerprint(gen)
                
#                 s_RDK = DataStructs.TanimotoSimilarity(gen_fp_RDK,target_fp_RDK)
#                 s_MACCS = DataStructs.TanimotoSimilarity(gen_fp_MACCS,target_fp_MACCS)
#                 s_Morgan = DataStructs.TanimotoSimilarity(gen_fp_Morgan,target_fp_Morgan)
                
#                 g.write(f"{reference_total[i][1]}\t{target}\t{gen_smiles}\t{s_RDK}\t{s_MACCS}\t{s_Morgan}\n".encode('utf-8'))

#                 if gen_smiles != "C":
#                     s_RDK_list.append(s_RDK)
#                     s_MACCS_list.append(s_MACCS)
#                     s_Morgan_list.append(s_Morgan)
#                     try:
#                         score =  Chem.MolToInchi(Chem.MolFromSmiles(gen_smiles)) == Chem.MolToInchi(Chem.MolFromSmiles(target))
#                     except:
#                         score = 0
#                     exact_canon_list.append(int(score))
#                 else:
#                     invalid += 1
#                     s_RDK_list.append(0)
#                     s_MACCS_list.append(0)
#                     s_Morgan_list.append(0)
#                     exact_canon_list.append(0)
                


#             avg_RDK = sum(s_RDK_list)/len(s_RDK_list)
#             avg_MACCS = sum(s_MACCS_list)/len(s_MACCS_list)
#             avg_Morgan = sum(s_Morgan_list)/len(s_Morgan_list)
#             exact_canon = sum(exact_canon_list)/len(exact_canon_list)

#             g.write(f"RDK FTS: {avg_RDK}\n".encode('utf-8'))
#             g.write(f"MACCS FTS: {avg_MACCS}\n".encode('utf-8'))
#             g.write(f"Morgan FTS: {avg_Morgan}\n".encode('utf-8'))
#             g.write(f"Exact: {exact_canon}\n".encode('utf-8'))

#             g.write(f'Invalid: {invalid}\n'.encode('utf-8'))
#         if prefix == "validation":
#             wandb.log({"validation_RDK": avg_RDK})
#             wandb.log({"validation_MACCS": avg_MACCS})
#             wandb.log({"validation_Morgan": avg_Morgan})
#             wandb.log({"validation_Exact": exact_canon})
#             wandb.log({"validation_Invalid": invalid})
#         if prefix == "test":
#             wandb.log({"test_RDK": avg_RDK})
#             wandb.log({"test_MACCS": avg_MACCS})
#             wandb.log({"test_Morgan": avg_Morgan})
#             wandb.log({"test_Exact": exact_canon})
#             wandb.log({"test_Invalid": invalid})
#     if args.test_task == 'mol2text':

#         s_rouge1_list = []
#         s_rouge2_list = []
#         s_rougel_list = []
#         s_meteor_list = []
        
#         method = args.data.data_dir.split("/")[-1].split()[0]
#         model_size = args.model.name.split("-")[-1].split()[0]
#         with open(f"/home/osikjs/BioT5/biot5/train_predictions/mol2text_eval_fts_ref_{args.current_train_step}_{method}_{model_size}_{args.wandb_name}.txt", "wb") as g:
#             references = [reference_total[i][0] for i in range(len(reference_total))]
#             predictions = prediction_total
#             text_tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')

#             meteor_scores = []

#             refs = []
#             preds = []

#             for gt, out in zip(references, predictions):

#                 gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=512,
#                                                     padding='max_length')
#                 gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
#                 gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
#                 gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

#                 out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=512,
#                                                     padding='max_length')
#                 out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
#                 out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
#                 out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

#                 refs.append([gt_tokens])
#                 preds.append(out_tokens)

#                 mscore = meteor_score([gt_tokens], out_tokens)
#                 meteor_scores.append(mscore)


#             _meteor_score = np.mean(meteor_scores)

#             scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

#             rouge_scores = []

#             refs = []
#             preds = []

#             for gt, out in zip(references, predictions):

#                 rs = scorer.score(out, gt)
#                 rouge_scores.append(rs)

#             # rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
#             # rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
#             # rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
#             i= 0
            
#             for rs, meteor in zip(rouge_scores, meteor_scores):
#                 s_rouge1 = rs['rouge1'].fmeasure
#                 s_rouge2 = rs['rouge2'].fmeasure
#                 s_rougel = rs['rougeL'].fmeasure
#                 s_meteor = meteor
#                 g.write(f"Input: {input_total[i]}\tReference: {references[i]}\tPrediction: {predictions[i]}\t{s_rouge1}\t{s_rouge2}\t{s_rougel}\t{s_meteor}\n".encode('utf-8'))
#                 s_rouge1_list.append(s_rouge1)
#                 s_rouge2_list.append(s_rouge2)
#                 s_rougel_list.append(s_rougel)
#                 s_meteor_list.append(s_meteor)
#                 i+= 1

                

#             avg_rouge1 = sum(s_rouge1_list)/len(s_rouge1_list)
#             avg_rouge2 = sum(s_rouge2_list)/len(s_rouge2_list)
#             avg_rougel = sum(s_rougel_list)/len(s_rougel_list)
#             avg_meteor = sum(s_meteor_list)/len(s_meteor_list)
            

#             g.write(f"Rouge-1: {avg_rouge1}\n".encode('utf-8'))
#             g.write(f"Rouge-2: {avg_rouge2}\n".encode('utf-8'))
#             g.write(f"Rouge-L: {avg_rougel}\n".encode('utf-8'))
#             g.write(f"Meteor: {avg_meteor}\n".encode('utf-8'))


#         if prefix == "validation":
#             wandb.log({"validation_bleu2": eval_metric["bleu2"]})
#             wandb.log({"validation_bleu4": eval_metric["bleu4"]})
#             wandb.log({"validation_rouge1": eval_metric["rouge1"]})
#             wandb.log({"validation_rogue2": eval_metric["rouge2"]})
#             wandb.log({"validation_rougeL": eval_metric["rougeL"]})
#             wandb.log({"validation_meteor": eval_metric["meteor"]})
#         if prefix == "test":
#             wandb.log({"test_bleu2": eval_metric["bleu2"]})
#             wandb.log({"test_bleu4": eval_metric["bleu4"]})
#             wandb.log({"test_rouge1": eval_metric["rouge1"]})
#             wandb.log({"test_rogue2": eval_metric["rouge2"]})
#             wandb.log({"test_rougeL": eval_metric["rougeL"]})
#             wandb.log({"test_meteor": eval_metric["meteor"]})
#         logger.log_stats(
#             stats={
#                 "bleu2": eval_metric["bleu2"],
#                 "bleu4": eval_metric["bleu4"],
#                 "rouge1": eval_metric["rouge1"],
#                 "rouge2": eval_metric["rouge2"],
#                 "rougeL": eval_metric["rougeL"],
#                 "meteor": eval_metric["meteor"],
#                 "time": time.time() - args.last_log,
#             },
#             step=args.current_train_step,
#             args=args,
#             prefix=f"{prefix}/",
#         )
#     elif args.test_task == 'text2mol':
#         logger.log_stats(
#             stats={
#                 "bleu": eval_metric["bleu"],
#                 "exact_match": eval_metric["exact_match"],
#                 "levenshtein": eval_metric["levenshtein"],
#                 "validity": eval_metric["validity"],
#                 "invalid selfies num": selfies_invalid,
#                 "RDK" : avg_RDK,
#                 "MACCS" : avg_MACCS,
#                 "Morgan" : avg_Morgan,
#                 "exact" : exact_canon,
#                 "time": time.time() - args.last_log,
#             },
#             step=args.current_train_step,
#             args=args,
#             prefix=f"{prefix}/",
#         )
#     elif args.test_task == 'text2frag':
#         logger.log_stats(
#             stats={
#                 "bleu": eval_metric["bleu"],
#                 "exact_match": eval_metric["exact_match"],
#                 "levenshtein": eval_metric["levenshtein"],
#                 "validity": eval_metric["validity"],
#                 "invalid selfies num": selfies_invalid,
#                 "time": time.time() - args.last_log,
#             },
#             step=args.current_train_step,
#             args=args,
#             prefix=f"{prefix}/",
#         )
#     elif args.test_task == 'dti' or args.test_task == 'peer' or args.test_task == 'molnet':
#         logger.log_stats(
#             stats={
#                 "accuracy": eval_metric["accuracy"],
#                 "auroc": eval_metric["auroc"],
#                 "auprc": eval_metric["auprc"],
#                 "sensitivity": eval_metric["sensitivity"],
#                 "specificity": eval_metric["specificity"],
#                 "f1": eval_metric["f1"],
#                 "thred_optim": eval_metric["thred_optim"],
#                 "precision": eval_metric["precision"],
#                 "time": time.time() - args.last_log,
#             },
#             step=args.current_train_step,
#             args=args,
#             prefix=f"{prefix}/",
#         )
#     else:
#         raise NotImplementedError

def maybe_save_checkpoint(accelerator: Accelerator, config: TrainConfig):
    if (
        config.current_state.train_step > config.optim_config.total_steps
        or config.current_state.train_step % config.checkpoint.every_steps == 0
    ):
        output_dir = f'checkpoint-{config.current_state.train_step}'
        accelerator.save_state(output_dir=output_dir)

def maybe_eval_predict(
    mode: str,
    model: Model,
    dataloader: DataLoader,
    config: TrainConfig,
    tokenizer: SpecialTokensMixin,
    accelerator: Accelerator,
    prefix='test'
):

    if (
        config.current_state.train_step > config.optim_config.total_steps
        or config.current_state.train_step % config.predict_config.every_steps == 0
    ):
        model.eval()

        with torch.no_grad():
            if mode == 'ft':
                predict(
                    model, dataloader, config, tokenizer, accelerator, prefix=prefix
                )

        config.current_state.last_log = time.time()
        model.train()
