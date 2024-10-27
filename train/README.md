# CAMT5 Training

This is task directory which contains the training (pretrain, finetune).

# Settings
Before start, should set `.env`, `.data`
```bash
# .env
EXTERNAL_PATH="path/to/your/external/dir" # Use for save/load model checkpoint
```

```
.data is directory for store/read train data (especially finetune data)
(ref: /home/osikjs/BioT5/biot5/biot5/data)

.data/
├── ni_dataset.py
├── splits/
└── tasks/
```

## Usage
To start training the model, use the following command:
```bash
python main.py task=train/{pretrain/finetune} task/train/exp={exp_name}
```
or
```bash
./scripts/{pretrain/finetune}.sh -e {exp_name} # -n {nnodes} -g {nproc_per_node} -p {master_port}
```

## Experiment
Experiment Configurations Path: `config/task/train/exp`

This configurations will override `config/task/train/{pretrain/finetune}.yaml`