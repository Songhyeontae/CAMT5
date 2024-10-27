from typing import List
import torch
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys, AllChem

RDLogger.DisableLog('rdApp.*')

def get_text2mol_metrics(
    predictions: List[torch.Tensor],
    references: List[torch.Tensor],
    is_valid: List[bool],
):
    #TODO(hyeontae): Remove hard-coded metrics
    s_rdk_list = []
    s_maccs_list = []
    s_morgan_list = []
    exact_canon_list = []
    
    invalid = 0
    
    for prediction, reference, valid in zip(predictions, references, is_valid):
        # gen_mol =  Chem.MolFromSmiles(prediction)
        # target_mol = Chem.MolFromSmiles(reference)
        # gen_smiles = Chem.MolToSmiles(gen_mol)
        # target_smiles = Chem.MolToSmiles(target_mol)

        if not valid:
            invalid += 1
            s_rdk_list.append(0)
            s_maccs_list.append(0)
            s_morgan_list.append(0)
            exact_canon_list.append(0)
            continue
        
        target_fp_rdk = FingerprintMols.GetRDKFingerprint(reference)
        gen_fp_rdk = FingerprintMols.GetRDKFingerprint(prediction)
        
        target_fp_maccs = MACCSkeys.GenMACCSKeys(reference)
        gen_fp_maccs = MACCSkeys.GenMACCSKeys(prediction)
        
        fpgen = AllChem.GetMorganGenerator(radius=2)
        target_fp_morgan = fpgen.GetSparseCountFingerprint(reference)
        gen_fp_morgan = fpgen.GetSparseCountFingerprint(prediction)
        
        exact_match = (
            1 if \
                Chem.MolToInchi(Chem.MolFromSmiles(prediction)) \
                    == Chem.MolToInchi(Chem.MolFromSmiles(reference)) else 0
        )
        
        s_rdk = DataStructs.TanimotoSimilarity(gen_fp_rdk, target_fp_rdk)
        s_maccs = DataStructs.TanimotoSimilarity(gen_fp_maccs, target_fp_maccs)
        s_morgan = DataStructs.TanimotoSimilarity(gen_fp_morgan, target_fp_morgan)
        
        s_rdk_list.append(s_rdk)
        s_maccs_list.append(s_maccs)
        s_morgan_list.append(s_morgan)
        exact_canon_list.append(exact_match)
       
    avg_rdk = sum(s_rdk_list) / (len(s_rdk_list)) 
    avg_maccs = sum(s_maccs_list) / (len(s_maccs_list))
    avg_morgan = sum(s_morgan_list) / (len(s_morgan_list))
    exact_canon = sum(exact_canon_list) / (len(exact_canon_list))
    ivalid_ratio = invalid / len(is_valid)
    
    return {
        "RDK": avg_rdk,
        "MACCS": avg_maccs,
        "Morgan": avg_morgan,
        "exact": exact_canon,
        "invalid_ratio": ivalid_ratio
    }
