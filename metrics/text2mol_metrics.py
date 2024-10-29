from typing import List

import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols

RDLogger.DisableLog('rdApp.*')
DUMMY_SMILES = "C"


def get_text2mol_metrics(
    predictions: List[torch.Tensor],
    references: List[torch.Tensor],
):
    #TODO(hyeontae): Remove hard-coded metrics
    s_rdk_list = []
    s_maccs_list = []
    s_morgan_list = []
    exact_canon_list = []

    invalid = 0

    for prediction, reference in zip(predictions, references):
        gen_mol = Chem.MolFromSmiles(prediction)
        if gen_mol == None:
            gen_mol = Chem.MolFromSmiles(DUMMY_SMILES)
        target_mol = Chem.MolFromSmiles(reference)
        gen_smiles = Chem.MolToSmiles(gen_mol)
        target_smiles = Chem.MolToSmiles(target_mol)

        if gen_smiles == DUMMY_SMILES:
            invalid += 1
            s_rdk_list.append(0)
            s_maccs_list.append(0)
            s_morgan_list.append(0)
            exact_canon_list.append(0)
            continue

        target_fp_rdk = FingerprintMols.GetRDKFingerprint(target_mol)
        gen_fp_rdk = FingerprintMols.GetRDKFingerprint(gen_mol)

        target_fp_maccs = MACCSkeys.GenMACCSKeys(target_mol)
        gen_fp_maccs = MACCSkeys.GenMACCSKeys(gen_mol)

        fpgen = AllChem.GetMorganGenerator(radius=2)
        target_fp_morgan = fpgen.GetSparseCountFingerprint(target_mol)
        gen_fp_morgan = fpgen.GetSparseCountFingerprint(gen_mol)

        exact_match = (
            1 if \
                Chem.MolToInchi(Chem.MolFromSmiles(gen_smiles)) \
                    == Chem.MolToInchi(Chem.MolFromSmiles(target_smiles)) else 0
        )

        s_rdk = DataStructs.TanimotoSimilarity(gen_fp_rdk, target_fp_rdk)
        s_maccs = DataStructs.TanimotoSimilarity(gen_fp_maccs, target_fp_maccs)
        s_morgan = DataStructs.TanimotoSimilarity(gen_fp_morgan,
                                                  target_fp_morgan)

        s_rdk_list.append(s_rdk)
        s_maccs_list.append(s_maccs)
        s_morgan_list.append(s_morgan)
        exact_canon_list.append(exact_match)

    avg_rdk = sum(s_rdk_list) / (len(s_rdk_list))
    avg_maccs = sum(s_maccs_list) / (len(s_maccs_list))
    avg_morgan = sum(s_morgan_list) / (len(s_morgan_list))
    exact_canon = sum(exact_canon_list) / (len(exact_canon_list))
    invalid_ratio = invalid / len(predictions)

    return {
        "RDK": avg_rdk,
        "MACCS": avg_maccs,
        "Morgan": avg_morgan,
        "exact": exact_canon,
        "invalid_ratio": invalid_ratio
    }
