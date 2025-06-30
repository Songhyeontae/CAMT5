from typing import List, Union, Optional

import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols

RDLogger.DisableLog('rdApp.*')
DUMMY_SMILES = "C"


def to_fingerprints(mols):
    """Convert molecules to Morgan fingerprints"""
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]
    return fps


def get_text2mol_metrics(
    predictions: Union[List[str], List[List[str]]],
    references: List[str],
    all_predictions: Optional[List[List[str]]] = None,
):
    #TODO(hyeontae): Remove hard-coded metrics
    s_rdk_list = []
    s_maccs_list = []
    s_morgan_list = []
    exact_canon_list = []

    invalid = 0

    # Handle both single predictions and multiple predictions per example
    if isinstance(predictions[0], str):
        # Single prediction per example
        predictions_list = [[pred] for pred in predictions]
    else:
        # Multiple predictions per example
        predictions_list = predictions

    # For metrics that need single predictions, use the first prediction
    single_predictions = [preds[0] if preds else DUMMY_SMILES for preds in predictions_list]

    for prediction, reference in zip(single_predictions, references):
        gen_mol = Chem.MolFromSmiles(prediction)
        if gen_mol == None:
            gen_mol = Chem.MolFromSmiles(DUMMY_SMILES)
        target_mol = Chem.MolFromSmiles(reference)
        if target_mol == None:
            target_mol = Chem.MolFromSmiles(DUMMY_SMILES)
        gen_smiles = Chem.MolToSmiles(gen_mol)
        target_smiles = Chem.MolToSmiles(target_mol)

        if gen_smiles == DUMMY_SMILES or target_smiles == DUMMY_SMILES:
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
    invalid_ratio = invalid / len(single_predictions)

    # Calculate new metrics
    similarity_score = calculate_similarity(single_predictions, references)  # Using MACCS + Cosine similarity
    validity_score = 1 - invalid_ratio
    
    # Calculate novelty and diversity
    novelty_score = 0.0
    diversity_score = 0.0
    
    # Use best predictions for novelty calculation
    if len(predictions_list) > 0:
        novelty_score = calculate_novelty(predictions_list, references)
    
    # Use all_predictions for diversity calculation if provided
    if all_predictions is not None and len(all_predictions) > 0:
        diversity_score = calculate_diversity(all_predictions)

    return {
        "RDK": avg_rdk,
        "MACCS": avg_maccs,
        "Morgan": avg_morgan,
        "exact": exact_canon,
        "example_cnt": len(exact_canon_list),
        "correct_cnt": sum(exact_canon_list),
        "invalid_ratio": invalid_ratio,
        "similarity": similarity_score,
        "novelty": novelty_score,
        "diversity": diversity_score,
        "validity": validity_score
    }


def calculate_similarity(predictions: List[str], references: List[str]) -> float:
    """
    Calculate similarity score using MACCS keys and Cosine similarity.
    Based on GitHub reference implementation.
    """
    if not predictions or not references:
        return 0.0
    
    similarity_scores = []
    
    for pred, ref in zip(predictions, references):
        try:
            pred_mol = Chem.MolFromSmiles(pred)
            ref_mol = Chem.MolFromSmiles(ref)
            
            if pred_mol is None or ref_mol is None:
                similarity_scores.append(0.0)
                continue
            
            # Kekulize molecules as in the reference code
            Chem.Kekulize(pred_mol)
            Chem.Kekulize(ref_mol)
            
            # Calculate MACCS similarity using Cosine similarity
            similarity = DataStructs.FingerprintSimilarity(
                MACCSkeys.GenMACCSKeys(ref_mol), 
                MACCSkeys.GenMACCSKeys(pred_mol), 
                metric=DataStructs.CosineSimilarity
            )
            
            similarity_scores.append(similarity)
            
        except Exception:
            similarity_scores.append(0.0)
    
    return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0


def calculate_novelty(predictions_list: List[List[str]], references: List[str]) -> float:
    """
    Calculate novelty score based on GitHub reference implementation.
    Novelty measures how different the generated molecules are from the reference molecules.
    Uses MACCS similarity thresholds: >=0.5 for qualified, >=0.5 and <0.8 for novel.
    """
    if not predictions_list or not references:
        return 0.0
    
    # For novelty calculation, we need to compare each prediction with its corresponding reference
    # Since we have multiple predictions per example, we'll use the first prediction for each example
    single_predictions = [preds[0] if preds else DUMMY_SMILES for preds in predictions_list]
    
    count_rd = 0  # Count of qualified samples (MACCS similarity >= 0.5)
    count_nv = 0  # Count of novel samples (MACCS similarity >= 0.5 and < 0.8)
    
    for pred, ref in zip(single_predictions, references):
        try:
            pred_mol = Chem.MolFromSmiles(pred)
            ref_mol = Chem.MolFromSmiles(ref)
            
            if pred_mol is None or ref_mol is None:
                continue
            
            # Kekulize molecules as in the reference code
            Chem.Kekulize(pred_mol)
            Chem.Kekulize(ref_mol)
            
            # Calculate MACCS similarity using Cosine similarity
            similarity = DataStructs.FingerprintSimilarity(
                MACCSkeys.GenMACCSKeys(ref_mol), 
                MACCSkeys.GenMACCSKeys(pred_mol), 
                metric=DataStructs.CosineSimilarity
            )
            
            if similarity >= 0.5:
                count_rd += 1
                if similarity < 0.8:
                    count_nv += 1
                    
        except Exception:
            continue
    
    # Calculate novelty as in the reference code
    novelty = count_nv / count_rd if count_rd > 0 else 0.0
    return novelty


def calculate_diversity(predictions_list: List[List[str]]) -> float:
    """
    Calculate diversity score for each example separately, then return the average.
    Diversity measures how different the generated molecules are from each other within each example.
    """
    if not predictions_list:
        return 0.0
    
    diversity_scores = []
    
    for example_predictions in predictions_list:
        # Convert predictions for this example to molecules
        mols = []
        for pred in example_predictions:
            mol = Chem.MolFromSmiles(pred)
            if mol is not None:
                mols.append(mol)
        
        if len(mols) <= 1:
            diversity_scores.append(0.0)
            continue
        
        # Calculate fingerprints for this example
        fps = to_fingerprints(mols)
        
        # Calculate diversity within this example
        example_diversity_scores = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            example_diversity_scores.extend([1 - s for s in sims])
        
        # Average diversity for this example
        if example_diversity_scores:
            example_avg_diversity = sum(example_diversity_scores) / len(example_diversity_scores)
            diversity_scores.append(example_avg_diversity)
        else:
            diversity_scores.append(0.0)
    
    # Return average diversity across all examples
    return sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0


def get_rdk_metric(prediction, reference) -> float:
    gen_mol = Chem.MolFromSmiles(prediction)
    if gen_mol == None:
        gen_mol = Chem.MolFromSmiles(DUMMY_SMILES)
    target_mol = Chem.MolFromSmiles(reference)
    gen_smiles = Chem.MolToSmiles(gen_mol)

    if gen_smiles == DUMMY_SMILES:
        return 0

    target_fp_rdk = FingerprintMols.GetRDKFingerprint(target_mol)
    gen_fp_rdk = FingerprintMols.GetRDKFingerprint(gen_mol)

    s_rdk = DataStructs.TanimotoSimilarity(gen_fp_rdk, target_fp_rdk)

    return s_rdk
