import logging
import re
from itertools import product
from typing import List, NewType, Protocol

import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops
from scipy.stats import rankdata

from model.config import RepresentationType

logger = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')

SMILES = NewType("SMILES", str)

DUMMY_SMILES = "C"
ATOM_FINDER = re.compile(
    r"""
(
 Cl? |             # Cl and Br are part of the organic subset
 Br? |
 [NOSPFIbcnosp] | # as are these single-letter elements
 \[[^]]+\]         # everything else must be in []s
)
""",
    re.X,
)

CHIRAL_TOKENS = {
    "[C@]": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    "[C@@]": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    "[C@H]": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    "[C@@H]": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    "[N@+]": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    "[N@@+]": Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
}


class Representation(Protocol):

    def encode(self, mol: SMILES, verbose=False) -> str:
        ...

    def decode(self, text_mol: str, verbose=False) -> SMILES:
        ...

    def get_size(self, text_mol: str) -> int:
        ...


class Smiles(Representation):

    def encode(self, mol: SMILES, verbose=False) -> str:
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol),
                                      kekuleSmiles=True)
        except:
            if verbose:
                logger.warning(f"Failed to encode SMILES: {mol}")
            smiles = DUMMY_SMILES
        return smiles

    def decode(self, text_mol: str, verbose=False) -> SMILES:
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(text_mol),
                                      kekuleSmiles=True)
        except:
            if verbose:
                logger.warning(f"Failed to decode SMILES: {text_mol}")
            smiles = DUMMY_SMILES
        return smiles

    def get_size(self, text_mol: str) -> int:
        smiles = self.decode(text_mol)
        mol = Chem.MolFromSmiles(smiles)
        atom_cnts = mol.GetNumAtoms()
        return atom_cnts


class Selfies(Representation):

    def encode(self, mol: SMILES, verbose=False) -> str:
        try:
            selfies = sf.encoder(
                Chem.MolToSmiles(Chem.MolFromSmiles(mol), kekuleSmiles=True))
        except:
            if verbose:
                logger.warning(f"Failed to encode SELFIES: {mol}")
            selfies = sf.encoder(DUMMY_SMILES)
        return selfies

    def decode(self, text_mol: str, verbose=False) -> SMILES:
        try:
            text_mol = text_mol.replace(" ", "")
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(sf.decoder(text_mol)),
                                      kekuleSmiles=True)
        except:
            if verbose:
                logger.warning(f"Failed to decode SELFIES: {text_mol}")
            try:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(
                    self._filter_selfies(text_mol)),
                                          kekuleSmiles=True)
            except:
                smiles = DUMMY_SMILES

        return smiles

    def get_size(self, text_mol: str) -> int:
        smiles = self.decode(text_mol)
        mol = Chem.MolFromSmiles(smiles)
        atom_cnts = mol.GetNumAtoms()
        return atom_cnts

    def _filter_selfies(selfies: str) -> str:
        pattern = r"(\[[^\]]+\]\.?)"
        matches = re.findall(pattern, selfies)
        return "".join(matches)


class Frag(Representation):

    def encode(self, mol: SMILES, verbose=False) -> str:
        try:
            linear_smiles = ""
            for smiles in mol.split("."):
                linear_smiles += linearize(smiles) + "[.]"
            linear_smiles = linear_smiles[:-3]
        except:
            if verbose:
                logger.warning(f"Failed to encode Frag: {mol}")
            linear_smiles = "[C]"
        return linear_smiles

    def decode(self, text_mol: str, verbose=False) -> SMILES:
        try:
            decoded_smiles = ""
            for smiles in text_mol.split("[.]"):
                decoded_smiles += decode_linear(smiles) + "."
            decoded_smiles = decoded_smiles[:-1]
        except:
            if verbose:
                logger.warning(f"Failed to decode Frag: {text_mol}")
            decoded_smiles = DUMMY_SMILES
        return decoded_smiles

    def get_size(self, text_mol: str) -> int:
        smiles = self.decode(text_mol)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        atom_cnts = mol.GetNumAtoms()
        return atom_cnts


def get_representation(representation_type) -> Representation:

    representation_dict = {
        RepresentationType.SMILES.value: Smiles,
        RepresentationType.SELFIES.value: Selfies,
        RepresentationType.FRAG.value: Frag,
    }

    return representation_dict[representation_type]()


def get_importance(
    tokens: List[str],
    representation: Representation,
) -> List[int]:
    token_importance = []
    for token in tokens:
        size = representation.get_size(token)
        token_importance.append(size)
    return token_importance


def find_order(smile):
    i = 0
    while True:
        if f"<{i}*>" not in smile:
            break
        i += 1
    return i


def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find + len(sub):]
    return s


def linearize(smile: SMILES) -> str:
    frag_dict = ["[.]"]
    chiral_order_list = []
    ref_chiral_order_list = []

    mol = Chem.MolFromSmiles(smile)
    find_chiral = Chem.FindMolChiralCenters(Chem.MolFromSmiles(smile),
                                            useLegacyImplementation=False)

    chiral_list = [None] * len(mol.GetAtoms())
    for idx, chiral in find_chiral:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetTotalNumHs() == 1:
            chiral_list[idx] = chiral + "H"
        else:
            chiral_list[idx] = chiral

    find_stereo_bond = []
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            bond_begin_atom_idx = bond.GetBeginAtomIdx()
            bond_end_atom_idx = bond.GetEndAtomIdx()
            find_stereo_bond.append(
                [bond_begin_atom_idx, bond_end_atom_idx,
                 bond.GetStereo()])

    orig_smile = smile

    smile = smile.replace("[C@H]", "C")
    smile = smile.replace("[C@@H]", "C")
    smile = smile.replace("[C@]", "C")
    smile = smile.replace("[C@@]", "C")
    smile = smile.replace("[N@+]", "[N+]")
    smile = smile.replace("[N@@+]", "[N+]")
    smile = smile.replace("[S@@]", "S")
    smile = smile.replace("[S@]", "S")
    smile = smile.replace("/", "")
    smile = smile.replace("\\", "")

    m = Chem.MolFromSmiles(smile)
    m_orig = Chem.MolFromSmiles(smile)

    res = []
    all_bonds = []
    for bond in m.GetBonds():
        bond_idx = bond.GetIdx()
        bond_begin_atom_idx = bond.GetBeginAtomIdx()
        bond_end_atom_idx = bond.GetEndAtomIdx()

        if (m.GetAtomWithIdx(bond_begin_atom_idx).IsInRing()
                and m.GetAtomWithIdx(bond_end_atom_idx).IsInRing()):
            continue

        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue

        res.append(((bond_begin_atom_idx, bond_end_atom_idx), (0, 0)))
        all_bonds.append(bond_idx)

    if len(all_bonds) == 0:
        frag_string = orig_smile
        frag_string = "[" + frag_string + "]"
        if frag_string not in frag_dict:
            frag_dict.append(frag_string)

        return frag_string

    m2 = Chem.FragmentOnBonds(m, all_bonds)

    frags = rdmolops.GetMolFrags(m2)
    for frag in frags:
        tmp = []
        for l in frag:
            if l < len(m_orig.GetAtoms()):
                tmp.append(l)
        ref_chiral_order_list.append(tmp)

    smiles_list = []
    permute_list = []
    for frag in frags:
        m = Chem.Mol()
        em = Chem.EditableMol(m)
        em = Chem.RWMol()
        map_dict = {}

        for idx in frag:
            new_idx = em.AddAtom(Chem.Atom(m2.GetAtoms()[idx].GetAtomicNum()))
            map_dict[idx] = new_idx

        for idx1, idx2 in product(frag, frag):
            if idx1 >= idx2:
                continue
            try:
                em.AddBond(
                    map_dict[idx1],
                    map_dict[idx2],
                    m2.GetBondBetweenAtoms(idx1, idx2).GetBondType(),
                )
                new_bond = em.GetBondBetweenAtoms(map_dict[idx1],
                                                  map_dict[idx2])
                new_bond.SetBondDir(
                    m2.GetBondBetweenAtoms(idx1, idx2).GetBondDir())
            except:
                continue
        m = em.GetMol()

        for idx in frag:
            m.GetAtomWithIdx(map_dict[idx]).SetNumExplicitHs(
                m2.GetAtoms()[idx].GetNumExplicitHs())
            m.GetAtomWithIdx(map_dict[idx]).SetFormalCharge(
                m2.GetAtoms()[idx].GetFormalCharge())
            if m2.GetAtoms()[idx].GetAtomicNum() != 0:
                m.GetAtomWithIdx(map_dict[idx]).SetNumRadicalElectrons(
                    m2.GetAtoms()[idx].GetNumRadicalElectrons())
                m.GetAtomWithIdx(map_dict[idx]).SetIsotope(
                    m2.GetAtoms()[idx].GetIsotope())

        Chem.SanitizeMol(m)

        canonical_atom_order = tuple(Chem.CanonicalRankAtoms(m))
        canonical_atom_order_inverted = list(
            tuple(
                zip(*sorted(
                    (j, i) for i, j in enumerate(canonical_atom_order))))[1])

        frag_smile = Chem.MolToSmiles(m, kekuleSmiles=True)
        ms = Chem.MolFromSmiles(frag_smile)
        canonical_atom_order2 = tuple(Chem.CanonicalRankAtoms(ms))
        canonical_atom_order_inverted2 = list(
            tuple(
                zip(*sorted(
                    (j, i) for i, j in enumerate(canonical_atom_order2))))[1])

        zero_idx = []
        for i, ord in enumerate(canonical_atom_order_inverted):
            if m.GetAtoms()[ord].GetAtomicNum() == 0:
                zero_idx.append(ord)
        zero_idx2 = []
        for i, ord in enumerate(canonical_atom_order_inverted2):
            if ms.GetAtoms()[ord].GetAtomicNum() == 0:
                zero_idx2.append(ord)

        nonzero_num = len(canonical_atom_order_inverted) - len(zero_idx)

        nonzero_idx = []
        for i in range(nonzero_num):
            corr_idx = canonical_atom_order_inverted.index(i)
            corr_idx2 = canonical_atom_order_inverted2[corr_idx]
            nonzero_idx.append(corr_idx2)

        # Create a list of tuples (index, value) and sort it based on the values
        sorted_nonzero_indices = sorted(enumerate(nonzero_idx),
                                        key=lambda x: x[1])

        # Create a dictionary to store ranks
        ranks_nonzero_dict = {
            index: i
            for i, (index, value) in enumerate(sorted_nonzero_indices)
        }

        # Create list B using the ranks_dict
        nonzero_idx_rank = [
            ranks_nonzero_dict[index]
            for index, value in enumerate(nonzero_idx)
        ]

        chiral_order_list.append(nonzero_idx_rank)

        # Create a list of tuples (index, value) and sort it based on the values
        sorted_indices = sorted(enumerate(zero_idx), key=lambda x: x[1])

        # Create a dictionary to store ranks
        ranks_dict = {
            index: i
            for i, (index, value) in enumerate(sorted_indices)
        }

        # Create list B using the ranks_dict
        zero_idx_rank = [
            ranks_dict[index] for index, value in enumerate(zero_idx)
        ]

        sorted_indices2 = sorted(enumerate(zero_idx2), key=lambda x: x[1])

        # Create a dictionary to store ranks
        ranks_dict2 = {
            index: i
            for i, (index, value) in enumerate(sorted_indices2)
        }

        # Create list B using the ranks_dict
        zero_idx_rank2 = [
            ranks_dict2[index] for index, value in enumerate(zero_idx2)
        ]

        permute = []

        for i in range(len(zero_idx_rank2)):
            cor_idx = zero_idx_rank.index(i)
            permute.append(zero_idx_rank2[cor_idx])

        permute_list.append(permute)

        smiles_list.append(frag_smile)

    dfs_output = []

    def dfs(graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)

        dfs_output.append(start)

        for next in graph[start] - visited:
            dfs(graph, next, visited)
        return visited

    graph = {}

    frag_indices = rdmolops.GetMolFrags(m2)
    atom_num = Chem.MolFromSmiles(smile).GetNumAtoms()

    idx = atom_num
    dummy_cnt = []

    for frag_index in frag_indices:
        tmp = []
        for idx in frag_index:
            if idx >= atom_num:
                neighbor = m2.GetAtomWithIdx(idx).GetNeighbors()
                assert len(neighbor) == 1

                tmp.append(neighbor[0].GetIdx())
        dummy_cnt.append(tmp)

    matching_list = []

    for ii in range(len(frag_indices)):
        graph[str(ii)] = []

    for indices, _ in res:
        atom_idx1, atom_idx2 = indices

        for ii in range(len(frag_indices)):
            if atom_idx1 in frag_indices[ii]:
                assert atom_idx1 in dummy_cnt[ii]
                index_in_frag1 = permute_list[ii][dummy_cnt[ii].index(
                    atom_idx1)]
                frag_idx1 = ii

            if atom_idx2 in frag_indices[ii]:
                assert atom_idx2 in dummy_cnt[ii]
                index_in_frag2 = permute_list[ii][dummy_cnt[ii].index(
                    atom_idx2)]

                frag_idx2 = ii

        matching_list.append(
            [frag_idx1, frag_idx2, index_in_frag1, index_in_frag2])

        graph[str(frag_idx1)].append(str(frag_idx2))
        graph[str(frag_idx2)].append(str(frag_idx1))

    tmp_matching = []
    for i, matching in enumerate(matching_list):
        frag_idx1, frag_idx2, index_in_frag1, index_in_frag2 = matching
        while [frag_idx1, index_in_frag1] in tmp_matching:
            index_in_frag1 += 1
        while [frag_idx2, index_in_frag2] in tmp_matching:
            index_in_frag2 += 1
        matching_list[i] = [
            frag_idx1, frag_idx2, index_in_frag1, index_in_frag2
        ]

        tmp_matching.append([frag_idx1, index_in_frag1])
        tmp_matching.append([frag_idx2, index_in_frag2])

    for ii in range(len(frag_indices)):
        graph[str(ii)] = set(graph[str(ii)])

    dfs(graph, "0")

    frag_list = []
    done_list = []
    processed_chiral_order_list = []
    for ref_chiral, chiral in zip(ref_chiral_order_list, chiral_order_list):
        tmp[:] = ref_chiral[:]
        for idx1, idx2 in enumerate(chiral):
            tmp[idx2] = ref_chiral[idx1]
        processed_chiral_order_list.append(tmp[:])

    processed_chiral_order_list_after_dfs = []

    for output in dfs_output:
        processed_chiral_order_list_after_dfs.append(
            processed_chiral_order_list[int(output)])

    visited_list = [0] * len(dfs_output)
    for order in dfs_output:
        smile = smiles_list[int(order)]

        visited_list[int(order)] = 1
        current_matching_list = []
        current_matching_list_order = []

        for matching in matching_list:
            idx1, idx2, ord1, ord2 = matching[0], matching[1], matching[
                2], matching[3]
            if int(order) == idx1:
                current_matching_list.append(
                    [matching[0], matching[1], matching[2], matching[3]])
                current_matching_list_order.append(dfs_output.index(str(idx2)))
            if int(order) == idx2:
                current_matching_list.append(
                    [matching[0], matching[1], matching[2], matching[3]])
                current_matching_list_order.append(dfs_output.index(str(idx1)))

        current_matching_list_order = [
            int(rank) - 1 for rank in rankdata(current_matching_list_order)
        ]
        ordered_current_matching_list = []

        for i in range(len(current_matching_list)):
            ordered_current_matching_list.append(
                current_matching_list[current_matching_list_order.index(i)])

        for matching in ordered_current_matching_list:
            idx1, idx2, ord1, ord2 = matching[0], matching[1], matching[
                2], matching[3]
            if int(order) == idx1:
                if [idx2, ord2, idx1, ord1] not in done_list and [
                        idx1,
                        ord1,
                        idx2,
                        ord2,
                ] not in done_list:
                    done_list.append([idx2, ord2, idx1, ord1])
                    done_list.append([idx1, ord1, idx2, ord2])

                    frag_order = find_order(smile)

                    smile = nth_repl(smile, "*", f"<{frag_order}*>", ord1 + 1)

                    smiles_list[int(order)] = smile

                    other_smile = smiles_list[idx2]
                    other_frag_order = find_order(other_smile)
                    other_smile = nth_repl(other_smile, "*",
                                           f"<{other_frag_order}*>", ord2 + 1)
                    smiles_list[idx2] = other_smile
            if int(order) == idx2:
                if [idx2, ord2, idx1, ord1] not in done_list and [
                        idx1,
                        ord1,
                        idx2,
                        ord2,
                ] not in done_list:
                    done_list.append([idx2, ord2, idx1, ord1])
                    done_list.append([idx1, ord1, idx2, ord2])
                    frag_order = find_order(smile)

                    smile = nth_repl(smile, "*", f"<{frag_order}*>", ord2 + 1)

                    smiles_list[int(order)] = smile

                    other_smile = smiles_list[idx1]
                    other_frag_order = find_order(other_smile)
                    other_smile = nth_repl(other_smile, "*",
                                           f"<{other_frag_order}*>", ord1 + 1)
                    smiles_list[idx1] = other_smile

        frag_list.append(smiles_list[int(order)])
    frag_string = ""

    for i, frag in enumerate(frag_list):
        ms = [x for x in ATOM_FINDER.finditer(frag)]
        individual_atoms = [frag[x.start():x.end()] for x in ms]

        chiral_tag_list = []
        for j in range(len(processed_chiral_order_list_after_dfs[i])):
            processed_idx = chiral_list[
                processed_chiral_order_list_after_dfs[i][j]]
            if individual_atoms[j] == "C" and processed_idx == "R":
                chiral_tag_list.append(["C", "[C@]", ms[j].start()])
            elif individual_atoms[j] == "C" and processed_idx == "S":
                chiral_tag_list.append(["C", "[C@@]", ms[j].start()])
            elif individual_atoms[j] == "C" and processed_idx == "RH":
                chiral_tag_list.append(["C", "[C@H]", ms[j].start()])
            elif individual_atoms[j] == "C" and processed_idx == "SH":
                chiral_tag_list.append(["C", "[C@@H]", ms[j].start()])
            elif individual_atoms[j] == "[N+]" and processed_idx == "R":
                chiral_tag_list.append(["[N+]", "[N@+]", ms[j].start()])
            elif individual_atoms[j] == "[N+]" and processed_idx == "S":
                chiral_tag_list.append(["[N+]", "[N@@+]", ms[j].start()])
            elif individual_atoms[j] == "S" and processed_idx == "R":
                chiral_tag_list.append(["S", "[S@]", ms[j].start()])
            elif individual_atoms[j] == "S" and processed_idx == "S":
                chiral_tag_list.append(["S", "[S@@]", ms[j].start()])
            elif individual_atoms[j] == "P" and processed_idx == "R":
                chiral_tag_list.append(["P", "[P@]", ms[j].start()])
            elif individual_atoms[j] == "P" and processed_idx == "S":
                chiral_tag_list.append(["P", "[P@@]", ms[j].start()])

        chiral_tag_list.sort(key=lambda x: -x[2])
        for tmp in chiral_tag_list:
            if tmp[0] == "C" and tmp[1] == "[C@]":
                frag = frag[:tmp[2]] + "[C@]" + frag[tmp[2] + 1:]
            elif tmp[0] == "C" and tmp[1] == "[C@@]":
                frag = frag[:tmp[2]] + "[C@@]" + frag[tmp[2] + 1:]
            elif tmp[0] == "C" and tmp[1] == "[C@H]":
                frag = frag[:tmp[2]] + "[C@H]" + frag[tmp[2] + 1:]
            elif tmp[0] == "C" and tmp[1] == "[C@@H]":
                frag = frag[:tmp[2]] + "[C@@H]" + frag[tmp[2] + 1:]
            elif tmp[0] == "[N+]" and tmp[1] == "[N@+]":
                frag = frag[:tmp[2] + 1] + "N@" + frag[tmp[2] + 2:]
            elif tmp[0] == "[N+]" and tmp[1] == "[N@@+]":
                frag = frag[:tmp[2] + 1] + "N@@" + frag[tmp[2] + 2:]
            elif tmp[0] == "S" and tmp[1] == "[S@]":
                frag = frag[:tmp[2]] + "[S@]" + frag[tmp[2] + 1:]
            elif tmp[0] == "S" and tmp[1] == "[S@@]":
                frag = frag[:tmp[2]] + "[S@@]" + frag[tmp[2] + 1:]
            elif tmp[0] == "P" and tmp[1] == "[P@]":
                frag = frag[:tmp[2]] + "[P@]" + frag[tmp[2] + 1:]
            elif tmp[0] == "P" and tmp[1] == "[P@@]":
                frag = frag[:tmp[2]] + "[P@@]" + frag[tmp[2] + 1:]

        ms = [x for x in ATOM_FINDER.finditer(frag)]
        individual_atoms = [frag[x.start():x.end()] for x in ms]
        bond_mark_list = []

        for bond in find_stereo_bond:
            cur_bond_list = []
            if bond[0] in processed_chiral_order_list_after_dfs[i]:
                cur_bond_list.append(bond)
            double_bond_atom_list = []
            for bond in cur_bond_list:
                bond_idx0 = processed_chiral_order_list_after_dfs[i].index(
                    bond[0])
                bond_idx1 = processed_chiral_order_list_after_dfs[i].index(
                    bond[1])
                if bond_idx0 > bond_idx1:
                    bond_idx0, bond_idx1 = bond_idx1, bond_idx0

                if (bond[0] not in double_bond_atom_list
                        and bond[1] not in double_bond_atom_list):
                    bond_mark_list.append([bond_idx0, bond_idx1, bond[2]])
                    bond_mark_list.append([bond_idx1, bond_idx0, bond[2]])

                double_bond_atom_list.append(bond[0])
                double_bond_atom_list.append(bond[1])
        bond_mark_list.sort(key=lambda x: -x[0])

        for bond in bond_mark_list:
            if bond[2] == Chem.rdchem.BondStereo.STEREOZ:
                frag = (
                    frag[:ms[bond[0]].start()] +
                    f"[{frag[ms[bond[0]].start():ms[bond[0]].end()]}:{bond[1]}:Z]"
                    + frag[ms[bond[0]].end():])
            if bond[2] == Chem.rdchem.BondStereo.STEREOE:
                frag = (
                    frag[:ms[bond[0]].start()] +
                    f"[{frag[ms[bond[0]].start():ms[bond[0]].end()]}:{bond[1]}:E]"
                    + frag[ms[bond[0]].end():])

        frag_string += "[" + frag + "]"

        if "[" + frag + "]" in frag_dict:
            continue
        else:
            frag_dict.append("[" + frag + "]")

    frag_string = frag_string.replace("[<1*>O<0*>]", "[<0*>O<1*>]")
    frag_string = frag_string.replace("[<1*>N<0*>]", "[<0*>N<1*>]")
    return frag_string


def decode_linear(linear_smiles: str) -> SMILES:
    frags = []
    opened = 0
    tmp_frag = ""
    chiral_list = []
    bond_stereo_list = []

    for s in linear_smiles:
        if opened == 0 and s != "[":
            continue
        if s == "[" and opened == 0:
            opened += 1
        elif s == "[" and opened != 0:
            opened += 1
            tmp_frag += s
        elif s == "]" and opened != 1:
            opened -= 1
            tmp_frag += s
        elif s == "]" and opened == 1:
            opened -= 1
            frags.append(tmp_frag)
            tmp_frag = ""
        else:
            tmp_frag += s

    links = [s.count("*") for s in frags]
    if sum(links) == 0:
        return linear_smiles[1:-1]
    tmp_frags = []
    order_list = []

    for x in frags:
        order = []
        i = 0
        while i + 1 < len(x):
            if x[i] == "<":
                k = 1
                while True:
                    if x[i + k] == "*":
                        break
                    k += 1

                order.append(x[i + 1:i + k])
            i += 1
        for i in range(x.count("*")):
            x = x.replace(f"<{i}*>", "*")
        try:
            smile = x
            ms = [x for x in ATOM_FINDER.finditer(smile)]
            individual_atoms = [smile[x.start():x.end()] for x in ms]

            for atom in individual_atoms:
                if atom in CHIRAL_TOKENS.keys():
                    chiral_list.append(atom)
                else:
                    chiral_list.append(None)

            total_atoms = len(ms)
            tmp_list = []
            offset = len(bond_stereo_list)
            for a_idx, m in enumerate(reversed(ms)):
                a_idx = total_atoms - a_idx - 1
                if ":" in smile[m.start():m.end()]:
                    splited = smile[m.start():m.end()].split(":")
                    atom = splited[0][1:]
                    target = splited[1]
                    attr = splited[2][:-1]
                    smile = smile[:m.start()] + atom + smile[m.end():]
                    tmp_list.append(
                        [atom, offset + a_idx, offset + int(target), attr])
                else:
                    tmp_list.append(None)

            bond_stereo_list += list(reversed(tmp_list))

            smile = smile.replace("[C@]", "C")
            smile = smile.replace("[C@@]", "C")
            smile = smile.replace("[C@H]", "C")
            smile = smile.replace("[C@@H]", "C")
            smile = smile.replace("[N@+]", "[N+]")
            smile = smile.replace("[N@@+]", "[N+]")
            smile = smile.replace("[S@]", "S")
            smile = smile.replace("[S@@]", "S")
            smile = smile.replace("[P@]", "P")
            smile = smile.replace("[P@@]", "P")
            smile = smile.replace("/", "")
            smile = smile.replace("\\", "")

            tmp_frags.append(Chem.MolFromSmiles(smile))
            order_list.append(order)

        except:
            continue
    frags = tmp_frags

    def dfs(ptr, path):
        orig_ptr = ptr
        ptr = ptr + 1
        for ii in range(links[orig_ptr]):
            try:
                links[orig_ptr] -= 1
                links[ptr] -= 1
            except:
                continue
            if links[ptr] < 0:
                continue
            path.append([orig_ptr, ptr, ii])
            ptr, path = dfs(ptr, path)

        return ptr, path

    _, paths = dfs(0, [])
    m = Chem.Mol()
    em = Chem.EditableMol(m)
    em = Chem.RWMol()
    map_dicts = []

    dummy_atoms = []
    dummy_bonds = []
    for i, frag in enumerate(frags):
        dummy_atom = []
        dummy_bond = []
        map_dict = {}

        for ii, atom in enumerate(frag.GetAtoms()):
            if atom.GetAtomicNum() != 0:
                new_idx = em.AddAtom(Chem.Atom(atom.GetAtomicNum()))

                map_dict[ii] = new_idx
            else:
                dummy_atom.append(ii)
        dummy_atoms.append(dummy_atom)
        for idx1, idx2 in product(range(len(frag.GetAtoms())),
                                  range(len(frag.GetAtoms()))):
            if idx1 in dummy_atom:
                try:
                    bond_type = frag.GetBondBetweenAtoms(idx1,
                                                         idx2).GetBondType()
                    bond_dir = frag.GetBondBetweenAtoms(idx1,
                                                        idx2).GetBondDir()
                    dummy_bond.append((bond_type, idx1, idx2, bond_dir))
                except:
                    continue
            if idx1 >= idx2:
                continue

            try:
                em.AddBond(
                    map_dict[idx1],
                    map_dict[idx2],
                    frag.GetBondBetweenAtoms(idx1, idx2).GetBondType(),
                )
                new_bond = em.GetBondBetweenAtoms(map_dict[idx1],
                                                  map_dict[idx2])
                new_bond.SetBondDir(
                    frag.GetBondBetweenAtoms(idx1, idx2).GetBondDir())
            except:
                continue
        ordered_dummy_bond = [None] * len(order_list[i])
        for j in range(len(order_list[i])):
            ordered_dummy_bond[j] = dummy_bond[order_list[i].index(str(j))]
        dummy_bond = ordered_dummy_bond

        dummy_bonds.append(ordered_dummy_bond)
        map_dicts.append(map_dict)

    for path in paths:
        pop0 = dummy_bonds[path[0]].pop(0)
        pop1 = dummy_bonds[path[1]].pop(0)

        bond_type = pop0[0]
        bond_dir = pop0[3]

        if pop0[0] != pop1[0]:
            bond_type = Chem.rdchem.BondType.SINGLE
        if bond_dir == Chem.rdchem.BondDir.NONE:
            bond_dir = pop1[3]

        em.AddBond(map_dicts[path[0]][pop0[2]], map_dicts[path[1]][pop1[2]],
                   bond_type)
        new_bond = em.GetBondBetweenAtoms(map_dicts[path[0]][pop0[2]],
                                          map_dicts[path[1]][pop1[2]])

        new_bond.SetBondDir(bond_dir)
    m = em.GetMol()

    for ii in range(len(frags)):
        for idx in map_dicts[ii].keys():
            m.GetAtomWithIdx(map_dicts[ii][idx]).SetNumExplicitHs(
                frags[ii].GetAtoms()[idx].GetNumExplicitHs())
            m.GetAtomWithIdx(map_dicts[ii][idx]).SetFormalCharge(
                frags[ii].GetAtoms()[idx].GetFormalCharge())
            m.GetAtomWithIdx(map_dicts[ii][idx]).SetNumRadicalElectrons(
                frags[ii].GetAtoms()[idx].GetNumRadicalElectrons())
            m.GetAtomWithIdx(map_dicts[ii][idx]).SetIsotope(
                frags[ii].GetAtoms()[idx].GetIsotope())
    try:
        Chem.SanitizeMol(m)
    except:
        for a in m.GetAtoms():
            m.UpdatePropertyCache(strict=False)
            a.SetNumExplicitHs(a.GetNumImplicitHs())
            a.SetNoImplicit(True)
        Chem.SanitizeMol(m)
    result_smiles = Chem.MolToSmiles(m, kekuleSmiles=True)

    canonical_atom_order = tuple(Chem.CanonicalRankAtoms(m))
    canonical_atom_order_inverted = list(
        tuple(zip(*sorted(
            (j, i) for i, j in enumerate(canonical_atom_order))))[1])

    result_smiles = Chem.MolToSmiles(m, kekuleSmiles=True)
    m_after_kekule = Chem.MolFromSmiles(result_smiles)
    canonical_atom_order2 = tuple(Chem.CanonicalRankAtoms(m_after_kekule))
    canonical_atom_order_inverted2 = list(
        tuple(zip(*sorted(
            (j, i) for i, j in enumerate(canonical_atom_order2))))[1])

    ms = [x for x in ATOM_FINDER.finditer(result_smiles)]
    individual_atoms = [result_smiles[x.start():x.end()] for x in ms]
    chiral_tag_list = []

    for i in range(len(chiral_list)):
        inverted_idx = chiral_list[canonical_atom_order_inverted[
            canonical_atom_order_inverted2.index(i)]]
        if individual_atoms[i] == "C" and inverted_idx == "[C@]":
            chiral_tag_list.append(["C", "[C@]", ms[i].start()])
        if individual_atoms[i] == "C" and inverted_idx == "[C@@]":
            chiral_tag_list.append(["C", "[C@@]", ms[i].start()])
        if individual_atoms[i] == "C" and inverted_idx == "[C@H]":
            chiral_tag_list.append(["C", "[C@H]", ms[i].start()])
        if individual_atoms[i] == "C" and inverted_idx == "[C@@H]":
            chiral_tag_list.append(["C", "[C@@H]", ms[i].start()])
        if individual_atoms[i] == "[N+]" and inverted_idx == "[N@+]":
            chiral_tag_list.append(["[N+]", "[N@+]", ms[i].start()])
        if individual_atoms[i] == "[N+]" and inverted_idx == "[N@@+]":
            chiral_tag_list.append(["[N+]", "[N@@+]", ms[i].start()])
        if individual_atoms[i] == "S" and inverted_idx == "[S@]":
            chiral_tag_list.append(["S", "[S@]", ms[i].start()])
        if individual_atoms[i] == "S" and inverted_idx == "[S@@]":
            chiral_tag_list.append(["S", "[S@@]", ms[i].start()])
        if individual_atoms[i] == "P" and inverted_idx == "[P@]":
            chiral_tag_list.append(["P", "[P@]", ms[i].start()])
        if individual_atoms[i] == "P" and inverted_idx == "[P@@]":
            chiral_tag_list.append(["P", "[P@@]", ms[i].start()])

    chiral_tag_list.sort(key=lambda x: -x[2])
    for tmp in chiral_tag_list:
        if tmp[0] == "C" and tmp[1] == "[C@]":
            result_smiles = (result_smiles[:tmp[2]] + "[C@]" +
                             result_smiles[tmp[2] + 1:])
        if tmp[0] == "C" and tmp[1] == "[C@@]":
            result_smiles = (result_smiles[:tmp[2]] + "[C@@]" +
                             result_smiles[tmp[2] + 1:])
        if tmp[0] == "C" and tmp[1] == "[C@H]":
            result_smiles = (result_smiles[:tmp[2]] + "[C@H]" +
                             result_smiles[tmp[2] + 1:])
        if tmp[0] == "C" and tmp[1] == "[C@@H]":
            result_smiles = (result_smiles[:tmp[2]] + "[C@@H]" +
                             result_smiles[tmp[2] + 1:])
        if tmp[0] == "[N+]" and tmp[1] == "[N@+]":
            result_smiles = (result_smiles[:tmp[2] + 1] + "N@" +
                             result_smiles[tmp[2] + 2:])
        if tmp[0] == "[N+]" and tmp[1] == "[N@@+]":
            result_smiles = (result_smiles[:tmp[2] + 1] + "N@@" +
                             result_smiles[tmp[2] + 2:])
        if tmp[0] == "S" and tmp[1] == "[S@]":
            result_smiles = (result_smiles[:tmp[2]] + "[S@]" +
                             result_smiles[tmp[2] + 1:])
        if tmp[0] == "S" and tmp[1] == "[S@@]":
            result_smiles = (result_smiles[:tmp[2]] + "[S@@]" +
                             result_smiles[tmp[2] + 1:])
        if tmp[0] == "P" and tmp[1] == "[P@]":
            result_smiles = (result_smiles[:tmp[2]] + "[P@]" +
                             result_smiles[tmp[2] + 1:])
        if tmp[0] == "P" and tmp[1] == "[P@@]":
            result_smiles = (result_smiles[:tmp[2]] + "[P@@]" +
                             result_smiles[tmp[2] + 1:])
    mol = Chem.MolFromSmiles(result_smiles)
    find_chiral = Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)

    ms = [x for x in ATOM_FINDER.finditer(result_smiles)]
    individual_atoms = [result_smiles[x.start():x.end()] for x in ms]
    find_chiral.sort(key=lambda x: -x[0])

    for idx, chiral in find_chiral:
        if "[C@@H]" == individual_atoms[idx] and "R" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[C@H]" +
                             result_smiles[ms[idx].start() + 6:])
        elif "[C@H]" == individual_atoms[idx] and "S" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[C@@H]" +
                             result_smiles[ms[idx].start() + 5:])
        elif "[C@@]" == individual_atoms[idx] and "R" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[C@]" +
                             result_smiles[ms[idx].start() + 5:])
        elif "[C@]" == individual_atoms[idx] and "S" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[C@@]" +
                             result_smiles[ms[idx].start() + 4:])
        elif "[N@@+]" == individual_atoms[idx] and "R" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[N@+]" +
                             result_smiles[ms[idx].start() + 6:])
        elif "[N@+]" == individual_atoms[idx] and "S" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[N@@+]" +
                             result_smiles[ms[idx].start() + 5:])
        elif "[S@@]" == individual_atoms[idx] and "R" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[S@]" +
                             result_smiles[ms[idx].start() + 5:])
        elif "[S@]" == individual_atoms[idx] and "S" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[S@@]" +
                             result_smiles[ms[idx].start() + 4:])
        elif "[P@@]" == individual_atoms[idx] and "R" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[P@]" +
                             result_smiles[ms[idx].start() + 5:])
        elif "[P@]" == individual_atoms[idx] and "S" in chiral:
            result_smiles = (result_smiles[:ms[idx].start()] + "[P@@]" +
                             result_smiles[ms[idx].start() + 4:])

    bond_before_kekule = []
    for i in range(len(bond_stereo_list)):
        ms = [x for x in ATOM_FINDER.finditer(result_smiles)]

        individual_atoms = [result_smiles[x.start():x.end()] for x in ms]
        bond_attr = bond_stereo_list[canonical_atom_order_inverted[
            canonical_atom_order_inverted2.index(i)]]

        if bond_attr != None:
            anchor_atom = canonical_atom_order_inverted2[
                canonical_atom_order_inverted.index(bond_attr[1])]
            connected_atom = canonical_atom_order_inverted2[
                canonical_atom_order_inverted.index(bond_attr[2])]
            if anchor_atom > connected_atom:
                continue
            bond_before_kekule.append(
                [anchor_atom, connected_atom, bond_attr[3]])
            result_smiles = (result_smiles[:ms[connected_atom + 1].start()] +
                             "/" +
                             result_smiles[ms[connected_atom + 1].start():])
            if bond_attr[3] == "Z":
                result_smiles = (result_smiles[:ms[anchor_atom].start()] +
                                 "/" + result_smiles[ms[anchor_atom].start():])
            elif bond_attr[3] == "E":
                result_smiles = (result_smiles[:ms[anchor_atom].start()] +
                                 "\\" +
                                 result_smiles[ms[anchor_atom].start():])

    ms = [x for x in ATOM_FINDER.finditer(result_smiles)]

    individual_atoms = [result_smiles[x.start():x.end()] for x in ms]

    result_smiles = result_smiles.replace("\\/", "/")
    result_smiles = result_smiles.replace("\\\\", "/")
    result_smiles = result_smiles.replace("//", "/")

    result_smiles = result_smiles.replace("/\\", "/")
    bond_stereo_list = [x for x in bond_stereo_list if x is not None]
    m = Chem.MolFromSmiles(result_smiles)

    result_smiles = Chem.MolToSmiles(m, kekuleSmiles=True)
    m_after_kekule = Chem.MolFromSmiles(result_smiles)

    canonical_atom_order = tuple(Chem.CanonicalRankAtoms(m))
    canonical_atom_order_inverted = list(
        tuple(zip(*sorted(
            (j, i) for i, j in enumerate(canonical_atom_order))))[1])

    canonical_atom_order2 = tuple(Chem.CanonicalRankAtoms(m_after_kekule))
    canonical_atom_order_inverted2 = list(
        tuple(zip(*sorted(
            (j, i) for i, j in enumerate(canonical_atom_order2))))[1])
    for bond in bond_before_kekule:
        bond_after_kekule = m_after_kekule.GetBondBetweenAtoms(
            canonical_atom_order_inverted2[canonical_atom_order_inverted.index(
                bond[0])],
            canonical_atom_order_inverted2[canonical_atom_order_inverted.index(
                bond[1])],
        )
        target = None
        if bond[2] == "E":
            target = Chem.rdchem.BondStereo.STEREOE
        if bond[2] == "Z":
            target = Chem.rdchem.BondStereo.STEREOZ
        if bond_after_kekule.GetStereo() != target:
            bond_after_kekule.SetStereo(target)

    result_smiles = Chem.MolToSmiles(m_after_kekule, kekuleSmiles=True)
    return result_smiles
