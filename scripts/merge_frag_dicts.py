import os
import sys

sys.path.append(os.getcwd())

if __name__ == "__main__":
    frag_set = set()
    with open(f"asset/mol_vocabs/frag_stereo_v2.txt") as f:
        for line in f:
            frag_set.add(line.strip())
    with open(f"asset/mol_vocabs/frag_pubchem_v3.txt") as f:
        for line in f:
            frag_set.add(line.strip())

    with open(f"asset/mol_vocabs/frag_camt5_v3.txt", "w") as f:
        frags = sorted(list(frag_set))
        for frag in frags:
            f.write(frag)
            f.write("\n")
