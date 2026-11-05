from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Make a fingerprint generator: look 2 bonds out, use 2048 slots
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)       # text -> molecule
    return gen.GetFingerprintAsNumPy(mol)  # molecule -> list of 0s and 1s

ethyl_acetate = fingerprint("CC(=O)OCC")   # fruity
vanillin = fingerprint("COc1cc(C=O)ccc1O") # vanilla

print("Length of fingerprint:", len(ethyl_acetate))
print("Ethyl acetate: slots switched on =", ethyl_acetate.sum())
print("Vanillin:      slots switched on =", vanillin.sum())
print("Slots switched on in BOTH:", (ethyl_acetate & vanillin).sum())