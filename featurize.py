import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

df = pd.read_csv("data/molecules.csv")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

fingerprints = []
kept_rows = []

for i, smiles in enumerate(df["nonStereoSMILES"]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:                      # RDKit couldn't read this molecule
        continue
    fingerprints.append(gen.GetFingerprintAsNumPy(mol))
    kept_rows.append(i)

X = np.array(fingerprints)               # the inputs
label_columns = df.columns[2:]           # the 138 smell-word columns
y = df.loc[kept_rows, label_columns].values   # the answers

print("Molecules kept:", len(kept_rows), "of", len(df))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Smell words:", list(label_columns[:8]), "...")
print("Average smell words per molecule:", round(y.sum(axis=1).mean(), 2))

np.savez_compressed("data/features.npz", X=X, y=y, words=np.array(label_columns))
print("Saved to data/features.npz")