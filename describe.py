import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

model = joblib.load("data/smell_model.joblib")
words = np.load("data/features.npz", allow_pickle=True)["words"]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def describe(smiles, top=5):
    mol = Chem.MolFromSmiles(smiles)
    fp = gen.GetFingerprintAsNumPy(mol).reshape(1, -1)   # one molecule as one row
    probs = model.predict_proba(fp)
    chance = np.array([p[0][1] for p in probs])          # chance of "yes" for each word
    best = np.argsort(chance)[::-1][:top]                # indices of the highest chances
    return [(words[i], round(float(chance[i]), 2)) for i in best]

molecules = {
    "Ethyl acetate":  "CC(=O)OCC",
    "Vanillin":       "COc1cc(C=O)ccc1O",
    "Menthol":        "CC(C)C1CCC(C)CC1O",
    "Limonene":       "CC1=CCC(CC1)C(C)=C",
    "Isoamyl acetate": "CC(C)CCOC(C)=O",
    "Guaiacol":       "COc1ccccc1O",
}

for name, smiles in molecules.items():
    print(name)
    for word, p in describe(smiles):
        print(f"   {word:<12} {p}")
    print()