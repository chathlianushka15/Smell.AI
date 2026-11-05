import json
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

with open("samples.json") as f:
    samples = json.load(f)

model = joblib.load("data/smell_model.joblib")
words = np.load("data/features.npz", allow_pickle=True)["words"]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

for name, molecules in samples.items():
    fps = []
    for mol_name, smiles in molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("BAD SMILES:", mol_name, smiles)
            continue
        fps.append(gen.GetFingerprintAsNumPy(mol))

    probs = model.predict_proba(np.array(fps))
    chance = np.array([p[:, 1] for p in probs]).T   # rows = molecules, columns = smell words
    average = chance.mean(axis=0)                   # average across the sample's molecules
    best = np.argsort(average)[::-1][:5]

    print(f"{name} ({len(fps)} molecules)")
    print("   ", ", ".join(words[i] for i in best))