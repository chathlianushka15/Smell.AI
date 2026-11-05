import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

N_SENSORS = 8

# Each sensor's fixed "recipe": how strongly it reacts to
# [size, oiliness, oxygen, sulfur]. Seed 7 means the same
# recipes every run, like 8 physical chips that never change.
_rng = np.random.default_rng(7)
SENSITIVITY = _rng.normal(0, 1, size=(N_SENSORS, 4))
HUMIDITY_EFFECT = _rng.normal(0.15, 0.05, size=N_SENSORS)


def molecule_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    size = Descriptors.MolWt(mol) / 100
    oiliness = Crippen.MolLogP(mol)
    oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    sulfur = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    return np.array([size, oiliness, oxygen, sulfur], dtype=float)


def sample_features(molecules):
    # Average the four properties over all molecules in the smell
    return np.mean([molecule_features(s) for s in molecules.values()], axis=0)


def clean_reading(features):
    # No noise: the ideal fingerprint of a smell
    response = SENSITIVITY @ features
    return np.exp(-0.15 * response)


def sniff(features, rng):
    # One realistic, messy sniff
    strength = rng.uniform(0.85, 1.15)
    humidity = rng.uniform(0.2, 0.8)
    response = (SENSITIVITY @ features) * strength
    reading = np.exp(-0.15 * response)
    reading = reading * (1 + HUMIDITY_EFFECT * (humidity - 0.5))
    reading = reading + rng.normal(0, 0.02, size=N_SENSORS)
    return reading


if __name__ == "__main__":
    with open("samples.json") as f:
        samples = json.load(f)

    rng = np.random.default_rng(0)

    print("Clean fingerprints (1.00 = clean air):")
    for name, molecules in samples.items():
        feats = sample_features(molecules)
        print(f"{name:<15}", np.round(clean_reading(feats), 2))

    print()
    coffee = sample_features(samples["Coffee"])
    print("Three noisy sniffs of Coffee:")
    for _ in range(3):
        print(np.round(sniff(coffee, rng), 2))