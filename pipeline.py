import json
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sensor import sample_features, sniff

# ---- Load everything once ----
with open("samples.json") as f:
    SAMPLES = json.load(f)

smell_model = joblib.load("data/smell_model.joblib")
WORDS = np.load("data/features.npz", allow_pickle=True)["words"]

nose_bundle = joblib.load("data/nose_model.joblib")
nose = nose_bundle["model"]
NAMES = nose_bundle["names"]

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def describe_sample(name, top=5):
    """Smell model: describe a smell in words from its molecules."""
    fps = []
    for smiles in SAMPLES[name].values():
        mol = Chem.MolFromSmiles(smiles)
        fps.append(gen.GetFingerprintAsNumPy(mol))

    probs = smell_model.predict_proba(np.array(fps))
    chance = np.array([p[:, 1] for p in probs]).T   # rows = molecules, columns = words
    average = chance.mean(axis=0)                   # one score per word
    best = np.argsort(average)[::-1][:top]
    return [(str(WORDS[i]), float(average[i])) for i in best]


def run_sniff(true_name, rng):
    """The whole device: sniff -> nose guesses -> smell model describes."""
    feats = sample_features(SAMPLES[true_name])
    reading = sniff(feats, rng)                       # 8 sensor numbers

    probs = nose.predict_proba(reading.reshape(1, -1))[0]
    best = int(np.argmax(probs))
    guess = NAMES[best]

    return {
        "reading": reading,
        "guess": guess,
        "confidence": float(probs[best]),
        "all_probs": {n: float(p) for n, p in zip(NAMES, probs)},
        "words": describe_sample(guess),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(123)   # a fresh seed: sniffs the nose never trained on
    for true_name in NAMES:
        result = run_sniff(true_name, rng)
        mark = "OK " if result["guess"] == true_name else "WRONG"
        print(f"[{mark}] Real: {true_name:<15} Nose guessed: {result['guess']:<15} "
              f"({result['confidence']:.0%})")
        print("      Smells like:", ", ".join(w for w, p in result["words"]))
        print()