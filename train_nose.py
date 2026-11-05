import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sensor import sample_features, sniff

with open("samples.json") as f:
    samples = json.load(f)

names = list(samples.keys())        # ["Coffee", "Vanilla candle", ...]
rng = np.random.default_rng(1)

# Build the training data: 200 noisy sniffs of each smell
X, y = [], []
for label, name in enumerate(names):
    feats = sample_features(samples[name])
    for _ in range(200):
        X.append(sniff(feats, rng))
        y.append(label)
X = np.array(X)
y = np.array(y)
print("Sniffs:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

nose = RandomForestClassifier(n_estimators=200, random_state=42)
nose.fit(X_train, y_train)

predicted = nose.predict(X_test)
print("Nose accuracy:", round(accuracy_score(y_test, predicted), 3))

print()
print("Confusion table (rows = real smell, columns = nose's guess):")
table = confusion_matrix(y_test, predicted)
print(pd.DataFrame(table, index=names, columns=names).to_string())

joblib.dump({"model": nose, "names": names}, "data/nose_model.joblib")
print()
print("Saved to data/nose_model.joblib")