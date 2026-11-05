import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = np.load("data/features.npz", allow_pickle=True)
X, y, words = data["X"], data["y"], data["words"]

# Same split as Step 10 (same random_state), so this is the SAME hidden test set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("data/smell_model.joblib")
probs = model.predict_proba(X_test)

rows = []
for k, word in enumerate(words):
    if len(np.unique(y_test[:, k])) < 2:
        continue
    auc = roc_auc_score(y_test[:, k], probs[k][:, 1])
    rows.append({"smell word": str(word),
                 "ROC-AUC": round(auc, 3),
                 "test examples": int(y_test[:, k].sum())})

table = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
table.to_csv("data/per_word_scores.csv", index=False)

print("Words scored:", len(table))
print("Average ROC-AUC:", round(table["ROC-AUC"].mean(), 3))
print()
print("BEST 8 words:")
print(table.head(8).to_string(index=False))
print()
print("WORST 8 words:")
print(table.tail(8).to_string(index=False))