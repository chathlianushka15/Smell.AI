import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

data = np.load("data/features.npz", allow_pickle=True)
X, y, words = data["X"], data["y"], data["words"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training molecules:", len(X_train), "| Test molecules:", len(X_test))

model = RandomForestClassifier(
    n_estimators=100,       # 100 trees
    min_samples_leaf=3,     # keeps trees small so the saved file isn't huge
    n_jobs=-1,              # use all CPU cores
    random_state=42,
)
print("Training... (this can take a few minutes)")
model.fit(X_train, y_train)

# probs[k] holds, for smell word k, the chance of "no" and "yes" per molecule
probs = model.predict_proba(X_test)

scores = []
for k in range(len(words)):
    if len(np.unique(y_test[:, k])) < 2:     # word never appears in test set
        continue
    scores.append(roc_auc_score(y_test[:, k], probs[k][:, 1]))

print("Average ROC-AUC:", round(np.mean(scores), 3))

joblib.dump(model, "data/smell_model.joblib", compress=3)
print("Saved to data/smell_model.joblib")