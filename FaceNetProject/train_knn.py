# train_knn.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ======================
# Load dataset dari embeddings.npz
# ======================
data = np.load("embeddings.npz")

X = data["embeddings"]   # ← sesuai key
y = data["labels"]       # ← sesuai key

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# ======================
# Training KNN
# ======================
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# ======================
# Simpan model
# ======================
joblib.dump(knn, "knn_model.pkl")
print("Model KNN berhasil disimpan! → knn_model.pkl")