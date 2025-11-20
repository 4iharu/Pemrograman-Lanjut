import numpy as np
import joblib
from utils_facenet import embed_from_path
import sys

MODEL_PATH = "knn_model.pkl"

# Load model KNN
knn = joblib.load(MODEL_PATH)

def predict_image(img_path):
    emb = embed_from_path(img_path)
    if emb is None:
        print("❌ Wajah tidak terdeteksi.")
        return
    
    emb = emb.reshape(1, -1)  # bentuk menjadi (1, 512)
    pred = knn.predict(emb)[0]

    # hitung jarak untuk confidence
    dist, idx = knn.kneighbors(emb, n_neighbors=1, return_distance=True)
    conf = 1 / (1 + dist[0][0])

    print("\nMemprediksi gambar:", img_path)
    print("Prediksi:", pred)
    print("Confidence:", round(conf, 3))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_knn.py <path_gambar>")
    else:
        predict_image(sys.argv[1])