from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss, joblib, json, numpy as np, os, sys, tqdm

in_json, out_dir = sys.argv[1], sys.argv[2]
texts = [m["what"] for m in json.load(open(in_json, encoding="utf-8"))]

# ---- SBERT ----
sbert = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
emb   = sbert.encode(texts, convert_to_numpy=True).astype("float32")
faiss_index = faiss.IndexFlatIP(emb.shape[1])
faiss_index.add(emb)

# ---- TF-IDF ----
vectorizer = TfidfVectorizer(max_features=50000, stop_words="english")
tfidf_mat  = vectorizer.fit_transform(texts).astype("float32")  # csr_matrix

os.makedirs(out_dir, exist_ok=True)
faiss.write_index(faiss_index,  str(Path(out_dir)/"index.faiss"))
joblib.dump(vectorizer,         Path(out_dir)/"vectorizer.joblib")
joblib.dump(tfidf_mat,          Path(out_dir)/"tfidf.npz")
