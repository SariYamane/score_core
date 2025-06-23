"""
usage: python scripts/build_faiss.py embedding_pool.json score_core/_faiss_index
"""
import sys
import json, os, numpy as np, faiss, pickle, joblib
from sklearn.feature_extraction.text import TfidfVectorizer

text_list = [m["what"] for m in json.load(open(sys.argv[1]))]
tfidf = TfidfVectorizer(max_features=50000, stop_words="english")
mat = tfidf.fit_transform(text_list).astype("float32")
faiss_index = faiss.IndexFlatIP(mat.shape[1])
faiss_index.add(mat.toarray())              # メモリで OK なら
os.makedirs(sys.argv[2], exist_ok=True)
faiss.write_index(faiss_index, f"{sys.argv[2]}/index.faiss")
joblib.dump(tfidf, f"{sys.argv[2]}/vectorizer.joblib")
