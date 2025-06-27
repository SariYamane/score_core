from datetime import datetime, UTC
from pathlib import Path
import shutil, tempfile, json
from importlib import reload

import numpy as np
import pytest

from score_core import MemoryEntry
from score_core import retriever

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from importlib import reload

import faiss, joblib


# ---- フィクスチャ ----------------------------------------------------
@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    tmpdir  = tmp_path_factory.mktemp("faiss_idx")
    idx_dir = tmpdir / "_faiss_index"
    idx_dir.mkdir()

    texts = [f"dummy event {i}" for i in range(10)]

    sbert = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    emb   = sbert.encode(texts, convert_to_numpy=True).astype("float32")

    ix = faiss.IndexFlatIP(emb.shape[1])      # 384 dims
    ix.add(emb)

    faiss.write_index(ix, str(idx_dir / "index.faiss"))   # ← str() が必須

    vec = TfidfVectorizer(max_features=1000)
    tfidf_mat = vec.fit_transform(texts).astype("float32")
    joblib.dump(vec, idx_dir / "vectorizer.joblib")
    joblib.dump(tfidf_mat, idx_dir / "tfidf.npz")   # TF-IDF 行列を保存

    reload(retriever)
    retriever._INDEX_DIR = idx_dir
    retriever._get_faiss.cache_clear()
    retriever._get_tfidf.cache_clear()
    retriever._get_tfidf_matrix.cache_clear()
    retriever._get_sbert.cache_clear()
    
    return tmpdir


# ---- 実テスト --------------------------------------------------------
def test_retrieve_topk(small_index):
    pool = [
        MemoryEntry(
            uuid=str(i),
            timestamp=datetime.now(UTC),
            who="A",
            what=f"dummy event {i}"
        )
        for i in range(10)
    ]
    topk = retriever.retrieve("event", pool, k=5)

    assert len(topk) == 5
    assert all(isinstance(m, MemoryEntry) for m in topk)

def test_kernel_monotonic():
    high = retriever.score_kernel(
        sim=np.array([0.9]),
        sent_delta=np.array([0]),
        recency=np.array([1]),
        importance=np.array([1])
    )
    low = retriever.score_kernel(
        sim=np.array([0.1]),
        sent_delta=np.array([0]),
        recency=np.array([1]),
        importance=np.array([1])
    )
    assert high > low