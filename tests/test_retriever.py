from datetime import datetime, UTC
from pathlib import Path
import shutil, tempfile, json
from importlib import reload

import numpy as np
import pytest

from score_core import MemoryEntry
from score_core import retriever

from sentence_transformers import SentenceTransformer
import faiss, joblib


# ---- フィクスチャ ----------------------------------------------------
@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    tmpdir  = tmp_path_factory.mktemp("faiss_idx")
    idx_dir = tmpdir / "_faiss_index"
    idx_dir.mkdir()

    # ------------- SBERT 384次元インデックスを生成 -------------
    texts = [f"dummy event {i}" for i in range(10)]
    sbert = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    emb   = sbert.encode(texts, convert_to_numpy=True).astype("float32")

    ix = faiss.IndexFlatIP(emb.shape[1])      # 384 dims
    ix.add(emb)

    faiss.write_index(ix, str(idx_dir / "index.faiss"))   # ← str() が必須
    joblib.dump(None, idx_dir / "vectorizer.joblib")      # ダミーファイル

    # ------------- retriever を切り替え -----------------------
    retriever._get_faiss.cache_clear()
    retriever._get_tfidf.cache_clear()
    retriever._get_sbert.cache_clear()

    reload(retriever)                         # ここで 384 次元をロード
    retriever._INDEX_DIR = idx_dir            # 新しい場所に更新
    retriever._get_faiss.cache_clear()
    
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