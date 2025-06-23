from __future__ import annotations

import pathlib
from typing import List
import numpy as np, faiss, joblib
from scipy.sparse import csr_matrix
from .models import MemoryEntry
from . import models

_INDEX_DIR = pathlib.Path(__file__).parent / "_faiss_index"
_TFIDF = joblib.load(_INDEX_DIR / "vectorizer.joblib")
_FAISS = faiss.read_index(str(_INDEX_DIR / "index.faiss"))

def _kernel(sim: np.ndarray, sent_delta: np.ndarray,
            gamma: float = .4, tau: float = .05) -> np.ndarray:
    """SCORE カーネル（論文 Eq.3）"""
    return np.exp((sim - gamma * sent_delta) / tau)

def retrieve(query: str,
             memories: List[MemoryEntry],
             k: int = 8) -> List[MemoryEntry]:
    """二段階検索＋SCORE rerank"""
    # ❶ TF-IDF ベクトル化
    q_vec = _TFIDF.transform([query]).astype("float32")
    # ❷ FAISS 近傍検索（上位 50 件）
    sim, idx = _FAISS.search(q_vec.toarray(), k=50)
    idx = idx[0]; sim = sim[0]
    cand = [memories[i] for i in idx]
    # ❸ 感情差 Δσ（ここでは 0 に）
    sent = np.zeros_like(sim)
    # ❹ SCORE カーネル
    score = _kernel(sim, sent)
    order = score.argsort()[::-1][:k]
    return [cand[i] for i in order]
