from __future__ import annotations

import functools
import pathlib
from datetime import datetime, UTC
from typing import List
import numpy as np, faiss, joblib

from .models import MemoryEntry
from . import models
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

import fasteners

import json

from contextlib import contextmanager


def score_kernel(sim: np.ndarray,
                 sent_delta: np.ndarray,
                 recency: np.ndarray,
                 importance: np.ndarray,
                 alpha=0.5, beta=0.3, gamma=0.4, tau=0.05):
    """
    S = α·sim + β·importance + (1-α-β)·recency - γ·sent_delta
    P = exp(S / τ)
    """
    s = alpha*sim + beta*importance + (1-alpha-beta)*recency - gamma*sent_delta
    s = np.clip(s / tau, -50, 50)        # overflow 対策
    return np.exp(s)

def _recency_score(mem: MemoryEntry, now: datetime) -> float:
    hours = (now - mem.timestamp).total_seconds() / 3600
    return max(0.0, 1 - hours / 48)          # 48h で 0 になる線形減衰

def _importance_score(mem: MemoryEntry) -> float:
    return min(max(mem.importance / 10, 0), 1)

def _sent_delta(mem: MemoryEntry, query_emotion: float) -> float:
    if mem.emotion is None: return 0.0
    return abs(mem.emotion - query_emotion)

# ❶ Sentence-BERT を lazy-load で用意
@functools.lru_cache(maxsize=1)
def _get_sbert() -> SentenceTransformer:
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # 軽量モデル

_INDEX_DIR = pathlib.Path(__file__).parent / "_faiss_index"
_INDEX_DIR.mkdir(exist_ok=True)          # ← ディレクトリだけは確実に作る

def _idx_path(name):
    return _INDEX_DIR / name

@contextmanager
def _file_lock():
    lock = fasteners.InterProcessLock(str(_INDEX_DIR / ".lock"))
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

def _ensure_index(memories: list[MemoryEntry]):
    """
    _INDEX_DIR に index.faiss / tfidf.npz / vectorizer.joblib が無ければ生成する
    """

    n_docs = len(memories)

    with _file_lock():
        #--- メタデータの読み込み -------------------------------------------------
        if _idx_path("meta.json").exists():
            with _idx_path("meta.json").open("r") as f:
                meta = json.load(f)
            indexed_docs = meta.get("document_count", -1)
        else:
            indexed_docs = -1  # 存在しない扱い

        #--- 既存インデックスが最新なら何もしない -------------------------------
        if (
            indexed_docs == n_docs
            and _idx_path("index.faiss").exists()
            and _idx_path("tfidf.npz").exists()
            and _idx_path("vectorizer.joblib").exists()
        ):
            return  # 早期リターン

    #-----------------------------------------------------------------------
    print(f"[score_core] building index (documents: {n_docs}) …")

    # --- TF-IDF -------------------------------------------------
    corpus = [m.what for m in memories]
    vec = TfidfVectorizer(max_features=30_000)
    tfmat = vec.fit_transform(corpus)
    joblib.dump(vec, _idx_path("vectorizer.joblib"), compress=0)
    joblib.dump(tfmat, _idx_path("tfidf.npz"), compress=0)

    # --- SBERT & FAISS ------------------------------------------
    sbert = _get_sbert()
    vecs  = sbert.encode(corpus, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)                       # 内積→cos 類似に合わせる
    index = faiss.IndexFlatIP(vecs.shape[1])       # 内積類似
    index.add(vecs)
    faiss.write_index(index, str(_idx_path("index.faiss")))
    
    _get_faiss.cache_clear()
    _get_tfidf.cache_clear()
    _get_tfidf_matrix.cache_clear()

    #--- メタデータを更新 ---------------------------------------------------
    with _idx_path("meta.json").open("w") as f:
        json.dump({"document_count": n_docs}, f)

# ❷ TF-IDF / FAISS も同じく遅延ロードにしておくとテストが楽
@functools.lru_cache(maxsize=1)
def _get_tfidf():
    return joblib.load(_idx_path("vectorizer.joblib"))

@functools.lru_cache(maxsize=1)
def _get_faiss():
    return faiss.read_index(str(_INDEX_DIR / "index.faiss"))

@functools.lru_cache(maxsize=1)
def _get_tfidf_matrix():
    tfidf_path = _idx_path("tfidf.npz")
    try:
        return joblib.load(tfidf_path)
    except FileNotFoundError:
        # インデックスが壊れている／書込み途中だった場合は作り直す
        print("[score_core] tfidf.npz missing – rebuilding index …")
        _ensure_index(_MEM_CACHE)       # _MEM_CACHE は直前に渡した memories を保持
        return joblib.load(tfidf_path)

def retrieve(query: str,
             memories: list[MemoryEntry],
             k: int = 8,
             now: datetime | None = None) -> list[MemoryEntry]:
    
    global _MEM_CACHE
    _MEM_CACHE = memories
    
    _ensure_index(memories)
    
    now = now or datetime.now(UTC)

    # ---------- Stage-0: TF-IDF で粗フィルタ ----------
    tfidf = _get_tfidf()
    tfmat = _get_tfidf_matrix()
    q_sparse = tfidf.transform([query])
    # cos類似 => dot / (||q||·||d||) でも良いが一次近似で OK
    scores = (q_sparse @ tfmat.T).toarray()[0]
    top_n = min(500, len(scores))
    top_mask = np.argpartition(-scores, top_n-1)[:top_n]
    cand_pool = [memories[i] for i in top_mask]
    cand_scores = scores[top_mask]
    
    
    # ---------- Stage-1: SBERT + FAISS ----------
    sbert   = _get_sbert()
    q_vec   = sbert.encode([query], convert_to_numpy=True).astype("float32")
    faiss_i = _get_faiss()
    n_total = faiss_i.ntotal
    k_faiss = min(50, n_total)
    # FAISS は元プール順なので cand_pool の id を使う
    _, idx_global = faiss_i.search(q_vec, k=k_faiss)
    idx_global = idx_global[0]
    cands =  [memories[i] for i in idx_global]
    sim = scores[idx_global]

    # --- 2) スコア要素計算 -------------------------
    rec  = np.array([_recency_score(m, now)       for m in cands])
    imp  = np.array([_importance_score(m)         for m in cands])
    sent = np.zeros_like(sim)                     # TODO: まず 0
    kern = score_kernel(sim, sent, rec, imp)

    # --- 3) 上位 k 件 -----------------------------
    order = kern.argsort()[::-1][:k]
    return [cands[i] for i in order]

