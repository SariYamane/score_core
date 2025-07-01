import faiss, joblib, os, time
import numpy as np
from datetime import datetime, UTC
from pathlib import Path
from score_core import MemoryEntry, retriever
from importlib import reload

def _reload_with_tmp(idx_dir):
    reload(retriever)
    retriever._INDEX_DIR = idx_dir
    retriever._get_faiss.cache_clear()
    retriever._get_tfidf.cache_clear()
    retriever._get_tfidf_matrix.cache_clear()
    retriever._get_sbert.cache_clear()

def _dummy_pool(n):
    now = datetime.now(UTC)
    return [MemoryEntry(uuid=str(i), timestamp=now, who="A", what=f"dummy {i}") for i in range(n)]

def test_build_when_missing(tmp_path):
    idx_dir = tmp_path / "_faiss_index"
    idx_dir.mkdir()
    _reload_with_tmp(idx_dir)

    retriever.retrieve("dummy", _dummy_pool(8), k=3)

    assert (idx_dir / "index.faiss").exists()
    assert (idx_dir / "vectorizer.joblib").exists()
    assert (idx_dir / "tfidf.npz").exists()

def test_rebuild_on_growth(tmp_path):
    idx_dir = tmp_path / "_faiss_index"
    idx_dir.mkdir()
    _reload_with_tmp(idx_dir)

    # ① 最初に 10 件で index 作成
    retriever.retrieve("dummy", _dummy_pool(10), k=3)
    ix_path = idx_dir / "index.faiss"
    ts_old  = ix_path.stat().st_mtime
    nt_old  = faiss.read_index(str(ix_path)).ntotal
    assert nt_old == 10

    # ② corpus を 25 件に増やす
    time.sleep(1)                      # mtime 変化を確実にする
    retriever.retrieve("dummy", _dummy_pool(25), k=5)
    ix      = faiss.read_index(str(ix_path))
    assert ix.ntotal == 25
    assert ix_path.stat().st_mtime > ts_old

def test_recover_broken_index(tmp_path):
    idx_dir = tmp_path / "_faiss_index"
    idx_dir.mkdir()
    _reload_with_tmp(idx_dir)

    retriever.retrieve("dummy", _dummy_pool(5), k=2)
    (idx_dir / "index.faiss").unlink()        # 壊す

    # 落ちずに再生成されるか
    retriever.retrieve("dummy", _dummy_pool(5), k=2)
    assert (idx_dir / "index.faiss").exists()
