from datetime import datetime, UTC
from pathlib import Path
import shutil, tempfile, json

import numpy as np
import pytest

from score_core import MemoryEntry
from score_core import retriever

# ---- フィクスチャ ----------------------------------------------------
@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    """
    テスト用に 10 件だけの FAISS + TF-IDF インデックスを作る。
    tmp_path は pytest が用意する一時ディレクトリ。
    """
    tmpdir = tmp_path_factory.mktemp("faiss_idx")
    index_dir = Path(retriever.__file__).parent / "_faiss_index"

    # ★ もし既に本物の index があるならコピーで済ませる
    if index_dir.exists():
        shutil.copytree(index_dir, tmpdir / "_faiss_index")
    else:
        # 例：ダミー行列をその場で作る場合
        import faiss, joblib
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [f"dummy event {i}" for i in range(10)]
        vectorizer = TfidfVectorizer(max_features=1000)
        mat = vectorizer.fit_transform(texts).astype("float32")
        faiss_index = faiss.IndexFlatIP(mat.shape[1])
        faiss_index.add(mat.toarray())

        (tmpdir / "_faiss_index").mkdir()
        faiss.write_index(faiss_index, tmpdir / "_faiss_index" / "index.faiss")
        joblib.dump(vectorizer, tmpdir / "_faiss_index" / "vectorizer.joblib")

    # retriever 内のグローバル変数を上書き
    retriever._INDEX_DIR = tmpdir / "_faiss_index"
    from importlib import reload
    reload(retriever)         # 再ロードして _TFIDF / _FAISS を読み直す
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
