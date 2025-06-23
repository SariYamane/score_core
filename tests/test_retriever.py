from score_core import MemoryEntry, retrieve
from datetime import datetime

def test_retrieve_returns_k():
    pool = [
        MemoryEntry(uuid=str(i), timestamp=datetime.utcnow(), who="A", what=f"event {i}")
        for i in range(10)
    ]
    out = retrieve("dummy", pool, k=5)
    assert len(out) == 5
