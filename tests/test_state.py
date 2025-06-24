from score_core.state import register_memory
from score_core.models import MemoryEntry
from datetime import datetime, UTC

def test_state_transition():
    pool=[]
    m1 = MemoryEntry(uuid="1", timestamp=datetime.now(UTC),
                     who="A", what="find box", entity_id="box1", state="active")
    assert register_memory(m1, pool)

    m2 = MemoryEntry(uuid="2", timestamp=datetime.now(UTC),
                     who="A", what="lose box", entity_id="box1", state="lost")
    assert register_memory(m2, pool)

    m3 = MemoryEntry(uuid="3", timestamp=datetime.now(UTC),
                     who="B", what="restore box", entity_id="box1", state="active")
    # active ← lost は不正遷移 → False
    assert not register_memory(m3, pool)
    assert m3.state == "lost"         # 上書きされている
