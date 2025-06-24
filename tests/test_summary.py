from score_core.summary import to_episode_summary
from score_core.models import MemoryEntry
from datetime import datetime, UTC

def test_episode_summary_basic():
    mems = [
        MemoryEntry(uuid="1", timestamp=datetime(2025,6,24,9,0, tzinfo=UTC),
                    who="A", what="open cafe"),
        MemoryEntry(uuid="2", timestamp=datetime(2025,6,24,9,5, tzinfo=UTC),
                    who="B", what="order coffee"),
    ]
    ep = to_episode_summary(mems, episode_id=1)
    assert ep.episode_id == 1
    assert ep.characters == ["A", "B"]
    assert "open cafe" in ep.summary
