from __future__ import annotations
from typing import Dict, List
from .models import MemoryEntry

# 合法遷移テーブル
LEGAL: dict[str, set[str]] = {
    "active":      {"active", "lost", "destroyed"},
    "lost":        {"lost", "destroyed"},
    "destroyed":   {"destroyed"},
}

# エンティティごとに最後に確定した状態を保持
_entity_state: Dict[str, str] = {}

def register_memory(mem: MemoryEntry, pool: List[MemoryEntry]) -> bool:
    """
    メモリをプールに追加。合法遷移なら True、不正なら False。
    """
    ent_id = mem.entity_id
    if not ent_id:
        pool.append(mem)
        return True                   # エンティティなし＝常に許可

    prev = _entity_state.get(ent_id, "active")
    if mem.state not in LEGAL[prev]:
        # 不正遷移→拒否して prev_state に書き換え
        mem.state = prev
        return False

    # 合法：状態を更新しプールへ
    _entity_state[ent_id] = mem.state
    pool.append(mem)
    return True
