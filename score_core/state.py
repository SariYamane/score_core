LEGAL_TRANSITIONS = {
    "active": {"active", "lost", "destroyed"},
    "lost": {"lost"},
    "destroyed": {"destroyed"},
}

def register_memory(mem: "MemoryEntry", pool: list["MemoryEntry"]) -> bool:
    """True を返せばプールに追加済み"""
    from .models import MemoryEntry                 # 循環参照回避
    assert isinstance(mem, MemoryEntry)
    for m in reversed(pool):
        if m.uuid == mem.uuid:
            if mem.state not in LEGAL_TRANSITIONS[m.state]:
                mem.state = m.state                 # 差し戻し
            break
    pool.append(mem)
    return True
