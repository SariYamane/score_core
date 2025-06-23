from typing import List
import random
from .models import MemoryEntry

def retrieve(query: str, memories: List[MemoryEntry], k: int = 8) -> List[MemoryEntry]:
    """とりあえずランダムに k 件返すスタブ"""
    return random.sample(memories, min(k, len(memories)))
