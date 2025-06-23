from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class MemoryEntry(BaseModel):
    uuid: str
    timestamp: datetime
    who: str
    what: str
    where: Optional[str] = None
    state: str = "active"
    relations: List[str] = []
    emotion: Optional[str] = None
    importance: float = 0.0
    embedding: Optional[list[float]] = Field(default=None, repr=False)

class EpisodeSummary(BaseModel):
    episode_id: int
    time_span: str
    characters: List[str]
    summary: str
