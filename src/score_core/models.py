from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field

class MemoryEntry(BaseModel):
    uuid: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    who: str
    what: str
    where: Optional[str] = None
    state: str = "active"
    relations: List[str] = []
    emotion: Optional[str] = None
    importance: float = 0.0
    embedding: Optional[list[float]] = Field(default=None, repr=False)
    entity_id: Optional[str] = None
    state: str = "active"

    @property
    def content(self) -> str:
        return self.what

    @content.setter
    def content(self, value: str) -> None:
        self.what = value

class EpisodeSummary(BaseModel):
    episode_id: int
    time_span: str
    characters: List[str]
    summary: str

#検索ハブ
class SearchIndex(BaseModel):
    tfidf: 'sklearn.feature_extraction.text.TfidfVectorizer'
    matrix: 'scipy.sparse.csr_matrix'            # TF-IDF 行列
    faiss_index_path: Optional[str] = None          # _faiss_index/index.faiss

class EpisodeSummary(BaseModel):
    episode_id: int
    time_span: dict[str, str]   # {"start": "...", "end": "..."}
    characters: list[str]
    locations: list[str]
    major_events: list[str]
    summary: str
