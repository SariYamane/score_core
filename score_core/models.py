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
    entity_id: str | None = None
    state: str = "active"

class EpisodeSummary(BaseModel):
    episode_id: int
    time_span: str
    characters: List[str]
    summary: str

#検索ハブ
class SearchIndex(BaseModel):
    tfidf: 'sklearn.feature_extraction.text.TfidfVectorizer'
    matrix: 'scipy.sparse.csr_matrix'            # TF-IDF 行列
    faiss_index_path: str | None = None          # _faiss_index/index.faiss
