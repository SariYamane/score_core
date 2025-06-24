from __future__ import annotations
from typing import List
from datetime import datetime, UTC
import json, itertools
from .models import MemoryEntry, EpisodeSummary

def to_episode_summary(memories: List[MemoryEntry],
                       episode_id: int) -> EpisodeSummary:
    """メモリ群 → EpisodeSummary 1 件"""
    start = min(m.timestamp for m in memories)
    end   = max(m.timestamp for m in memories)

    chars = sorted({m.who for m in memories})
    locs  = sorted({m.where for m in memories if m.where})
    events = [f"({m.who}, {m.predicate}, {m.object})"
              for m in memories if hasattr(m, "predicate")]

    summary_text = _llm_summarize(memories)   # ★LLM 呼び出し or ルール
    return EpisodeSummary(
        episode_id=episode_id,
        time_span={"start": start.isoformat(), "end": end.isoformat()},
        characters=chars,
        locations=locs,
        major_events=events[:10],
        summary=summary_text
    )

def _llm_summarize(memories: List[MemoryEntry]) -> str:
    """最小版：直近3件を繋げるだけ"""
    lines = [m.what for m in memories[-3:]]
    return " ".join(lines)
