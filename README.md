## SCORE 実装

## 機能
- TF-IDF → SBERT → カーネル結合の三段階検索
- 重要度・新規性・時間経過を考慮したスコアリング
- FAISS

## 構成
```
score_core/          │  パッケージ本体
├── __init__.py      │  公開 API (Retriever など) を re-export
├── retriever.py     │  TF-IDF → SBERT → カーネル検索のエンジン
├── kernel.py        │  一貫性スコア α·sim + β·novelty + γ·recency
└── utils/           │  補助モジュール
    ├── indexing.py  │   └─ FAISS インデックス構築／ロード
    └── logger.py    │      ログ設定（色付き ConsoleHandler など）
tests/               │  ユニットテスト
└── test_retriever.py│   └─ 取得順位・閾値の検証
scripts/             │  CLI ツール群
└── build_index.py   │   └─ jsonl → インデックス生成
```
