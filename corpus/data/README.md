This project reuses schema assets from the repository root to maintain a single source of truth.

Use these paths when validating or importing:
- Validator: ../data/schema_validator.py
- Taxonomy: ../data/emotion_taxonomy.json

Example (run from repo root):
python data/schema_validator.py "oviya-corpus/normalized/*.parquet"

Alternative (run from within oviya-corpus/):
python ../data/schema_validator.py "normalized/*.parquet"


