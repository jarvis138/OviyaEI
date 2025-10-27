import json
import hashlib
from pathlib import Path


class TaxonomyManager:
    def __init__(self, taxonomy_path: str = "data/emotion_taxonomy.json"):
        self.path = Path(taxonomy_path)
        self.taxonomy = self._load_and_verify()

    def _load_and_verify(self) -> dict:
        with open(self.path) as f:
            data = json.load(f)
        if data.get('frozen'):
            # Exclude checksum field from hash computation
            payload = dict(data)
            payload.pop('checksum', None)
            content = json.dumps(payload, sort_keys=True).encode()
            computed = hashlib.sha256(content).hexdigest()
            stored = data.get('checksum', '').split(':')[-1]
            if computed != stored:
                raise ValueError("Taxonomy checksum mismatch")
        return data

    def map_emotion(self, source_dataset: str, emotion: str) -> str:
        m = self.taxonomy.get('mappings', {}).get(source_dataset)
        if not m:
            raise ValueError(f"Unknown dataset: {source_dataset}")
        if emotion not in m:
            raise ValueError(f"Unknown emotion '{emotion}' for {source_dataset}")
        return m[emotion]

    def get_version(self) -> str:
        return self.taxonomy.get('version', '0.0')


