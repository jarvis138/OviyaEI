import json
import hashlib
from pathlib import Path


class TaxonomyManager:
    def __init__(self, taxonomy_path: str = "data/emotion_taxonomy.json"):
        """
        Initialize the TaxonomyManager and load the emotion taxonomy from the given file.
        
        Parameters:
            taxonomy_path (str): Filesystem path to the JSON taxonomy file; the taxonomy is loaded and verified during initialization and stored on the instance.
        """
        self.path = Path(taxonomy_path)
        self.taxonomy = self._load_and_verify()

    def _load_and_verify(self) -> dict:
        """
        Load the taxonomy JSON from self.path and verify its checksum when the taxonomy is marked frozen.
        
        If the loaded data contains a truthy 'frozen' field, the method verifies the SHA-256 checksum stored in the taxonomy's 'checksum' field (using the last colon-separated segment). Raises ValueError if the computed checksum does not match the stored checksum.
        
        Returns:
            dict: The parsed taxonomy dictionary.
        """
        with open(self.path) as f:
            data = json.load(f)
        if data.get('frozen'):
            content = json.dumps(data, sort_keys=True).encode()
            computed = hashlib.sha256(content).hexdigest()
            stored = (data.get('checksum', '').split(':') + [''])[-1]
            if computed != stored:
                raise ValueError("Taxonomy checksum mismatch")
        return data

    def map_emotion(self, source_dataset: str, emotion: str) -> str:
        """
        Map an emotion label from a source dataset to the canonical taxonomy label.
        
        Parameters:
            source_dataset (str): Identifier of the source dataset whose mapping to use.
            emotion (str): Emotion label from the source dataset to be mapped.
        
        Returns:
            str: The corresponding label in the taxonomy.
        
        Raises:
            ValueError: If the specified dataset is not known.
            ValueError: If the specified emotion is not present in the dataset's mappings.
        """
        m = self.taxonomy.get('mappings', {}).get(source_dataset)
        if not m:
            raise ValueError(f"Unknown dataset: {source_dataset}")
        if emotion not in m:
            raise ValueError(f"Unknown emotion '{emotion}' for {source_dataset}")
        return m[emotion]

    def get_version(self) -> str:
        """
        Return the taxonomy version.
        
        Returns:
            version (str): The taxonomy 'version' value, or "0.0" if the field is absent.
        """
        return self.taxonomy.get('version', '0.0')

