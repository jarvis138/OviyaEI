from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, List, Literal


class EmotionDimensions(BaseModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)

    @root_validator(skip_on_failure=True)
    def check_consistency(cls, values):
        # Placeholder for cross-checks with emotion_label
        """
        Perform cross-field consistency checks for emotion dimension values relative to other sample fields.
        
        Parameters:
            cls: The EmotionDimensions class.
            values (dict): A mapping of field names to their validated values for the entire model instance.
        
        Returns:
            dict: The (possibly adjusted) field-value mapping to continue model validation.
        """
        return values


class ToneFeatures(BaseModel):
    warmth: float = Field(ge=0.0, le=1.0)
    directness: float = Field(ge=0.0, le=1.0)
    formality: float = Field(ge=0.0, le=1.0)


class OviyaSample(BaseModel):
    utterance_id: str = Field(pattern=r'^[a-f0-9\-]{8,36}$')
    text: str = Field(min_length=1, max_length=2048)
    audio_path: Optional[str]

    emotion_label: Literal[
        'joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral', 'trust', 'anticipation'
    ]
    emotion_aliases: List[str] = Field(default_factory=list)
    emotion_dim: EmotionDimensions

    culture: str = Field(pattern=r'^[a-z]{2}_[a-z]{2}$')
    context: Optional[str]
    speaker_role: Literal['expresser', 'listener', 'observer']

    response: Optional[str]
    empathy_type: Optional[str]
    tone_features: Optional[ToneFeatures]

    source_dataset: Literal[
        'IEMOCAP', 'MELD', 'EmpatheticDialogues', 'DailyDialog',
        'GoEmotions', 'XED', 'EmoEvent', 'RAVDESS', 'CREMA-D',
        'SEMAINE', 'OpenSubtitles', 'Synthetic', 'Prolific'
    ]
    quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
    rater_agreement: Optional[float] = Field(ge=0.0, le=1.0)

    class Config:
        extra = 'forbid'

    @validator('text', 'response')
    def check_text_quality(cls, v):
        """
        Validate that a text value has at least two words and is not all uppercase.
        
        Returns the trimmed string when valid; returns None unchanged.
        
        Parameters:
            v (str | None): The text value to validate.
        
        Returns:
            str | None: The trimmed text value, or None if input was None.
        
        Raises:
            ValueError: If the text contains fewer than two words.
            ValueError: If the text is entirely uppercase.
        """
        if v is None:
            return v
        if len(v.split()) < 2:
            raise ValueError("Text too short")
        if v.isupper():
            raise ValueError("All caps text detected")
        return v.strip()

    @root_validator(skip_on_failure=True)
    def check_response_completeness(cls, values):
        """
        Ensure that when a `response` value is provided the corresponding `empathy_type` is also present.
        
        Parameters:
            values (dict): Mapping of field names to their current validated values.
        
        Returns:
            dict: The unchanged values mapping.
        
        Raises:
            ValueError: If `response` is present and `empathy_type` is missing.
        """
        if values.get('response') and not values.get('empathy_type'):
            raise ValueError("Response must have empathy_type")
        return values


def validate_parquet_batch(filepath: str, sample_rate: float = 1.0, strict: bool = True):
    """
    Validate a Parquet file of records against the OviyaSample schema and collect summary statistics.
    
    Parameters:
        filepath (str): Path to the Parquet file to validate.
        sample_rate (float): Fraction of rows to randomly sample from the file (0 < sample_rate <= 1.0).
        strict (bool): If True, raise a ValueError on the first validation failure; if False, collect errors and continue.
    
    Returns:
        result (dict): Summary of validation containing:
            - filepath (str): The input filepath.
            - valid (bool): True if no validation errors were found, False otherwise.
            - errors (List[str]): Up to 100 error messages, each indicating row index and error text.
            - warnings (List[str]): High-level dataset warnings (e.g., imbalance alerts).
            - statistics (dict): Aggregated statistics including:
                - total_rows (int): Number of rows processed (after sampling).
                - validated_rows (int): Number of rows that passed validation.
                - error_count (int): Number of rows that failed validation.
                - culture_distribution (dict): Counts per `culture` value if present.
                - emotion_distribution (dict): Counts per `emotion_label` value if present.
                - missing_responses (int): Count of missing `response` values if present.
                - avg_quality_score (float): Mean of `quality_score` if present, otherwise 0.0.
    """
    import pandas as pd
    import json as _json
    df = pd.read_parquet(filepath)
    if sample_rate < 1.0:
        df = df.sample(frac=sample_rate, random_state=42)
    errors, warnings = [], []
    validated = 0
    for idx, row in df.iterrows():
        try:
            record = row.to_dict()
            if 'emotion_dim' in record and isinstance(record['emotion_dim'], str):
                record['emotion_dim'] = _json.loads(record['emotion_dim'])
            OviyaSample(**record)
            validated += 1
        except Exception as e:
            msg = f"Row {idx}: {str(e)}"
            errors.append(msg)
            if strict:
                raise ValueError(f"Validation failed at {filepath}:{idx}\n{e}")
    stats = {
        'total_rows': len(df),
        'validated_rows': validated,
        'error_count': len(errors),
        'culture_distribution': df['culture'].value_counts().to_dict() if 'culture' in df else {},
        'emotion_distribution': df['emotion_label'].value_counts().to_dict() if 'emotion_label' in df else {},
        'missing_responses': int(df['response'].isna().sum()) if 'response' in df else 0,
        'avg_quality_score': float(df['quality_score'].mean()) if 'quality_score' in df else 0.0,
    }
    # Simple imbalance warnings
    if 'culture' in df and not df.empty:
        vc = df['culture'].value_counts()
        if vc.min() > 0 and (vc.max() / vc.min()) > 10:
            warnings.append("Severe culture imbalance detected")
    if 'emotion_label' in df and not df.empty:
        vc = df['emotion_label'].value_counts()
        if vc.min() > 0 and (vc.max() / vc.min()) > 5:
            warnings.append("Emotion imbalance detected")
    return {'filepath': filepath, 'valid': len(errors) == 0, 'errors': errors[:100], 'warnings': warnings, 'statistics': stats}


if __name__ == '__main__':
    import sys, glob
    if len(sys.argv) < 2:
        print("Usage: python data/schema_validator.py <path_pattern>")
        sys.exit(1)
    files = glob.glob(sys.argv[1], recursive=True)
    all_valid = True
    for f in files:
        if f.endswith('.parquet'):
            res = validate_parquet_batch(f, strict=False)
            ok = res['valid']
            print(("✓" if ok else "✗"), f, res['statistics'])
            if not ok:
                all_valid = False
    sys.exit(0 if all_valid else 1)

