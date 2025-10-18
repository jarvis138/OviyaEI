from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, List, Literal


class EmotionDimensions(BaseModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)

    @root_validator
    def check_consistency(cls, values):
        # Placeholder for cross-checks with emotion_label
        return values


class ToneFeatures(BaseModel):
    warmth: float = Field(ge=0.0, le=1.0)
    directness: float = Field(ge=0.0, le=1.0)
    formality: float = Field(ge=0.0, le=1.0)


class OviyaSample(BaseModel):
    utterance_id: str = Field(regex=r'^[a-f0-9\-]{8,36}$')
    text: str = Field(min_length=1, max_length=2048)
    audio_path: Optional[str]

    emotion_label: Literal[
        'joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral', 'trust', 'anticipation'
    ]
    emotion_aliases: List[str] = Field(default_factory=list)
    emotion_dim: EmotionDimensions

    culture: str = Field(regex=r'^[a-z]{2}_[a-z]{2}$')
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
        if v is None:
            return v
        if len(v.split()) < 2:
            raise ValueError("Text too short")
        if v.isupper():
            raise ValueError("All caps text detected")
        return v.strip()

    @root_validator
    def check_response_completeness(cls, values):
        if values.get('response') and not values.get('empathy_type'):
            raise ValueError("Response must have empathy_type")
        return values


def validate_parquet_batch(filepath: str, sample_rate: float = 1.0, strict: bool = True):
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


