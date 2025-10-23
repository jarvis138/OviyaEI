import hashlib
from datetime import datetime
from typing import Dict


class PrivacyPreservingFeedbackStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def hash_user_id(self, user_id: str, salt: str) -> str:
        return hashlib.sha256(f"{user_id}{salt}".encode()).hexdigest()[:16]

    def anonymize_sample(self, sample: Dict, salt: str) -> Dict:
        user_hash = self.hash_user_id(sample['user_id'], salt)
        user_emb = self.embedding_model.encode(sample['user_message']).tolist()
        resp_emb = self.embedding_model.encode(sample['oviya_response']).tolist()
        ts = sample.get('timestamp') or datetime.utcnow().isoformat()
        meta = {
            'message_length': len(sample['user_message'].split()),
            'response_length': len(sample['oviya_response'].split()),
            'emotion_detected': sample.get('emotion_detected'),
            'culture_inferred': sample.get('culture'),
            'session_number': sample.get('session_number', 1)
        }
        return {
            'session_hash': user_hash,
            'user_embedding': user_emb,
            'response_embedding': resp_emb,
            'cultural_weights': sample.get('cultural_weights', {}),
            'feedback_scores': sample['feedback'],
            'timestamp': ts,
            'metadata': meta
        }




