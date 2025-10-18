import hashlib
from datetime import datetime
from typing import Dict


class PrivacyPreservingFeedbackStore:
    def __init__(self, embedding_model):
        """
        Store the embedding model used to encode text into vector embeddings.
        
        Parameters:
            embedding_model: An object that converts text to numeric embeddings (must provide an encode or similar method used by other class methods).
        """
        self.embedding_model = embedding_model

    def hash_user_id(self, user_id: str, salt: str) -> str:
        """
        Produce a deterministic anonymized identifier derived from a user identifier and a salt.
        
        Parameters:
            user_id (str): The original user identifier to anonymize.
            salt (str): A secret or non-secret string mixed into the identifier to vary or harden the output.
        
        Returns:
            str: The anonymized identifier as the first 16 hexadecimal characters of the SHA-256 digest of the concatenated inputs.
        """
        return hashlib.sha256(f"{user_id}{salt}".encode()).hexdigest()[:16]

    def anonymize_sample(self, sample: Dict, salt: str) -> Dict:
        """
        Create an anonymized, embedding-based representation of a feedback sample for privacy-preserving storage.
        
        Parameters:
            sample (Dict): A mapping containing the raw feedback. Required keys:
                - 'user_id' (str): identifier to be hashed.
                - 'user_message' (str): text to encode as the user embedding.
                - 'oviya_response' (str): text to encode as the response embedding.
                - 'feedback' (any): feedback scores or labels to preserve.
              Optional keys:
                - 'timestamp' (str): ISO timestamp to use instead of current time.
                - 'emotion_detected' (any): emotion metadata to include if present.
                - 'culture' (any): inferred culture to include if present.
                - 'session_number' (int): session index (defaults to 1).
                - 'cultural_weights' (dict): cultural weighting info (defaults to {}).
            salt (str): Salt value used when hashing the user identifier.
        
        Returns:
            Dict: An anonymized sample containing:
                - 'session_hash' (str): deterministic hash of the user identifier and salt.
                - 'user_embedding' (list): embedding vector of `user_message`.
                - 'response_embedding' (list): embedding vector of `oviya_response`.
                - 'cultural_weights' (dict): preserved cultural weights or an empty dict.
                - 'feedback_scores' (any): the original `feedback` value from `sample`.
                - 'timestamp' (str): provided timestamp or current UTC ISO timestamp.
                - 'metadata' (dict): auxiliary fields:
                    - 'message_length' (int): word count of `user_message`.
                    - 'response_length' (int): word count of `oviya_response`.
                    - 'emotion_detected' (any): copied from `sample` if present.
                    - 'culture_inferred' (any): copied from `sample` if present.
                    - 'session_number' (int): session number (defaults to 1).
        """
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

