"""
Acoustic Emotion Detection using Wav2Vec2
Detects emotion from audio features (pitch, tone, energy, rhythm)
Complements text-based emotion detection for higher accuracy
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available - install with: pip install transformers")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not available - install with: pip install librosa")


class AcousticEmotionDetector:
    """
    Detect emotion from acoustic features using pre-trained Wav2Vec2 model
    Provides arousal (energy) and valence (positive/negative) scores
    """
    
    # Emotion mapping from model outputs to Oviya's 49-emotion taxonomy
    EMOTION_MAPPING = {
        'angry': ['frustrated', 'irritated_annoyed', 'defensive'],
        'happy': ['joyful_excited', 'playful', 'grateful'],
        'sad': ['empathetic_sad', 'melancholic', 'disappointed'],
        'neutral': ['calm_supportive', 'neutral', 'thoughtful_reflective'],
        'fear': ['concerned_anxious', 'nervous', 'vulnerable'],
        'disgust': ['skeptical', 'dismissive'],
        'surprise': ['surprised', 'curious']
    }
    
    def __init__(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initialize acoustic emotion detector
        
        Args:
            model_name: HuggingFace model for emotion recognition
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.model_name = model_name
        self.sample_rate = 16000
        
        # Fallback to feature-based detection if model unavailable
        self.use_model = TRANSFORMERS_AVAILABLE and LIBROSA_AVAILABLE
        
        if self.use_model:
            self._load_model()
        else:
            print("⚠️  Using fallback feature-based emotion detection")
    
    def _load_model(self):
        """Load pre-trained Wav2Vec2 emotion model"""
        try:
            print(f"🎤 Loading acoustic emotion model: {self.model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Acoustic emotion model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            self.use_model = False
    
    def detect_emotion(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Detect emotion from audio
        
        Args:
            audio: Audio array (mono, float32)
            sample_rate: Sample rate of audio
            
        Returns:
            {
                'emotion': 'happy',
                'confidence': 0.85,
                'arousal': 0.7,  # Energy level (0-1)
                'valence': 0.8,  # Positive/negative (-1 to 1)
                'oviya_emotions': ['joyful_excited', 'playful'],  # Mapped to 49 emotions
                'acoustic_features': {...}
            }
        """
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            if LIBROSA_AVAILABLE:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            else:
                print("⚠️  Cannot resample without librosa")
        
        if self.use_model and self.model is not None:
            return self._detect_with_model(audio)
        else:
            return self._detect_with_features(audio)
    
    def _detect_with_model(self, audio: np.ndarray) -> Dict:
        """Detect emotion using Wav2Vec2 model"""
        try:
            # Process audio
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top emotion
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
            
            # Map to emotion label (model-specific)
            emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise']
            emotion = emotion_labels[predicted_id] if predicted_id < len(emotion_labels) else 'neutral'
            
            # Calculate arousal and valence
            arousal, valence = self._calculate_arousal_valence(audio)
            
            # Map to Oviya's 49 emotions
            oviya_emotions = self.EMOTION_MAPPING.get(emotion, ['neutral'])
            
            # Get acoustic features
            features = self._extract_acoustic_features(audio)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'arousal': arousal,
                'valence': valence,
                'oviya_emotions': oviya_emotions,
                'acoustic_features': features,
                'method': 'model'
            }
            
        except Exception as e:
            print(f"⚠️  Model detection failed: {e}")
            return self._detect_with_features(audio)
    
    def _detect_with_features(self, audio: np.ndarray) -> Dict:
        """Fallback: Detect emotion using acoustic features"""
        features = self._extract_acoustic_features(audio)
        
        # Simple rule-based emotion detection
        arousal = features['energy']
        valence = features['spectral_centroid_normalized']
        
        # Map to basic emotions
        if arousal > 0.7 and valence > 0.6:
            emotion = 'happy'
        elif arousal > 0.7 and valence < 0.4:
            emotion = 'angry'
        elif arousal < 0.3 and valence < 0.4:
            emotion = 'sad'
        elif arousal < 0.3 and valence > 0.6:
            emotion = 'neutral'
        elif arousal > 0.6 and valence < 0.5:
            emotion = 'fear'
        else:
            emotion = 'neutral'
        
        oviya_emotions = self.EMOTION_MAPPING.get(emotion, ['neutral'])
        
        return {
            'emotion': emotion,
            'confidence': 0.6,  # Lower confidence for feature-based
            'arousal': arousal,
            'valence': valence,
            'oviya_emotions': oviya_emotions,
            'acoustic_features': features,
            'method': 'features'
        }
    
    def _extract_acoustic_features(self, audio: np.ndarray) -> Dict:
        """Extract acoustic features from audio"""
        if not LIBROSA_AVAILABLE:
            return {
                'energy': 0.5,
                'pitch_mean': 0.5,
                'pitch_std': 0.1,
                'spectral_centroid_normalized': 0.5,
                'zero_crossing_rate': 0.5,
                'tempo': 120.0
            }
        
        try:
            # Energy (RMS)
            energy = np.sqrt(np.mean(audio**2))
            energy_normalized = min(energy * 10, 1.0)  # Normalize to 0-1
            
            # Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            pitch_mean = np.mean(pitch_values) if pitch_values else 0
            pitch_std = np.std(pitch_values) if pitch_values else 0
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_centroid_mean = np.mean(spectral_centroid)
            spectral_centroid_normalized = min(spectral_centroid_mean / 4000, 1.0)  # Normalize
            
            # Zero crossing rate (noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            return {
                'energy': float(energy_normalized),
                'pitch_mean': float(pitch_mean),
                'pitch_std': float(pitch_std),
                'spectral_centroid_normalized': float(spectral_centroid_normalized),
                'zero_crossing_rate': float(zcr_mean),
                'tempo': float(tempo)
            }
            
        except Exception as e:
            print(f"⚠️  Feature extraction failed: {e}")
            return {
                'energy': 0.5,
                'pitch_mean': 0.5,
                'pitch_std': 0.1,
                'spectral_centroid_normalized': 0.5,
                'zero_crossing_rate': 0.5,
                'tempo': 120.0
            }
    
    def _calculate_arousal_valence(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Calculate arousal (energy) and valence (positive/negative) from audio
        
        Returns:
            (arousal, valence) both in range 0-1
        """
        features = self._extract_acoustic_features(audio)
        
        # Arousal = energy + tempo (normalized)
        arousal = (features['energy'] + min(features['tempo'] / 200, 1.0)) / 2
        
        # Valence = spectral brightness - pitch variance
        valence = features['spectral_centroid_normalized'] - (features['pitch_std'] / 100)
        valence = max(0, min(1, valence))  # Clamp to 0-1
        
        return arousal, valence
    
    def combine_with_text_emotion(
        self, 
        acoustic_result: Dict, 
        text_emotion: str, 
        text_confidence: float = 0.8,
        acoustic_weight: float = 0.6
    ) -> Dict:
        """
        Combine acoustic and text-based emotion detection
        
        Args:
            acoustic_result: Result from detect_emotion()
            text_emotion: Emotion detected from text
            text_confidence: Confidence of text detection
            acoustic_weight: Weight for acoustic (0-1), text gets (1-acoustic_weight)
            
        Returns:
            Combined emotion result with weighted confidence
        """
        acoustic_emotions = acoustic_result['oviya_emotions']
        acoustic_confidence = acoustic_result['confidence']
        
        # If text emotion matches acoustic, boost confidence
        if text_emotion in acoustic_emotions:
            combined_confidence = (acoustic_confidence * acoustic_weight + 
                                 text_confidence * (1 - acoustic_weight))
            return {
                'emotion': text_emotion,
                'confidence': combined_confidence,
                'source': 'combined_match',
                'acoustic_result': acoustic_result,
                'text_emotion': text_emotion
            }
        
        # If mismatch, use weighted average
        if acoustic_confidence * acoustic_weight > text_confidence * (1 - acoustic_weight):
            # Acoustic wins
            return {
                'emotion': acoustic_emotions[0],
                'confidence': acoustic_confidence * acoustic_weight,
                'source': 'acoustic_dominant',
                'acoustic_result': acoustic_result,
                'text_emotion': text_emotion
            }
        else:
            # Text wins
            return {
                'emotion': text_emotion,
                'confidence': text_confidence * (1 - acoustic_weight),
                'source': 'text_dominant',
                'acoustic_result': acoustic_result,
                'text_emotion': text_emotion
            }


def test_acoustic_emotion():
    """Test acoustic emotion detection"""
    print("=" * 60)
    print("Testing Acoustic Emotion Detection")
    print("=" * 60)
    
    detector = AcousticEmotionDetector()
    
    # Test with synthetic audio
    print("\n📝 Test 1: Synthetic happy audio (high energy, high pitch)")
    happy_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.5  # 2 seconds
    result = detector.detect_emotion(happy_audio)
    
    print(f"   Emotion: {result['emotion']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Arousal: {result['arousal']:.2f}")
    print(f"   Valence: {result['valence']:.2f}")
    print(f"   Oviya emotions: {result['oviya_emotions']}")
    print(f"   Method: {result['method']}")
    
    print("\n📝 Test 2: Combine with text emotion")
    combined = detector.combine_with_text_emotion(
        result,
        text_emotion='joyful_excited',
        text_confidence=0.85
    )
    print(f"   Combined emotion: {combined['emotion']}")
    print(f"   Combined confidence: {combined['confidence']:.2f}")
    print(f"   Source: {combined['source']}")
    
    print("\n✅ Acoustic emotion detection test complete!")


if __name__ == "__main__":
    test_acoustic_emotion()

