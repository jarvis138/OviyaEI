#!/usr/bin/env python3
"""
Phase 4: Emotion Detection from Speech

Uses acoustic emotion models to detect emotional tone, prosody, and intensity
from speech audio and transcribed text.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AcousticEmotionDetector:
    """
    Phase 4: Acoustic Emotion Detection

    Uses wav2vec2-xlsr-speech-emotion model to detect emotions from speech audio.
    """

    def __init__(self):
        """Initialize acoustic emotion detector"""
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Emotion labels (wav2vec2-xlsr-speech-emotion)
        self.emotion_labels = [
            'angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'
        ]

        # Load model
        self._load_model()

    def _load_model(self):
        """Load wav2vec2 emotion detection model"""
        try:
            from transformers import AutoProcessor, AutoModelForAudioClassification

            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info("✅ Acoustic emotion detector loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load emotion detection model: {e}")
            # Fallback to simple rule-based detection
            self.model = None
            self.processor = None

    def detect_emotion_from_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, any]:
        """
        Detect emotion from speech audio

        Args:
            audio: Audio array (16kHz)
            sample_rate: Audio sample rate

        Returns:
            Emotion detection results
        """
        if self.model is None or self.processor is None:
            # Fallback to basic detection
            return self._fallback_emotion_detection(audio)

        try:
            # Ensure correct format
            if len(audio.shape) > 1:
                audio = audio.flatten()

            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

            # Get top emotion
            confidence, predicted_idx = torch.max(probabilities, dim=-1)
            predicted_emotion = self.emotion_labels[predicted_idx.item()]

            # Get all emotion probabilities
            emotion_scores = {}
            for i, label in enumerate(self.emotion_labels):
                emotion_scores[label] = probabilities[i].item()

            # Calculate intensity (variance in emotion probabilities)
            intensity = float(torch.std(probabilities).item())

            return {
                'primary_emotion': predicted_emotion,
                'confidence': float(confidence.item()),
                'intensity': intensity,
                'emotion_scores': emotion_scores,
                'all_emotions': self.emotion_labels
            }

        except Exception as e:
            logger.error(f"❌ Emotion detection failed: {e}")
            return self._fallback_emotion_detection(audio)

    def _fallback_emotion_detection(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Fallback emotion detection using audio features
        """
        # Simple rule-based detection based on audio energy and zero crossings
        energy = np.mean(np.abs(audio))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / len(audio)

        # Mock emotion detection
        if energy > 0.1:  # High energy = anger or excitement
            primary_emotion = 'angry' if zero_crossings > 0.1 else 'happy'
            confidence = min(0.7, energy * 10)
        elif energy > 0.05:  # Medium energy = calm or neutral
            primary_emotion = 'neutral'
            confidence = 0.5
        else:  # Low energy = sadness or calm
            primary_emotion = 'sad'
            confidence = 0.6

        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'intensity': float(energy),
            'emotion_scores': {emotion: 0.1 for emotion in self.emotion_labels},
            'all_emotions': self.emotion_labels,
            'fallback': True
        }


class TextEmotionAnalyzer:
    """
    Text-based emotion analysis from transcription
    """

    def __init__(self):
        """Initialize text emotion analyzer"""
        # Simple keyword-based emotion detection
        self.emotion_keywords = {
            'anxiety': ['worried', 'anxious', 'nervous', 'scared', 'fear', 'panic'],
            'sadness': ['sad', 'depressed', 'lonely', 'alone', 'grief', 'cry'],
            'anger': ['angry', 'mad', 'frustrated', 'hate', 'furious', 'rage'],
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing'],
            'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'quiet', 'tranquil'],
            'confusion': ['confused', 'lost', 'unsure', 'uncertain', 'puzzled'],
            'gratitude': ['thankful', 'grateful', 'appreciate', 'thanks'],
            'hope': ['hope', 'optimistic', 'future', 'better', 'positive']
        }

    def analyze_text_emotion(self, text: str) -> Dict[str, any]:
        """
        Analyze emotion from transcribed text

        Args:
            text: Transcribed speech text

        Returns:
            Text-based emotion analysis
        """
        text_lower = text.lower()
        emotion_scores = {}

        # Count keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword)
            emotion_scores[emotion] = min(1.0, score / 5.0)  # Normalize

        # Find primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
        else:
            primary_emotion = 'neutral'
            confidence = 0.5

        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'emotion_scores': emotion_scores,
            'text_length': len(text)
        }


class ComprehensiveEmotionDetector:
    """
    Complete Phase 4: Comprehensive Emotion Detection

    Combines acoustic and text-based emotion detection for accurate emotional analysis.
    """

    def __init__(self):
        """Initialize comprehensive emotion detector"""
        self.acoustic_detector = AcousticEmotionDetector()
        self.text_analyzer = TextEmotionAnalyzer()

        logger.info("🎭 Comprehensive emotion detector initialized")

    def detect_emotion(self, audio: np.ndarray, text: str, sample_rate: int = 16000) -> Dict[str, any]:
        """
        Comprehensive emotion detection from both audio and text

        Args:
            audio: Speech audio array
            text: Transcribed speech text
            sample_rate: Audio sample rate

        Returns:
            Comprehensive emotion analysis
        """
        # Phase 4.1: Acoustic emotion detection
        acoustic_result = self.acoustic_detector.detect_emotion_from_audio(audio, sample_rate)

        # Phase 4.2: Text emotion analysis
        text_result = self.text_analyzer.analyze_text_emotion(text)

        # Phase 4.3: Combine acoustic and text analysis
        combined_emotion = self._combine_emotion_analyses(acoustic_result, text_result)

        return {
            'primary_emotion': combined_emotion['emotion'],
            'confidence': combined_emotion['confidence'],
            'intensity': acoustic_result.get('intensity', 0.5),
            'acoustic_analysis': acoustic_result,
            'text_analysis': text_result,
            'combined_score': combined_emotion,
            'timestamp': torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else 0
        }

    def _combine_emotion_analyses(self, acoustic: Dict, text: Dict) -> Dict[str, any]:
        """
        Combine acoustic and text emotion analyses

        Args:
            acoustic: Acoustic emotion detection results
            text: Text emotion analysis results

        Returns:
            Combined emotion with confidence
        """
        # Weight acoustic analysis more heavily (60%) than text (40%)
        acoustic_weight = 0.6
        text_weight = 0.4

        # Get emotion scores from both analyses
        acoustic_scores = acoustic.get('emotion_scores', {})
        text_scores = text.get('emotion_scores', {})

        # Combine scores for each emotion
        combined_scores = {}
        all_emotions = set(acoustic_scores.keys()) | set(text_scores.keys())

        for emotion in all_emotions:
            acoustic_score = acoustic_scores.get(emotion, 0.0)
            text_score = text_scores.get(emotion, 0.0)
            combined_scores[emotion] = acoustic_weight * acoustic_score + text_weight * text_score

        # Find primary emotion
        if combined_scores:
            primary_emotion = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[primary_emotion]
        else:
            primary_emotion = 'neutral'
            confidence = 0.5

        return {
            'emotion': primary_emotion,
            'confidence': confidence,
            'combined_scores': combined_scores
        }


# Global emotion detector instance
_emotion_detector = None

def get_emotion_detector() -> ComprehensiveEmotionDetector:
    """Get or create global emotion detector"""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = ComprehensiveEmotionDetector()
    return _emotion_detector


# Test function
def test_emotion_detection():
    """Test emotion detection components"""
    print("🧪 TESTING EMOTION DETECTION")
    print("=" * 50)

    detector = get_emotion_detector()

    # Test with sample audio (silence = neutral)
    sample_audio = np.random.normal(0, 0.01, 16000)  # 1 second of near-silence
    test_text = "I feel really anxious about what might happen"

    print("Testing emotion detection...")
    result = detector.detect_emotion(sample_audio, test_text)

    print(f"✅ Primary emotion: {result['primary_emotion']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Intensity: {result['intensity']:.2f}")

    if 'acoustic_analysis' in result:
        acoustic = result['acoustic_analysis']
        print(f"   Acoustic: {acoustic.get('primary_emotion', 'unknown')}")

    if 'text_analysis' in result:
        text_analysis = result['text_analysis']
        print(f"   Text: {text_analysis.get('primary_emotion', 'unknown')}")

    print("✅ Emotion detection test completed")


if __name__ == "__main__":
    test_emotion_detection()
