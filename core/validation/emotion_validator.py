"""
Emotion Validation Framework for Oviya
Validates 49-emotion mapping consistency using test cases
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import statistics

# Import emotion detector
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from production.emotion_detector.detector import EmotionDetector


@dataclass
class EmotionTestCase:
    """Single emotion validation test case"""
    text: str
    expected_emotion: str
    min_confidence: float
    context: str = ""
    alternative_emotions: List[str] = None


class EmotionValidator:
    """
    Validate 49-emotion mapping consistency
    Uses test cases to measure accuracy and identify misclassifications
    """
    
    # Comprehensive test cases for 49 emotions
    TEST_CASES = [
        # Tier 1: Core Emotions (15 emotions)
        EmotionTestCase("I'm feeling really calm and peaceful right now", "calm_supportive", 0.7),
        EmotionTestCase("I'm so sorry you're going through this", "empathetic_sad", 0.7),
        EmotionTestCase("This is amazing! I'm so excited!", "joyful_excited", 0.8),
        EmotionTestCase("Haha, that's hilarious!", "playful", 0.7),
        EmotionTestCase("I know exactly what to do", "confident", 0.7),
        EmotionTestCase("I'm really worried about this", "concerned_anxious", 0.7),
        EmotionTestCase("Tell me more about that", "curious", 0.6),
        EmotionTestCase("I'm feeling really frustrated", "frustrated", 0.7),
        EmotionTestCase("This is so romantic", "romantic", 0.7),
        EmotionTestCase("I'm feeling a bit shy", "shy", 0.6),
        EmotionTestCase("I'm so grateful for your help", "grateful", 0.7),
        EmotionTestCase("I'm feeling really tired", "tired", 0.6),
        EmotionTestCase("Let me think about this carefully", "thoughtful_reflective", 0.6),
        EmotionTestCase("I'm feeling pretty neutral about it", "neutral", 0.5),
        EmotionTestCase("I'm feeling really motivated", "motivated", 0.7),
        
        # Tier 2: Nuanced Emotions (17 emotions)
        EmotionTestCase("I'm feeling really hopeful about the future", "hopeful", 0.7),
        EmotionTestCase("I'm so disappointed", "disappointed", 0.7),
        EmotionTestCase("This is making me nostalgic", "nostalgic", 0.6),
        EmotionTestCase("I'm feeling a bit jealous", "jealous", 0.6),
        EmotionTestCase("I'm really proud of you", "proud", 0.7),
        EmotionTestCase("I'm feeling guilty about that", "guilty", 0.6),
        EmotionTestCase("I'm feeling a bit embarrassed", "embarrassed", 0.6),
        EmotionTestCase("I'm feeling really lonely", "lonely", 0.7),
        EmotionTestCase("I'm feeling content with life", "content", 0.6),
        EmotionTestCase("I'm feeling really bored", "bored", 0.6),
        EmotionTestCase("That's really surprising!", "surprised", 0.7),
        EmotionTestCase("I'm feeling a bit confused", "confused", 0.6),
        EmotionTestCase("I'm feeling really relieved", "relieved", 0.7),
        EmotionTestCase("I'm feeling a bit irritated", "irritated_annoyed", 0.7),
        EmotionTestCase("I'm feeling really passionate about this", "passionate", 0.7),
        EmotionTestCase("I'm feeling a bit melancholic", "melancholic", 0.6),
        EmotionTestCase("I'm feeling really enthusiastic", "enthusiastic", 0.7),
        
        # Tier 3: Complex Emotions (17 emotions)
        EmotionTestCase("I'm feeling really vulnerable right now", "vulnerable", 0.6),
        EmotionTestCase("I'm feeling quite defensive", "defensive", 0.6),
        EmotionTestCase("I'm feeling a bit skeptical", "skeptical", 0.6),
        EmotionTestCase("I'm feeling really determined", "determined", 0.7),
        EmotionTestCase("I'm feeling a bit resentful", "resentful", 0.6),
        EmotionTestCase("I'm feeling really compassionate", "compassionate", 0.7),
        EmotionTestCase("I'm feeling quite indifferent", "indifferent", 0.5),
        EmotionTestCase("I'm feeling really overwhelmed", "overwhelmed", 0.7),
        EmotionTestCase("I'm feeling quite serene", "serene", 0.6),
        EmotionTestCase("I'm feeling really mischievous", "mischievous", 0.6),
        EmotionTestCase("I'm feeling a bit apprehensive", "apprehensive", 0.6),
        EmotionTestCase("I'm feeling really affectionate", "affectionate", 0.7),
        EmotionTestCase("I'm feeling quite dismissive", "dismissive", 0.6),
        EmotionTestCase("I'm feeling really inspired", "inspired", 0.7),
        EmotionTestCase("I'm feeling a bit nervous", "nervous", 0.7),
        EmotionTestCase("I'm feeling really sympathetic", "sympathetic", 0.7),
        EmotionTestCase("I'm feeling quite bitter", "bitter", 0.6),
    ]
    
    def __init__(self):
        """Initialize emotion validator"""
        self.detector = EmotionDetector()
        print("✅ Emotion Validator initialized")
    
    def validate_mapping(self) -> Dict:
        """
        Validate 49-emotion mapping consistency
        
        Returns:
            {
                'accuracy': float,
                'total_tests': int,
                'passed': int,
                'failed': int,
                'confusion_matrix': Dict,
                'misclassifications': List[Dict],
                'emotion_accuracy': Dict  # Per-emotion accuracy
            }
        """
        print("🧪 Validating 49-emotion mapping...")
        
        results = []
        misclassifications = []
        emotion_stats = {}
        
        for test_case in self.TEST_CASES:
            # Detect emotion
            detected = self.detector.detect_emotion(test_case.text)
            confidence = 0.8  # Default confidence (detector doesn't return confidence)
            
            # Check if correct
            is_correct = detected == test_case.expected_emotion
            
            # Track result
            results.append(is_correct)
            
            # Track per-emotion stats
            if test_case.expected_emotion not in emotion_stats:
                emotion_stats[test_case.expected_emotion] = {'correct': 0, 'total': 0}
            
            emotion_stats[test_case.expected_emotion]['total'] += 1
            if is_correct:
                emotion_stats[test_case.expected_emotion]['correct'] += 1
            
            # Track misclassifications
            if not is_correct:
                misclassifications.append({
                    'text': test_case.text,
                    'expected': test_case.expected_emotion,
                    'detected': detected,
                    'confidence': confidence
                })
        
        # Calculate overall accuracy
        accuracy = sum(results) / len(results) if results else 0.0
        
        # Calculate per-emotion accuracy
        emotion_accuracy = {}
        for emotion, stats in emotion_stats.items():
            emotion_accuracy[emotion] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        # Build confusion matrix
        confusion_matrix = self._build_confusion_matrix(misclassifications)
        
        return {
            'accuracy': accuracy,
            'total_tests': len(results),
            'passed': sum(results),
            'failed': len(results) - sum(results),
            'confusion_matrix': confusion_matrix,
            'misclassifications': misclassifications,
            'emotion_accuracy': emotion_accuracy
        }
    
    def _build_confusion_matrix(self, misclassifications: List[Dict]) -> Dict:
        """Build confusion matrix from misclassifications"""
        matrix = {}
        
        for mis in misclassifications:
            expected = mis['expected']
            detected = mis['detected']
            
            if expected not in matrix:
                matrix[expected] = {}
            
            matrix[expected][detected] = matrix[expected].get(detected, 0) + 1
        
        return matrix
    
    def print_report(self, results: Dict):
        """
        Print validation report
        
        Args:
            results: Results from validate_mapping()
        """
        print("\n" + "=" * 60)
        print("EMOTION VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\n📊 Overall Results:")
        print(f"   Accuracy: {results['accuracy']:.1%}")
        print(f"   Passed: {results['passed']}/{results['total_tests']}")
        print(f"   Failed: {results['failed']}/{results['total_tests']}")
        
        print(f"\n📊 Per-Emotion Accuracy:")
        # Sort by accuracy
        sorted_emotions = sorted(
            results['emotion_accuracy'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for emotion, accuracy in sorted_emotions[:10]:  # Top 10
            print(f"   {emotion:30s}: {accuracy:.1%}")
        
        if results['failed'] > 0:
            print(f"\n❌ Misclassifications ({results['failed']}):")
            for mis in results['misclassifications'][:10]:  # Show first 10
                print(f"   Text: {mis['text'][:50]}...")
                print(f"   Expected: {mis['expected']}")
                print(f"   Detected: {mis['detected']}")
                print()
        
        if results['confusion_matrix']:
            print(f"\n🔀 Confusion Matrix (Top Confusions):")
            # Flatten and sort
            confusions = []
            for expected, detected_dict in results['confusion_matrix'].items():
                for detected, count in detected_dict.items():
                    confusions.append((expected, detected, count))
            
            confusions.sort(key=lambda x: x[2], reverse=True)
            
            for expected, detected, count in confusions[:5]:  # Top 5
                print(f"   {expected} → {detected}: {count} times")
        
        print("\n" + "=" * 60)
    
    def export_results(self, results: Dict, output_file: Path):
        """
        Export validation results to JSON
        
        Args:
            results: Results from validate_mapping()
            output_file: Output JSON file path
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Exported validation results to {output_file}")
    
    def add_test_case(self, test_case: EmotionTestCase):
        """
        Add a custom test case
        
        Args:
            test_case: EmotionTestCase instance
        """
        self.TEST_CASES.append(test_case)
        print(f"✅ Added test case for emotion: {test_case.expected_emotion}")
    
    def validate_single_emotion(self, emotion: str) -> Dict:
        """
        Validate a single emotion
        
        Args:
            emotion: Emotion to validate
            
        Returns:
            Validation results for that emotion
        """
        relevant_cases = [tc for tc in self.TEST_CASES if tc.expected_emotion == emotion]
        
        if not relevant_cases:
            return {
                'emotion': emotion,
                'test_cases': 0,
                'message': 'No test cases found for this emotion'
            }
        
        results = []
        for test_case in relevant_cases:
            detected = self.detector.detect_emotion(test_case.text)
            results.append(detected == test_case.expected_emotion)
        
        return {
            'emotion': emotion,
            'test_cases': len(results),
            'passed': sum(results),
            'accuracy': sum(results) / len(results) if results else 0.0
        }


def test_emotion_validator():
    """Test emotion validation framework"""
    print("=" * 60)
    print("Testing Emotion Validation Framework")
    print("=" * 60)
    
    validator = EmotionValidator()
    
    # Run validation
    results = validator.validate_mapping()
    
    # Print report
    validator.print_report(results)
    
    # Export results
    output_dir = Path("validation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    validator.export_results(results, output_dir / "emotion_validation.json")
    
    print("\n✅ Emotion validation complete!")


if __name__ == "__main__":
    test_emotion_validator()

