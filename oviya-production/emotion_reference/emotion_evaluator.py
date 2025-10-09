"""
Emotion Transfer Evaluator

Tests CSM's ability to reproduce emotions from OpenVoice V2 references.
This is Stage 0 - baseline evaluation before any fine-tuning.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
import time


class EmotionTransferEvaluator:
    """
    Evaluates emotion transfer from OpenVoice V2 (teacher) to CSM (student).
    
    This tests if CSM can reproduce emotional prosody when given
    OpenVoice V2's emotional reference audio as context.
    """
    
    def __init__(
        self,
        teacher,  # OpenVoiceEmotionTeacher
        student,  # HybridVoiceEngine (CSM)
        config_path: str = "config/emotion_reference.json"
    ):
        """Initialize evaluator with teacher and student models."""
        self.teacher = teacher
        self.student = student
        self.config = self._load_config(config_path)
        self.emotion_classifier = None  # Optional: load if available
    
    def _load_config(self, config_path: str) -> dict:
        """Load test configuration."""
        try:
            with open(config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Default test sentences for each emotion."""
        return {
            "test_sentences": {
                "calm_supportive": "Take a deep breath, I'm here with you.",
                "empathetic_sad": "I understand how you feel, I'm so sorry.",
                "joyful_excited": "That's amazing! I'm so proud of you!",
                "playful": "You're so funny, I love your sense of humor!",
                "confident": "You've got this, I believe in you.",
                "concerned_anxious": "I'm worried about you, are you okay?",
                "angry_firm": "That's not acceptable.",
                "neutral": "Hello, how are you doing today?"
            }
        }
    
    def test_emotion_transfer(
        self,
        emotion: str,
        text: Optional[str] = None
    ) -> Dict:
        """
        Test if CSM can reproduce emotion from OpenVoice V2 reference.
        
        Flow:
        1. Get OpenVoice V2 reference audio for emotion
        2. Feed reference + text to CSM
        3. Save both for comparison
        4. (Optional) Score similarity
        
        Args:
            emotion: Emotion label to test
            text: Text to speak (uses default if None)
        
        Returns:
            Dict with test results
        """
        # Use default test sentence if not provided
        if text is None:
            text = self.config["test_sentences"].get(
                emotion,
                "This is a test sentence."
            )
        
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {emotion}")
        print(f"   Text: {text}")
        print(f"{'='*60}")
        
        # Create output directory
        output_dir = Path("output/emotion_transfer")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Get teacher reference from OpenVoice V2
            print("\n[1/3] Getting OpenVoice V2 reference...")
            teacher_audio, sr = self.teacher.get_reference_audio(emotion)
            
            # Save teacher reference
            teacher_path = output_dir / f"teacher_{emotion}.wav"
            torchaudio.save(str(teacher_path), teacher_audio.unsqueeze(0), sr)
            print(f"   ✅ Teacher reference saved: {teacher_path}")
            
            # Step 2: Generate CSM audio conditioned on reference
            print("\n[2/3] Generating CSM audio with reference...")
            
            # Prepare emotion parameters for CSM
            emotion_params = {
                "emotion_label": emotion,
                "intensity": 0.7,
                "pitch_scale": 1.0,
                "rate_scale": 1.0,
                "energy_scale": 1.0
            }
            
            start_time = time.time()
            
            # Generate with CSM using reference
            csm_audio = self.student.generate_with_reference(
                text=text,
                reference_audio=teacher_audio,
                emotion_params=emotion_params
            )
            
            generation_time = time.time() - start_time
            
            # Save CSM output
            csm_path = output_dir / f"csm_{emotion}.wav"
            self.student.save_audio(csm_audio, str(csm_path))
            print(f"   ✅ CSM output saved: {csm_path}")
            print(f"   ⚡ Generation time: {generation_time*1000:.0f}ms")
            
            # Step 3: Basic audio analysis
            print("\n[3/3] Analyzing outputs...")
            
            teacher_duration = teacher_audio.shape[0] / sr
            csm_duration = csm_audio.shape[0] / 24000  # CSM sample rate
            
            print(f"   Teacher duration: {teacher_duration:.2f}s")
            print(f"   CSM duration: {csm_duration:.2f}s")
            
            # Calculate basic similarity (optional)
            similarity_score = self._calculate_basic_similarity(
                teacher_audio,
                csm_audio
            )
            
            print(f"   Similarity score: {similarity_score:.3f}")
            
            result = {
                "emotion": emotion,
                "text": text,
                "teacher_audio": str(teacher_path),
                "csm_audio": str(csm_path),
                "teacher_duration": float(teacher_duration),
                "csm_duration": float(csm_duration),
                "generation_time_ms": float(generation_time * 1000),
                "similarity_score": float(similarity_score),
                "success": True
            }
            
            print(f"\n✅ Test completed successfully")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                "emotion": emotion,
                "text": text,
                "error": str(e),
                "success": False
            }
        
        return result
    
    def _calculate_basic_similarity(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor
    ) -> float:
        """
        Calculate basic audio similarity.
        
        This is a simple spectral correlation measure.
        For production, use HuBERT or other emotion classifiers.
        """
        try:
            # Ensure same length (truncate to shorter)
            min_len = min(audio1.shape[0], audio2.shape[0])
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # Normalize
            audio1 = audio1 / (torch.max(torch.abs(audio1)) + 1e-8)
            audio2 = audio2 / (torch.max(torch.abs(audio2)) + 1e-8)
            
            # Calculate correlation
            correlation = torch.corrcoef(torch.stack([audio1, audio2]))[0, 1]
            
            # Return absolute correlation (0-1 range)
            return abs(float(correlation.item()))
        
        except:
            return 0.0
    
    def run_full_evaluation(self) -> Dict:
        """
        Run emotion transfer test across all emotions.
        
        Returns:
            Dict with comprehensive results
        """
        print("\n" + "="*60)
        print("🚀 STAGE 0: EMOTION TRANSFER EVALUATION")
        print("   Teacher: OpenVoice V2")
        print("   Student: CSM")
        print("="*60)
        
        results = {}
        
        for emotion in self.config["test_sentences"].keys():
            results[emotion] = self.test_emotion_transfer(emotion)
            print()  # Spacing
        
        # Calculate aggregate metrics
        successful = [r for r in results.values() if r.get("success")]
        failed = [r for r in results.values() if not r.get("success")]
        
        if successful:
            avg_similarity = np.mean([
                r["similarity_score"] for r in successful
            ])
            avg_gen_time = np.mean([
                r["generation_time_ms"] for r in successful
            ])
        else:
            avg_similarity = 0.0
            avg_gen_time = 0.0
        
        summary = {
            "total_tests": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "average_similarity": float(avg_similarity),
            "average_generation_time_ms": float(avg_gen_time),
            "results": results
        }
        
        # Print summary
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Average Similarity: {summary['average_similarity']:.3f}")
        print(f"Average Generation Time: {summary['average_generation_time_ms']:.0f}ms")
        print("="*60)
        
        # Interpret results
        self._interpret_results(summary)
        
        return summary
    
    def _interpret_results(self, summary: Dict):
        """Interpret and provide recommendations based on results."""
        print("\n💡 INTERPRETATION:")
        
        avg_sim = summary["average_similarity"]
        success_rate = summary["successful"] / summary["total_tests"]
        
        if success_rate < 0.5:
            print("❌ Low success rate - check CSM service connectivity")
        elif avg_sim > 0.6:
            print("✅ Strong emotion transfer - CSM is responsive to references!")
            print("   → Recommended: Proceed to Stage 1 (Fine-tuning with Oviya voice)")
        elif avg_sim > 0.4:
            print("⚠️ Moderate emotion transfer - CSM shows some responsiveness")
            print("   → Consider: Test with longer/clearer references")
        else:
            print("❌ Weak emotion transfer - CSM may need architectural changes")
            print("   → Consider: Explicit emotion conditioning or different approach")
        
        print("\n📁 Audio files saved to: output/emotion_transfer/")
        print("   Listen to teacher vs CSM outputs to validate results")


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from voice.emotion_teacher import OpenVoiceEmotionTeacher
    from voice.openvoice_tts import HybridVoiceEngine
    
    # Initialize teacher and student
    teacher = OpenVoiceEmotionTeacher()
    student = HybridVoiceEngine(
        csm_url="http://localhost:6006/generate",
        default_engine="csm"
    )
    
    # Create evaluator
    evaluator = EmotionTransferEvaluator(teacher, student)
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Save results
    output_path = "output/emotion_transfer/evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")

