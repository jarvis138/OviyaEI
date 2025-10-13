"""
A/B Testing Framework for Oviya
Tests different voice configurations and collects MOS scores
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics


@dataclass
class VoiceVariant:
    """Voice configuration variant for A/B testing"""
    variant_id: str
    name: str
    prosody_intensity: float  # 0.5-1.5
    breath_frequency: float  # 0.5-2.0
    emotion_intensity_scale: float  # 0.8-1.2
    pause_duration_scale: float  # 0.8-1.2
    description: str


@dataclass
class MOSScore:
    """Mean Opinion Score for a voice sample"""
    variant_id: str
    sample_id: str
    rater_id: str
    naturalness: int  # 1-5
    expressiveness: int  # 1-5
    empathy: int  # 1-5
    overall: int  # 1-5
    comments: str
    timestamp: str


class ABTestFramework:
    """
    A/B test different voice configurations
    Collects MOS scores and determines winning variant
    """
    
    # Predefined variants
    VARIANTS = {
        'A_baseline': VoiceVariant(
            variant_id='A',
            name='Baseline',
            prosody_intensity=1.0,
            breath_frequency=1.0,
            emotion_intensity_scale=1.0,
            pause_duration_scale=1.0,
            description='Current production configuration'
        ),
        'B_expressive': VoiceVariant(
            variant_id='B',
            name='More Expressive',
            prosody_intensity=1.3,
            breath_frequency=1.5,
            emotion_intensity_scale=1.1,
            pause_duration_scale=1.1,
            description='Enhanced prosody and breathing'
        ),
        'C_subtle': VoiceVariant(
            variant_id='C',
            name='More Subtle',
            prosody_intensity=0.7,
            breath_frequency=0.8,
            emotion_intensity_scale=0.9,
            pause_duration_scale=0.9,
            description='Reduced prosody for more neutral tone'
        ),
        'D_dramatic': VoiceVariant(
            variant_id='D',
            name='Dramatic',
            prosody_intensity=1.5,
            breath_frequency=2.0,
            emotion_intensity_scale=1.2,
            pause_duration_scale=1.2,
            description='Maximum expressiveness'
        )
    }
    
    def __init__(self, test_dir: Path = Path("testing/ab_tests")):
        """
        Initialize A/B testing framework
        
        Args:
            test_dir: Directory to store test results
        """
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        self.scores_file = self.test_dir / "mos_scores.jsonl"
        self.results_file = self.test_dir / "test_results.json"
        
        print(f"🧪 A/B Testing Framework initialized: {self.test_dir}")
    
    def create_test_pairs(
        self, 
        sample_texts: List[str],
        variant_ids: List[str] = None
    ) -> List[Dict]:
        """
        Create test pairs for MOS evaluation
        
        Args:
            sample_texts: List of text samples to synthesize
            variant_ids: List of variant IDs to test (default: all)
            
        Returns:
            List of test pairs with sample IDs
        """
        if variant_ids is None:
            variant_ids = list(self.VARIANTS.keys())
        
        test_pairs = []
        
        for i, text in enumerate(sample_texts):
            for variant_id in variant_ids:
                test_pairs.append({
                    'sample_id': f"sample_{i}_{variant_id}",
                    'text': text,
                    'variant_id': variant_id,
                    'variant_name': self.VARIANTS[variant_id].name
                })
        
        # Randomize order
        random.shuffle(test_pairs)
        
        return test_pairs
    
    def submit_mos_score(self, score: MOSScore):
        """
        Submit a MOS score
        
        Args:
            score: MOSScore instance
        """
        # Append to JSONL file
        with open(self.scores_file, 'a') as f:
            f.write(json.dumps(asdict(score)) + '\n')
        
        print(f"📊 Submitted MOS score: {score.variant_id} - Overall: {score.overall}/5")
    
    def load_all_scores(self) -> List[MOSScore]:
        """Load all MOS scores from file"""
        if not self.scores_file.exists():
            return []
        
        scores = []
        with open(self.scores_file, 'r') as f:
            for line in f:
                try:
                    score_dict = json.loads(line.strip())
                    scores.append(MOSScore(**score_dict))
                except json.JSONDecodeError:
                    continue
        
        return scores
    
    def calculate_results(self) -> Dict:
        """
        Calculate A/B test results
        
        Returns:
            {
                'variant_scores': Dict,  # Average scores per variant
                'winner': str,  # Winning variant ID
                'confidence': float,  # Statistical confidence (0-1)
                'total_ratings': int,
                'detailed_scores': Dict
            }
        """
        scores = self.load_all_scores()
        
        if not scores:
            return {
                'variant_scores': {},
                'winner': None,
                'confidence': 0.0,
                'total_ratings': 0,
                'message': 'No scores available yet'
            }
        
        # Group scores by variant
        variant_scores = {}
        for score in scores:
            variant_id = score.variant_id
            
            if variant_id not in variant_scores:
                variant_scores[variant_id] = {
                    'naturalness': [],
                    'expressiveness': [],
                    'empathy': [],
                    'overall': []
                }
            
            variant_scores[variant_id]['naturalness'].append(score.naturalness)
            variant_scores[variant_id]['expressiveness'].append(score.expressiveness)
            variant_scores[variant_id]['empathy'].append(score.empathy)
            variant_scores[variant_id]['overall'].append(score.overall)
        
        # Calculate averages
        variant_averages = {}
        for variant_id, scores_dict in variant_scores.items():
            variant_averages[variant_id] = {
                'naturalness': statistics.mean(scores_dict['naturalness']),
                'expressiveness': statistics.mean(scores_dict['expressiveness']),
                'empathy': statistics.mean(scores_dict['empathy']),
                'overall': statistics.mean(scores_dict['overall']),
                'count': len(scores_dict['overall'])
            }
        
        # Determine winner (highest overall score)
        winner = max(variant_averages.items(), key=lambda x: x[1]['overall'])
        winner_id = winner[0]
        winner_score = winner[1]['overall']
        
        # Calculate confidence (simple heuristic based on sample size and score difference)
        min_samples = min(v['count'] for v in variant_averages.values())
        score_spread = max(v['overall'] for v in variant_averages.values()) - min(v['overall'] for v in variant_averages.values())
        
        confidence = min(1.0, (min_samples / 20) * (score_spread / 1.0))  # Normalize to 0-1
        
        return {
            'variant_scores': variant_averages,
            'winner': winner_id,
            'winner_score': winner_score,
            'confidence': confidence,
            'total_ratings': len(scores),
            'detailed_scores': variant_scores
        }
    
    def print_results(self, results: Dict):
        """
        Print A/B test results
        
        Args:
            results: Results from calculate_results()
        """
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)
        
        if not results.get('variant_scores'):
            print("\n⚠️  No results available yet")
            return
        
        print(f"\n📊 Total Ratings: {results['total_ratings']}")
        
        print(f"\n🏆 Winner: Variant {results['winner']}")
        print(f"   Overall Score: {results['winner_score']:.2f}/5")
        print(f"   Confidence: {results['confidence']:.1%}")
        
        print(f"\n📊 Variant Scores:")
        for variant_id, scores in results['variant_scores'].items():
            variant_name = self.VARIANTS.get(variant_id, VoiceVariant('', '', 0, 0, 0, 0, '')).name
            print(f"\n   {variant_id} - {variant_name}:")
            print(f"      Naturalness:     {scores['naturalness']:.2f}/5")
            print(f"      Expressiveness:  {scores['expressiveness']:.2f}/5")
            print(f"      Empathy:         {scores['empathy']:.2f}/5")
            print(f"      Overall:         {scores['overall']:.2f}/5")
            print(f"      Ratings:         {scores['count']}")
        
        print("\n" + "=" * 60)
    
    def export_results(self, results: Dict, output_file: Path):
        """
        Export results to JSON
        
        Args:
            results: Results from calculate_results()
            output_file: Output JSON file path
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Exported results to {output_file}")
    
    def run_mos_survey(
        self,
        sample_texts: List[str],
        num_raters: int = 20
    ) -> Dict:
        """
        Simulate MOS survey (for testing)
        In production, this would be a web interface for real users
        
        Args:
            sample_texts: List of text samples
            num_raters: Number of simulated raters
            
        Returns:
            Survey results
        """
        print(f"🧪 Simulating MOS survey with {num_raters} raters...")
        
        test_pairs = self.create_test_pairs(sample_texts)
        
        # Simulate ratings
        for rater_id in range(num_raters):
            for pair in test_pairs:
                # Simulate scores (with some bias towards variant B)
                base_score = 3.5
                if pair['variant_id'] == 'B_expressive':
                    base_score = 4.0
                elif pair['variant_id'] == 'D_dramatic':
                    base_score = 3.2  # Too dramatic
                
                # Add noise
                naturalness = max(1, min(5, int(base_score + random.gauss(0, 0.5))))
                expressiveness = max(1, min(5, int(base_score + random.gauss(0, 0.5))))
                empathy = max(1, min(5, int(base_score + random.gauss(0, 0.5))))
                overall = max(1, min(5, int(base_score + random.gauss(0, 0.5))))
                
                score = MOSScore(
                    variant_id=pair['variant_id'],
                    sample_id=pair['sample_id'],
                    rater_id=f"rater_{rater_id}",
                    naturalness=naturalness,
                    expressiveness=expressiveness,
                    empathy=empathy,
                    overall=overall,
                    comments="",
                    timestamp=datetime.now().isoformat()
                )
                
                self.submit_mos_score(score)
        
        # Calculate and return results
        results = self.calculate_results()
        return results


def test_ab_framework():
    """Test A/B testing framework"""
    print("=" * 60)
    print("Testing A/B Testing Framework")
    print("=" * 60)
    
    framework = ABTestFramework(test_dir=Path("testing/test_ab"))
    
    # Sample texts for testing
    sample_texts = [
        "I'm here for you, and I understand how you're feeling.",
        "That's wonderful news! I'm so happy for you!",
        "Let's think about this carefully and find a solution together."
    ]
    
    # Run simulated MOS survey
    print("\n📝 Running simulated MOS survey...")
    results = framework.run_mos_survey(sample_texts, num_raters=20)
    
    # Print results
    framework.print_results(results)
    
    # Export results
    framework.export_results(results, Path("testing/test_ab/results.json"))
    
    print("\n✅ A/B testing framework test complete!")


if __name__ == "__main__":
    test_ab_framework()

