#!/usr/bin/env python3
"""
Production Monitoring Metrics
Track MOS, emotion accuracy, persona drift, and latency
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from collections import deque


class MetricsCollector:
    """
    Collect and aggregate production metrics
    
    Target Metrics:
    - Mean Opinion Score (MOS): ≥ 4.4
    - Emotion classification accuracy: ≥ 85%
    - Persona drift: ≤ 0.15
    - Latency: ≤ 1.5s
    """
    
    def __init__(self, log_dir: Path = Path("logs/metrics")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.mos_scores = deque(maxlen=1000)
        self.emotion_accuracy = deque(maxlen=1000)
        self.persona_drifts = deque(maxlen=1000)
        self.latencies = deque(maxlen=1000)
        
        # Turn embeddings for persona drift
        self.turn_embeddings = deque(maxlen=50)
        
        # Session info
        self.session_start = datetime.now()
        self.total_turns = 0
        self.total_errors = 0
        
        print("✅ Metrics collector initialized")
        print(f"   📁 Logging to: {self.log_dir}")
    
    def record_mos_score(self, score: float, user_id: Optional[str] = None):
        """
        Record Mean Opinion Score from user feedback
        Scale: 1.0 (bad) to 5.0 (excellent)
        """
        if not 1.0 <= score <= 5.0:
            print(f"⚠️  Invalid MOS score: {score}")
            return
        
        self.mos_scores.append(score)
        
        # Log to file
        self._log_metric("mos", {
            "score": score,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_emotion_match(self, predicted: str, expected: str):
        """
        Record emotion classification accuracy
        """
        is_match = predicted == expected
        self.emotion_accuracy.append(1.0 if is_match else 0.0)
        
        self._log_metric("emotion_accuracy", {
            "predicted": predicted,
            "expected": expected,
            "match": is_match,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_persona_drift(self, current_embedding: List[float]):
        """
        Record persona drift (cosine distance between turn embeddings)
        """
        if len(current_embedding) == 0:
            return
        
        current_embedding = np.array(current_embedding)
        self.turn_embeddings.append(current_embedding)
        
        if len(self.turn_embeddings) >= 2:
            # Calculate cosine distance from previous turn
            prev_embedding = self.turn_embeddings[-2]
            
            # Cosine similarity
            dot_product = np.dot(current_embedding, prev_embedding)
            norm_product = np.linalg.norm(current_embedding) * np.linalg.norm(prev_embedding)
            
            if norm_product > 0:
                cosine_similarity = dot_product / norm_product
                cosine_distance = 1.0 - cosine_similarity
                
                self.persona_drifts.append(cosine_distance)
                
                self._log_metric("persona_drift", {
                    "distance": float(cosine_distance),
                    "timestamp": datetime.now().isoformat()
                })
    
    def record_latency(self, latency_seconds: float, component: str = "total"):
        """
        Record system latency
        """
        self.latencies.append(latency_seconds)
        
        self._log_metric("latency", {
            "latency": latency_seconds,
            "component": component,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_turn(self):
        """Record conversation turn"""
        self.total_turns += 1
    
    def record_error(self, error_type: str, details: str):
        """Record system error"""
        self.total_errors += 1
        
        self._log_metric("error", {
            "type": error_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_current_metrics(self) -> Dict:
        """Get current metric values"""
        metrics = {
            "mos": {
                "current": float(np.mean(self.mos_scores)) if self.mos_scores else 0.0,
                "target": 4.4,
                "samples": len(self.mos_scores),
                "status": "✅" if (self.mos_scores and np.mean(self.mos_scores) >= 4.4) else "⚠️"
            },
            "emotion_accuracy": {
                "current": float(np.mean(self.emotion_accuracy) * 100) if self.emotion_accuracy else 0.0,
                "target": 85.0,
                "samples": len(self.emotion_accuracy),
                "status": "✅" if (self.emotion_accuracy and np.mean(self.emotion_accuracy) >= 0.85) else "⚠️"
            },
            "persona_drift": {
                "current": float(np.mean(self.persona_drifts)) if self.persona_drifts else 0.0,
                "target": 0.15,
                "samples": len(self.persona_drifts),
                "status": "✅" if (self.persona_drifts and np.mean(self.persona_drifts) <= 0.15) else "⚠️"
            },
            "latency": {
                "current": float(np.mean(self.latencies)) if self.latencies else 0.0,
                "target": 1.5,
                "samples": len(self.latencies),
                "status": "✅" if (self.latencies and np.mean(self.latencies) <= 1.5) else "⚠️"
            },
            "session": {
                "total_turns": self.total_turns,
                "total_errors": self.total_errors,
                "error_rate": (self.total_errors / self.total_turns * 100) if self.total_turns > 0 else 0.0,
                "uptime": str(datetime.now() - self.session_start)
            }
        }
        
        return metrics
    
    def print_dashboard(self):
        """Print metrics dashboard"""
        metrics = self.get_current_metrics()
        
        print("\n" + "="*70)
        print("  📊 PRODUCTION METRICS DASHBOARD")
        print("="*70 + "\n")
        
        # MOS
        mos = metrics['mos']
        print(f"1. Mean Opinion Score (MOS)")
        print(f"   {mos['status']} Current: {mos['current']:.2f} / Target: {mos['target']}")
        print(f"   📈 Samples: {mos['samples']}")
        
        # Emotion Accuracy
        emotion = metrics['emotion_accuracy']
        print(f"\n2. Emotion Classification Accuracy")
        print(f"   {emotion['status']} Current: {emotion['current']:.1f}% / Target: {emotion['target']:.0f}%")
        print(f"   📈 Samples: {emotion['samples']}")
        
        # Persona Drift
        drift = metrics['persona_drift']
        print(f"\n3. Persona Drift (Cosine Distance)")
        print(f"   {drift['status']} Current: {drift['current']:.3f} / Target: ≤ {drift['target']}")
        print(f"   📈 Samples: {drift['samples']}")
        
        # Latency
        latency = metrics['latency']
        print(f"\n4. System Latency")
        print(f"   {latency['status']} Current: {latency['current']:.2f}s / Target: ≤ {latency['target']}s")
        print(f"   📈 Samples: {latency['samples']}")
        
        # Session Info
        session = metrics['session']
        print(f"\n5. Session Statistics")
        print(f"   📊 Total turns: {session['total_turns']}")
        print(f"   ⚠️  Total errors: {session['total_errors']} ({session['error_rate']:.1f}%)")
        print(f"   ⏱️  Uptime: {session['uptime']}")
        
        print("\n" + "="*70 + "\n")
    
    def export_report(self, filename: Optional[str] = None) -> Path:
        """Export metrics report to JSON"""
        if filename is None:
            filename = f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = self.log_dir / filename
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "session_start": self.session_start.isoformat(),
            "metrics": self.get_current_metrics(),
            "raw_data": {
                "mos_scores": list(self.mos_scores),
                "emotion_accuracy": list(self.emotion_accuracy),
                "persona_drifts": list(self.persona_drifts),
                "latencies": list(self.latencies)
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Metrics report exported to: {report_path}")
        return report_path
    
    def _log_metric(self, metric_type: str, data: Dict):
        """Log metric to file"""
        log_file = self.log_dir / f"{metric_type}.jsonl"
        
        with open(log_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')


class EmotionDistributionMonitor:
    """
    Monitor emotion distribution to ensure no bias toward neutral
    """
    def __init__(self):
        self.emotion_counts = {}
        self.total_generations = 0
        print("✅ Emotion distribution monitor initialized")
    
    def record_emotion(self, emotion: str):
        """Record generated emotion"""
        self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + 1
        self.total_generations += 1
    
    def get_distribution(self) -> Dict:
        """Get emotion distribution"""
        if self.total_generations == 0:
            return {}
        
        distribution = {}
        for emotion, count in self.emotion_counts.items():
            distribution[emotion] = {
                "count": count,
                "percentage": (count / self.total_generations) * 100
            }
        
        return distribution
    
    def check_bias(self, threshold: float = 30.0) -> Dict:
        """
        Check if any emotion is over-represented
        Returns: {"biased": bool, "dominant_emotion": str, "percentage": float}
        """
        distribution = self.get_distribution()
        
        if not distribution:
            return {"biased": False, "dominant_emotion": None, "percentage": 0.0}
        
        # Find most common emotion
        dominant = max(distribution.items(), key=lambda x: x[1]['percentage'])
        dominant_emotion = dominant[0]
        dominant_pct = dominant[1]['percentage']
        
        is_biased = dominant_pct > threshold
        
        return {
            "biased": is_biased,
            "dominant_emotion": dominant_emotion,
            "percentage": dominant_pct
        }
    
    def print_report(self, top_n: int = 10):
        """Print distribution report"""
        distribution = self.get_distribution()
        bias_check = self.check_bias()
        
        print("\n" + "="*70)
        print("  🎭 EMOTION DISTRIBUTION REPORT")
        print("="*70 + "\n")
        
        print(f"Total generations: {self.total_generations}\n")
        
        # Sort by percentage
        sorted_emotions = sorted(
            distribution.items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )[:top_n]
        
        print(f"Top {top_n} emotions:")
        for emotion, data in sorted_emotions:
            bar = '█' * int(data['percentage'] / 2)
            print(f"   {emotion:20s} {bar} {data['percentage']:.1f}% ({data['count']})")
        
        print(f"\n📊 Bias Check:")
        if bias_check['biased']:
            print(f"   ⚠️  BIASED toward '{bias_check['dominant_emotion']}' ({bias_check['percentage']:.1f}%)")
        else:
            print(f"   ✅ No significant bias (max: {bias_check['percentage']:.1f}%)")
        
        print("\n" + "="*70 + "\n")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("  MONITORING SYSTEMS TEST")
    print("="*70)
    
    # Test 1: Metrics Collector
    print("\n1. Testing Metrics Collector:")
    collector = MetricsCollector()
    
    # Simulate some metrics
    for i in range(10):
        collector.record_mos_score(4.5 + np.random.uniform(-0.3, 0.3))
        collector.record_emotion_match("joyful_excited", "joyful_excited" if np.random.rand() > 0.1 else "calm_supportive")
        collector.record_latency(1.2 + np.random.uniform(-0.3, 0.3))
        collector.record_persona_drift(np.random.rand(128).tolist())
        collector.record_turn()
    
    collector.print_dashboard()
    
    # Test 2: Emotion Distribution Monitor
    print("\n2. Testing Emotion Distribution Monitor:")
    monitor = EmotionDistributionMonitor()
    
    # Simulate emotion generation
    emotions = ["joyful_excited", "calm_supportive", "neutral", "empathetic_sad", "confident"]
    weights = [0.2, 0.25, 0.15, 0.25, 0.15]
    
    for _ in range(100):
        emotion = np.random.choice(emotions, p=weights)
        monitor.record_emotion(emotion)
    
    monitor.print_report()
    
    # Export report
    collector.export_report("test_metrics_report.json")
    
    print("="*70)
    print("✅ All monitoring systems tested successfully!")
    print("="*70)


