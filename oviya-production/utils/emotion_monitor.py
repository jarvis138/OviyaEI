"""
Emotion Distribution Monitor
Tracks and validates emotion usage across the system
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import numpy as np
from datetime import datetime


class EmotionDistributionMonitor:
    """Monitors emotion distribution to ensure it matches design targets"""
    
    def __init__(self, log_dir: Path = Path("logs/emotion_distribution")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Target distribution
        self.target_distribution = {
            "tier1_core": 0.70,
            "tier2_contextual": 0.25,
            "tier3_expressive": 0.05
        }
        
        # Emotion to tier mapping
        self.emotion_tiers = {
            # Tier 1
            "calm_supportive": "tier1_core",
            "empathetic_sad": "tier1_core",
            "joyful_excited": "tier1_core",
            "confident": "tier1_core",
            "neutral": "tier1_core",
            "comforting": "tier1_core",
            "encouraging": "tier1_core",
            "thoughtful": "tier1_core",
            "affectionate": "tier1_core",
            "reassuring": "tier1_core",
            
            # Tier 2
            "playful": "tier2_contextual",
            "concerned_anxious": "tier2_contextual",
            "melancholy": "tier2_contextual",
            "wistful": "tier2_contextual",
            "tired": "tier2_contextual",
            "curious": "tier2_contextual",
            "dreamy": "tier2_contextual",
            "relieved": "tier2_contextual",
            "proud": "tier2_contextual",
            
            # Tier 3
            "angry_firm": "tier3_expressive",
            "sarcastic": "tier3_expressive",
            "mischievous": "tier3_expressive",
            "tender": "tier3_expressive",
            "amused": "tier3_expressive",
            "sympathetic": "tier3_expressive",
            "reflective": "tier3_expressive",
            "grateful": "tier3_expressive",
            "apologetic": "tier3_expressive"
        }
        
        # Tracking
        self.emotion_history = []
        self.session_start = datetime.now()
        
    def record_emotion(self, emotion: str):
        """Record an emotion usage"""
        self.emotion_history.append({
            "emotion": emotion,
            "tier": self.emotion_tiers.get(emotion, "unknown"),
            "timestamp": datetime.now().isoformat()
        })
        
        # Auto-save every 100 emotions
        if len(self.emotion_history) % 100 == 0:
            self.save_checkpoint()
    
    def get_distribution(self) -> Dict:
        """Get current emotion distribution"""
        if not self.emotion_history:
            return {"tier1_core": 0, "tier2_contextual": 0, "tier3_expressive": 0}
        
        # Count by tier
        tier_counts = Counter([e["tier"] for e in self.emotion_history])
        total = len(self.emotion_history)
        
        distribution = {
            tier: count / total 
            for tier, count in tier_counts.items()
            if tier != "unknown"
        }
        
        return distribution
    
    def check_distribution_health(self) -> Dict:
        """Check if distribution matches targets"""
        current = self.get_distribution()
        
        health_check = {
            "status": "healthy",
            "issues": [],
            "current_distribution": current,
            "target_distribution": self.target_distribution,
            "sample_size": len(self.emotion_history)
        }
        
        # Need at least 100 samples for meaningful check
        if len(self.emotion_history) < 100:
            health_check["status"] = "insufficient_data"
            return health_check
        
        # Check each tier
        for tier, target in self.target_distribution.items():
            actual = current.get(tier, 0)
            deviation = abs(actual - target)
            
            # Allow 10% deviation
            if deviation > 0.10:
                health_check["status"] = "unhealthy"
                health_check["issues"].append({
                    "tier": tier,
                    "target": target,
                    "actual": actual,
                    "deviation": deviation
                })
        
        return health_check
    
    def generate_histogram(self) -> Dict:
        """Generate emotion histogram data"""
        if not self.emotion_history:
            return {}
        
        # Count individual emotions
        emotion_counts = Counter([e["emotion"] for e in self.emotion_history])
        
        # Sort by tier and count
        histogram = {
            "tier1_core": {},
            "tier2_contextual": {},
            "tier3_expressive": {}
        }
        
        for emotion, count in emotion_counts.items():
            tier = self.emotion_tiers.get(emotion, "unknown")
            if tier != "unknown":
                histogram[tier][emotion] = count
        
        return histogram
    
    def save_checkpoint(self):
        """Save current state to disk"""
        checkpoint = {
            "session_start": self.session_start.isoformat(),
            "checkpoint_time": datetime.now().isoformat(),
            "total_emotions": len(self.emotion_history),
            "distribution": self.get_distribution(),
            "histogram": self.generate_histogram(),
            "health_check": self.check_distribution_health()
        }
        
        # Save with timestamp
        filename = f"emotion_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"📊 Emotion distribution saved to {filepath}")
        
        # Also save latest
        latest_path = self.log_dir / "latest.json"
        with open(latest_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def print_summary(self):
        """Print distribution summary"""
        print("\n" + "=" * 60)
        print("📊 EMOTION DISTRIBUTION SUMMARY")
        print("=" * 60)
        
        if not self.emotion_history:
            print("No emotions recorded yet")
            return
        
        print(f"Total emotions: {len(self.emotion_history)}")
        print("\nDistribution by tier:")
        
        current = self.get_distribution()
        for tier, actual in current.items():
            target = self.target_distribution.get(tier, 0)
            status = "✅" if abs(actual - target) <= 0.10 else "⚠️"
            print(f"  {status} {tier}: {actual:.1%} (target: {target:.1%})")
        
        # Top emotions
        print("\nTop 5 emotions:")
        emotion_counts = Counter([e["emotion"] for e in self.emotion_history])
        for emotion, count in emotion_counts.most_common(5):
            print(f"  - {emotion}: {count} ({count/len(self.emotion_history):.1%})")
        
        # Health check
        health = self.check_distribution_health()
        print(f"\nHealth status: {health['status'].upper()}")
        if health["issues"]:
            print("Issues found:")
            for issue in health["issues"]:
                print(f"  - {issue['tier']}: {issue['deviation']:.1%} deviation")
        
        print("=" * 60)


# Global monitor instance
_emotion_monitor = None

def get_emotion_monitor() -> EmotionDistributionMonitor:
    """Get global emotion monitor instance (singleton)"""
    global _emotion_monitor
    if _emotion_monitor is None:
        _emotion_monitor = EmotionDistributionMonitor()
    return _emotion_monitor


def test_distribution(num_samples: int = 1000):
    """Test emotion distribution with random sampling"""
    from voice.emotion_library import get_emotion_library
    
    print(f"🧪 Testing emotion distribution with {num_samples} samples\n")
    
    library = get_emotion_library()
    monitor = EmotionDistributionMonitor()
    
    # Generate random emotions
    for i in range(num_samples):
        emotion = library.sample_emotion()
        monitor.record_emotion(emotion)
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1} emotions...")
    
    # Print summary
    monitor.print_summary()
    monitor.save_checkpoint()
    
    return monitor


if __name__ == "__main__":
    # Run distribution test
    test_distribution(1000)
