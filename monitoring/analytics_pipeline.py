"""
Structured Analytics Pipeline for Oviya
Tracks conversation metrics, user satisfaction, and system performance
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import statistics


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation"""
    user_id: str
    session_id: str
    turn_count: int
    avg_latency: float  # Average response latency in seconds
    emotions_used: List[str]  # Oviya emotions used
    sentiment_trajectory: List[float]  # User sentiment over time (-1 to +1)
    user_satisfaction: Optional[float]  # User rating (0-5)
    total_duration: float  # Total conversation duration in seconds
    timestamp: str
    metadata: Dict


class AnalyticsPipeline:
    """
    Structured analytics for Oviya conversations
    Logs metrics, generates insights, and provides dashboard data
    """
    
    def __init__(self, log_dir: Path = Path("logs/analytics")):
        """
        Initialize analytics pipeline
        
        Args:
            log_dir: Directory to store analytics logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / "conversation_metrics.jsonl"
        self.summary_file = self.log_dir / "analytics_summary.json"
        
        print(f"📊 Analytics Pipeline initialized: {self.log_dir}")
    
    def log_conversation(self, metrics: ConversationMetrics):
        """
        Log conversation metrics to file
        
        Args:
            metrics: ConversationMetrics instance
        """
        # Convert to dict
        metrics_dict = asdict(metrics)
        
        # Append to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_dict) + '\n')
        
        print(f"📊 Logged conversation: {metrics.session_id}")
        
        # Update summary
        self._update_summary()
    
    def _update_summary(self):
        """Update analytics summary with latest data"""
        # Load all metrics
        metrics_list = self._load_all_metrics()
        
        if not metrics_list:
            return
        
        # Calculate summary statistics
        summary = {
            'total_conversations': len(metrics_list),
            'total_users': len(set(m['user_id'] for m in metrics_list)),
            'avg_turns_per_conversation': statistics.mean([m['turn_count'] for m in metrics_list]),
            'avg_latency': statistics.mean([m['avg_latency'] for m in metrics_list]),
            'avg_conversation_duration': statistics.mean([m['total_duration'] for m in metrics_list]),
            'most_used_emotions': self._get_top_emotions(metrics_list, top_n=10),
            'avg_sentiment_improvement': self._calculate_sentiment_improvement(metrics_list),
            'satisfaction_scores': self._get_satisfaction_stats(metrics_list),
            'last_updated': datetime.now().isoformat()
        }
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _load_all_metrics(self) -> List[Dict]:
        """Load all conversation metrics from file"""
        if not self.metrics_file.exists():
            return []
        
        metrics_list = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    metrics_list.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return metrics_list
    
    def _get_top_emotions(self, metrics_list: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get most frequently used emotions"""
        emotion_counts = {}
        
        for metrics in metrics_list:
            for emotion in metrics['emotions_used']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Sort by count
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'emotion': emotion, 'count': count, 'percentage': count / len(metrics_list) * 100}
            for emotion, count in sorted_emotions[:top_n]
        ]
    
    def _calculate_sentiment_improvement(self, metrics_list: List[Dict]) -> float:
        """Calculate average sentiment improvement (end - start)"""
        improvements = []
        
        for metrics in metrics_list:
            trajectory = metrics['sentiment_trajectory']
            if len(trajectory) >= 2:
                improvement = trajectory[-1] - trajectory[0]
                improvements.append(improvement)
        
        return statistics.mean(improvements) if improvements else 0.0
    
    def _get_satisfaction_stats(self, metrics_list: List[Dict]) -> Dict:
        """Get satisfaction score statistics"""
        scores = [m['user_satisfaction'] for m in metrics_list if m['user_satisfaction'] is not None]
        
        if not scores:
            return {
                'avg_score': None,
                'total_ratings': 0,
                'distribution': {}
            }
        
        # Calculate distribution
        distribution = {}
        for score in scores:
            score_int = int(score)
            distribution[score_int] = distribution.get(score_int, 0) + 1
        
        return {
            'avg_score': statistics.mean(scores),
            'median_score': statistics.median(scores),
            'total_ratings': len(scores),
            'distribution': distribution
        }
    
    def get_dashboard_data(self) -> Dict:
        """
        Get dashboard data for visualization
        
        Returns:
            {
                'total_conversations': int,
                'total_users': int,
                'avg_turns_per_conversation': float,
                'avg_latency': float,
                'most_used_emotions': List[Dict],
                'sentiment_improvement': float,
                'satisfaction_scores': Dict,
                'recent_conversations': List[Dict]
            }
        """
        if not self.summary_file.exists():
            return {
                'total_conversations': 0,
                'total_users': 0,
                'message': 'No data available yet'
            }
        
        with open(self.summary_file, 'r') as f:
            summary = json.load(f)
        
        # Add recent conversations
        metrics_list = self._load_all_metrics()
        recent = metrics_list[-10:] if metrics_list else []
        
        summary['recent_conversations'] = recent
        
        return summary
    
    def get_user_analytics(self, user_id: str) -> Dict:
        """
        Get analytics for a specific user
        
        Args:
            user_id: User identifier
            
        Returns:
            User-specific analytics
        """
        metrics_list = self._load_all_metrics()
        user_metrics = [m for m in metrics_list if m['user_id'] == user_id]
        
        if not user_metrics:
            return {
                'user_id': user_id,
                'total_conversations': 0,
                'message': 'No conversations found for this user'
            }
        
        return {
            'user_id': user_id,
            'total_conversations': len(user_metrics),
            'avg_turns': statistics.mean([m['turn_count'] for m in user_metrics]),
            'total_duration': sum([m['total_duration'] for m in user_metrics]),
            'favorite_emotions': self._get_top_emotions(user_metrics, top_n=5),
            'avg_sentiment_improvement': self._calculate_sentiment_improvement(user_metrics),
            'satisfaction_history': [m['user_satisfaction'] for m in user_metrics if m['user_satisfaction'] is not None]
        }
    
    def export_to_csv(self, output_file: Path):
        """
        Export metrics to CSV for external analysis
        
        Args:
            output_file: Output CSV file path
        """
        import csv
        
        metrics_list = self._load_all_metrics()
        
        if not metrics_list:
            print("⚠️  No metrics to export")
            return
        
        # Define CSV columns
        columns = [
            'user_id', 'session_id', 'turn_count', 'avg_latency',
            'total_duration', 'sentiment_start', 'sentiment_end',
            'sentiment_improvement', 'user_satisfaction', 'timestamp'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for metrics in metrics_list:
                trajectory = metrics['sentiment_trajectory']
                row = {
                    'user_id': metrics['user_id'],
                    'session_id': metrics['session_id'],
                    'turn_count': metrics['turn_count'],
                    'avg_latency': metrics['avg_latency'],
                    'total_duration': metrics['total_duration'],
                    'sentiment_start': trajectory[0] if trajectory else 0,
                    'sentiment_end': trajectory[-1] if trajectory else 0,
                    'sentiment_improvement': (trajectory[-1] - trajectory[0]) if len(trajectory) >= 2 else 0,
                    'user_satisfaction': metrics['user_satisfaction'],
                    'timestamp': metrics['timestamp']
                }
                writer.writerow(row)
        
        print(f"📊 Exported {len(metrics_list)} conversations to {output_file}")


class ConversationTracker:
    """
    Tracks metrics for an ongoing conversation
    Used by the WebSocket server to collect data in real-time
    """
    
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.turn_count = 0
        self.latencies = []
        self.emotions_used = []
        self.sentiment_trajectory = []
        self.start_time = datetime.now()
        self.metadata = {}
    
    def add_turn(self, latency: float, emotion: str, sentiment: float):
        """
        Add a conversation turn
        
        Args:
            latency: Response latency in seconds
            emotion: Oviya emotion used
            sentiment: User sentiment (-1 to +1)
        """
        self.turn_count += 1
        self.latencies.append(latency)
        self.emotions_used.append(emotion)
        self.sentiment_trajectory.append(sentiment)
    
    def finalize(self, user_satisfaction: Optional[float] = None) -> ConversationMetrics:
        """
        Finalize conversation and return metrics
        
        Args:
            user_satisfaction: Optional user rating (0-5)
            
        Returns:
            ConversationMetrics instance
        """
        total_duration = (datetime.now() - self.start_time).total_seconds()
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0.0
        
        return ConversationMetrics(
            user_id=self.user_id,
            session_id=self.session_id,
            turn_count=self.turn_count,
            avg_latency=avg_latency,
            emotions_used=self.emotions_used,
            sentiment_trajectory=self.sentiment_trajectory,
            user_satisfaction=user_satisfaction,
            total_duration=total_duration,
            timestamp=datetime.now().isoformat(),
            metadata=self.metadata
        )


def test_analytics_pipeline():
    """Test analytics pipeline"""
    print("=" * 60)
    print("Testing Analytics Pipeline")
    print("=" * 60)
    
    pipeline = AnalyticsPipeline(log_dir=Path("logs/test_analytics"))
    
    # Test 1: Log sample conversations
    print("\n📝 Test 1: Log sample conversations")
    for i in range(5):
        metrics = ConversationMetrics(
            user_id=f"user_{i % 2}",
            session_id=f"session_{i}",
            turn_count=5 + i,
            avg_latency=2.5 + (i * 0.1),
            emotions_used=['calm_supportive', 'empathetic_sad', 'joyful_excited'][:i+1],
            sentiment_trajectory=[-0.2, 0.0, 0.3, 0.5, 0.7],
            user_satisfaction=4.0 + (i * 0.1),
            total_duration=60.0 + (i * 10),
            timestamp=datetime.now().isoformat(),
            metadata={'test': True}
        )
        pipeline.log_conversation(metrics)
    
    # Test 2: Get dashboard data
    print("\n📝 Test 2: Get dashboard data")
    dashboard = pipeline.get_dashboard_data()
    print(f"   Total conversations: {dashboard['total_conversations']}")
    print(f"   Avg latency: {dashboard['avg_latency']:.2f}s")
    print(f"   Sentiment improvement: {dashboard['avg_sentiment_improvement']:.2f}")
    
    # Test 3: Get user analytics
    print("\n📝 Test 3: Get user analytics")
    user_analytics = pipeline.get_user_analytics("user_0")
    print(f"   User conversations: {user_analytics['total_conversations']}")
    
    # Test 4: Export to CSV
    print("\n📝 Test 4: Export to CSV")
    pipeline.export_to_csv(Path("logs/test_analytics/export.csv"))
    
    print("\n✅ Analytics pipeline test complete!")


if __name__ == "__main__":
    test_analytics_pipeline()

