"""
Persistent Personality Storage for Oviya
Stores user-specific personality state across sessions
Enables long-term relationship building and context continuity
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib


class PersonalityStore:
    """
    Store and retrieve user-specific personality state
    Enables cross-session memory and relationship continuity
    """
    
    def __init__(self, db_path: Path = Path("data/personalities")):
        """
        Initialize personality store
        
        Args:
            db_path: Directory to store personality files
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Personality Store initialized: {self.db_path}")
    
    def _get_user_file(self, user_id: str) -> Path:
        """Get file path for user personality"""
        # Hash user_id for privacy
        hashed_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return self.db_path / f"{hashed_id}.json"
    
    def save_personality(self, user_id: str, state: Dict):
        """
        Save personality state for user
        
        Args:
            user_id: Unique user identifier
            state: Personality state dictionary
                {
                    'user_id': str,
                    'conversation_history': List[Dict],  # Recent conversations
                    'emotion_preferences': Dict,  # Preferred emotional responses
                    'interaction_style': str,  # 'formal' | 'casual' | 'playful'
                    'topics_discussed': List[str],  # Topics user has discussed
                    'relationship_level': float,  # 0.0-1.0 (stranger to close friend)
                    'user_traits': Dict,  # Observed user personality traits
                    'preferences': Dict,  # User preferences (response length, humor, etc.)
                    'last_interaction': str,  # ISO timestamp
                    'total_interactions': int,
                    'metadata': Dict  # Additional metadata
                }
        """
        # Add metadata
        state['user_id'] = user_id
        state['last_updated'] = datetime.now().isoformat()
        
        # Ensure required fields
        state.setdefault('conversation_history', [])
        state.setdefault('emotion_preferences', {})
        state.setdefault('interaction_style', 'casual')
        state.setdefault('topics_discussed', [])
        state.setdefault('relationship_level', 0.0)
        state.setdefault('user_traits', {})
        state.setdefault('preferences', {})
        state.setdefault('total_interactions', 0)
        state.setdefault('metadata', {})
        
        # Limit conversation history to last 50 turns
        if len(state['conversation_history']) > 50:
            state['conversation_history'] = state['conversation_history'][-50:]
        
        # Save to file
        user_file = self._get_user_file(user_id)
        with open(user_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved personality for user: {user_id[:8]}...")
    
    def load_personality(self, user_id: str) -> Optional[Dict]:
        """
        Load personality state for user
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Personality state dict or None if not found
        """
        user_file = self._get_user_file(user_id)
        
        if not user_file.exists():
            print(f"New user: {user_id[:8]}...")
            return None
        
        try:
            with open(user_file, 'r') as f:
                state = json.load(f)
            
            print(f"Loaded personality for user: {user_id[:8]}...")
            print(f"   Interactions: {state.get('total_interactions', 0)}")
            print(f"   Relationship level: {state.get('relationship_level', 0.0):.2f}")
            print(f"   Last interaction: {state.get('last_interaction', 'N/A')}")
            
            return state
            
        except Exception as e:
            print(f"Warning: Failed to load personality: {e}")
            return None
    
    def update_personality(self, user_id: str, updates: Dict):
        """
        Update specific fields in personality state
        
        Args:
            user_id: Unique user identifier
            updates: Dictionary of fields to update
        """
        state = self.load_personality(user_id)
        
        if state is None:
            state = {}
        
        # Merge updates
        state.update(updates)
        
        # Save
        self.save_personality(user_id, state)
    
    def add_conversation_turn(self, user_id: str, turn: Dict):
        """
        Add a conversation turn to history
        
        Args:
            user_id: Unique user identifier
            turn: {
                'user_message': str,
                'oviya_response': str,
                'user_emotion': str,
                'oviya_emotion': str,
                'timestamp': str
            }
        """
        state = self.load_personality(user_id) or {}
        
        # Add timestamp if not present
        turn.setdefault('timestamp', datetime.now().isoformat())
        
        # Add to history
        history = state.get('conversation_history', [])
        history.append(turn)
        
        # Update state
        state['conversation_history'] = history
        state['total_interactions'] = state.get('total_interactions', 0) + 1
        state['last_interaction'] = turn['timestamp']
        
        # Update relationship level (grows slowly over time)
        current_level = state.get('relationship_level', 0.0)
        state['relationship_level'] = min(1.0, current_level + 0.01)
        
        # Save
        self.save_personality(user_id, state)
    
    def get_conversation_summary(self, user_id: str, last_n: int = 10) -> str:
        """
        Get summary of recent conversations for context
        
        Args:
            user_id: Unique user identifier
            last_n: Number of recent turns to include
            
        Returns:
            Formatted conversation summary
        """
        state = self.load_personality(user_id)
        
        if not state or not state.get('conversation_history'):
            return "No previous conversations."
        
        history = state['conversation_history'][-last_n:]
        
        summary_lines = [
            f"Previous conversations with {user_id[:8]}...",
            f"Total interactions: {state.get('total_interactions', 0)}",
            f"Relationship level: {state.get('relationship_level', 0.0):.2f}",
            f"Interaction style: {state.get('interaction_style', 'casual')}",
            "",
            "Recent conversation:"
        ]
        
        for turn in history:
            summary_lines.append(f"  User: {turn.get('user_message', '')[:50]}...")
            summary_lines.append(f"  Oviya: {turn.get('oviya_response', '')[:50]}...")
        
        return "\n".join(summary_lines)
    
    def analyze_user_traits(self, user_id: str) -> Dict:
        """
        Analyze user personality traits from conversation history
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            {
                'emotional_tendency': str,  # Most common emotion
                'conversation_topics': List[str],  # Common topics
                'response_preference': str,  # Long/short responses
                'humor_appreciation': float,  # 0-1
                'formality_preference': float  # 0-1
            }
        """
        state = self.load_personality(user_id)
        
        if not state or not state.get('conversation_history'):
            return {
                'emotional_tendency': 'neutral',
                'conversation_topics': [],
                'response_preference': 'medium',
                'humor_appreciation': 0.5,
                'formality_preference': 0.5
            }
        
        history = state['conversation_history']
        
        # Analyze emotions
        emotions = [turn.get('user_emotion', 'neutral') for turn in history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        # Analyze topics (simple keyword extraction)
        all_messages = ' '.join([turn.get('user_message', '') for turn in history])
        topics = []
        common_topics = ['work', 'family', 'health', 'hobbies', 'feelings', 'goals', 'relationships']
        for topic in common_topics:
            if topic in all_messages.lower():
                topics.append(topic)
        
        # Analyze response length preference
        avg_user_length = np.mean([len(turn.get('user_message', '')) for turn in history])
        if avg_user_length < 50:
            response_preference = 'short'
        elif avg_user_length < 150:
            response_preference = 'medium'
        else:
            response_preference = 'long'
        
        return {
            'emotional_tendency': most_common_emotion,
            'conversation_topics': topics,
            'response_preference': response_preference,
            'humor_appreciation': state.get('user_traits', {}).get('humor_appreciation', 0.5),
            'formality_preference': state.get('user_traits', {}).get('formality_preference', 0.5)
        }
    
    def get_all_users(self) -> List[str]:
        """Get list of all user IDs with stored personalities"""
        user_files = list(self.db_path.glob("*.json"))
        user_ids = []
        
        for file in user_files:
            try:
                with open(file, 'r') as f:
                    state = json.load(f)
                    user_ids.append(state.get('user_id', 'unknown'))
            except Exception:
                continue
        
        return user_ids
    
    def delete_personality(self, user_id: str):
        """Delete personality data for user (GDPR compliance)"""
        user_file = self._get_user_file(user_id)
        
        if user_file.exists():
            user_file.unlink()
            print(f"Deleted personality for user: {user_id[:8]}...")
        else:
            print(f"Warning: No personality found for user: {user_id[:8]}...")


# Import numpy for analysis
try:
    import numpy as np
except ImportError:
    # Fallback without numpy
    class np:
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0


def test_personality_store():
    """Test personality storage system"""
    print("=" * 60)
    print("Testing Personality Store")
    print("=" * 60)
    
    store = PersonalityStore(db_path=Path("data/test_personalities"))
    
    # Test 1: Save new personality
    print("\nTest 1: Save new user personality")
    user_id = "test_user_123"
    
    state = {
        'interaction_style': 'casual',
        'relationship_level': 0.1,
        'preferences': {
            'humor': True,
            'response_length': 'medium'
        }
    }
    
    store.save_personality(user_id, state)
    
    # Test 2: Load personality
    print("\nTest 2: Load user personality")
    loaded_state = store.load_personality(user_id)
    print(f"   Loaded: {loaded_state is not None}")
    
    # Test 3: Add conversation turns
    print("\nTest 3: Add conversation turns")
    for i in range(3):
        turn = {
            'user_message': f'Test message {i+1}',
            'oviya_response': f'Test response {i+1}',
            'user_emotion': 'neutral',
            'oviya_emotion': 'calm_supportive'
        }
        store.add_conversation_turn(user_id, turn)
    
    # Test 4: Get conversation summary
    print("\n📝 Test 4: Get conversation summary")
    summary = store.get_conversation_summary(user_id)
    print(summary)
    
    # Test 5: Analyze user traits
    print("\n📝 Test 5: Analyze user traits")
    traits = store.analyze_user_traits(user_id)
    print(f"   Traits: {traits}")
    
    # Test 6: Get all users
    print("\n📝 Test 6: Get all users")
    users = store.get_all_users()
    print(f"   Total users: {len(users)}")
    
    print("\n✅ Personality store test complete!")


if __name__ == "__main__":
    test_personality_store()

