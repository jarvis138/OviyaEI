"""
Emotion Transition Smoothing Module
Provides smooth emotional transitions between states using embedding interpolation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class EmotionTransitionSmoother:
    """Smooths emotion transitions using embedding space interpolation"""
    
    # Emotion compatibility matrix (how well emotions blend)
    EMOTION_COMPATIBILITY = {
        "joyful_excited": {
            "playful": 0.9, "confident": 0.8, "encouraging": 0.7,
            "sad": 0.2, "angry": 0.3  # Jarring transitions
        },
        "calm_supportive": {
            "comforting": 0.95, "thoughtful": 0.9, "reassuring": 0.9,
            "angry": 0.2, "sarcastic": 0.3
        },
        "empathetic_sad": {
            "comforting": 0.9, "melancholy": 0.95, "wistful": 0.8,
            "joyful_excited": 0.3, "playful": 0.2
        },
        "angry_firm": {
            "confident": 0.7, "sarcastic": 0.6,
            "joyful_excited": 0.2, "playful": 0.1
        }
    }
    
    # Transition speeds (how fast to transition)
    TRANSITION_SPEEDS = {
        "instant": 0.0,     # No smoothing
        "fast": 0.3,        # 30% old, 70% new
        "normal": 0.5,      # 50/50 blend
        "slow": 0.7,        # 70% old, 30% new
        "gradual": 0.85     # 85% old, 15% new
    }
    
    def __init__(self, embedding_dir: Optional[Path] = None):
        """
        Initialize the emotion transition smoother
        
        Args:
            embedding_dir: Directory containing emotion embeddings
        """
        self.embedding_dir = embedding_dir or Path("emotion_embeddings")
        self.current_embedding = None
        self.current_emotion = "neutral"
        self.previous_emotion = "neutral"
        self.transition_history = []
        
        # Load emotion embeddings if available
        self.embeddings = self._load_embeddings()
        
        # Transition parameters
        self.min_transition_steps = 2  # Minimum steps for gradual transition
        self.max_transition_steps = 5  # Maximum steps
        self.current_transition_step = 0
        self.target_emotion = None
        self.transition_trajectory = []
    
    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load pre-computed emotion embeddings"""
        embeddings = {}
        
        # Try to load from files
        if self.embedding_dir.exists():
            for emotion_file in self.embedding_dir.glob("*.npy"):
                emotion_name = emotion_file.stem
                try:
                    embedding = np.load(emotion_file)
                    embeddings[emotion_name] = embedding
                except Exception as e:
                    print(f"Failed to load {emotion_name}: {e}")
        
        # If no embeddings found, create synthetic ones
        if not embeddings:
            embeddings = self._create_synthetic_embeddings()
        
        return embeddings
    
    def _create_synthetic_embeddings(self) -> Dict[str, np.ndarray]:
        """Create synthetic emotion embeddings for testing"""
        
        # Define emotions in a 3D space (valence, arousal, dominance)
        emotion_coords = {
            "neutral": [0.5, 0.5, 0.5],
            "joyful_excited": [0.9, 0.9, 0.7],
            "calm_supportive": [0.7, 0.3, 0.6],
            "empathetic_sad": [0.2, 0.4, 0.3],
            "angry_firm": [0.1, 0.8, 0.9],
            "playful": [0.8, 0.7, 0.5],
            "thoughtful": [0.6, 0.4, 0.6],
            "confident": [0.7, 0.6, 0.8],
            "melancholy": [0.3, 0.3, 0.3],
            "comforting": [0.6, 0.4, 0.5],
            "encouraging": [0.8, 0.6, 0.7],
            "wistful": [0.4, 0.4, 0.4],
            "sarcastic": [0.5, 0.6, 0.7],
            "reassuring": [0.7, 0.4, 0.6]
        }
        
        # Convert to higher-dimensional embeddings
        embeddings = {}
        embedding_dim = 256  # Typical emotion embedding dimension
        
        for emotion, coords in emotion_coords.items():
            # Create base embedding from coordinates
            base = np.array(coords)
            
            # Expand to full dimension with noise
            full_embedding = np.zeros(embedding_dim)
            full_embedding[:3] = base
            
            # Add structured noise for uniqueness
            noise = np.random.randn(embedding_dim) * 0.1
            full_embedding += noise
            
            # Normalize
            full_embedding = full_embedding / np.linalg.norm(full_embedding)
            
            embeddings[emotion] = full_embedding
        
        return embeddings
    
    def smooth_transition(
        self,
        target_emotion: str,
        intensity: float = 1.0,
        speed: str = "normal"
    ) -> Tuple[str, np.ndarray, Dict]:
        """
        Create smooth transition to target emotion
        
        Args:
            target_emotion: Target emotion to transition to
            intensity: Intensity of the emotion (0-1)
            speed: Transition speed (instant/fast/normal/slow/gradual)
            
        Returns:
            Tuple of (interpolated_emotion_name, embedding, transition_info)
        """
        
        # Get embeddings
        if target_emotion not in self.embeddings:
            print(f"Warning: {target_emotion} not in embeddings, using neutral")
            target_emotion = "neutral"
        
        target_embedding = self.embeddings[target_emotion]
        
        # Initialize if first call
        if self.current_embedding is None:
            self.current_embedding = self.embeddings.get(
                self.current_emotion, 
                self.embeddings["neutral"]
            )
        
        # Get transition parameters
        blend_ratio = self.TRANSITION_SPEEDS.get(speed, 0.5)
        
        # Check compatibility for smooth transition
        compatibility = self._get_compatibility(self.current_emotion, target_emotion)
        
        # Adjust blend ratio based on compatibility
        if compatibility < 0.5:  # Incompatible emotions
            # Slower transition for jarring changes
            blend_ratio = min(0.8, blend_ratio + 0.2)
        
        # Perform interpolation
        if speed == "instant":
            interpolated = target_embedding * intensity
            self.current_embedding = interpolated
            self.current_emotion = target_emotion
        else:
            # Smooth interpolation
            interpolated = (
                blend_ratio * self.current_embedding + 
                (1 - blend_ratio) * target_embedding * intensity
            )
            
            # Normalize
            interpolated = interpolated / np.linalg.norm(interpolated)
            
            # Update current
            self.current_embedding = interpolated
            
            # Determine current emotion name (closest in embedding space)
            self.current_emotion = self._find_closest_emotion(interpolated)
        
        # Record transition
        transition_info = {
            "from": self.previous_emotion,
            "to": target_emotion,
            "current": self.current_emotion,
            "blend_ratio": blend_ratio,
            "compatibility": compatibility,
            "intensity": intensity
        }
        
        self.transition_history.append(transition_info)
        self.previous_emotion = self.current_emotion
        
        return self.current_emotion, interpolated, transition_info
    
    def _get_compatibility(self, emotion1: str, emotion2: str) -> float:
        """Get compatibility score between two emotions"""
        
        # Check direct compatibility
        if emotion1 in self.EMOTION_COMPATIBILITY:
            if emotion2 in self.EMOTION_COMPATIBILITY[emotion1]:
                return self.EMOTION_COMPATIBILITY[emotion1][emotion2]
        
        # Check reverse
        if emotion2 in self.EMOTION_COMPATIBILITY:
            if emotion1 in self.EMOTION_COMPATIBILITY[emotion2]:
                return self.EMOTION_COMPATIBILITY[emotion2][emotion1]
        
        # Calculate based on embedding distance
        if emotion1 in self.embeddings and emotion2 in self.embeddings:
            emb1 = self.embeddings[emotion1]
            emb2 = self.embeddings[emotion2]
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2)
            return (similarity + 1) / 2  # Normalize to 0-1
        
        return 0.5  # Default neutral compatibility
    
    def _find_closest_emotion(self, embedding: np.ndarray) -> str:
        """Find closest emotion in embedding space"""
        
        min_distance = float('inf')
        closest_emotion = "neutral"
        
        for emotion, emb in self.embeddings.items():
            distance = np.linalg.norm(embedding - emb)
            if distance < min_distance:
                min_distance = distance
                closest_emotion = emotion
        
        return closest_emotion
    
    def plan_trajectory(
        self,
        target_emotion: str,
        steps: Optional[int] = None
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Plan multi-step transition trajectory
        
        Args:
            target_emotion: Final target emotion
            steps: Number of steps (None for automatic)
            
        Returns:
            List of (emotion, embedding) tuples for trajectory
        """
        
        if steps is None:
            # Determine steps based on compatibility
            compatibility = self._get_compatibility(self.current_emotion, target_emotion)
            if compatibility > 0.7:
                steps = self.min_transition_steps
            elif compatibility < 0.3:
                steps = self.max_transition_steps
            else:
                steps = 3
        
        trajectory = []
        
        # Get start and end embeddings
        start_emb = self.current_embedding if self.current_embedding is not None else self.embeddings["neutral"]
        end_emb = self.embeddings.get(target_emotion, self.embeddings["neutral"])
        
        # Generate trajectory points
        for i in range(steps + 1):
            t = i / steps  # 0 to 1
            
            # Use ease-in-out curve for natural transition
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            
            # Interpolate
            interp_emb = (1 - t_smooth) * start_emb + t_smooth * end_emb
            interp_emb = interp_emb / np.linalg.norm(interp_emb)
            
            # Find closest emotion
            emotion = self._find_closest_emotion(interp_emb)
            
            trajectory.append((emotion, interp_emb))
        
        self.transition_trajectory = trajectory
        return trajectory
    
    def get_transition_parameters(self, from_emotion: str, to_emotion: str) -> Dict:
        """
        Get optimal transition parameters
        
        Returns:
            Dict with transition recommendations
        """
        
        compatibility = self._get_compatibility(from_emotion, to_emotion)
        
        params = {
            "compatibility": compatibility,
            "recommended_speed": "normal",
            "needs_intermediate": False,
            "intermediate_emotion": None,
            "estimated_steps": 1
        }
        
        if compatibility < 0.3:
            # Very incompatible - needs intermediate
            params["recommended_speed"] = "gradual"
            params["needs_intermediate"] = True
            params["intermediate_emotion"] = self._find_intermediate(from_emotion, to_emotion)
            params["estimated_steps"] = 3
            
        elif compatibility < 0.5:
            # Somewhat incompatible
            params["recommended_speed"] = "slow"
            params["estimated_steps"] = 2
            
        elif compatibility > 0.8:
            # Very compatible
            params["recommended_speed"] = "fast"
            params["estimated_steps"] = 1
        
        return params
    
    def _find_intermediate(self, emotion1: str, emotion2: str) -> str:
        """Find intermediate emotion for smooth transition"""
        
        # Common intermediates for difficult transitions
        intermediates = {
            ("joyful_excited", "empathetic_sad"): "thoughtful",
            ("angry_firm", "joyful_excited"): "neutral",
            ("playful", "empathetic_sad"): "calm_supportive",
            ("sarcastic", "comforting"): "thoughtful"
        }
        
        # Check predefined intermediates
        if (emotion1, emotion2) in intermediates:
            return intermediates[(emotion1, emotion2)]
        if (emotion2, emotion1) in intermediates:
            return intermediates[(emotion2, emotion1)]
        
        # Default to neutral for difficult transitions
        return "neutral"


def test_emotion_transitions():
    """Test the emotion transition system"""
    
    smoother = EmotionTransitionSmoother()
    
    print("🎭 Testing Emotion Transition Smoothing\n")
    print("=" * 60)
    
    # Test direct transitions
    print("\n📊 Direct Transitions")
    print("-" * 40)
    
    transitions = [
        ("neutral", "joyful_excited", "fast"),
        ("joyful_excited", "empathetic_sad", "slow"),
        ("angry_firm", "calm_supportive", "gradual"),
        ("playful", "thoughtful", "normal")
    ]
    
    for from_emotion, to_emotion, speed in transitions:
        # Set current emotion
        smoother.current_emotion = from_emotion
        smoother.current_embedding = smoother.embeddings[from_emotion]
        
        # Perform transition
        current, embedding, info = smoother.smooth_transition(to_emotion, 1.0, speed)
        
        print(f"{from_emotion} → {to_emotion} ({speed})")
        print(f"  Current: {current}")
        print(f"  Compatibility: {info['compatibility']:.2f}")
        print(f"  Blend ratio: {info['blend_ratio']:.2f}")
        print()
    
    # Test trajectory planning
    print("\n🗺️ Trajectory Planning")
    print("-" * 40)
    
    smoother.current_emotion = "angry_firm"
    trajectory = smoother.plan_trajectory("joyful_excited", steps=4)
    
    print("Trajectory from angry_firm to joyful_excited:")
    for i, (emotion, _) in enumerate(trajectory):
        print(f"  Step {i}: {emotion}")
    
    # Test transition parameters
    print("\n⚙️ Transition Parameters")
    print("-" * 40)
    
    test_pairs = [
        ("joyful_excited", "playful"),
        ("joyful_excited", "empathetic_sad"),
        ("angry_firm", "comforting")
    ]
    
    for from_em, to_em in test_pairs:
        params = smoother.get_transition_parameters(from_em, to_em)
        print(f"{from_em} → {to_em}:")
        print(f"  Compatibility: {params['compatibility']:.2f}")
        print(f"  Speed: {params['recommended_speed']}")
        if params['needs_intermediate']:
            print(f"  Via: {params['intermediate_emotion']}")
        print()


if __name__ == "__main__":
    test_emotion_transitions()
