#!/usr/bin/env python3
"""
Oviya Emotion Prosody MCP Server
Provides real-time voice emotion detection and prosody analysis
"""

import asyncio
import json
import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import librosa
import torch

# Add project paths for standalone execution
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class OviyaEmotionProsodyServer:
    """
    MCP Server for voice emotion detection and prosody analysis

    Provides:
    - Real-time F0 (pitch) analysis
    - Energy/volume analysis
    - Speech rate detection
    - Emotion classification from voice
    - Prosody target generation for synthesis
    """

    def __init__(self):
        self.sample_rate = 16000  # Standard for voice processing
        self.frame_length = 0.025  # 25ms frames
        self.frame_step = 0.01     # 10ms step

        # Emotion-voice correlations (based on research)
        self.emotion_prosody_profiles = {
            "calm": {
                "f0_mean": 180.0, "f0_std": 20.0, "energy": 0.6, "rate": 0.9,
                "pause_freq": 0.3, "breath_freq": 0.2
            },
            "sad": {
                "f0_mean": 160.0, "f0_std": 15.0, "energy": 0.4, "rate": 0.8,
                "pause_freq": 0.5, "breath_freq": 0.4
            },
            "anxious": {
                "f0_mean": 200.0, "f0_std": 35.0, "energy": 0.7, "rate": 1.2,
                "pause_freq": 0.2, "breath_freq": 0.6
            },
            "angry": {
                "f0_mean": 220.0, "f0_std": 40.0, "energy": 0.9, "rate": 1.3,
                "pause_freq": 0.1, "breath_freq": 0.3
            },
            "joyful": {
                "f0_mean": 210.0, "f0_std": 30.0, "energy": 0.8, "rate": 1.1,
                "pause_freq": 0.2, "breath_freq": 0.1
            },
            "neutral": {
                "f0_mean": 185.0, "f0_std": 25.0, "energy": 0.65, "rate": 1.0,
                "pause_freq": 0.25, "breath_freq": 0.25
            }
        }

    def _extract_prosody_features(self, audio_data: bytes) -> Dict[str, float]:
        """
        Extract prosody features from audio data

        Args:
            audio_data: Raw PCM audio bytes (16-bit, 16kHz, mono)

        Returns:
            Dictionary of prosody features
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) == 0:
                return {"error": "Empty audio data"}

            # Extract pitch (F0) using librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_array,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                frame_length=int(self.frame_length * self.sample_rate),
                hop_length=int(self.frame_step * self.sample_rate)
            )

            # Filter out unvoiced frames
            f0_voiced = f0[voiced_flag]

            if len(f0_voiced) == 0:
                f0_mean, f0_std = 185.0, 25.0  # Default values
            else:
                f0_mean = float(np.mean(f0_voiced))
                f0_std = float(np.std(f0_voiced))

            # Extract energy (RMS)
            rms = librosa.feature.rms(
                y=audio_array,
                frame_length=int(self.frame_length * self.sample_rate),
                hop_length=int(self.frame_step * self.sample_rate)
            )[0]
            energy = float(np.mean(rms))

            # Estimate speech rate (syllables per second)
            # Simple heuristic: higher zero crossings = faster speech
            zero_crossings = librosa.zero_crossings(audio_array)
            speech_rate = float(np.mean(zero_crossings)) * 10  # Rough approximation

            # Detect pauses (low energy periods)
            pause_threshold = np.percentile(rms, 25)  # Lower quartile
            pauses = np.sum(rms < pause_threshold) / len(rms)
            pause_freq = float(pauses)

            # Estimate breath frequency (very rough heuristic)
            # Look for low-frequency energy patterns
            breath_freq = min(0.5, pause_freq * 0.8)  # Simplified

            return {
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "energy": energy,
                "speech_rate": speech_rate,
                "pause_frequency": pause_freq,
                "breath_frequency": breath_freq,
                "duration_seconds": len(audio_array) / self.sample_rate,
                "confidence": 0.85 if len(audio_array) > self.sample_rate else 0.6
            }

        except Exception as e:
            return {
                "error": str(e),
                "f0_mean": 185.0,
                "f0_std": 25.0,
                "energy": 0.65,
                "speech_rate": 1.0,
                "pause_frequency": 0.25,
                "breath_frequency": 0.25,
                "confidence": 0.3
            }

    def _classify_emotion_from_prosody(self, prosody_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify emotion from prosody features using rule-based approach
        """
        f0_mean = prosody_features.get("f0_mean", 185.0)
        f0_std = prosody_features.get("f0_std", 25.0)
        energy = prosody_features.get("energy", 0.65)
        speech_rate = prosody_features.get("speech_rate", 1.0)
        pause_freq = prosody_features.get("pause_frequency", 0.25)

        # Emotion classification based on prosody research
        scores = {}

        for emotion, profile in self.emotion_prosody_profiles.items():
            # Calculate similarity score
            f0_diff = abs(f0_mean - profile["f0_mean"]) / 50.0  # Normalized difference
            f0_std_diff = abs(f0_std - profile["f0_std"]) / 20.0
            energy_diff = abs(energy - profile["energy"]) / 0.5
            rate_diff = abs(speech_rate - profile["rate"]) / 0.5
            pause_diff = abs(pause_freq - profile["pause_freq"]) / 0.5

            # Weighted combination
            total_diff = (f0_diff * 0.3 + f0_std_diff * 0.2 + energy_diff * 0.25 +
                         rate_diff * 0.15 + pause_diff * 0.1)

            scores[emotion] = max(0, 1 - total_diff)  # Convert to similarity score

        # Get top emotion
        top_emotion = max(scores, key=scores.get)
        confidence = scores[top_emotion]

        return {
            "emotion": top_emotion,
            "confidence": confidence,
            "all_scores": scores,
            "prosody_features": prosody_features
        }

    def _detect_emotion_shift(self, previous_prosody: Dict[str, float],
                            current_prosody: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect significant emotion shifts between audio segments
        """
        if not previous_prosody or not current_prosody:
            return {"shift_detected": False, "shift_magnitude": 0}

        # Calculate shift magnitude across key features
        f0_shift = abs(current_prosody.get("f0_mean", 185) - previous_prosody.get("f0_mean", 185)) / 50.0
        energy_shift = abs(current_prosody.get("energy", 0.65) - previous_prosody.get("energy", 0.65)) / 0.5
        rate_shift = abs(current_prosody.get("speech_rate", 1.0) - previous_prosody.get("speech_rate", 1.0)) / 0.5

        # Weighted shift magnitude
        shift_magnitude = (f0_shift * 0.4 + energy_shift * 0.4 + rate_shift * 0.2)

        # Classify previous and current emotions
        prev_emotion = self._classify_emotion_from_prosody(previous_prosody)
        curr_emotion = self._classify_emotion_from_prosody(current_prosody)

        emotion_changed = prev_emotion["emotion"] != curr_emotion["emotion"]
        significant_shift = shift_magnitude > 0.3  # Threshold for significance

        return {
            "shift_detected": emotion_changed or significant_shift,
            "shift_magnitude": shift_magnitude,
            "emotion_changed": emotion_changed,
            "previous_emotion": prev_emotion["emotion"],
            "current_emotion": curr_emotion["emotion"],
            "significant_shift": significant_shift,
            "feature_changes": {
                "f0_change": current_prosody.get("f0_mean", 185) - previous_prosody.get("f0_mean", 185),
                "energy_change": current_prosody.get("energy", 0.65) - previous_prosody.get("energy", 0.65),
                "rate_change": current_prosody.get("speech_rate", 1.0) - previous_prosody.get("speech_rate", 1.0)
            }
        }

    def _compute_prosody_targets(self, personality_vector: Dict[str, float],
                               emotion: str, intensity: float = 0.7) -> Dict[str, Any]:
        """
        Compute prosody targets for voice synthesis based on personality and emotion
        """
        # Base prosody from emotion profile
        base_prosody = self.emotion_prosody_profiles.get(emotion, self.emotion_prosody_profiles["neutral"])

        # Adjust based on personality vector
        personality_adjustments = {
            "Ma": {"f0_variation": +0.1, "energy": +0.05, "speech_rate": +0.1},  # Creative = more expressive
            "Ahimsa": {"energy": -0.1, "speech_rate": -0.1, "pause_freq": +0.1},  # Safe = calmer, more pauses
            "Jeong": {"f0_mean": -10, "energy": -0.05, "pause_freq": +0.05},  # Connected = warmer, thoughtful
            "Logos": {"speech_rate": -0.1, "pause_freq": +0.1},  # Logical = deliberate, clear
            "Lagom": {"f0_std": -5, "energy": -0.05, "speech_rate": -0.05}  # Balanced = moderate
        }

        # Apply personality adjustments
        targets = base_prosody.copy()
        for pillar, weight in personality_vector.items():
            if pillar in personality_adjustments and weight > 0.2:
                adjustments = personality_adjustments[pillar]
                for feature, adjustment in adjustments.items():
                    if feature in targets:
                        targets[feature] += adjustment * weight * intensity

        # Apply intensity scaling
        for feature in ["f0_std", "energy"]:
            if feature in targets:
                targets[feature] *= intensity

        # Ensure reasonable bounds
        targets["f0_mean"] = max(150, min(300, targets["f0_mean"]))
        targets["f0_std"] = max(10, min(60, targets["f0_std"]))
        targets["energy"] = max(0.3, min(1.0, targets["energy"]))
        targets["speech_rate"] = max(0.6, min(1.8, targets["speech_rate"]))

        return {
            "prosody_targets": targets,
            "personality_influence": personality_vector,
            "emotion": emotion,
            "intensity": intensity,
            "voice_style": self._map_to_voice_style(targets)
        }

    def _map_to_voice_style(self, prosody_targets: Dict[str, float]) -> str:
        """Map prosody targets to voice synthesis style"""
        f0_mean = prosody_targets.get("f0_mean", 185)
        energy = prosody_targets.get("energy", 0.65)
        speech_rate = prosody_targets.get("speech_rate", 1.0)

        if energy > 0.8 and speech_rate > 1.1:
            return "excited_energetic"
        elif energy < 0.5 and speech_rate < 0.9:
            return "calm_soothing"
        elif f0_mean > 200 and energy > 0.7:
            return "warm_enthusiastic"
        elif speech_rate < 0.8 and energy < 0.6:
            return "thoughtful_deliberate"
        else:
            return "balanced_natural"

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""

        if request.get("method") == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                if tool_name == "analyze_prosody":
                    # Expect base64 encoded audio or raw bytes
                    audio_data = arguments.get("audio_chunk", b"")
                    if isinstance(audio_data, str):
                        # Assume base64 if string
                        import base64
                        audio_data = base64.b64decode(audio_data)

                    prosody_features = self._extract_prosody_features(audio_data)
                    emotion_analysis = self._classify_emotion_from_prosody(prosody_features)

                    return {
                        "content": [{"type": "text", "text": json.dumps({
                            "prosody_analysis": prosody_features,
                            "emotion_detection": emotion_analysis,
                            "confidence": emotion_analysis.get("confidence", 0)
                        })}]
                    }

                elif tool_name == "detect_emotion_shift":
                    previous_prosody = arguments.get("previous_prosody", {})
                    current_prosody = arguments.get("current_prosody", {})

                    shift_analysis = self._detect_emotion_shift(previous_prosody, current_prosody)

                    return {
                        "content": [{"type": "text", "text": json.dumps(shift_analysis)}]
                    }

                elif tool_name == "compute_prosody_targets":
                    personality_vector = arguments.get("personality_vector", {})
                    emotion = arguments.get("emotion", "neutral")
                    intensity = arguments.get("intensity", 0.7)

                    targets = self._compute_prosody_targets(personality_vector, emotion, intensity)

                    return {
                        "content": [{"type": "text", "text": json.dumps(targets)}]
                    }

                elif tool_name == "analyze_voice_emotion_trend":
                    # Analyze emotion trends over multiple audio segments
                    audio_segments = arguments.get("audio_segments", [])
                    trend_analysis = []

                    previous_prosody = None
                    for i, segment in enumerate(audio_segments):
                        if isinstance(segment, str):
                            import base64
                            segment = base64.b64decode(segment)

                        current_prosody = self._extract_prosody_features(segment)
                        current_emotion = self._classify_emotion_from_prosody(current_prosody)

                        trend_analysis.append({
                            "segment": i,
                            "prosody": current_prosody,
                            "emotion": current_emotion["emotion"],
                            "confidence": current_emotion["confidence"]
                        })

                        if previous_prosody:
                            shift = self._detect_emotion_shift(previous_prosody, current_prosody)
                            trend_analysis[-1]["shift_from_previous"] = shift

                        previous_prosody = current_prosody

                    return {
                        "content": [{"type": "text", "text": json.dumps({
                            "trend_analysis": trend_analysis,
                            "total_segments": len(trend_analysis),
                            "emotion_stability": self._calculate_emotion_stability(trend_analysis)
                        })}]
                    }

                else:
                    return {"error": f"Unknown tool: {tool_name}"}

            except Exception as e:
                return {"error": str(e)}

        elif request.get("method") == "tools/list":
            return {
                "tools": [
                    {
                        "name": "analyze_prosody",
                        "description": "Extract prosody features and detect emotion from voice audio",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "audio_chunk": {
                                    "type": "string",
                                    "description": "Base64-encoded PCM audio data (16-bit, 16kHz, mono)"
                                }
                            },
                            "required": ["audio_chunk"]
                        }
                    },
                    {
                        "name": "detect_emotion_shift",
                        "description": "Detect significant emotion changes between audio segments",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "previous_prosody": {"type": "object"},
                                "current_prosody": {"type": "object"}
                            },
                            "required": ["previous_prosody", "current_prosody"]
                        }
                    },
                    {
                        "name": "compute_prosody_targets",
                        "description": "Generate prosody targets for voice synthesis",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "personality_vector": {"type": "object"},
                                "emotion": {"type": "string"},
                                "intensity": {"type": "number", "default": 0.7}
                            },
                            "required": ["personality_vector", "emotion"]
                        }
                    },
                    {
                        "name": "analyze_voice_emotion_trend",
                        "description": "Analyze emotion trends across multiple audio segments",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "audio_segments": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of base64-encoded audio segments"
                                }
                            },
                            "required": ["audio_segments"]
                        }
                    }
                ]
            }

        elif request.get("method") == "resources/list":
            return {
                "resources": [
                    {
                        "uri": "prosody://emotion_profiles",
                        "name": "Emotion Prosody Profiles",
                        "description": "Voice characteristics for different emotions",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "prosody://personality_voice_map",
                        "name": "Personality to Voice Mapping",
                        "description": "How personality pillars affect voice synthesis",
                        "mimeType": "application/json"
                    }
                ]
            }

        elif request.get("method") == "resources/read":
            uri = request["params"]["uri"]

            if uri == "prosody://emotion_profiles":
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "emotion_profiles": self.emotion_prosody_profiles,
                            "features_extracted": [
                                "f0_mean", "f0_std", "energy", "speech_rate",
                                "pause_frequency", "breath_frequency"
                            ]
                        })
                    }]
                }

            elif uri == "prosody://personality_voice_map":
                personality_voice_map = {
                    "Ma": "More expressive, varied pitch, energetic",
                    "Ahimsa": "Calmer, gentler, more pauses for safety",
                    "Jeong": "Warmer tone, thoughtful pacing, connected feeling",
                    "Logos": "Clear articulation, deliberate pace, rational tone",
                    "Lagom": "Balanced modulation, moderate energy, harmonious"
                }
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({"personality_voice_map": personality_voice_map})
                    }]
                }

        return {"error": "Method not supported"}

    def _calculate_emotion_stability(self, trend_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate emotion stability across segments"""
        if len(trend_analysis) < 2:
            return {"stability_score": 1.0, "assessment": "insufficient_data"}

        emotions = [segment["emotion"] for segment in trend_analysis]
        confidences = [segment["confidence"] for segment in trend_analysis]

        # Calculate consistency (how often same emotion appears)
        most_common_emotion = max(set(emotions), key=emotions.count)
        consistency = emotions.count(most_common_emotion) / len(emotions)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)

        # Calculate shifts
        shifts = sum(1 for segment in trend_analysis if segment.get("shift_from_previous", {}).get("shift_detected", False))
        shift_rate = shifts / (len(trend_analysis) - 1)

        # Overall stability score (higher = more stable)
        stability_score = (consistency * 0.5 + avg_confidence * 0.3 + (1 - shift_rate) * 0.2)

        if stability_score > 0.8:
            assessment = "highly_stable"
        elif stability_score > 0.6:
            assessment = "moderately_stable"
        elif stability_score > 0.4:
            assessment = "variable"
        else:
            assessment = "highly_variable"

        return {
            "stability_score": stability_score,
            "assessment": assessment,
            "most_common_emotion": most_common_emotion,
            "consistency": consistency,
            "average_confidence": avg_confidence,
            "shift_rate": shift_rate
        }

async def main():
    """Main MCP server loop"""
    server = OviyaEmotionProsodyServer()

    # Read from stdin, write to stdout (MCP stdio protocol)
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = await server.handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            print(json.dumps({"error": "Invalid JSON"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
