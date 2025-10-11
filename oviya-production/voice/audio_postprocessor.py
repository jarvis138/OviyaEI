"""
Audio Post-Processor for Oviya Realism
Handles breath injection, EQ, mastering, and human imperfection modeling
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import random
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

try:
    import pyloudnorm as pyln
    LOUDNORM_AVAILABLE = True
except ImportError:
    LOUDNORM_AVAILABLE = False
    print("⚠️  pyloudnorm not available - install with: pip install pyloudnorm")


class BreathSampleManager:
    """Manages breath samples for natural speech injection"""
    
    def __init__(self, breath_dir: Path = Path("audio_assets/breath_samples")):
        self.breath_dir = Path(breath_dir)
        self.breath_samples = {}
        self.sample_rate = 24000
        self._load_breath_samples()
    
    def _load_breath_samples(self):
        """Load breath samples from directory"""
        if not self.breath_dir.exists():
            print(f"⚠️  Breath samples directory not found: {self.breath_dir}")
            self._generate_synthetic_breaths()
            return
        
        # Load existing breath samples
        breath_files = list(self.breath_dir.glob("*.wav"))
        if not breath_files:
            print("⚠️  No breath samples found, generating synthetic ones")
            self._generate_synthetic_breaths()
            return
        
        for breath_file in breath_files:
            try:
                audio, sr = torchaudio.load(breath_file)
                
                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    audio = resampler(audio)
                
                # Convert to mono
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                
                # Normalize
                audio = audio / (audio.abs().max() + 1e-8)
                
                breath_type = breath_file.stem
                self.breath_samples[breath_type] = audio.squeeze(0)
                
            except Exception as e:
                print(f"⚠️  Failed to load {breath_file}: {e}")
        
        print(f"✅ Loaded {len(self.breath_samples)} breath samples")
    
    def _generate_synthetic_breaths(self):
        """Generate synthetic breath samples"""
        print("🔄 Generating synthetic breath samples...")
        
        # Create directory
        self.breath_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate different types of breaths (shorter durations to minimize drift)
        breath_types = {
            "soft_inhale": {"duration": 0.15, "freq_range": (50, 200), "amplitude": 0.08},
            "gentle_exhale": {"duration": 0.2, "freq_range": (40, 150), "amplitude": 0.06},
            "quick_breath": {"duration": 0.1, "freq_range": (80, 300), "amplitude": 0.1},
            "sigh": {"duration": 0.25, "freq_range": (30, 120), "amplitude": 0.12},
            "pause_breath": {"duration": 0.15, "freq_range": (60, 180), "amplitude": 0.05}
        }
        
        for breath_name, params in breath_types.items():
            # Generate noise-based breath
            duration = params["duration"]
            samples = int(duration * self.sample_rate)
            
            # Pink noise base
            noise = torch.randn(samples)
            
            # Apply frequency shaping (low-pass for breath-like sound)
            freq_low, freq_high = params["freq_range"]
            sos = signal.butter(4, [freq_low, freq_high], 'band', fs=self.sample_rate, output='sos')
            shaped_noise = signal.sosfilt(sos, noise.numpy())
            
            # Apply amplitude envelope
            envelope = np.exp(-np.linspace(0, 3, samples))  # Exponential decay
            if "inhale" in breath_name:
                envelope = envelope[::-1]  # Reverse for inhale
            
            breath_audio = torch.tensor(shaped_noise * envelope * params["amplitude"], dtype=torch.float32)
            
            # Save
            output_path = self.breath_dir / f"{breath_name}.wav"
            torchaudio.save(str(output_path), breath_audio.unsqueeze(0), self.sample_rate)
            
            self.breath_samples[breath_name] = breath_audio
        
        print(f"✅ Generated {len(breath_types)} synthetic breath samples")
    
    def get_breath_sample(self, breath_type: str = "random") -> torch.Tensor:
        """Get a breath sample"""
        if not self.breath_samples:
            return torch.zeros(int(0.5 * self.sample_rate))  # Silent fallback
        
        if breath_type == "random":
            breath_type = random.choice(list(self.breath_samples.keys()))
        
        if breath_type in self.breath_samples:
            return self.breath_samples[breath_type].clone()
        else:
            # Fallback to first available
            return list(self.breath_samples.values())[0].clone()


class AudioMaster:
    """Audio mastering for natural speech quality"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        
    def master_audio(self, audio: torch.Tensor, target_lufs: float = -16.0) -> torch.Tensor:
        """Apply mastering to audio for natural speech quality"""
        
        # Convert to numpy for processing
        audio_np = audio.cpu().numpy()
        
        # 1. Loudness normalization
        if LOUDNORM_AVAILABLE:
            try:
                meter = pyln.Meter(self.sample_rate)
                loudness = meter.integrated_loudness(audio_np)
                
                if not np.isnan(loudness) and loudness > -70:  # Valid loudness
                    audio_np = pyln.normalize.loudness(audio_np, loudness, target_lufs)
            except Exception as e:
                print(f"⚠️  Loudness normalization failed: {e}")
        
        # 2. High-frequency enhancement (presence boost)
        try:
            # Gentle high-shelf EQ at 4kHz (+2dB)
            sos = signal.butter(2, 4000, 'high', fs=self.sample_rate, output='sos')
            audio_np = signal.sosfilt(sos, audio_np)
            audio_np = audio_np * 1.02  # +2dB boost
        except Exception as e:
            print(f"⚠️  EQ processing failed: {e}")
        
        # 3. Gentle compression (reduce dynamic range)
        audio_np = self._gentle_compress(audio_np)
        
        # 4. Final limiting
        peak = np.abs(audio_np).max()
        if peak > 0.95:
            audio_np = audio_np * (0.95 / peak)
        
        return torch.tensor(audio_np, dtype=torch.float32)
    
    def _gentle_compress(self, audio: np.ndarray, threshold: float = 0.7, ratio: float = 3.0) -> np.ndarray:
        """Apply gentle compression to reduce dynamic range"""
        # Simple soft-knee compression
        abs_audio = np.abs(audio)
        
        # Find samples above threshold
        above_threshold = abs_audio > threshold
        
        if np.any(above_threshold):
            # Calculate compression
            excess = abs_audio[above_threshold] - threshold
            compressed_excess = excess / ratio
            
            # Apply compression while preserving sign
            sign = np.sign(audio[above_threshold])
            audio[above_threshold] = sign * (threshold + compressed_excess)
        
        return audio
    
    def add_room_reverb(self, audio: torch.Tensor, wet_level: float = 0.1) -> torch.Tensor:
        """Add subtle room reverb for spatial realism"""
        # Simple algorithmic reverb using multiple delays
        reverb_delays = [0.03, 0.05, 0.08, 0.12, 0.18]  # seconds
        reverb_gains = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        reverb_audio = torch.zeros_like(audio)
        
        for delay, gain in zip(reverb_delays, reverb_gains):
            delay_samples = int(delay * self.sample_rate)
            if delay_samples < len(audio):
                # Create delayed version
                delayed = torch.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain
                reverb_audio += delayed
        
        # Mix dry and wet signals
        return (1 - wet_level) * audio + wet_level * reverb_audio


class ProsodyProcessor:
    """Processes prosodic markup in text for audio modification"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.breath_manager = BreathSampleManager()
    
    def process_prosodic_text(self, audio: torch.Tensor, prosodic_text: str) -> torch.Tensor:
        """Process audio based on prosodic markup"""
        
        # Extract prosodic markers
        markers = self._extract_markers(prosodic_text)
        
        if not markers:
            return audio  # No markers to process
        
        # Apply timing modifications
        modified_audio = self._apply_timing_modifications(audio, markers)
        
        # Add breath samples
        final_audio = self._add_breath_samples(modified_audio, markers)
        
        return final_audio
    
    def _extract_markers(self, prosodic_text: str) -> List[Dict]:
        """Extract prosodic markers from text"""
        import re
        
        markers = []
        
        # Find all markers
        marker_patterns = {
            'breath': r'<breath>',
            'pause': r'<pause>',
            'long_pause': r'<long_pause>',
            'smile': r'<smile>',
            'gentle': r'<gentle>(.*?)</gentle>',
            'strong': r'<strong>(.*?)</strong>'
        }
        
        for marker_type, pattern in marker_patterns.items():
            matches = re.finditer(pattern, prosodic_text)
            for match in matches:
                markers.append({
                    'type': marker_type,
                    'position': match.start(),
                    'text': match.group(1) if marker_type in ['gentle', 'strong'] else None
                })
        
        return sorted(markers, key=lambda x: x['position'])
    
    def _apply_timing_modifications(self, audio: torch.Tensor, markers: List[Dict]) -> torch.Tensor:
        """Apply timing modifications based on markers"""
        
        # For now, just add small random variations to make speech more human
        # In a full implementation, this would map markers to specific audio positions
        
        # Add subtle timing jitter (±5% duration variation)
        jitter_factor = 1.0 + (random.random() - 0.5) * 0.1  # ±5%
        
        if jitter_factor != 1.0:
            # Simple time-stretching using interpolation
            original_length = len(audio)
            new_length = int(original_length * jitter_factor)
            
            # Linear interpolation for time stretching
            indices = torch.linspace(0, original_length - 1, new_length)
            indices_floor = indices.long()
            indices_ceil = torch.clamp(indices_floor + 1, max=original_length - 1)
            
            weight = indices - indices_floor.float()
            
            stretched_audio = (1 - weight) * audio[indices_floor] + weight * audio[indices_ceil]
            return stretched_audio
        
        return audio
    
    def _add_breath_samples(self, audio: torch.Tensor, markers: List[Dict]) -> torch.Tensor:
        """Add breath samples based on markers with robust error handling"""
        
        try:
            breath_markers = [m for m in markers if m['type'] in ['breath', 'pause', 'long_pause']]
            
            if not breath_markers:
                return audio
            
            # For simplicity, add breath at the beginning if breath markers exist
            has_breath = any(m['type'] == 'breath' for m in breath_markers)
            has_pause = any(m['type'] in ['pause', 'long_pause'] for m in breath_markers)
            
            segments = [audio]
            
            # Limit breath injection to prevent excessive drift
            max_breaths = 1  # Only add one breath per audio to limit drift
            breaths_added = 0
            
            if has_breath and breaths_added < max_breaths:
                try:
                    # Add breath at beginning (only if audio is long enough)
                    if len(audio) / self.sample_rate > 0.5:  # Only for audio > 0.5s
                        breath_sample = self.breath_manager.get_breath_sample("quick_breath")
                        if breath_sample is not None and len(breath_sample) > 0:
                            # Scale breath to be quieter
                            breath_sample = breath_sample * 0.3
                            segments.insert(0, breath_sample)
                            breaths_added += 1
                except Exception as e:
                    print(f"⚠️  Failed to add breath at beginning: {e}")
            
            if has_pause and breaths_added < max_breaths:
                try:
                    # Add very short pause (silence) instead of breath sample
                    # This maintains pause effect without adding duration
                    if len(audio) / self.sample_rate > 1.0:  # Only for audio > 1s
                        split_point = int(len(audio) * 0.6)
                        if split_point > 0 and split_point < len(audio):
                            # Add tiny silence (50ms)
                            silence = torch.zeros(int(0.05 * self.sample_rate))
                            segments = [
                                audio[:split_point],
                                silence,
                                audio[split_point:]
                            ]
                            breaths_added += 1
                except Exception as e:
                    print(f"⚠️  Failed to add pause: {e}")
            
            # Concatenate all segments with error handling
            try:
                if len(segments) > 1:
                    return torch.cat(segments, dim=0)
                else:
                    return audio
            except Exception as e:
                print(f"⚠️  Failed to concatenate audio segments: {e}")
                return audio
                
        except Exception as e:
            print(f"⚠️  Breath injection failed, returning original audio: {e}")
            return audio  # Return unmodified audio on any error


class AudioPostProcessor:
    """Main audio post-processor combining all enhancements"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.prosody_processor = ProsodyProcessor(sample_rate)
        self.audio_master = AudioMaster(sample_rate)
        
        print("✅ Audio Post-Processor initialized")
        print("   🫁 Breath sample system ready")
        print("   🎚️  Audio mastering system ready")
        print("   🎭 Prosody processing system ready")
    
    def process(
        self, 
        audio: torch.Tensor, 
        prosodic_text: str = "",
        emotional_state: Optional[Dict] = None,
        add_reverb: bool = True,
        master_audio: bool = True,
        check_drift: bool = True
    ) -> torch.Tensor:
        """
        Full audio post-processing pipeline
        
        Args:
            audio: Input audio tensor
            prosodic_text: Text with prosodic markers
            emotional_state: Emotional state dict from brain
            add_reverb: Whether to add room reverb
            master_audio: Whether to apply mastering
            check_drift: Whether to check for audio duration drift
            
        Returns:
            Processed audio tensor
        """
        
        # Track original duration for drift detection
        original_duration = len(audio) / self.sample_rate
        
        # 1. Process prosodic markup (breath, pauses, timing)
        if prosodic_text:
            audio = self.prosody_processor.process_prosodic_text(audio, prosodic_text)
        
        # 2. Apply emotional modulations
        if emotional_state:
            audio = self._apply_emotional_modulations(audio, emotional_state)
        
        # 3. Add room reverb for spatial realism
        if add_reverb:
            audio = self.audio_master.add_room_reverb(audio, wet_level=0.08)
        
        # 4. Master audio for natural quality
        if master_audio:
            audio = self.audio_master.master_audio(audio, target_lufs=-15.0)  # -15 LUFS for web
        
        # 5. Check for audio drift
        if check_drift:
            processed_duration = len(audio) / self.sample_rate
            drift_percent = abs((processed_duration - original_duration) / original_duration) * 100
            
            if drift_percent > 3.0:
                print(f"⚠️  Audio drift detected: {drift_percent:.1f}%")
                print(f"   Original: {original_duration:.2f}s → Processed: {processed_duration:.2f}s")
            
        return audio
    
    def _apply_emotional_modulations(self, audio: torch.Tensor, emotional_state: Dict) -> torch.Tensor:
        """Apply subtle emotional modulations to audio"""
        
        # Get emotional parameters
        energy_level = emotional_state.get("energy_level", 0.5)
        pace = emotional_state.get("pace", 1.0)
        warmth = emotional_state.get("warmth", 0.5)
        
        # Apply energy-based volume modulation (±10%)
        volume_mod = 0.9 + (energy_level * 0.2)  # 0.9 to 1.1
        audio = audio * volume_mod
        
        # Apply warmth-based EQ (warmer = more low-mids)
        if warmth != 0.5:
            try:
                # Boost/cut around 400Hz based on warmth
                warmth_factor = (warmth - 0.5) * 4  # -2 to +2 dB
                sos = signal.butter(2, [300, 600], 'band', fs=self.sample_rate, output='sos')
                eq_audio = signal.sosfilt(sos, audio.numpy())
                eq_boost = torch.tensor(eq_audio * (10 ** (warmth_factor / 20)), dtype=torch.float32)
                audio = audio + (eq_boost - audio) * 0.3  # Blend 30%
            except Exception:
                pass  # Skip EQ if it fails
        
        return audio


def main():
    """Test the audio post-processor"""
    print("🧪 Testing Audio Post-Processor\n")
    
    # Create test audio (1 second of sine wave)
    sample_rate = 24000
    duration = 1.0
    frequency = 440  # A4
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.3 * torch.sin(2 * np.pi * frequency * t)
    
    # Initialize processor
    processor = AudioPostProcessor(sample_rate)
    
    # Test prosodic text
    prosodic_text = "<breath> Hello there! <smile> How are you doing today? <pause> I hope you're well. <breath>"
    
    # Test emotional state
    emotional_state = {
        "energy_level": 0.7,
        "pace": 1.1,
        "warmth": 0.8,
        "dominant_emotion": "joyful_excited"
    }
    
    # Process audio
    processed_audio = processor.process(
        test_audio,
        prosodic_text=prosodic_text,
        emotional_state=emotional_state
    )
    
    print(f"✅ Processed audio:")
    print(f"   Original length: {len(test_audio)} samples ({len(test_audio)/sample_rate:.2f}s)")
    print(f"   Processed length: {len(processed_audio)} samples ({len(processed_audio)/sample_rate:.2f}s)")
    print(f"   Length change: {len(processed_audio)/len(test_audio):.2f}x")
    
    # Save test output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    torchaudio.save(str(output_dir / "test_original.wav"), test_audio.unsqueeze(0), sample_rate)
    torchaudio.save(str(output_dir / "test_processed.wav"), processed_audio.unsqueeze(0), sample_rate)
    
    print(f"\n💾 Saved test files to {output_dir}/")
    print("   - test_original.wav")
    print("   - test_processed.wav")


if __name__ == "__main__":
    main()
