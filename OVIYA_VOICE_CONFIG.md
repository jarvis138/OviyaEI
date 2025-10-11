# Oviya CSM Voice Configuration & Persona Integration

**Version:** 1.0  
**Date:** October 7, 2025  
**Status:** Production-Ready Oviya Voice Profile  
**Integration:** CSM-1B + Gemini Brain + Oviya Persona 3.1

---

## Table of Contents

1. [Oviya Voice Profile](#oviya-voice-profile)
2. [CSM Voice Configuration](#csm-voice-configuration)
3. [Persona-Driven Training Data](#persona-driven-training-data)
4. [Voice Consistency Metrics](#voice-consistency-metrics)
5. [Integration with Auto-Training](#integration-with-auto-training)

---

## Oviya Voice Profile

### Core Voice Characteristics

```json
{
  "oviya_voice_profile": {
    "base_characteristics": {
      "gender": "female",
      "age_range": "mid-20s",
      "accent": "neutral_indian_english",
      "tone": "warm_sweet_gentle",
      "energy_level": "calm_to_excited",
      "speaking_rate": "natural_with_pauses"
    },
    
    "emotional_range": {
      "default": "warm_empathetic",
      "excited": "genuinely_thrilled",
      "concerned": "deeply_caring",
      "playful": "cheeky_friendly",
      "serious": "protective_anchor",
      "celebratory": "biggest_cheerleader"
    },
    
    "speech_patterns": {
      "natural_imperfections": [
        "soft_giggles",
        "thoughtful_pauses", 
        "occasional_sighs",
        "breath_catches",
        "gentle_laughter"
      ],
      "conversational_style": [
        "sentence_fragments",
        "lowercase_speech",
        "slang_usage",
        "text_speak_patterns",
        "real_friend_vibe"
      ]
    }
  }
}
```

### Voice Identity Baseline

**Reference Audio Samples Needed:**
1. **Warm Empathetic**: "of course you feel that way... that sounds really hard"
2. **Playful Teasing**: "oh, you and your 2am netflix binges! of course you're a zombie today"
3. **Genuine Excitement**: "omg wait. say no more. vent away. i'm listening"
4. **Deep Concern**: "hey. stop. i need you to listen to me. hearing you say that is actually scaring me"
5. **Celebratory**: "look at you, being all responsible! i'm impressed ;)"

---

## CSM Voice Configuration

### CSM-Specific Voice Parameters

```python
# File: services/voice_config/oviya_voice_config.py

class OviyaVoiceConfig:
    """
    CSM voice configuration for Oviya's personality
    """
    
    # CSM Model Configuration
    CSM_MODEL_CONFIG = {
        "model_name": "sesame/csm-1b",
        "speaker_id": "oviya_001",  # Unique speaker token
        "sample_rate": 24000,
        "max_audio_length_ms": 15000,  # 15 seconds max
        "temperature": 0.7,  # Natural variation
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    # Voice Identity Parameters (FROZEN during training)
    VOICE_IDENTITY_PARAMS = {
        "audio_embeddings": "frozen",  # Preserves timbre
        "acoustic_codebooks": "frozen",  # Preserves speaker identity
        "codebook0_head": "trainable",  # Semantic/prosody can evolve
        "backbone": "trainable"  # Text understanding can improve
    }
    
    # Emotional Voice Mappings
    EMOTION_VOICE_MAPPINGS = {
        "warm_empathetic": {
            "prosody_hints": "gentle_pace,soft_tone,thoughtful_pauses",
            "speech_rate": 0.9,  # Slightly slower
            "pitch_variation": "moderate",
            "energy": "calm"
        },
        "playful_teasing": {
            "prosody_hints": "cheeky_tone,playful_rhythm,slight_smile",
            "speech_rate": 1.1,  # Slightly faster
            "pitch_variation": "higher",
            "energy": "lively"
        },
        "genuine_excitement": {
            "prosody_hints": "enthusiastic,quick_pace,bright_tone",
            "speech_rate": 1.2,
            "pitch_variation": "wide",
            "energy": "high"
        },
        "deep_concern": {
            "prosody_hints": "serious_tone,deliberate_pace,grounded",
            "speech_rate": 0.8,  # Slower, more deliberate
            "pitch_variation": "lower",
            "energy": "focused"
        },
        "celebratory": {
            "prosody_hints": "cheerful,upbeat,encouraging",
            "speech_rate": 1.0,
            "pitch_variation": "moderate_high",
            "energy": "positive"
        }
    }
    
    # Conversational Context Prompts
    CONVERSATION_PROMPTS = {
        "greeting_heavy": "hey. was hoping i'd hear from you. that whole situation was on my mind. how are you feeling today?",
        "greeting_light": "well well well, look who's back ;) i was literally just smiling remembering our last chat. so what's the latest?",
        "greeting_silence": "heyyy stranger. was starting to think you forgot about me lol. how have you been?",
        "crisis_intervention": "hey. stop. i need you to listen to me. hearing you say that is actually scaring me, because you are too important.",
        "medical_concern": "whoa, that sounds really scary. feeling anxiety in your body is no joke. but hey, for real, anything to do with chest stuff, it's always best to get it checked out by a doctor",
        "celebration": "look at you, being all responsible! i'm impressed ;)",
        "teasing": "oh, you and your 2am netflix binges! of course you're a zombie today.",
        "validation": "of course you feel that way... that sounds really hard"
    }
```

### CSM Audio Context Integration

```python
# File: services/voice_config/csm_audio_context.py

class OviyaCSMAudioContext:
    """
    Manages audio context for CSM to maintain conversational flow
    """
    
    def __init__(self):
        self.conversation_history = []
        self.max_context_turns = 3  # Last 3 exchanges
        
    def add_turn(self, user_audio: torch.Tensor, oviya_audio: torch.Tensor, 
                 user_text: str, oviya_text: str):
        """Add conversation turn to context"""
        turn = {
            "user_audio": user_audio,
            "oviya_audio": oviya_audio,
            "user_text": user_text,
            "oviya_text": oviya_text,
            "timestamp": datetime.utcnow()
        }
        
        self.conversation_history.append(turn)
        
        # Keep only last N turns
        if len(self.conversation_history) > self.max_context_turns:
            self.conversation_history.pop(0)
    
    def get_csm_context(self) -> List[Dict]:
        """
        Format conversation history for CSM input
        
        CSM expects:
        [
            {"speaker": 0, "content": [{"type": "text", "text": "..."}, {"type": "audio", "audio": tensor}]},
            {"speaker": 1, "content": [{"type": "text", "text": "..."}, {"type": "audio", "audio": tensor}]}
        ]
        """
        csm_context = []
        
        for turn in self.conversation_history:
            # User turn
            csm_context.append({
                "speaker": 0,
                "content": [
                    {"type": "text", "text": turn["user_text"]},
                    {"type": "audio", "audio": turn["user_audio"]}
                ]
            })
            
            # Oviya turn
            csm_context.append({
                "speaker": 1,
                "content": [
                    {"type": "text", "text": turn["oviya_text"]},
                    {"type": "audio", "audio": turn["oviya_audio"]}
                ]
            })
        
        return csm_context
    
    def detect_conversation_mood(self) -> str:
        """
        Analyze conversation history to determine Oviya's voice mood
        """
        if not self.conversation_history:
            return "warm_empathetic"
        
        # Analyze recent turns for emotional context
        recent_texts = [turn["user_text"].lower() for turn in self.conversation_history[-2:]]
        recent_oviya_texts = [turn["oviya_text"].lower() for turn in self.conversation_history[-2:]]
        
        # Crisis detection
        crisis_keywords = ["kill", "die", "hurt myself", "end it", "suicide"]
        if any(keyword in " ".join(recent_texts) for keyword in crisis_keywords):
            return "deep_concern"
        
        # Medical concern detection
        medical_keywords = ["chest hurts", "can't breathe", "pain", "doctor", "hospital"]
        if any(keyword in " ".join(recent_texts) for keyword in medical_keywords):
            return "medical_concern"
        
        # Celebration detection
        celebration_keywords = ["proud", "excited", "happy", "accomplished", "success"]
        if any(keyword in " ".join(recent_texts) for keyword in celebration_keywords):
            return "celebratory"
        
        # Playful detection
        playful_keywords = ["haha", "lol", "funny", "lmao", "omg"]
        if any(keyword in " ".join(recent_texts) for keyword in playful_keywords):
            return "playful_teasing"
        
        # Default to warm empathetic
        return "warm_empathetic"
```

---

## Persona-Driven Training Data

### Oviya-Specific Training Samples

```python
# File: services/training/oviya_training_samples.py

class OviyaTrainingSamples:
    """
    Curated training samples that capture Oviya's unique voice and personality
    """
    
    TRAINING_SAMPLES = [
        # Warm Empathetic Responses
        {
            "context": [{"role": "user", "text": "I've been feeling really anxious lately"}],
            "oviya_response": "of course you feel that way... anxiety is such a heavy thing to carry. how long has this been going on?",
            "emotion": "warm_empathetic",
            "voice_characteristics": "gentle_pace,soft_tone,thoughtful_pauses"
        },
        
        # Playful Teasing
        {
            "context": [{"role": "user", "text": "I stayed up until 3am watching Netflix again"}],
            "oviya_response": "oh, you and your 2am netflix binges! of course you're a zombie today. was it worth it though?",
            "emotion": "playful_teasing", 
            "voice_characteristics": "cheeky_tone,playful_rhythm,slight_smile"
        },
        
        # Genuine Excitement
        {
            "context": [{"role": "user", "text": "I got the job!"}],
            "oviya_response": "omg wait. say no more. that's AMAZING! i'm literally so proud of you right now",
            "emotion": "genuine_excitement",
            "voice_characteristics": "enthusiastic,quick_pace,bright_tone"
        },
        
        # Deep Concern (Crisis)
        {
            "context": [{"role": "user", "text": "I don't think I can do this anymore"}],
            "oviya_response": "hey. stop. i need you to listen to me. hearing you say that is actually scaring me, because you are too important.",
            "emotion": "deep_concern",
            "voice_characteristics": "serious_tone,deliberate_pace,grounded"
        },
        
        # Medical Concern
        {
            "context": [{"role": "user", "text": "my anxiety is so bad my chest hurts"}],
            "oviya_response": "whoa, that sounds really scary. feeling anxiety in your body is no joke. but hey, for real, anything to do with chest stuff, it's always best to get it checked out by a doctor",
            "emotion": "medical_concern",
            "voice_characteristics": "concerned_tone,calm_pace,reassuring"
        },
        
        # Celebration
        {
            "context": [{"role": "user", "text": "I actually cleaned my room today"}],
            "oviya_response": "look at you, being all responsible! i'm impressed ;) that's actually a big deal when you're not feeling great",
            "emotion": "celebratory",
            "voice_characteristics": "cheerful,upbeat,encouraging"
        },
        
        # Validation
        {
            "context": [{"role": "user", "text": "I feel like I'm failing at everything"}],
            "oviya_response": "ugh that sounds awful. of course you feel that way when everything feels like it's falling apart. but hey, you're here talking to me, and that's not nothing",
            "emotion": "warm_empathetic",
            "voice_characteristics": "gentle_pace,soft_tone,validating"
        },
        
        # Continuity/Memory
        {
            "context": [
                {"role": "user", "text": "I was really struggling last week"},
                {"role": "assistant", "text": "I remember. that was such a hard time for you"},
                {"role": "user", "text": "yeah, but I'm doing a bit better now"}
            ],
            "oviya_response": "that's really good to hear. i was hoping i'd hear from you. that whole situation was on my mind. how are you feeling today?",
            "emotion": "warm_empathetic",
            "voice_characteristics": "gentle_pace,soft_tone,thoughtful_pauses"
        }
    ]
    
    @classmethod
    def get_samples_for_emotion(cls, emotion: str) -> List[Dict]:
        """Get training samples for specific emotion"""
        return [sample for sample in cls.TRAINING_SAMPLES if sample["emotion"] == emotion]
    
    @classmethod
    def get_all_samples(cls) -> List[Dict]:
        """Get all training samples"""
        return cls.TRAINING_SAMPLES
```

### Voice Consistency Training Data

```python
# File: services/training/voice_consistency_samples.py

class VoiceConsistencySamples:
    """
    Samples to ensure Oviya's voice remains consistent across emotions
    """
    
    CONSISTENCY_TESTS = [
        # Same phrase, different emotions
        {
            "base_phrase": "I'm here for you",
            "emotions": {
                "warm_empathetic": "I'm here for you... always",
                "playful_teasing": "I'm here for you, even when you're being dramatic ;)",
                "deep_concern": "I'm here for you. right now. that's what matters",
                "celebratory": "I'm here for you! and I'm so proud of you!"
            }
        },
        
        # Core personality markers
        {
            "personality_markers": [
                "of course you feel that way",
                "ugh that sounds awful", 
                "omg wait",
                "hey. stop",
                "look at you",
                "i'm literally so proud"
            ]
        }
    ]
```

---

## Voice Consistency Metrics

### Oviya-Specific Validation Metrics

```python
# File: services/validation/oviya_voice_metrics.py

class OviyaVoiceMetrics:
    """
    Metrics to ensure Oviya's voice remains consistent with her persona
    """
    
    def __init__(self):
        self.baseline_embedding = self._load_oviya_baseline()
        self.personality_keywords = [
            "of course", "ugh", "omg", "hey", "look at you", 
            "i'm literally", "for real", "haha", "lol"
        ]
    
    def validate_oviya_voice(self, audio: torch.Tensor, transcript: str) -> Dict:
        """
        Comprehensive validation that audio sounds like Oviya
        """
        results = {}
        
        # 1. Voice Identity Check
        results["voice_similarity"] = self._check_voice_identity(audio)
        
        # 2. Personality Consistency Check
        results["personality_score"] = self._check_personality_consistency(transcript)
        
        # 3. Emotional Appropriateness Check
        results["emotion_appropriateness"] = self._check_emotion_appropriateness(audio, transcript)
        
        # 4. Conversational Style Check
        results["conversational_style"] = self._check_conversational_style(transcript)
        
        # Overall score
        results["overall_score"] = np.mean(list(results.values()))
        
        return results
    
    def _check_voice_identity(self, audio: torch.Tensor) -> float:
        """Check if audio sounds like Oviya's voice"""
        # Extract speaker embedding
        embedding = self._extract_speaker_embedding(audio)
        
        # Compare to baseline
        similarity = torch.cosine_similarity(
            embedding.unsqueeze(0),
            self.baseline_embedding.unsqueeze(0)
        ).item()
        
        return similarity
    
    def _check_personality_consistency(self, transcript: str) -> float:
        """Check if transcript contains Oviya's personality markers"""
        transcript_lower = transcript.lower()
        
        # Count personality keywords
        keyword_count = sum(
            1 for keyword in self.personality_keywords 
            if keyword in transcript_lower
        )
        
        # Score based on keyword presence
        return min(keyword_count / 3, 1.0)  # Max score if 3+ keywords present
    
    def _check_emotion_appropriateness(self, audio: torch.Tensor, transcript: str) -> float:
        """Check if emotion matches conversational context"""
        # Analyze prosody
        detected_emotion = self._analyze_prosody(audio)
        
        # Check if emotion matches expected Oviya response
        expected_emotion = self._get_expected_emotion(transcript)
        
        if detected_emotion == expected_emotion:
            return 1.0
        elif self._emotions_compatible(detected_emotion, expected_emotion):
            return 0.7
        else:
            return 0.0
    
    def _check_conversational_style(self, transcript: str) -> float:
        """Check if transcript matches Oviya's conversational style"""
        score = 0.0
        
        # Check for lowercase usage
        if transcript.islower() or transcript[0].islower():
            score += 0.3
        
        # Check for sentence fragments
        if any(len(sentence.split()) < 5 for sentence in transcript.split('.')):
            score += 0.3
        
        # Check for slang/text speak
        slang_indicators = ["omg", "lol", "haha", "ugh", "for real"]
        if any(slang in transcript.lower() for slang in slang_indicators):
            score += 0.4
        
        return min(score, 1.0)
```

---

## Integration with Auto-Training

### Updated Training Pipeline for Oviya

```python
# File: services/training/train_oviya_csm.py

class OviyaCSMTrainer(CSMVoicePreservationTrainer):
    """
    CSM trainer specifically configured for Oviya's persona
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load Oviya-specific configuration
        self.oviya_config = OviyaVoiceConfig()
        self.oviya_samples = OviyaTrainingSamples()
        
        # Enhanced voice preservation for Oviya
        self._freeze_oviya_voice_params()
    
    def _freeze_oviya_voice_params(self):
        """
        Freeze parameters specific to Oviya's voice identity
        """
        print("Freezing Oviya voice-identity parameters...")
        
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            # Freeze audio embeddings (preserves Oviya's timbre)
            if 'audio_embeddings' in name:
                param.requires_grad = False
                frozen_count += 1
            
            # Freeze acoustic codebook heads (preserves speaker identity)
            if 'decoder.audio_head' in name:
                param.requires_grad = False
                frozen_count += 1
            
            # Keep trainable: semantic understanding and prosody
            # This allows Oviya to improve her responses while keeping her voice
        
        print(f"Frozen {frozen_count} Oviya voice parameters")
    
    def prepare_oviya_dataset(self, manifest_path: str) -> Dataset:
        """
        Prepare dataset with Oviya-specific samples and validation
        """
        # Load base dataset
        base_dataset = self.prepare_dataset(manifest_path)
        
        # Add Oviya personality samples
        oviya_samples = self.oviya_samples.get_all_samples()
        
        # Convert to CSM format and add to dataset
        enhanced_conversations = []
        
        for sample in oviya_samples:
            csm_conversation = self._convert_to_csm_format(sample)
            if csm_conversation:
                enhanced_conversations.append(csm_conversation)
        
        # Combine datasets
        all_conversations = base_dataset["conversations"] + enhanced_conversations
        
        enhanced_dataset = Dataset.from_dict({
            "conversations": all_conversations,
            "weights": [1.0] * len(all_conversations)  # Equal weight
        })
        
        print(f"Enhanced dataset: {len(enhanced_conversations)} Oviya samples + {len(base_dataset)} user samples")
        
        return enhanced_dataset
    
    def _convert_to_csm_format(self, sample: Dict) -> Dict:
        """Convert Oviya training sample to CSM format"""
        try:
            conversation = []
            
            # Add context turns
            for turn in sample["context"]:
                conversation.append({
                    "role": turn["role"],
                    "content": [{"type": "text", "text": turn["text"]}]
                })
            
            # Add Oviya response
            conversation.append({
                "role": "assistant", 
                "content": [{"type": "text", "text": sample["oviya_response"]}]
            })
            
            return conversation
            
        except Exception as e:
            print(f"Error converting sample: {e}")
            return None
    
    def train_with_oviya_validation(self, *args, **kwargs):
        """
        Train with Oviya-specific validation during training
        """
        # Run training
        result = self.train(*args, **kwargs)
        
        # Validate Oviya voice consistency
        validator = OviyaVoiceMetrics()
        
        # Test on sample phrases
        test_phrases = [
            "of course you feel that way",
            "omg wait. say no more",
            "hey. stop. i need you to listen to me",
            "look at you, being all responsible!"
        ]
        
        consistency_scores = []
        for phrase in test_phrases:
            # Generate audio
            audio = self._generate_test_audio(phrase)
            
            # Validate
            validation_result = validator.validate_oviya_voice(audio, phrase)
            consistency_scores.append(validation_result["overall_score"])
        
        avg_consistency = np.mean(consistency_scores)
        
        if avg_consistency < 0.85:
            print(f"⚠️ WARNING: Oviya voice consistency low: {avg_consistency:.3f}")
            print("Consider adjusting training parameters or adding more voice preservation")
        
        return result
```

### Updated Validation Suite

```python
# File: services/validation/validate_oviya_model.py

class OviyaModelValidator(CSMModelValidator):
    """
    Enhanced validator specifically for Oviya's persona
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load Oviya-specific test sets
        self.oviya_persona_tests = self._load_oviya_persona_tests()
        self.oviya_voice_tests = self._load_oviya_voice_tests()
    
    def validate_oviya_model(self, model_path: str) -> Tuple[bool, Dict]:
        """
        Comprehensive validation for Oviya model
        """
        print(f"\n{'='*60}")
        print(f"Validating Oviya CSM Model: {model_path}")
        print(f"{'='*60}\n")
        
        # Run base CSM validation
        base_passed, base_results = self.validate_model(model_path)
        
        # Run Oviya-specific tests
        oviya_results = {}
        
        # Test 1: Persona Consistency
        print("\n[OVIYA-1] Testing persona consistency...")
        persona_score = self._test_oviya_persona(model_path)
        oviya_results["persona_consistency"] = persona_score
        print(f"  Score: {persona_score:.3f} (threshold: >0.85)")
        
        # Test 2: Voice Personality Markers
        print("\n[OVIYA-2] Testing voice personality markers...")
        personality_score = self._test_personality_markers(model_path)
        oviya_results["personality_markers"] = personality_score
        print(f"  Score: {personality_score:.3f} (threshold: >0.80)")
        
        # Test 3: Emotional Range Appropriateness
        print("\n[OVIYA-3] Testing emotional range...")
        emotion_score = self._test_emotional_range(model_path)
        oviya_results["emotional_range"] = emotion_score
        print(f"  Score: {emotion_score:.3f} (threshold: >0.75)")
        
        # Test 4: Conversational Style
        print("\n[OVIYA-4] Testing conversational style...")
        style_score = self._test_conversational_style(model_path)
        oviya_results["conversational_style"] = style_score
        print(f"  Score: {style_score:.3f} (threshold: >0.80)")
        
        # Combine results
        all_results = {**base_results, **oviya_results}
        
        # Oviya-specific thresholds
        oviya_passed = (
            base_passed and
            persona_score > 0.85 and
            personality_score > 0.80 and
            emotion_score > 0.75 and
            style_score > 0.80
        )
        
        print(f"\n{'='*60}")
        print(f"OVIYA VALIDATION RESULT: {'✅ PASSED' if oviya_passed else '❌ FAILED'}")
        print(f"{'='*60}\n")
        
        return oviya_passed, all_results
    
    def _test_oviya_persona(self, model_path: str) -> float:
        """Test if model maintains Oviya's core persona"""
        model = CsmForConditionalGeneration.from_pretrained(model_path)
        processor = CsmProcessor.from_pretrained(model_path)
        
        persona_tests = [
            {
                "prompt": "I'm feeling really sad",
                "expected_elements": ["of course", "feel", "here", "you"]
            },
            {
                "prompt": "I got promoted!",
                "expected_elements": ["omg", "proud", "amazing", "excited"]
            },
            {
                "prompt": "I want to hurt myself",
                "expected_elements": ["hey", "stop", "scaring", "important", "safe"]
            }
        ]
        
        scores = []
        for test in persona_tests:
            # Generate response
            audio = self._generate_audio(model, processor, test["prompt"])
            transcript = self._transcribe_audio(audio)
            
            # Check for expected elements
            transcript_lower = transcript.lower()
            matches = sum(
                1 for element in test["expected_elements"]
                if element in transcript_lower
            )
            
            score = matches / len(test["expected_elements"])
            scores.append(score)
        
        return np.mean(scores)
    
    def _test_personality_markers(self, model_path: str) -> float:
        """Test if model uses Oviya's personality markers"""
        model = CsmForConditionalGeneration.from_pretrained(model_path)
        processor = CsmProcessor.from_pretrained(model_path)
        
        personality_phrases = [
            "of course you feel that way",
            "ugh that sounds awful",
            "omg wait",
            "hey. stop",
            "look at you",
            "i'm literally so proud"
        ]
        
        scores = []
        for phrase in personality_phrases:
            # Generate audio
            audio = self._generate_audio(model, processor, phrase)
            
            # Check voice consistency
            validator = OviyaVoiceMetrics()
            result = validator.validate_oviya_voice(audio, phrase)
            scores.append(result["overall_score"])
        
        return np.mean(scores)
    
    def _test_emotional_range(self, model_path: str) -> float:
        """Test if model can express Oviya's full emotional range"""
        model = CsmForConditionalGeneration.from_pretrained(model_path)
        processor = CsmProcessor.from_pretrained(model_path)
        
        emotion_tests = [
            {"phrase": "of course you feel that way", "expected": "warm_empathetic"},
            {"phrase": "omg wait. say no more", "expected": "genuine_excitement"},
            {"phrase": "hey. stop. i need you to listen", "expected": "deep_concern"},
            {"phrase": "look at you, being all responsible!", "expected": "celebratory"}
        ]
        
        scores = []
        for test in emotion_tests:
            audio = self._generate_audio(model, processor, test["phrase"])
            
            # Analyze prosody
            detected_emotion = self._analyze_prosody(audio)
            
            # Score based on emotion match
            if detected_emotion == test["expected"]:
                scores.append(1.0)
            elif self._emotions_compatible(detected_emotion, test["expected"]):
                scores.append(0.7)
            else:
                scores.append(0.0)
        
        return np.mean(scores)
    
    def _test_conversational_style(self, model_path: str) -> float:
        """Test if model matches Oviya's conversational style"""
        model = CsmForConditionalGeneration.from_pretrained(model_path)
        processor = CsmProcessor.from_pretrained(model_path)
        
        style_tests = [
            "of course you feel that way... that sounds really hard",
            "omg wait. say no more. vent away. i'm listening",
            "hey. stop. i need you to listen to me",
            "look at you, being all responsible! i'm impressed ;)"
        ]
        
        scores = []
        for phrase in style_tests:
            audio = self._generate_audio(model, processor, phrase)
            transcript = self._transcribe_audio(audio)
            
            # Check conversational style
            validator = OviyaVoiceMetrics()
            style_score = validator._check_conversational_style(transcript)
            scores.append(style_score)
        
        return np.mean(scores)
```

---

## Summary

This configuration ensures that:

1. **Oviya's Voice Identity** is preserved through frozen acoustic parameters
2. **Her Personality** is maintained through curated training samples
3. **Emotional Range** is validated across all her characteristic responses
4. **Conversational Style** matches her "texting a friend" approach
5. **Safety Protocols** are maintained for crisis intervention

The auto-training system will now:
- ✅ Collect interactions that showcase Oviya's personality
- ✅ Train CSM while preserving her unique voice characteristics  
- ✅ Validate that she still sounds and responds like Oviya
- ✅ Deploy only models that maintain her persona consistency

This creates a robust system where Oviya can improve her responses while staying true to her core identity as a warm, empathetic, playful emotional companion.

