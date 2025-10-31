# Emotion Reference Directory

## Purpose
This directory is reserved for emotion reference data and training corpora to support advanced emotion detection and synthesis models in Oviya EI.

## Intended Contents

### **emotion_taxonomy.json**
- Comprehensive emotion classification system
- Hierarchical emotion categories (basic → complex emotions)
- Cross-cultural emotion mappings
- Intensity scales and thresholds

### **emotion_intensity_scales.json**
- Standardized intensity measurements (0.0-1.0)
- Emotion-specific intensity mappings
- Cultural variations in intensity perception
- Clinical thresholds for therapeutic interventions

### **cultural_emotion_mappings/**
- **japanese_emotions.json**: Wa (harmony), Mono no aware (sensitivity to ephemera)
- **korean_emotions.json**: Jeong (deep emotional connection), Han (enduring sorrow)
- **indian_emotions.json**: Ahimsa (non-violence), Dharma (duty/righteousness)
- **greek_emotions.json**: Philotimo (love of honor), Kefi (joyful exuberance)
- **scandinavian_emotions.json**: Lagom (just the right amount), Fika (cozy social time)

### **emotion_transition_graphs/**
- Valid emotion transition patterns
- Therapeutic transition pathways
- Cultural transition preferences
- Risk assessment for rapid transitions

### **reference_audio_samples/**
- Gold-standard emotion expression recordings
- Multi-cultural voice samples
- Clinical validation recordings
- Intensity variation samples

## Status
- **Currently empty** (intentional placeholder)
- **Implementation planned** for v2.0 advanced emotion modeling
- **Will integrate** with emotion detection pipeline when ready

## Technical Requirements
- Audio samples: 24kHz, 16-bit PCM, mono
- JSON schemas: Versioned with backward compatibility
- Cultural data: Ethnographically validated
- Clinical data: IRB-approved and de-identified

## Integration Points
- `production/voice/emotion_library.py`: Load emotion taxonomy
- `production/brain/emotional_reciprocity.py`: Use cultural mappings
- `production/emotion_detector/`: Reference validation data
- `production/voice/emotion_blender.py`: Use intensity scales

## Contact
For questions about emotion reference implementation, contact the ML team.

## Future Roadmap
- Q1 2025: Basic emotion taxonomy implementation
- Q2 2025: Cultural emotion mapping integration
- Q3 2025: Audio reference corpus development
- Q4 2025: Clinical validation and deployment
