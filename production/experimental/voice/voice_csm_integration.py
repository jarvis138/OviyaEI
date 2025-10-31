# Simple CSM-1B integration for Oviya
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, MimiModel
import numpy as np
import os

# Set HuggingFace cache directory
os.environ['HF_HOME'] = '/root/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'

# Use HuggingFace model IDs instead of local paths
MODEL_ID = "sesame/csm-1b"
MIMI_MODEL_ID = "kyutai/mimi"

_csm_model = None
_processor = None
_mimi_decoder = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_csm_models():
    global _csm_model, _processor, _mimi_decoder
    if _csm_model is None:
        print("Loading CSM-1B models for utility...")
        try:
            _processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            _csm_model = CsmForConditionalGeneration.from_pretrained(MODEL_ID, local_files_only=True, device_map=_device, torch_dtype=torch.float16)
            _csm_model.eval()
            _mimi_decoder = MimiModel.from_pretrained(MIMI_MODEL_ID, local_files_only=True).to(_device).eval()
            print("CSM-1B models loaded for utility.")
        except Exception as e:
            print(f"Failed to load models locally: {e}")
            print("Models may need to be downloaded first...")
            raise

def generate_csm_audio(text: str, emotion: str = "neutral") -> tuple:
    global _csm_model, _processor, _mimi_decoder
    if _csm_model is None:
        load_csm_models()

    prompt = f"[{emotion}] {text}"
    inputs = _processor(prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        audio = _csm_model.generate(**inputs, output_audio=True)
    
    # Handle potential list of tensors or single tensor
    if isinstance(audio, list):
        audio_np = np.concatenate([a.cpu().numpy() for a in audio])
    else:
        audio_np = audio.cpu().numpy()
        
    return audio_np, 24000

if __name__ == "__main__":
    load_csm_models()
    print("CSM-1B integration ready")
    test_text = "Hello from Oviya. How are you feeling today?"
    audio_data, sr = generate_csm_audio(test_text, "joyful")
    print(f"✅ SUCCESS: {len(audio_data)} samples at {sr}Hz")
