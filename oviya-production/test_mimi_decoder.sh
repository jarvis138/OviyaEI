#!/bin/bash

python3 << 'EOFMIMI'
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, MimiModel
import torchaudio

print("🔍 Testing CSM-1B + Mimi decoder...")

# Load models
processor = AutoProcessor.from_pretrained(
    "sesame/csm-1b",
    cache_dir="/workspace/.cache/huggingface"
)

model = CsmForConditionalGeneration.from_pretrained(
    "sesame/csm-1b",
    cache_dir="/workspace/.cache/huggingface",
    device_map="auto",
    torch_dtype=torch.float16
)

mimi = MimiModel.from_pretrained(
    "kyutai/mimi",
    cache_dir="/workspace/.cache/huggingface"
).to("cuda").eval()

print("✅ Models loaded!")

# Generate speech
text = "[joyful] Hello, how are you today?"
print(f"\n🎵 Generating: '{text}'")

inputs = processor(text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    # Generate RVQ codes
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(f"   Codes shape: {outputs.shape}")
    
    # Decode with Mimi
    codes = outputs.transpose(1, 2)
    decoder_output = mimi.decode(codes)
    
    # Extract audio tensor
    if hasattr(decoder_output, 'audio_values'):
        audio = decoder_output.audio_values
    elif hasattr(decoder_output, 'audio'):
        audio = decoder_output.audio
    elif isinstance(decoder_output, tuple):
        audio = decoder_output[0]
    else:
        audio = decoder_output
    
    print(f"✅ Audio shape: {audio.shape}")
    print(f"   Duration: {audio.shape[-1] / 24000:.2f}s")
    
    # Save
    torchaudio.save('/tmp/test_real_csm.wav', audio.cpu(), 24000)
    print(f"   Saved: /tmp/test_real_csm.wav")

print("\n🎉 SUCCESS! Real CSM-1B + Mimi working!")
EOFMIMI
