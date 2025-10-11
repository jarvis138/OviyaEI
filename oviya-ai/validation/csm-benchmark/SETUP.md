# CSM Model Setup Guide - Day 2

## 🎯 Goal
Get CSM model running and test basic generation

---

## Option 1: Quick Test with RunPod (Recommended) ⭐

### Why RunPod?
- No local GPU needed
- Professional setup
- Pay only when using (~$0.34/hour)
- Perfect for validation

### Steps:

#### 1. Create RunPod Account
1. Go to: https://www.runpod.io/
2. Sign up with email
3. Add payment method (credit card)
4. Get $10 free credit (usually)

#### 2. Deploy a Pod
1. Go to: https://www.runpod.io/console/pods
2. Click "Deploy" → "GPU Pods"
3. Choose template: **PyTorch 2.0**
4. Select GPU: **RTX A4000** (16GB, $0.34/hr) or **A40** (48GB, $0.76/hr)
5. Disk: 50GB
6. Expose HTTP Ports: 8000
7. Click "Deploy On-Demand"

#### 3. Connect to Pod
```bash
# Copy SSH command from RunPod dashboard
# It looks like:
ssh root@your-pod-id.runpod.io -p 22

# Or use Jupyter Lab (easier):
# Click "Connect" → "Start Jupyter Lab"
```

#### 4. Install CSM on Pod
```bash
# In Pod terminal:
cd /workspace

# Clone CSM
git clone https://github.com/SesameAILabs/csm.git
cd csm

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login
# Paste your token: hf_...

# Test CSM
python run_csm.py
```

---

## Option 2: Local GPU Setup

### Requirements:
- NVIDIA GPU with 12GB+ VRAM
- CUDA 12.0+
- 50GB free disk space

### Steps:

#### 1. Check GPU
```bash
nvidia-smi
```

#### 2. Install CSM
```bash
cd ~/Documents/Oviya\ EI/oviya-ai/validation/csm-benchmark

# Clone CSM
git clone https://github.com/SesameAILabs/csm.git
cd csm

# Create virtual environment
python3 -m venv csm-venv
source csm-venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login
```

---

## 🔑 Get Hugging Face Token

### Step 1: Create Account
- Go to: https://huggingface.co/join
- Sign up and verify email

### Step 2: Request CSM Access
- Go to: https://huggingface.co/sesame-ai/csm-1b
- Click "Request Access"
- Usually approved instantly

### Step 3: Get Token
- Go to: https://huggingface.co/settings/tokens
- Click "New token"
- Name: "oviya-csm"
- Type: Read
- Copy token (starts with `hf_...`)

### Step 4: Save Token
```bash
# Add to .env file
echo "HUGGINGFACE_TOKEN=hf_your_token_here" >> ../../.env
```

---

## ✅ Verify Setup

After installation, test CSM:

```bash
python run_csm.py
```

Expected output:
```
Loading CSM model...
Model loaded successfully!
Generating audio...
Audio saved to: audio.wav
```

---

## 🐛 Troubleshooting

### Error: "No module named 'generator'"
**Solution**: Make sure you're in the csm directory
```bash
cd csm
python run_csm.py
```

### Error: "CUDA out of memory"
**Solution**: Your GPU doesn't have enough VRAM. Use RunPod with A40 (48GB)

### Error: "Access denied to model"
**Solution**: Request access at https://huggingface.co/sesame-ai/csm-1b

### Error: "huggingface-cli: command not found"
**Solution**: Install transformers
```bash
pip install transformers
```

---

## 📊 Next Steps

After CSM is working, run our benchmark tests:

```bash
cd /Users/jarvis/Documents/Oviya\ EI/oviya-ai/validation/csm-benchmark
python basic_test.py
```

This will measure:
- Cold start latency
- Warm inference latency
- Audio quality

---

## 💰 Cost Estimate

### RunPod Costs:
- RTX A4000: $0.34/hour
- A40: $0.76/hour
- Day 2 testing: ~2 hours = $0.68-1.52

### Total Sprint 0 Budget: $50
### Day 2 Target: <$2

---

## 🎯 Success Criteria

- [ ] CSM model loads successfully
- [ ] Generates audio file
- [ ] Audio quality is acceptable
- [ ] Warm inference <2 seconds

Once these pass, we move to audio context testing!

