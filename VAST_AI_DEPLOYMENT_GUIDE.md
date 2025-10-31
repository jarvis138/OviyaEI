# 🚀 OVIYA EI VAST AI DEPLOYMENT GUIDE

## Prerequisites
1. Vast AI account with sufficient credits
2. RTX 5880 Ada instance (recommended) or similar GPU instance
3. SSH access to Vast AI instance

## Step 1: Provision Vast AI Instance
- GPU: RTX 5880 Ada (48GB VRAM) or RTX 4090 (24GB VRAM)
- RAM: 64GB minimum
- Storage: 100GB+ SSD
- OS: Ubuntu 22.04

## Step 2: Clone Repository
```bash
git clone https://github.com/jarvis138/OviyaEI.git
cd OviyaEI
```

## Step 3: Install Dependencies
```bash
# System packages
sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg build-essential

# Python environment
python3.11 -m venv oviya_env
source oviya_env/bin/activate
pip install -r production/requirements.txt
```

## Step 4: Download Models
```bash
# Ollama LLM
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull qwen2.5:7b

# Voice models
cd production
./download_emotion_datasets.sh
./download_openvoice_models.sh
```

## Step 5: Configure Environment
```bash
# Create .env file
cat > .env << EOF
OLLAMA_URL=http://localhost:11434/api/generate
CSM_URL=http://localhost:19517/generate
WHISPERX_URL=http://localhost:1111
CUDA_VISIBLE_DEVICES=0
EOF
```

## Step 6: Start Services
```bash
# Start Ollama (background)
/usr/local/bin/ollama serve &

# Start CSM Voice Server
python -c "
from voice.csm_1b_generator_optimized import CSMGenerator
generator = CSMGenerator()
generator.start_server(port=19517)
" &

# Start WhisperX (if needed)
# python -c "from voice.whisper_client import WhisperXClient; client = WhisperXClient(); client.start_server(port=1111)" &

# Start main Oviya server
python websocket_server.py
```

## Step 7: Setup Cloudflare Tunnel (Optional)
For external access, install cloudflared:
```bash
# Install cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# Create tunnel
cloudflared tunnel login
cloudflared tunnel create oviya-tunnel
cloudflared tunnel route dns oviya-tunnel yourdomain.com
```

## Monitoring
- Check GPU usage: `nvidia-smi`
- Monitor processes: `htop`
- View logs: `tail -f /path/to/logs/*.log`

## Cost Optimization
- RTX 5880 Ada: ~.50/hour
- Use spot instances when available
- Auto-shutdown when not in use

