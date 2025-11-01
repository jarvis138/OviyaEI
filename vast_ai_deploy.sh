#!/bin/bash
# 🚀 Oviya EI Vast AI Deployment Script
# Automated deployment for RTX 5880 Ada instances

set -e

echo "🚀 Starting Oviya EI Vast AI Deployment"
echo "======================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if we're on Vast AI
check_vast_ai() {
    print_status "Checking Vast AI environment..."
    if ! nvidia-smi &>/dev/null; then
        print_error "NVIDIA GPU not detected. Are you on a Vast AI GPU instance?"
        exit 1
    fi
    
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU detected: $GPU_INFO"
}

# System setup
setup_system() {
    print_status "Setting up system dependencies..."
    
    # Update system
    sudo apt update && sudo apt upgrade -y
    
    # Install dependencies
    sudo apt install -y python3.11 python3.11-venv python3.11-dev ffmpeg build-essential curl wget git htop nvtop
    
    # Install Ollama
    if ! command -v ollama &>/dev/null; then
        print_status "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
    print_success "System setup complete"
}

# Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    python3.11 -m venv oviya_env
    source oviya_env/bin/activate
    
    # Install requirements
    pip install --upgrade pip
    pip install -r production/requirements.txt
    
    print_success "Python environment ready"
}

# Download models
download_models() {
    print_status "Downloading AI models..."
    
    # Start Ollama and download model
    /usr/local/bin/ollama serve &
    sleep 5
    ollama pull qwen2.5:7b
    
    # Download voice models (if scripts exist)
    cd production
    [ -f "download_emotion_datasets.sh" ] && ./download_emotion_datasets.sh
    [ -f "download_openvoice_models.sh" ] && ./download_openvoice_models.sh
    cd ..
    
    print_success "Model downloads complete"
}

# Configure environment
configure_env() {
    print_status "Configuring environment..."
    
    cat > .env << EOF
OLLAMA_URL=http://localhost:11434/api/generate
CSM_URL=http://localhost:19517/generate
CSM_STREAM_URL=http://localhost:19517/generate/stream
WHISPERX_URL=http://localhost:1111
CUDA_VISIBLE_DEVICES=0
PYTHONUNBUFFERED=1
EOF
    
    print_success "Environment configured"
}

# Start services
start_services() {
    print_status "Starting Oviya services..."
    
    # Start Ollama (background)
    /usr/local/bin/ollama serve &
    sleep 10
    
    # Start CSM voice server (background)
    source oviya_env/bin/activate
    python -c "
from production.voice.csm_1b_generator_optimized import CSMGenerator
generator = CSMGenerator()
generator.start_server(port=19517)
" &
    
    # Wait for services to start
    sleep 15
    
    # Start main Oviya server
    python production/websocket_server.py &
    
    print_success "Oviya services started"
}

# Health checks
health_check() {
    print_status "Running health checks..."
    
    # Check Ollama
    if curl -f http://localhost:11434/api/tags &>/dev/null; then
        print_success "Ollama: OK"
    else
        print_warning "Ollama: Not responding"
    fi
    
    # Check CSM
    if curl -f http://localhost:19517/health &>/dev/null; then
        print_success "CSM Voice: OK"
    else
        print_warning "CSM Voice: Not responding"
    fi
    
    # Check Oviya
    if curl -f http://localhost:8000/health &>/dev/null; then
        print_success "Oviya Backend: OK"
    else
        print_warning "Oviya Backend: Not responding"
    fi
}

# Main deployment
main() {
    check_vast_ai
    setup_system
    setup_python
    download_models
    configure_env
    start_services
    health_check
    
    echo ""
    print_success "🎉 Oviya EI deployment complete!"
    echo ""
    echo "🌐 Access your Oviya instance:"
    echo "   WebSocket: ws://localhost:8000/ws/conversation"
    echo "   Health: http://localhost:8000/health"
    echo ""
    echo "📊 Monitor with:"
    echo "   nvidia-smi    # GPU usage"
    echo "   htop         # System resources"
    echo "   nvtop        # GPU processes"
}

# Run main function
main "$@"
