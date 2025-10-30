#!/bin/bash
# Oviya MCP Ecosystem Deployment Script
# This script deploys the complete MCP infrastructure for Oviya EI

set -e

echo "🚀 Starting Oviya MCP Ecosystem Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCP_ROOT="$PROJECT_ROOT/mcp-ecosystem"
ENV_FILE="$MCP_ROOT/.env"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js first."
        exit 1
    fi

    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Create environment file
create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        print_status "Creating environment configuration file..."

        cat > "$ENV_FILE" << EOF
# Oviya MCP Ecosystem Environment Configuration
# Copy this file and customize the values for your environment

# Database Configuration
POSTGRES_DB=oviya_db
POSTGRES_USER=oviya
POSTGRES_PASSWORD=oviya_password
DATABASE_URL=postgresql://oviya:oviya_password@localhost:5432/oviya_db

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Vector Database Configuration
QDRANT_URL=http://localhost:6333

# MCP Server URLs (internal)
OPENMEMORY_URL=http://localhost:3001
OVIYA_PERSONALITY_URL=http://localhost:3002
OVIYA_EMOTION_PROSODY_URL=http://localhost:3003
OVIYA_SITUATIONAL_EMPATHY_URL=http://localhost:3004
WHATSAPP_MCP_URL=http://localhost:3005
STRIPE_MCP_URL=http://localhost:3006
MONITORING_URL=http://localhost:3007

# External API Keys (required for full functionality)
# Get these from the respective service providers

# WhatsApp Business API (for WhatsApp integration)
WHATSAPP_API_KEY=your_whatsapp_api_key_here
WHATSAPP_WEBHOOK_SECRET=your_whatsapp_webhook_secret_here

# Stripe (for monetization)
STRIPE_SECRET_KEY=your_stripe_secret_key_here
STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key_here

# Optional: Hugging Face (for additional AI capabilities)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# OpenAI (for enhanced AI features)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude (for MCP integrations)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
EOF

        print_warning "Environment file created at $ENV_FILE"
        print_warning "Please edit this file with your actual API keys and configuration"
        print_warning "Press Enter to continue with default/demo configuration, or Ctrl+C to configure first"
        read -r || true
    else
        print_success "Environment file already exists"
    fi
}

# Install Tier 1 MCP servers
install_tier1_servers() {
    print_status "Installing Tier 1 MCP servers..."

    # AI Therapist MCP
    if [ ! -d "node_modules/@danieldunderfelt" ]; then
        print_status "Installing AI Therapist MCP..."
        npm install @danieldunderfelt/ai-therapist-mcp
        print_success "AI Therapist MCP installed"
    else
        print_success "AI Therapist MCP already installed"
    fi

    # MCP Thinking server
    if [ ! -d "servers/tier1/mcp-thinking" ]; then
        print_error "MCP Thinking server not found. Please ensure it's cloned."
        exit 1
    else
        print_success "MCP Thinking server found"
    fi

    print_success "Tier 1 servers ready"
}

# Build custom Oviya MCP servers
build_custom_servers() {
    print_status "Building custom Oviya MCP servers..."

    # Create directories if they don't exist
    mkdir -p servers/custom-oviya/personality
    mkdir -p servers/custom-oviya/emotion-prosody
    mkdir -p servers/custom-oviya/situational-empathy

    # Copy server implementations (assuming they exist)
    if [ -f "servers/custom-oviya/personality/server.py" ]; then
        print_success "Personality server found"
    else
        print_warning "Personality server not found - will be created during build"
    fi

    if [ -f "servers/custom-oviya/emotion-prosody/server.py" ]; then
        print_success "Emotion prosody server found"
    else
        print_warning "Emotion prosody server not found - will be created during build"
    fi

    if [ -f "servers/custom-oviya/situational-empathy/server.py" ]; then
        print_success "Situational empathy server found"
    else
        print_warning "Situational empathy server not found - will be created during build"
    fi

    print_success "Custom servers prepared"
}

# Start Docker services
start_docker_services() {
    print_status "Starting Docker services..."

    cd "$MCP_ROOT"

    # Check if docker-compose or docker compose is available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    # Start core infrastructure first
    print_status "Starting core infrastructure (PostgreSQL, Redis, Qdrant)..."
    $COMPOSE_CMD up -d postgres redis qdrant

    # Wait for services to be healthy
    print_status "Waiting for core services to be healthy..."
    sleep 30

    # Check service health
    if $COMPOSE_CMD ps postgres | grep -q "Up"; then
        print_success "PostgreSQL is running"
    else
        print_error "PostgreSQL failed to start"
        exit 1
    fi

    if $COMPOSE_CMD ps redis | grep -q "Up"; then
        print_success "Redis is running"
    else
        print_error "Redis failed to start"
        exit 1
    fi

    if $COMPOSE_CMD ps qdrant | grep -q "Up"; then
        print_success "Qdrant is running"
    else
        print_error "Qdrant failed to start"
        exit 1
    fi

    # Start MCP servers
    print_status "Starting MCP servers..."
    $COMPOSE_CMD up -d openmemory oviya-personality oviya-emotion-prosody oviya-situational-empathy

    # Wait for MCP servers
    print_status "Waiting for MCP servers to be healthy..."
    sleep 20

    # Start optional services if API keys are configured
    if grep -q "your_whatsapp_api_key_here" "$ENV_FILE"; then
        print_warning "WhatsApp API key not configured - skipping WhatsApp MCP"
    else
        print_status "Starting WhatsApp MCP..."
        $COMPOSE_CMD up -d whatsapp-mcp
    fi

    if grep -q "your_stripe_secret_key_here" "$ENV_FILE"; then
        print_warning "Stripe API key not configured - skipping Stripe MCP"
    else
        print_status "Starting Stripe MCP..."
        $COMPOSE_CMD up -d stripe-mcp
    fi

    # Start monitoring
    print_status "Starting monitoring service..."
    $COMPOSE_CMD up -d monitoring

    print_success "All services started successfully"
}

# Run tests
run_tests() {
    print_status "Running MCP integration tests..."

    cd "$MCP_ROOT"

    # Install test dependencies
    pip install pytest pytest-asyncio

    # Run tests
    if python -m pytest tests/test_mcp_integrations.py -v; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed. Please check the output above."
        exit 1
    fi
}

# Update Oviya configuration
update_oviya_config() {
    print_status "Updating Oviya MCP configuration..."

    # Update the main MCP config file
    MCP_CONFIG_FILE="$HOME/.cursor/mcp.json"

    # Create backup
    if [ -f "$MCP_CONFIG_FILE" ]; then
        cp "$MCP_CONFIG_FILE" "${MCP_CONFIG_FILE}.backup"
        print_status "Created backup of existing MCP configuration"
    fi

    # Update MCP configuration with all services
    cat > "$MCP_CONFIG_FILE" << EOF
{
  "mcpServers": {
    "openmemory": {
      "command": "docker",
      "args": ["exec", "openmemory", "python", "mcp_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://oviya:oviya_password@localhost:5432/openmemory",
        "QDRANT_URL": "http://localhost:6333"
      }
    },
    "ai-therapist": {
      "command": "npx",
      "args": ["@danieldunderfelt/ai-therapist-mcp"],
      "env": {}
    },
    "mcp-thinking": {
      "command": "node",
      "args": ["$MCP_ROOT/servers/tier1/mcp-thinking/dist/index.js"],
      "env": {}
    },
    "chroma": {
      "command": "python",
      "args": ["-c", "from chroma_mcp import server; server.run()"],
      "env": {
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000",
        "CHROMA_PERSIST_DIR": "$MCP_ROOT/data/chroma"
      }
    },
    "oviya-personality": {
      "command": "python",
      "args": ["$MCP_ROOT/servers/custom-oviya/personality/server.py"],
      "env": {
        "MODEL_PATH": "/models",
        "CONFIG_PATH": "$PROJECT_ROOT/production/config"
      }
    },
    "oviya-emotion-prosody": {
      "command": "python",
      "args": ["$MCP_ROOT/servers/custom-oviya/emotion-prosody/server.py"],
      "env": {
        "VOICE_MODEL_PATH": "$PROJECT_ROOT/production/voice"
      }
    },
    "oviya-situational-empathy": {
      "command": "python",
      "args": ["$MCP_ROOT/servers/custom-oviya/situational-empathy/server.py"],
      "env": {
        "SAFETY_CONFIG": "$PROJECT_ROOT/production/brain/safety_router.py"
      }
    }
  }
}
EOF

    print_success "Oviya MCP configuration updated"
}

# Display status
show_status() {
    print_status "Checking deployment status..."

    echo ""
    echo "🔍 MCP Ecosystem Status:"
    echo "========================"

    # Check Docker services
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    cd "$MCP_ROOT"
    $COMPOSE_CMD ps

    echo ""
    echo "🌐 Service Endpoints:"
    echo "===================="
    echo "PostgreSQL:     localhost:5432"
    echo "Redis:          localhost:6379"
    echo "Qdrant:         localhost:6333"
    echo "OpenMemory:     localhost:3001"
    echo "Personality:    localhost:3002"
    echo "Emotion:        localhost:3003"
    echo "Empathy:        localhost:3004"
    echo "WhatsApp:       localhost:3005 (if configured)"
    echo "Stripe:         localhost:3006 (if configured)"
    echo "Monitoring:     localhost:3007"

    echo ""
    echo "📊 Next Steps:"
    echo "=============="
    echo "1. Start Oviya with: cd production && python websocket_server.py"
    echo "2. Test MCP integrations: python mcp-ecosystem/tests/test_mcp_integrations.py"
    echo "3. Monitor logs: docker-compose logs -f"
    echo "4. Configure API keys in .env file for full functionality"
}

# Main deployment function
main() {
    echo "🤖 Oviya EI - MCP Ecosystem Deployment"
    echo "======================================"
    echo ""

    check_prerequisites
    create_env_file
    install_tier1_servers
    build_custom_servers
    start_docker_services
    run_tests
    update_oviya_config
    show_status

    echo ""
    print_success "🎉 Oviya MCP Ecosystem deployment completed successfully!"
    print_success "Your emotional AI companion now has enterprise-grade memory, safety, and intelligence capabilities."
    echo ""
    print_status "To start Oviya: cd production && python websocket_server.py"
}

# Handle command line arguments
case "${1:-}" in
    "status")
        show_status
        ;;
    "test")
        run_tests
        ;;
    "stop")
        cd "$MCP_ROOT"
        if command -v docker-compose &> /dev/null; then
            docker-compose down
        else
            docker compose down
        fi
        print_success "MCP ecosystem stopped"
        ;;
    "restart")
        cd "$MCP_ROOT"
        if command -v docker-compose &> /dev/null; then
            docker-compose restart
        else
            docker compose restart
        fi
        print_success "MCP ecosystem restarted"
        ;;
    *)
        main
        ;;
esac
