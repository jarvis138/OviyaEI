#!/bin/bash

# Oviya AI Production Deployment Script
# This script deploys the complete Oviya AI system to production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="oviya-ai"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if NVIDIA Docker is available (for GPU support)
    if ! docker info | grep -q nvidia; then
        log_warning "NVIDIA Docker runtime not detected. GPU acceleration may not be available."
    fi
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file $ENV_FILE not found. Please create it first."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build CSM service
    log_info "Building CSM service..."
    docker build -f infrastructure/docker/Dockerfile.csm -t oviya-csm:latest services/csm-streaming/
    
    # Build ASR service
    log_info "Building ASR service..."
    docker build -f infrastructure/docker/Dockerfile.asr -t oviya-asr:latest services/asr-realtime/
    
    # Build Orchestrator service
    log_info "Building Orchestrator service..."
    docker build -f infrastructure/docker/Dockerfile.orchestrator -t oviya-orchestrator:latest services/orchestrator/
    
    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose -f infrastructure/$DOCKER_COMPOSE_FILE down || true
    
    # Start services
    log_info "Starting services..."
    docker-compose -f infrastructure/$DOCKER_COMPOSE_FILE --env-file $ENV_FILE up -d
    
    log_success "Services deployed successfully"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    # Wait for CSM service
    log_info "Waiting for CSM service..."
    timeout 300 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done' || {
        log_error "CSM service failed to start"
        exit 1
    }
    
    # Wait for ASR service
    log_info "Waiting for ASR service..."
    timeout 300 bash -c 'until curl -f http://localhost:8001/health; do sleep 5; done' || {
        log_error "ASR service failed to start"
        exit 1
    }
    
    # Wait for Orchestrator service
    log_info "Waiting for Orchestrator service..."
    timeout 300 bash -c 'until curl -f http://localhost:8002/health; do sleep 5; done' || {
        log_error "Orchestrator service failed to start"
        exit 1
    }
    
    log_success "All services are healthy"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check CSM service
    log_info "Checking CSM service..."
    curl -f http://localhost:8000/health || {
        log_error "CSM service health check failed"
        exit 1
    }
    
    # Check ASR service
    log_info "Checking ASR service..."
    curl -f http://localhost:8001/health || {
        log_error "ASR service health check failed"
        exit 1
    }
    
    # Check Orchestrator service
    log_info "Checking Orchestrator service..."
    curl -f http://localhost:8002/health || {
        log_error "Orchestrator service health check failed"
        exit 1
    }
    
    log_success "All health checks passed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo ""
    echo "Services:"
    docker-compose -f infrastructure/$DOCKER_COMPOSE_FILE ps
    echo ""
    echo "Service URLs:"
    echo "  CSM Service: http://localhost:8000"
    echo "  ASR Service: http://localhost:8001"
    echo "  Orchestrator: http://localhost:8002"
    echo "  Nginx: http://localhost:80"
    echo ""
    echo "Logs:"
    echo "  docker-compose -f infrastructure/$DOCKER_COMPOSE_FILE logs -f"
    echo ""
    echo "Stop services:"
    echo "  docker-compose -f infrastructure/$DOCKER_COMPOSE_FILE down"
}

# Main deployment function
main() {
    log_info "Starting Oviya AI deployment..."
    
    check_prerequisites
    build_images
    deploy_services
    wait_for_services
    run_health_checks
    show_status
    
    log_success "Oviya AI deployment completed successfully!"
    log_info "You can now access the services at the URLs shown above."
}

# Run main function
main "$@"


