#!/bin/bash
# Oviya EI Environment Setup Script
# Sets up required environment variables for secure operation

echo "🔐 OVIYA EI ENVIRONMENT SETUP"
echo "=============================="
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists. Backing up..."
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
fi

# Create .env file
cat > .env << 'EOF'
# Oviya EI Environment Configuration
# This file contains sensitive configuration - DO NOT COMMIT TO VERSION CONTROL

# HuggingFace API Token (Required for CSM-1B model access)
# Get your token from: https://huggingface.co/settings/tokens
# Format: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_TOKEN=

# Oviya Secret for JWT authentication (generate a secure random key)
OVIYA_SECRET=

# Cloud GPU Configuration
CLOUD_GPU_AVAILABLE=false

# Environment Detection
OVIYA_ENV=development

# Database Configuration (optional)
# DATABASE_URL=sqlite:///./oviya.db

# Logging Configuration
LOG_LEVEL=INFO
EOF

echo "✅ .env template created"
echo ""
echo "📝 NEXT STEPS:"
echo "1. Edit the .env file with your actual values:"
echo "   nano .env"
echo ""
echo "2. Get your HuggingFace token:"
echo "   - Visit: https://huggingface.co/settings/tokens"
echo "   - Create a new token with 'Read' permissions"
echo "   - Copy the token (starts with 'hf_')"
echo ""
echo "3. Generate a secure secret key:"
echo "   python3 -c \"import secrets; print(secrets.token_hex(32))\""
echo ""
echo "4. Test your configuration:"
echo "   source .env && python3 tests/test_config_loading.py"
echo ""
echo "🔒 SECURITY NOTES:"
echo "- Never commit .env files to version control"
echo "- Keep your HuggingFace token secure"
echo "- Use strong, unique secret keys"
echo "- Regularly rotate tokens and keys"
echo ""

# Make script executable
chmod +x "$0"
