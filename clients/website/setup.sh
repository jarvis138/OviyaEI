#!/bin/bash

# Oviya Website Deployment Script
# This script sets up and deploys the new Oviya website

echo "🚀 Setting up Oviya Website..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the oviya-website directory."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Create environment file if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo "⚙️ Creating environment configuration..."
    cat > .env.local << EOF
# Oviya Website Environment Configuration
NEXT_PUBLIC_ORCHESTRATOR_URL=http://localhost:8000
NODE_ENV=development
EOF
    echo "✅ Created .env.local file"
else
    echo "✅ Environment file already exists"
fi

# Build the project
echo "🔨 Building the project..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "🎉 Oviya Website is ready!"
    echo ""
    echo "To start the development server:"
    echo "  npm run dev"
    echo ""
    echo "To start the production server:"
    echo "  npm run start"
    echo ""
    echo "Make sure your oviya-production backend is running on port 8000:"
    echo "  cd ../oviya-production"
    echo "  python websocket_server.py"
    echo ""
    echo "Then visit: http://localhost:3000"
else
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi
