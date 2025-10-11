# Expo Bare Mobile App Setup Script

#!/bin/bash
echo "🚀 Setting up Oviya Mobile App..."

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Create assets directory
echo "📁 Creating assets directory..."
mkdir -p assets

# Create placeholder assets
echo "🎨 Creating placeholder assets..."
echo "Placeholder icon" > assets/icon.png
echo "Placeholder splash" > assets/splash.png
echo "Placeholder adaptive icon" > assets/adaptive-icon.png
echo "Placeholder favicon" > assets/favicon.png

echo "✅ Mobile app setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Update WebSocket URL in App.tsx to your RunPod IP"
echo "2. Run: npm start"
echo "3. Scan QR code with Expo Go app"
echo ""
echo "📱 The app will connect to your Oviya backend services!"


