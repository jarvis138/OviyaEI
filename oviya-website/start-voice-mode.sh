#!/bin/bash

echo "🎙️ Starting Oviya Voice Mode..."
echo ""

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is already running on port 8000"
else
    echo "🚀 Starting Oviya backend..."
    cd "../oviya-production" && python3 websocket_server.py &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    sleep 3
fi

# Check if frontend is running
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is already running on port 3000"
else
    echo "🚀 Starting Oviya frontend..."
    npm run dev &
    FRONTEND_PID=$!
    echo "   Frontend PID: $FRONTEND_PID"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "🎉 Oviya Voice Mode is ready!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "🌐 Open your browser: http://localhost:3000"
echo ""
echo "🎤 How to use:"
echo "   1. Wait for the 'Connected' status"
echo "   2. Click the orb or press Space to talk"
echo "   3. Speak naturally"
echo "   4. Listen to Oviya's response"
echo ""
echo "⌨️  Keyboard shortcuts:"
echo "   Space - Toggle voice recording"
echo ""
echo "To stop: Press Ctrl+C"
echo "═══════════════════════════════════════════════════"

# Keep script running
wait
