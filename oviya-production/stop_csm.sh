#!/bin/bash
# Stop the old placeholder CSM server

echo "🛑 Stopping old CSM server..."

# Find and kill process on port 19517
PID=$(lsof -ti:19517)

if [ -z "$PID" ]; then
    echo "   ℹ️  No process running on port 19517"
else
    echo "   Found PID: $PID"
    kill $PID
    sleep 2
    
    # Force kill if still running
    if kill -0 $PID 2>/dev/null; then
        echo "   Force killing..."
        kill -9 $PID
    fi
    
    echo "   ✅ Stopped"
fi

# Verify port is free
sleep 1
if lsof -ti:19517 >/dev/null 2>&1; then
    echo "   ⚠️  Port 19517 still in use!"
    lsof -i:19517
else
    echo "   ✅ Port 19517 is free"
fi

