#!/bin/bash
# Monitor CSM-1B server performance

echo "═══════════════════════════════════════════════════════════════════════════"
echo "🔍 OVIYA CSM-1B SERVER MONITOR"
echo "═══════════════════════════════════════════════════════════════════════════"

# Check if server is running
if ! lsof -ti:19517 >/dev/null 2>&1; then
    echo "❌ CSM-1B server is not running on port 19517"
    echo ""
    echo "Start it with: ./start_csm_1b.sh"
    exit 1
fi

PID=$(lsof -ti:19517)
echo "✅ Server is running (PID: $PID)"
echo ""

# Show GPU status
echo "───────────────────────────────────────────────────────────────────────────"
echo "🎮 GPU STATUS"
echo "───────────────────────────────────────────────────────────────────────────"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r idx name temp gpu_util mem_util mem_used mem_total; do
        echo "GPU $idx: $name"
        echo "   Temperature: ${temp}°C"
        echo "   GPU Util: ${gpu_util}%"
        echo "   Memory: ${mem_used}MB / ${mem_total}MB (${mem_util}% used)"
    done
else
    echo "⚠️  nvidia-smi not available"
fi

echo ""

# Show process stats
echo "───────────────────────────────────────────────────────────────────────────"
echo "📊 PROCESS STATS (PID: $PID)"
echo "───────────────────────────────────────────────────────────────────────────"

if command -v ps &> /dev/null; then
    ps -p $PID -o pid,ppid,%cpu,%mem,vsz,rss,etime,cmd 2>/dev/null | tail -n +2 | while read -r line; do
        echo "$line"
    done
else
    echo "⚠️  ps command not available"
fi

echo ""

# Show health check
echo "───────────────────────────────────────────────────────────────────────────"
echo "🏥 HEALTH CHECK"
echo "───────────────────────────────────────────────────────────────────────────"

HEALTH=$(curl -s http://localhost:19517/health 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
else
    echo "⚠️  Health check failed (server may be starting up)"
fi

echo ""

# Show recent logs
echo "───────────────────────────────────────────────────────────────────────────"
echo "📝 RECENT LOGS (last 20 lines)"
echo "───────────────────────────────────────────────────────────────────────────"

if [ -f "/workspace/csm_1b.log" ]; then
    tail -n 20 /workspace/csm_1b.log
else
    echo "⚠️  Log file not found: /workspace/csm_1b.log"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "💡 Commands:"
echo "   • Watch logs: tail -f /workspace/csm_1b.log"
echo "   • Test server: python3 verify_csm_1b.py"
echo "   • Restart: ./stop_csm.sh && ./start_csm_1b.sh"
echo "   • GPU monitor: watch -n 1 nvidia-smi"
echo "═══════════════════════════════════════════════════════════════════════════"

