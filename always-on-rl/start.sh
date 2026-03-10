#!/bin/bash
# Start Always-On Memory + RL System

set -e

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "Always-On Memory + RL System"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 not found!${NC}"
    exit 1
fi

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import fastapi" 2>/dev/null || {
    echo "Installing FastAPI..."
    pip install fastapi uvicorn httpx pydantic
}

python3 -c "import mlx_lm" 2>/dev/null || {
    echo -e "${YELLOW}MLX not available - will run in mock mode${NC}"
    echo "To install MLX: pip install mlx mlx-lm"
}

# Create data directory
mkdir -p data

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    pkill -f "memory_server.py" 2>/dev/null || true
    pkill -f "rl_server.py" 2>/dev/null || true
    pkill -f "agent.py" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

# Start RL Server (needs to start first for model loading)
echo -e "${GREEN}Starting RL Server on port 30000...${NC}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
python3 rl/rl_server.py &
RL_PID=$!
sleep 3

# Start Memory Server
echo -e "${GREEN}Starting Memory Server on port 8888...${NC}"
export LLM_ENDPOINT="http://localhost:30000/v1"
export RL_ENDPOINT="http://localhost:30000"
python3 memory/memory_server.py &
MEMORY_PID=$!
sleep 2

# Start Integration Layer (optional)
if [ "$NO_AGENT" != "1" ]; then
    echo -e "${GREEN}Starting Integrated Agent on port 8080...${NC}"
    export MEMORY_URL="http://localhost:8888"
    export RL_URL="http://localhost:30000"
    python3 integration/agent.py &
    AGENT_PID=$!
fi

echo ""
echo "========================================"
echo -e "${GREEN}All services running!${NC}"
echo "========================================"
echo ""
echo "Endpoints:"
echo "  Memory:    http://localhost:8888"
echo "  RL:        http://localhost:30000"
if [ "$NO_AGENT" != "1" ]; then
    echo "  Agent:     http://localhost:8080"
fi
echo ""
echo "Quick tests:"
echo ""
echo "# Ingest memory:"
echo "curl -X POST http://localhost:8888/ingest \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"text\": \"User prefers short answers\", \"source\": \"feedback\"}'"
echo ""
echo "# Query memory:"
echo "curl -X POST http://localhost:8888/query \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"question\": \"What are the user preferences?\"}'"
echo ""
echo "# Trigger consolidation (generates training samples):"
echo "curl -X POST http://localhost:8888/consolidate"
echo ""
echo "# Check training samples:"
echo "curl http://localhost:30000/samples"
echo ""
echo "# View learned patterns:"
echo "curl http://localhost:30000/patterns"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait
wait
