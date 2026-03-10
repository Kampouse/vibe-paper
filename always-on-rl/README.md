# Always-On Memory + RL Training System

Automatic training from memory consolidation.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn httpx pydantic

# Optional: Install MLX for Apple Silicon
pip install mlx mlx-lm

# Start everything
./start.sh
```

## How It Works

```
Conversation → Memory → Consolidation (30 min) → Patterns → Training Samples → RL
```

### Consolidation-Driven Training

Every 30 minutes, the system:

1. **Reviews** recent conversations
2. **Finds patterns** using LLM:
   - User preferences ("User likes short answers")
   - Successes ("Checking file first worked well")
   - Failures ("Forgot to check PATH")
   - Improvements ("Should ask clarifying questions")
3. **Generates training samples**:
   - Positive patterns → Reward +1
   - Negative patterns → Reward -1
   - Improvements → Hint for OPD
4. **Sends to RL** for training

## Endpoints

| Service | Port | URL |
|---------|------|-----|
| Memory | 8888 | http://localhost:8888 |
| RL | 30000 | http://localhost:30000 |

## API

### Memory

```bash
# Ingest
curl -X POST http://localhost:8888/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers short answers", "source": "feedback"}'

# Query
curl -X POST http://localhost:8888/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are user preferences?"}'

# Consolidate (generates training samples)
curl -X POST http://localhost:8888/consolidate

# Stats
curl http://localhost:8888/stats
```

### RL

```bash
# Chat
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Check samples
curl http://localhost:30000/samples

# View learned patterns
curl http://localhost:30000/patterns

# Train
curl -X POST http://localhost:30000/train
```

## Test

```bash
# Run full test
python3 test_consolidation.py
```

## Files

```
always-on-rl/
├── memory/
│   ├── memory_db.py         # SQLite database
│   └── memory_server.py     # FastAPI server + consolidation
├── rl/
│   └── rl_server.py         # MLX + training
├── integration/
│   └── agent.py             # Combined agent
├── data/
│   └── memory.db            # Auto-created
├── start.sh                 # Start everything
├── test_consolidation.py    # Test script
└── requirements.txt
```

## Pattern Detection

The system detects:

| Pattern Type | Example | Reward |
|--------------|---------|--------|
| **Positive** | "User liked short answer" | +1.0 |
| **Negative** | "Agent forgot to check file" | -1.0 |
| **Improvement** | "Should ask clarifying questions" | 0.0 + hint |

## Next Steps

1. Add actual MLX LoRA training
2. Improve pattern detection prompts
3. Add PRM judge for automatic rewards
4. Integrate with NullClaw (Zig)
