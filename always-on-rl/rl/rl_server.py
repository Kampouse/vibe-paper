"""RL Server using MLX for Apple Silicon
With consolidation-driven training support
"""
import os
from typing import List, Optional, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from collections import defaultdict
import time

# MLX
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("[Warning] MLX not available, using mock responses")

# Initialize
app = FastAPI(title="RL Server (MLX)")

# Model
model = None
tokenizer = None

# Session tracking
sessions: Dict[str, List[dict]] = {}

# Training samples
feedback_samples: List[dict] = []  # Manual feedback
consolidation_samples: List[dict] = []  # From memory consolidation

# Training statistics
training_stats = {
    "total_samples_received": 0,
    "feedback_samples": 0,
    "consolidation_samples": 0,
    "training_runs": 0,
    "last_training": None
}

# ============== Models ==============

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1024

class FeedbackRequest(BaseModel):
    session_id: str
    turn_id: int
    reward: float
    hint: Optional[str] = None

class ConsolidationSamples(BaseModel):
    samples: List[dict]

# ============== Generation ==============

def format_messages(messages: List[Message]) -> str:
    """Format messages for the model"""
    formatted = ""
    for msg in messages:
        if msg.role == "system":
            formatted += f" system\n{msg.content}\n"
        elif msg.role == "user":
            formatted += f" user\n{msg.content}\n"
        elif msg.role == "assistant":
            formatted += f" assistant\n{msg.content}\n"
    formatted += " assistant\n"
    return formatted

def generate_response(prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Generate response with MLX"""
    global model, tokenizer
    
    if not MLX_AVAILABLE or model is None:
        # Mock response
        return "I understand. Let me help you with that."
    
    sampler = make_sampler(temp=temperature, top_p=0.9)
    
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler
    )
    
    return response

# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "service": "RL Server (MLX)", 
        "status": "running",
        "mlx_available": MLX_AVAILABLE,
        "model_loaded": model is not None,
        "training_stats": training_stats
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    """OpenAI-compatible chat endpoint"""
    prompt = format_messages(request.messages)
    
    response_text = generate_response(
        prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    if request.session_id not in sessions:
        sessions[request.session_id] = []
    
    turn = {
        "turn_id": len(sessions[request.session_id]),
        "messages": [m.dict() for m in request.messages],
        "response": response_text,
        "timestamp": time.time()
    }
    sessions[request.session_id].append(turn)
    
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "session_id": request.session_id,
        "turn_id": turn["turn_id"]
    }

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Receive manual feedback for training"""
    session = sessions.get(request.session_id, [])
    
    if request.turn_id >= len(session):
        return {"status": "error", "message": "Invalid turn_id"}
    
    turn = session[request.turn_id]
    
    sample = {
        "type": "manual_feedback",
        "session_id": request.session_id,
        "turn_id": request.turn_id,
        "messages": turn["messages"],
        "response": turn["response"],
        "reward": request.reward,
        "hint": request.hint,
        "timestamp": time.time()
    }
    feedback_samples.append(sample)
    
    training_stats["feedback_samples"] += 1
    training_stats["total_samples_received"] += 1
    
    return {
        "status": "received", 
        "pending_feedback": len(feedback_samples),
        "pending_consolidation": len(consolidation_samples)
    }

@app.post("/consolidation-samples")
async def receive_consolidation_samples(data: ConsolidationSamples):
    """Receive training samples from memory consolidation"""
    global consolidation_samples
    
    for sample in data.samples:
        sample["timestamp"] = time.time()
        consolidation_samples.append(sample)
    
    training_stats["consolidation_samples"] += len(data.samples)
    training_stats["total_samples_received"] += len(data.samples)
    
    return {
        "status": "received",
        "new_samples": len(data.samples),
        "total_consolidation": len(consolidation_samples),
        "total_feedback": len(feedback_samples)
    }

@app.get("/samples")
async def get_samples():
    """Get pending training samples"""
    return {
        "feedback_samples": len(feedback_samples),
        "consolidation_samples": len(consolidation_samples),
        "total": len(feedback_samples) + len(consolidation_samples),
        "stats": training_stats
    }

@app.get("/samples/detail")
async def get_samples_detail(limit: int = 10):
    """Get detailed view of pending samples"""
    return {
        "feedback": feedback_samples[:limit],
        "consolidation": consolidation_samples[:limit],
        "stats": training_stats
    }

@app.get("/patterns")
async def get_learned_patterns():
    """Get patterns learned from consolidation"""
    patterns = defaultdict(list)
    
    for sample in consolidation_samples:
        pattern_type = sample.get("pattern_type", sample.get("type", "unknown"))
        patterns[pattern_type].append({
            "pattern": sample.get("pattern", sample.get("insight", "")),
            "reward": sample.get("reward", 0),
            "confidence": sample.get("confidence", 0.5),
            "hint": sample.get("hint")
        })
    
    return {
        "patterns_by_type": dict(patterns),
        "total_patterns": len(consolidation_samples)
    }

@app.post("/train")
async def train():
    """Train on accumulated samples"""
    global feedback_samples, consolidation_samples
    
    all_samples = feedback_samples + consolidation_samples
    
    if len(all_samples) < 4:
        return {
            "status": "not enough samples", 
            "count": len(all_samples),
            "needed": 4
        }
    
    # Prioritize: use consolidation samples first (more data)
    # Then add feedback samples for quality
    samples_to_use = []
    
    # Take up to 6 consolidation samples
    samples_to_use.extend(consolidation_samples[:6])
    
    # Take up to 2 feedback samples
    samples_to_use.extend(feedback_samples[:2])
    
    # Remove used samples
    used_consolidation = min(6, len(consolidation_samples))
    used_feedback = min(2, len(feedback_samples))
    
    consolidation_samples = consolidation_samples[used_consolidation:]
    feedback_samples = feedback_samples[used_feedback:]
    
    # Calculate average reward
    rewards = [s.get("reward", 0) for s in samples_to_use]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    
    # Group by type
    types = defaultdict(int)
    for s in samples_to_use:
        types[s.get("type", "unknown")] += 1
    
    # Update stats
    training_stats["training_runs"] += 1
    training_stats["last_training"] = time.time()
    
    # TODO: Actual MLX LoRA training
    # For now, just log what would be trained
    print(f"\n[Training] Would train on {len(samples_to_use)} samples:")
    print(f"  - Consolidation: {types.get('consolidation_pattern', 0) + types.get('consolidation_connection', 0)}")
    print(f"  - Manual feedback: {types.get('manual_feedback', 0)}")
    print(f"  - Average reward: {avg_reward:.2f}")
    print(f"  - Hints provided: {sum(1 for s in samples_to_use if s.get('hint'))}")
    
    return {
        "status": "training_placeholder",
        "samples_used": len(samples_to_use),
        "sample_types": dict(types),
        "average_reward": avg_reward,
        "hints_count": sum(1 for s in samples_to_use if s.get('hint')),
        "remaining_feedback": len(feedback_samples),
        "remaining_consolidation": len(consolidation_samples),
        "training_runs_total": training_stats["training_runs"],
        "note": "MLX LoRA training not yet implemented - samples logged"
    }

@app.get("/sessions")
async def list_sessions():
    """List all sessions"""
    return {
        "sessions": [
            {"id": sid, "turns": len(turns)}
            for sid, turns in sessions.items()
        ],
        "total_sessions": len(sessions)
    }

@app.post("/clear-samples")
async def clear_samples(sample_type: str = "all"):
    """Clear pending samples"""
    global feedback_samples, consolidation_samples
    
    cleared = 0
    if sample_type in ["all", "feedback"]:
        cleared += len(feedback_samples)
        feedback_samples = []
    if sample_type in ["all", "consolidation"]:
        cleared += len(consolidation_samples)
        consolidation_samples = []
    
    return {"status": "cleared", "samples_removed": cleared}

# ============== Startup ==============

@app.on_event("startup")
async def startup():
    global model, tokenizer
    
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    
    if MLX_AVAILABLE:
        print(f"[RL Server] Loading model: {model_name}")
        try:
            model, tokenizer = load(model_name)
            print(f"[RL Server] Model loaded successfully!")
        except Exception as e:
            print(f"[RL Server] Failed to load model: {e}")
            print("[RL Server] Running in mock mode")
    else:
        print("[RL Server] MLX not available, running in mock mode")
    
    print("[RL Server] Started on port 30000")
    print("[RL Server] Consolidation training ENABLED")

# ============== Run ==============

if __name__ == "__main__":
    uvicorn.run(
        "rl_server:app",
        host="0.0.0.0",
        port=30000,
        reload=False
    )
