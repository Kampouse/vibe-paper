"""Always-On Memory Server - Stores, consolidates, and queries memories
With consolidation-driven training signals
"""
import asyncio
import os
import re
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx

from memory_db import MemoryDB

# Initialize
app = FastAPI(title="Always-On Memory Server")
db: Optional[MemoryDB] = None

# LLM client (will connect to RL server)
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:30000/v1")
RL_ENDPOINT = os.getenv("RL_ENDPOINT", "http://localhost:30000")

# ============== Models ==============

class IngestRequest(BaseModel):
    text: str
    source: str = "manual"
    importance: float = 0.5

class QueryRequest(BaseModel):
    question: str
    include_insights: bool = True
    limit: int = 10

# ============== LLM Helpers ==============

async def llm_generate(prompt: str, max_tokens: int = 1024) -> str:
    """Call LLM for extraction/summarization"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{LLM_ENDPOINT}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                }
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[LLM Error] {e}")
        return f"SUMMARY: {prompt[:100]}\nENTITIES: \nTOPICS: \nIMPORTANCE: 0.5"

async def extract_structure(text: str) -> dict:
    """Extract structured info from text"""
    prompt = f"""Analyze this text and extract structured information.

Text: {text}

Respond in this exact format:
SUMMARY: <one sentence summary>
ENTITIES: <comma-separated list of named entities>
TOPICS: <comma-separated list of topics>
IMPORTANCE: <number 0.0-1.0>

Keep it concise."""
    
    response = await llm_generate(prompt, max_tokens=256)
    
    result = {
        "summary": text[:100],
        "entities": [],
        "topics": [],
        "importance": 0.5
    }
    
    for line in response.split('\n'):
        if line.startswith('SUMMARY:'):
            result['summary'] = line.replace('SUMMARY:', '').strip()
        elif line.startswith('ENTITIES:'):
            entities = line.replace('ENTITIES:', '').strip()
            result['entities'] = [e.strip() for e in entities.split(',') if e.strip()]
        elif line.startswith('TOPICS:'):
            topics = line.replace('TOPICS:', '').strip()
            result['topics'] = [t.strip() for t in topics.split(',') if t.strip()]
        elif line.startswith('IMPORTANCE:'):
            try:
                result['importance'] = float(line.replace('IMPORTANCE:', '').strip())
            except:
                pass
    
    return result

async def find_connections(memories: List[dict]) -> List[dict]:
    """Find connections between memories"""
    if len(memories) < 2:
        return []
    
    memory_texts = "\n".join([
        f"[{m['id']}] {m['summary']}"
        for m in memories[:20]
    ])
    
    prompt = f"""Analyze these memories and find connections between them.

Memories:
{memory_texts}

For each connection found, respond in this format:
CONNECTION: [id1, id2] - <type of connection>
INSIGHT: <brief insight about this connection>

Find 2-5 most meaningful connections."""
    
    response = await llm_generate(prompt, max_tokens=512)
    
    connections = []
    current_conn = None
    
    for line in response.split('\n'):
        if line.startswith('CONNECTION:'):
            if current_conn:
                connections.append(current_conn)
            ids = re.findall(r'\[(\d+),\s*(\d+)\]', line)
            if ids:
                conn_type = line.split('-')[1].strip() if '-' in line else 'related'
                current_conn = {
                    'memory_ids': [int(ids[0][0]), int(ids[0][1])],
                    'connection_type': conn_type,
                    'insight': ''
                }
        elif line.startswith('INSIGHT:') and current_conn:
            current_conn['insight'] = line.replace('INSIGHT:', '').strip()
    
    if current_conn:
        connections.append(current_conn)
    
    return connections

async def synthesize_answer(question: str, memories: List[dict], insights: List[dict]) -> str:
    """Synthesize an answer from memories and insights"""
    if not memories:
        return "I don't have any relevant information about that."
    
    context = "Relevant memories:\n"
    for m in memories[:5]:
        context += f"- [{m['id']}] {m['summary']}\n"
    
    if insights:
        context += "\nRelated insights:\n"
        for i in insights[:3]:
            context += f"- {i.get('insight', '')}\n"
    
    prompt = f"""Answer the question based on the context.

{context}

Question: {question}

Provide a concise answer with citations like [1], [2] referencing memory IDs."""
    
    return await llm_generate(prompt, max_tokens=512)

# ============== SMARTER PATTERN DETECTION ==============

async def analyze_training_patterns(memories: List[dict]) -> List[dict]:
    """Analyze memories to extract training signals using LLM"""
    if len(memories) < 2:
        return []
    
    memory_texts = "\n".join([
        f"[{i}] {m.get('raw_content', m['summary'])}"
        for i, m in enumerate(memories[:30])
    ])
    
    prompt = f"""Analyze these conversation memories and extract training signals for an AI agent.

Memories:
{memory_texts}

For each pattern you find, respond in this EXACT format:

PATTERN: <description of the pattern>
TYPE: <positive|negative|neutral|improvement>
CONFIDENCE: <0.0-1.0>
REWARD: <+1.0|0.0|-1.0>
HINT: <actionable advice for the agent>
MEMORY_IDS: [<id1>, <id2>, ...]

Look for:
1. User preferences (e.g., "User likes short answers")
2. Successful behaviors (e.g., "Checking file before editing worked")
3. Failed behaviors (e.g., "Running command without checking PATH failed")
4. Improvement opportunities (e.g., "Should ask clarifying questions more often")
5. Error patterns (e.g., "Common mistake: not handling edge cases")

Extract 3-10 meaningful patterns. Be specific and actionable."""
    
    response = await llm_generate(prompt, max_tokens=1024)
    
    # Parse patterns
    patterns = []
    current = None
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('PATTERN:'):
            if current and current.get('pattern'):
                patterns.append(current)
            current = {
                'pattern': line.replace('PATTERN:', '').strip(),
                'type': 'neutral',
                'confidence': 0.5,
                'reward': 0.0,
                'hint': None,
                'memory_ids': []
            }
        elif current:
            if line.startswith('TYPE:'):
                current['type'] = line.replace('TYPE:', '').strip().lower()
            elif line.startswith('CONFIDENCE:'):
                try:
                    current['confidence'] = float(line.replace('CONFIDENCE:', '').strip())
                except:
                    pass
            elif line.startswith('REWARD:'):
                try:
                    val = line.replace('REWARD:', '').strip()
                    current['reward'] = float(val)
                except:
                    pass
            elif line.startswith('HINT:'):
                current['hint'] = line.replace('HINT:', '').strip()
            elif line.startswith('MEMORY_IDS:'):
                ids_str = line.replace('MEMORY_IDS:', '').strip()
                current['memory_ids'] = [
                    int(x.strip()) 
                    for x in re.findall(r'\d+', ids_str)
                ]
    
    if current and current.get('pattern'):
        patterns.append(current)
    
    return patterns

async def generate_training_samples(memories: List[dict], connections: List[dict]) -> List[dict]:
    """Generate RL training samples from consolidation"""
    samples = []
    
    # Method 1: Use LLM to analyze patterns
    patterns = await analyze_training_patterns(memories)
    
    for p in patterns:
        sample = {
            "type": "consolidation_pattern",
            "pattern": p['pattern'],
            "pattern_type": p['type'],
            "confidence": p['confidence'],
            "reward": p['reward'],
            "hint": p['hint'],
            "memory_ids": p['memory_ids'],
            "source": "consolidation"
        }
        samples.append(sample)
    
    # Method 2: Analyze connections for additional signals
    for conn in connections:
        insight = conn.get('insight', '').lower()
        
        # Skip if already captured in patterns
        sample = {
            "type": "consolidation_connection",
            "insight": conn['insight'],
            "connection_type": conn['connection_type'],
            "memory_ids": conn['memory_ids'],
            "reward": 0.0,
            "hint": None,
            "source": "consolidation"
        }
        
        # Smart reward detection
        # Positive indicators
        positive_words = [
            'successful', 'works well', 'good approach', 'effective',
            'correctly', 'properly', 'best practice', 'ideal',
            'user liked', 'user appreciated', 'positive feedback'
        ]
        
        # Negative indicators  
        negative_words = [
            'error', 'failed', 'mistake', 'wrong', 'incorrect',
            'should have', 'could have', 'missed', 'forgot',
            'user complained', 'user corrected', 'problem'
        ]
        
        # Improvement indicators
        improvement_words = [
            'should', 'better', 'improve', 'next time',
            'consider', 'might want to', 'could try'
        ]
        
        # Calculate reward
        pos_score = sum(1 for w in positive_words if w in insight)
        neg_score = sum(1 for w in negative_words if w in insight)
        imp_score = sum(1 for w in improvement_words if w in insight)
        
        if pos_score > neg_score:
            sample['reward'] = min(1.0, 0.3 + pos_score * 0.2)
            sample['hint'] = f"Good pattern: {conn['insight']}"
        elif neg_score > pos_score:
            sample['reward'] = max(-1.0, -0.3 - neg_score * 0.2)
            sample['hint'] = f"Avoid this: {conn['insight']}"
        elif imp_score > 0:
            sample['reward'] = 0.0
            sample['hint'] = conn['insight']
        
        if sample['reward'] != 0.0 or sample['hint']:
            samples.append(sample)
    
    return samples

async def send_to_rl(samples: List[dict]):
    """Send training samples to RL server"""
    if not samples:
        return {"sent": 0}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{RL_ENDPOINT}/consolidation-samples",
                json={"samples": samples}
            )
            return response.json()
    except Exception as e:
        print(f"[Warning] Failed to send training samples: {e}")
        return {"sent": 0, "error": str(e)}

# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {"service": "Always-On Memory", "status": "running"}

@app.get("/stats")
async def stats():
    return db.get_stats()

@app.post("/ingest")
async def ingest(request: IngestRequest):
    """Ingest new text into memory"""
    structure = await extract_structure(request.text)
    
    memory_id = db.store_memory(
        summary=structure['summary'],
        entities=structure['entities'],
        topics=structure['topics'],
        source=request.source,
        importance=structure['importance'],
        raw_content=request.text
    )
    
    return {
        "id": memory_id,
        "summary": structure['summary'],
        "entities": structure['entities'],
        "topics": structure['topics']
    }

@app.get("/memories")
async def list_memories(limit: int = 50, unconsolidated_only: bool = False):
    """List stored memories"""
    if unconsolidated_only:
        memories = db.get_unconsolidated(limit)
    else:
        memories = db.search("", limit)
    return {"memories": memories, "count": len(memories)}

@app.post("/query")
async def query(request: QueryRequest):
    """Query memory with a question"""
    memories = db.search(request.question, limit=request.limit)
    
    if len(memories) < request.limit:
        recent = db.get_recent(hours=168, limit=request.limit - len(memories))
        existing_ids = {m['id'] for m in memories}
        for m in recent:
            if m['id'] not in existing_ids:
                memories.append(m)
    
    insights = []
    if request.include_insights:
        insights = db.get_all_insights(limit=5)
    
    answer = await synthesize_answer(request.question, memories, insights)
    
    return {
        "answer": answer,
        "sources": [{"id": m['id'], "summary": m['summary']} for m in memories[:5]],
        "insights_used": len(insights)
    }

@app.post("/consolidate")
async def consolidate():
    """Run consolidation and generate training samples"""
    memories = db.get_unconsolidated(limit=50)
    
    if len(memories) < 2:
        return {"status": "not enough memories", "count": len(memories)}
    
    # Find connections
    connections = await find_connections(memories)
    
    # Store insights
    for conn in connections:
        db.store_insight(
            memory_ids=conn['memory_ids'],
            connection_type=conn['connection_type'],
            insight=conn['insight']
        )
    
    # Generate training samples (NEW!)
    training_samples = await generate_training_samples(memories, connections)
    
    # Send to RL server
    send_result = await send_to_rl(training_samples)
    
    # Mark as consolidated
    memory_ids = [m['id'] for m in memories]
    db.mark_consolidated(memory_ids)
    
    return {
        "status": "consolidated",
        "memories_processed": len(memories),
        "insights_created": len(connections),
        "training_samples": len(training_samples),
        "samples_sent": send_result.get("sent", 0),
        "patterns": [
            {
                "pattern": s.get('pattern', s.get('insight', '')),
                "reward": s.get('reward', 0),
                "type": s.get('pattern_type', s.get('type', 'unknown'))
            }
            for s in training_samples[:5]  # Show first 5
        ]
    }

@app.post("/analyze-patterns")
async def analyze_patterns():
    """Manually trigger pattern analysis without consolidation"""
    memories = db.get_unconsolidated(limit=50)
    
    if len(memories) < 2:
        memories = db.get_recent(hours=168, limit=30)
    
    patterns = await analyze_training_patterns(memories)
    
    return {
        "patterns": patterns,
        "memories_analyzed": len(memories)
    }

@app.post("/feedback")
async def store_feedback(memory_id: int, reward: float, hint: str = None):
    """Store feedback for RL training"""
    db.store_feedback(memory_id, reward, hint)
    return {"status": "stored"}

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    db.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.conn.commit()
    return {"status": "deleted"}

@app.post("/clear")
async def clear_all():
    db.conn.execute("DELETE FROM memories")
    db.conn.execute("DELETE FROM insights")
    db.conn.execute("DELETE FROM feedback")
    db.conn.commit()
    return {"status": "cleared"}

# ============== Background Tasks ==============

async def periodic_consolidation():
    """Run consolidation every 30 minutes"""
    while True:
        await asyncio.sleep(1800)  # 30 minutes
        try:
            result = await consolidate()
            print(f"[Consolidation] {result}")
        except Exception as e:
            print(f"[Consolidation Error] {e}")

@app.on_event("startup")
async def startup():
    global db
    db = MemoryDB("data/memory.db")
    asyncio.create_task(periodic_consolidation())
    print("[Memory Server] Started on port 8888")
    print(f"[Memory Server] LLM Endpoint: {LLM_ENDPOINT}")
    print(f"[Memory Server] RL Endpoint: {RL_ENDPOINT}")
    print("[Memory Server] Consolidation training ENABLED")

@app.on_event("shutdown")
async def shutdown():
    if db:
        db.close()

if __name__ == "__main__":
    uvicorn.run(
        "memory_server:app",
        host="0.0.0.0",
        port=8888,
        reload=True
    )
