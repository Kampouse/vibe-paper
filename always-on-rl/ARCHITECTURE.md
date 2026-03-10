# Always-On Memory + RL Architecture

Language-agnostic specification for implementing a self-learning agent system.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   USER INPUT                                                            │
│       │                                                                 │
│       ▼                                                                 │
│   ┌───────────────┐         ┌───────────────┐         ┌─────────────┐ │
│   │  MEMORY LAYER │ ◀─────▶ │   AGENT CORE  │ ◀─────▶ │  RL LAYER   │ │
│   │               │         │               │         │             │ │
│   │ • Store       │         │ • Route       │         │ • Generate  │ │
│   │ • Consolidate │         │ • Orchestrate │         │ • Train     │ │
│   │ • Query       │         │ • Coordinate  │         │ • Improve   │ │
│   └───────┬───────┘         └───────────────┘         └──────┬──────┘ │
│           │                                                  │        │
│           ▼                                                  ▼        │
│   ┌───────────────┐                                 ┌───────────────┐ │
│   │   DATABASE    │                                 │ MODEL WEIGHTS │ │
│   │   (SQLite)    │                                 │  (MLX/PyTorch)│ │
│   └───────────────┘                                 └───────────────┘ │
│                                                                         │
│   AUTOMATIC LEARNING LOOP:                                              │
│   Conversations → Consolidation → Patterns → Training → Better Model   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Memory Layer

**Purpose:** Persistent storage with active consolidation

**Key Functions:**
- `ingest(text, source)` → Store structured memory
- `consolidate()` → Find patterns, generate insights
- `query(question)` → Retrieve relevant context
- `get_training_samples()` → Extract learning signals

### 2. RL Layer

**Purpose:** Model serving and training

**Key Functions:**
- `generate(messages)` → Produce response
- `receive_feedback(reward, hint)` → Store manual feedback
- `receive_consolidation(samples)` → Store auto-generated samples
- `train()` → Update model weights

### 3. Integration

**Purpose:** Orchestrate both layers

**Key Functions:**
- `chat(message)` → Full pipeline
- `give_feedback(reward, hint)` → Training signal

---

## Data Models

### Memory

```typescript
interface Memory {
  id: number;
  summary: string;           // One-sentence summary
  entities: string[];        // Named entities (people, places, things)
  topics: string[];          // Topics/tags
  source: string;            // Where it came from (session/123, feedback)
  importance: number;        // 0.0 - 1.0
  raw_content: string;       // Original text
  created_at: timestamp;
  consolidated: boolean;     // Has been processed?
}
```

### Insight (from consolidation)

```typescript
interface Insight {
  id: number;
  memory_ids: number[];      // Which memories are connected
  connection_type: string;   // related, cause_effect, contradiction
  insight: string;           // What was learned
  created_at: timestamp;
}
```

### Training Sample

```typescript
interface TrainingSample {
  type: "manual_feedback" | "consolidation_pattern" | "consolidation_connection";
  
  // For manual feedback
  session_id?: string;
  turn_id?: number;
  messages?: Message[];
  response?: string;
  
  // For consolidation
  pattern?: string;          // Detected pattern description
  pattern_type?: "positive" | "negative" | "improvement" | "neutral";
  confidence?: number;       // 0.0 - 1.0
  memory_ids?: number[];
  
  // Common
  reward: number;            // -1.0, 0.0, +1.0
  hint: string | null;       // Actionable improvement
  timestamp: number;
}
```

### Message

```typescript
interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}
```

---

## API Specification

### Memory Layer API

**Base URL:** `http://localhost:8888`

#### POST /ingest

Store new memory with automatic structuring.

**Request:**
```json
{
  "text": "User prefers short answers without unnecessary explanations",
  "source": "feedback",
  "importance": 0.7
}
```

**Response:**
```json
{
  "id": 42,
  "summary": "User prefers concise responses",
  "entities": ["User"],
  "topics": ["preferences", "communication style"]
}
```

#### POST /query

Query memory for relevant context.

**Request:**
```json
{
  "question": "What are the user's preferences?",
  "include_insights": true,
  "limit": 10
}
```

**Response:**
```json
{
  "answer": "Based on stored memories:\n- User prefers short answers [1]\n- User likes direct responses [3]",
  "sources": [
    {"id": 1, "summary": "User prefers short answers"},
    {"id": 3, "summary": "User likes direct responses"}
  ],
  "insights_used": 2
}
```

#### POST /consolidate

Run consolidation on unprocessed memories. **This generates training samples.**

**Request:** Empty body

**Response:**
```json
{
  "status": "consolidated",
  "memories_processed": 15,
  "insights_created": 3,
  "training_samples": 5,
  "samples_sent": 5,
  "patterns": [
    {
      "pattern": "User consistently prefers brief responses",
      "type": "positive",
      "reward": 1.0
    },
    {
      "pattern": "Agent should check file existence before reading",
      "type": "improvement",
      "reward": 0.0
    }
  ]
}
```

#### GET /stats

Get memory statistics.

**Response:**
```json
{
  "total_memories": 150,
  "total_insights": 23,
  "unconsolidated": 12,
  "total_feedback": 8
}
```

---

### RL Layer API

**Base URL:** `http://localhost:30000`

#### POST /v1/chat/completions

OpenAI-compatible chat endpoint.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "[Context] User likes short answers"},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "session_id": "session-123",
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Response:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "4"
    },
    "finish_reason": "stop"
  }],
  "session_id": "session-123",
  "turn_id": 0
}
```

#### POST /feedback

Submit manual feedback for training.

**Request:**
```json
{
  "session_id": "session-123",
  "turn_id": 0,
  "reward": 1.0,
  "hint": "Good, kept it short as requested"
}
```

**Response:**
```json
{
  "status": "received",
  "pending_feedback": 3,
  "pending_consolidation": 12
}
```

#### POST /consolidation-samples

Receive training samples from memory consolidation.

**Request:**
```json
{
  "samples": [
    {
      "type": "consolidation_pattern",
      "pattern": "User prefers short answers",
      "pattern_type": "positive",
      "confidence": 0.9,
      "reward": 1.0,
      "hint": "Keep responses concise",
      "memory_ids": [1, 5, 8]
    }
  ]
}
```

**Response:**
```json
{
  "status": "received",
  "new_samples": 1,
  "total_consolidation": 12,
  "total_feedback": 3
}
```

#### GET /samples

Get pending training samples.

**Response:**
```json
{
  "feedback_samples": 3,
  "consolidation_samples": 12,
  "total": 15,
  "stats": {
    "total_samples_received": 45,
    "training_runs": 5
  }
}
```

#### GET /patterns

View learned patterns from consolidation.

**Response:**
```json
{
  "patterns_by_type": {
    "positive": [
      {
        "pattern": "User prefers short answers",
        "reward": 1.0,
        "confidence": 0.9,
        "hint": "Keep responses concise"
      }
    ],
    "improvement": [
      {
        "pattern": "Should check file exists before reading",
        "reward": 0.0,
        "confidence": 0.8,
        "hint": "Add existence check before file operations"
      }
    ]
  },
  "total_patterns": 12
}
```

#### POST /train

Trigger training on accumulated samples.

**Response:**
```json
{
  "status": "training_complete",
  "samples_used": 8,
  "sample_types": {
    "consolidation_pattern": 6,
    "manual_feedback": 2
  },
  "average_reward": 0.65,
  "hints_count": 5,
  "remaining_consolidation": 4,
  "remaining_feedback": 1
}
```

---

## Database Schema

### SQLite Tables

```sql
-- Core memory storage
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary TEXT NOT NULL,
    entities JSON DEFAULT '[]',
    topics JSON DEFAULT '[]',
    source TEXT,
    importance REAL DEFAULT 0.5,
    raw_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    consolidated BOOLEAN DEFAULT 0
);

-- Consolidation insights
CREATE TABLE insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_ids JSON NOT NULL,
    connection_type TEXT,
    insight TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Manual feedback storage
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER,
    reward REAL,
    hint TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

-- Indexes for performance
CREATE INDEX idx_memories_created ON memories(created_at);
CREATE INDEX idx_memories_consolidated ON memories(consolidated);
CREATE INDEX idx_memories_source ON memories(source);
```

---

## Algorithms

### 1. Consolidation Algorithm

```
function consolidate():
    memories = get_unconsolidated_memories(limit=50)
    
    if len(memories) < 2:
        return "not enough memories"
    
    // Step 1: Find connections between memories
    connections = find_connections(memories)
    
    // Step 2: Store insights
    for connection in connections:
        store_insight(connection)
    
    // Step 3: Generate training samples
    training_samples = []
    
    // Method A: LLM pattern analysis
    patterns = analyze_patterns_with_llm(memories)
    for pattern in patterns:
        sample = {
            type: "consolidation_pattern",
            pattern: pattern.description,
            pattern_type: pattern.type,  // positive/negative/improvement
            confidence: pattern.confidence,
            reward: calculate_reward(pattern),
            hint: pattern.hint,
            memory_ids: pattern.related_memories
        }
        training_samples.append(sample)
    
    // Method B: Connection-based signals
    for connection in connections:
        reward = calculate_reward_from_connection(connection)
        if reward != 0 or connection.insight:
            sample = {
                type: "consolidation_connection",
                insight: connection.insight,
                reward: reward,
                hint: extract_hint(connection.insight),
                memory_ids: connection.memory_ids
            }
            training_samples.append(sample)
    
    // Step 4: Send to RL layer
    send_to_rl_server(training_samples)
    
    // Step 5: Mark as consolidated
    mark_consolidated(memories)
    
    return training_samples
```

### 2. Pattern Detection Algorithm

```
function analyze_patterns_with_llm(memories):
    prompt = build_analysis_prompt(memories)
    response = llm_generate(prompt)
    
    patterns = []
    for block in parse_response_blocks(response):
        pattern = {
            description: extract_field(block, "PATTERN"),
            type: extract_field(block, "TYPE"),  // positive/negative/improvement
            confidence: extract_field(block, "CONFIDENCE"),
            reward: extract_field(block, "REWARD"),
            hint: extract_field(block, "HINT"),
            related_memories: extract_field(block, "MEMORY_IDS")
        }
        patterns.append(pattern)
    
    return patterns
```

### 3. Reward Calculation

```
function calculate_reward_from_connection(connection):
    insight = connection.insight.lower()
    
    // Positive indicators
    positive_words = [
        "successful", "works well", "good approach",
        "correctly", "properly", "user liked", "positive"
    ]
    
    // Negative indicators
    negative_words = [
        "error", "failed", "mistake", "wrong",
        "should have", "missed", "user corrected", "problem"
    ]
    
    // Count matches
    pos_score = count_matches(insight, positive_words)
    neg_score = count_matches(insight, negative_words)
    
    // Calculate reward
    if pos_score > neg_score:
        return min(1.0, 0.3 + pos_score * 0.2)
    elif neg_score > pos_score:
        return max(-1.0, -0.3 - neg_score * 0.2)
    else:
        return 0.0
```

### 4. Query Algorithm

```
function query_memory(question):
    // Step 1: Text search (simple)
    memories = search_memories(question, limit=10)
    
    // Step 2: Add recent context
    if len(memories) < limit:
        recent = get_recent_memories(hours=168)
        memories.extend(deduplicate(recent, memories))
    
    // Step 3: Get related insights
    insights = get_recent_insights(limit=5)
    
    // Step 4: Synthesize answer with LLM
    context = build_context(memories, insights)
    answer = llm_synthesize(question, context)
    
    return {
        answer: answer,
        sources: memories,
        insights_used: insights
    }
```

---

## Training Loop

### Automatic Schedule

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONSOLIDATION TIMELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  T+0min:   User has conversation                                │
│  T+1min:   Conversation stored in memory                        │
│  T+2min:   User gives feedback (optional)                       │
│  ...                                                            │
│  T+30min:  CONSOLIDATION TRIGGERS                               │
│            ├─ Review last 30 min of memories                    │
│            ├─ Find patterns                                     │
│            ├─ Generate training samples                         │
│            └─ Send to RL server                                 │
│  ...                                                            │
│  T+60min:  Next consolidation                                   │
│  ...                                                            │
│  T+120min: TRAINING TRIGGERS (when samples >= 8)                │
│            ├─ Combine consolidation + feedback samples          │
│            ├─ Run GRPO/OPD training                             │
│            └─ Update model weights                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sample Prioritization

When training:

```
1. Take 6 consolidation samples (high volume, automated)
2. Take 2 manual feedback samples (high quality, explicit)
3. Mix and shuffle
4. Calculate average reward
5. Apply GRPO or OPD training
```

---

## LLM Prompts

### Pattern Analysis Prompt

```
Analyze these conversation memories and extract training signals for an AI agent.

Memories:
{memory_list}

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

Extract 3-10 meaningful patterns. Be specific and actionable.
```

### Query Synthesis Prompt

```
Answer the question based on the context.

{context_with_memories_and_insights}

Question: {question}

Provide a concise answer with citations like [1], [2] referencing memory IDs.
```

### Connection Finding Prompt

```
Analyze these memories and find connections between them.

Memories:
{memory_list}

For each connection found, respond in this format:
CONNECTION: [id1, id2] - <type of connection>
INSIGHT: <brief insight about this connection>

Find 2-5 most meaningful connections. Focus on:
- Related topics or entities
- Temporal patterns
- Cause and effect relationships
- Contradictions or confirmations
```

---

## Integration Patterns

### Request Flow

```
1. USER MESSAGE
   │
   ├─▶ MEMORY: query(question) → context
   │
   ├─▶ BUILD: messages + context
   │
   ├─▶ RL: generate(messages) → response
   │
   ├─▶ MEMORY: ingest(conversation)
   │
   └─▶ RETURN response
```

### Feedback Flow

```
1. USER FEEDBACK
   │
   ├─▶ RL: receive_feedback(reward, hint)
   │
   ├─▶ MEMORY: ingest(feedback_text)
   │
   └─▶ (Later) CONSOLIDATION processes feedback
```

### Consolidation Flow

```
1. TIMER (30 min)
   │
   ├─▶ MEMORY: get_unconsolidated()
   │
   ├─▶ ANALYZE: find_patterns()
   │
   ├─▶ GENERATE: training_samples
   │
   ├─▶ RL: receive_consolidation(samples)
   │
   └─▶ MEMORY: mark_consolidated()
```

---

## Implementation Checklist

### Memory Layer

- [ ] SQLite database with schema
- [ ] REST API server
- [ ] `POST /ingest` - Store memory
- [ ] `POST /query` - Search and synthesize
- [ ] `POST /consolidate` - Pattern detection + training samples
- [ ] `GET /stats` - Statistics
- [ ] Background consolidation timer (30 min)

### RL Layer

- [ ] Model loading (MLX/PyTorch/other)
- [ ] REST API server
- [ ] `POST /v1/chat/completions` - OpenAI-compatible generation
- [ ] `POST /feedback` - Manual feedback
- [ ] `POST /consolidation-samples` - Auto-generated samples
- [ ] `GET /samples` - View pending samples
- [ ] `GET /patterns` - View learned patterns
- [ ] `POST /train` - Training loop
- [ ] Session tracking

### Integration

- [ ] Combined agent that uses both layers
- [ ] Context injection from memory to RL
- [ ] Automatic memory storage after conversations
- [ ] End-to-end test

---

## Language-Specific Notes

### Zig (NullClaw)

```zig
// Use std.http for API calls
// Use sqlite wrapper for database
// Use std.Thread for background consolidation

const MemoryClient = struct {
    allocator: Allocator,
    http_client: http.Client,
    
    pub fn ingest(self: *Self, text: []const u8, source: []const u8) !Memory {
        const response = try self.http_client.post(
            "http://localhost:8888/ingest",
            .{ .text = text, .source = source }
        );
        return parse_memory(response);
    }
};
```

### Rust

```rust
// Use reqwest for HTTP
// Use rusqlite for database
// Use tokio for async

struct MemoryClient {
    client: reqwest::Client,
    base_url: String,
}

impl MemoryClient {
    async fn ingest(&self, text: &str, source: &str) -> Result<Memory> {
        let response = self.client
            .post(format!("{}/ingest", self.base_url))
            .json(&IngestRequest { text, source })
            .send()
            .await?;
        response.json().await
    }
}
```

### Go

```go
// Use net/http for API calls
// Use modernc.org/sqlite for database
// Use time.Ticker for background tasks

type MemoryClient struct {
    baseURL    string
    httpClient *http.Client
}

func (c *MemoryClient) Ingest(text, source string) (*Memory, error) {
    payload := IngestRequest{Text: text, Source: source}
    resp, err := c.httpClient.Post(
        c.baseURL + "/ingest",
        "application/json",
        bytes.NewReader(toJSON(payload)),
    )
    // ...
}
```

---

## Testing

### Unit Tests

```bash
# Test memory storage
curl -X POST http://localhost:8888/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Test memory", "source": "test"}'

# Test query
curl -X POST http://localhost:8888/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is test memory?"}'

# Test consolidation
curl -X POST http://localhost:8888/consolidate

# Test training
curl -X POST http://localhost:30000/train
```

### Integration Test

```bash
# Full loop test
python3 test_consolidation.py
```

---

## Deployment

### Single Machine (Mac/Linux)

```bash
./start.sh
```

### Docker

```dockerfile
FROM python:3.12

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["./start.sh"]
```

### Environment Variables

```bash
# Memory server
export LLM_ENDPOINT="http://localhost:30000/v1"
export RL_ENDPOINT="http://localhost:30000"

# RL server
export MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# Ports
export MEMORY_PORT=8888
export RL_PORT=30000
```

---

## Next Steps

1. **Add actual MLX LoRA training** in RL layer
2. **Improve pattern detection** with better prompts
3. **Add PRM judge** for automatic reward scoring
4. **Implement in other languages** (Zig, Rust, Go)
5. **Scale horizontally** with distributed training

---

## References

- OpenClaw-RL: https://github.com/Gen-Verse/OpenClaw-RL
- Slime Framework: https://github.com/THUDM/slime
- MLX: https://github.com/ml-explore/mlx
- Google Always-On Memory: https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/agents/always-on-memory-agent
