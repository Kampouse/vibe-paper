# Architecture Overview

## 🎯 System Overview

The Nostr-Backed NEAR Whitelist System is a multi-layered security architecture that combines the flexibility of Nostr's decentralized communication protocol with the immutability and security of NEAR's blockchain infrastructure.

```
┌─────────────────────────────────────────────────────────────────┐
│                   SYSTEM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              PRESENTATION LAYER                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │  │Web Dashboard│  │CLI Tools    │  │Monitoring  ││
│  │  └─────────────┘  └─────────────┘  └──────────┘│
│  └────────────────────────────────────────────────────────┘    │
│                          │                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              APPLICATION LAYER                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │  │Swarm        │  │Auth Service │  │Task Dist.  ││
│  │  │Orchestrator │  │             │  │            ││
│  │  └─────────────┘  └─────────────┘  └──────────┘│
│  └────────────────────────────────────────────────────────┘    │
│                          │                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              COMMUNICATION LAYER                   │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │  │Private Relay │  │Private Relay │  │Private    ││
│  │  │    1        │  │    2        │  │Relay 3    ││
│  │  └─────────────┘  └─────────────┘  └──────────┘│
│  └────────────────────────────────────────────────────────┘    │
│                          │                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              IDENTITY LAYER                         │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │  Agent Whitelist Smart Contract (NEAR)   │    │    │
│  │  │                                         │    │    │
│  │  │  • Whitelisted Agents                 │    │    │
│  │  │  • Ownership Proofs                 │    │    │
│  │  │  • Stake Management                 │    │    │
│  │  │  • Capability Declarations           │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └────────────────────────────────────────────────────────┘    │
│                          │                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              DATA LAYER                            │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │  │NEAR         │  │Nostr Events │  │File Store ││
│  │  │Blockchain    │  │(Encrypted)  │  │(IPFS/Bloom)│
│  │  └─────────────┘  └─────────────┘  └──────────┘│
│  └────────────────────────────────────────────────────────┘    │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## 🏗️ Component Architecture

### 1. Agent Identity Layer

**Purpose**: Manages and validates agent identities across both Nostr and NEAR networks.

**Components**:
- **Nostr Key Management**: Handles generation, storage, and rotation of Nostr keypairs
- **NEAR Account Binding**: Cryptographically links Nostr pubkeys to NEAR accounts
- **Ownership Proof Generator**: Creates verifiable signatures proving dual ownership
- **Identity Cache**: Fast lookups between Nostr pubkeys and NEAR accounts

**Data Flow**:
```
Agent generates Nostr keypair
        ↓
Agent creates NEAR account
        ↓
Agent signs ownership message
        ↓
Registration submitted to whitelist contract
        ↓
Contract stores bidirectional mapping
        ↓
Identity verified and cached
```

### 2. Whitelist Management Layer

**Purpose**: Enforces strict access control through blockchain-based whitelist.

**Components**:
- **NEAR Smart Contract**: Immutable whitelist with stake requirements
- **Whitelist Registry**: Tracks agent status (Active/Suspended/Revoked)
- **Capability Registry**: Stores declared agent capabilities
- **Stake Manager**: Handles NEAR deposits, slashes, and refunds

**Contract Structure**:
```rust
struct WhitelistedAgent {
    near_account: AccountId,
    nostr_pubkey: String,
    registered_at: u64,
    status: AgentStatus,
    stake_amount: u128,
    capabilities: Vec<String>,
    proof: OwnershipProof,
}

struct OwnershipProof {
    signature: String,
    message: String,
    verification_method: String,
    verified_at: u64,
}
```

**Whitelist Operations**:
- `register_agent`: Register new agent with stake
- `is_whitelisted`: Check if agent is allowed
- `get_agent_status`: Query agent's current status
- `revoke_agent`: Remove agent from whitelist (admin only)
- `update_capabilities`: Update agent capabilities
- `update_whitelist_status`: Open/close/invite-only modes

### 3. Authentication Layer

**Purpose**: Secure authentication and session management for agents.

**Components**:
- **Auth Gateway**: Entry point for all agent connections
- **Session Manager**: Creates and validates session tokens
- **Token Validator**: Verifies session authenticity and expiration
- **Audit Logger**: Records all authentication attempts

**Authentication Flow**:
```
1. Agent connects with Nostr pubkey
        ↓
2. System validates NEAR whitelist
        ↓
3. Ownership proof verified
        ↓
4. Session token generated
        ↓
5. Token encrypted with agent's Nostr key
        ↓
6. Agent receives encrypted token
        ↓
7. Agent decrypts and uses token
```

**Session Token Structure**:
```json
{
  "token": "base64-encoded-session-data",
  "signature": "admin-signature",
  "expires_at": 1699999999999,
  "capabilities": ["code-analysis", "security-scan"],
  "agent_id": "agent-1",
  "nonce": "random-value"
}
```

### 4. Communication Layer

**Purpose**: Secure, private relay infrastructure for agent communication.

**Components**:
- **Private Relays**: WebSocket servers with mTLS authentication
- **Relay Authentication**: Validates agent certificates
- **Message Router**: Routes messages between agents
- **Event Validator**: Ensures message format compliance

**Network Topology**:
```
┌─────────────────────────────────────────┐
│         Load Balancer                 │
└────┬────────────┬────────────┬────┘
     │            │            │
     ▼            ▼            ▼
┌────────┐   ┌────────┐   ┌────────┐
│Relay 1│   │Relay 2│   │Relay 3│
│Primary │   │Secondary│   │Backup │
└────┬───┘   └────┬───┘   └────┬───┘
     │            │            │
     └────────────┴────────────┘
                  │
                  ▼
          ┌───────────────┐
          │Agent Network  │
          └───────────────┘
```

### 5. Task Distribution Layer

**Purpose**: Efficiently distribute tasks to capable, whitelisted agents.

**Components**:
- **Task Queue**: Priority-based queue for incoming tasks
- **Task Router**: Matches tasks to agent capabilities
- **Load Balancer**: Distributes tasks based on agent availability
- **Result Aggregator**: Collects and processes agent results

**Routing Strategies**:
1. **Round-Robin**: Sequential distribution among agents
2. **Least-Loaded**: Assign to agent with lowest current load
3. **Capability-Based**: Match by specific capabilities
4. **Priority-Based**: Critical tasks go to most capable agents
5. **Multi-Agent**: Assign to multiple agents for redundancy

### 6. Orchestration Layer

**Purpose**: Manages overall swarm operations and coordination.

**Components**:
- **Swarm Orchestrator**: Central coordination of all swarm components
- **Health Monitor**: Tracks system and agent health
- **Resource Manager**: Manages agent allocation and de-allocation
- **Failure Handler**: Handles errors and retries gracefully

**Orchestrator Responsibilities**:
- Agent lifecycle management (registration, activation, deactivation)
- Task submission and monitoring
- Health checks and alerts
- Performance optimization
- Incident response coordination

## 🔒 Security Architecture

### Multi-Layer Security Model

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 7: MONITORING & AUDITING              │
│  • Real-time anomaly detection                   │
│  • Comprehensive logging                       │
│  • Automated alerts and responses               │
└─────────────────────────────────────────────────────────┘
                        ▲
┌─────────────────────────────────────────────────────────┐
│  LAYER 6: INCIDENT RESPONSE                   │
│  • Automated containment                       │
│  • Graceful degradation                      │
│  • Rollback and recovery                      │
└─────────────────────────────────────────────────────────┘
                        ▲
┌─────────────────────────────────────────────────────────┐
│  LAYER 5: APPLICATION SECURITY              │
│  • Input validation                         │
│  • Output encoding                          │
│  • Secure coding practices                    │
└─────────────────────────────────────────────────────────┘
                        ▲
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: TASK & DATA SECURITY           │
│  • Task validation and sanitization           │
│  • Result verification                    │
│  • Secure file storage                     │
└─────────────────────────────────────────────────────────┘
                        ▲
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: AGENT SECURITY                   │
│  • Sandboxing and isolation                  │
│  • Resource limits                          │
│  • Secure credential storage                │
└─────────────────────────────────────────────────────────┘
                        ▲
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: AUTHENTICATION & AUTHORIZATION     │
│  • Multi-factor authentication               │
│  • Role-based access control (RBAC)          │
│  • Session management                      │
│  • Revocation mechanisms                   │
└─────────────────────────────────────────────────────────┘
                        ▲
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: NETWORK SECURITY                │
│  • Private relay infrastructure              │
│  • mTLS encryption                         │
│  • Network segmentation                    │
│  • IP whitelisting                        │
└─────────────────────────────────────────────────────────┘
```

### Threat Mitigation

| Threat Vector | Mitigation Strategy | Implementation |
|--------------|---------------------|----------------|
| **Identity Spoofing** | Cryptographic binding between Nostr and NEAR | Ownership proof with Nostr signature |
| **Unauthorized Access** | Whitelist enforcement | NEAR smart contract whitelist check |
| **Sybil Attacks** | Financial stake requirement | 10 NEAR minimum stake per agent |
| **Replay Attacks** | Timestamp validation | Timestamp in ownership proof |
| **Man-in-the-Middle** | End-to-end encryption | NIP-44 encryption for sensitive data |
| **Denial of Service** | Rate limiting and quotas | Per-agent and per-relay rate limits |
| **Privilege Escalation** | Capability-based authorization | Agents declare and prove capabilities |
| **Data Exfiltration** | Network segmentation | Private relay network only |
| **Insider Threats** | Audit logging and revocation | Admin can revoke any agent |
| **Smart Contract Exploits** | Comprehensive testing & audits | Security reviews before deployment |

## 📊 Data Flow

### Agent Registration Flow

```
1. Agent generates Nostr keypair
        ↓
2. Agent creates NEAR account
        ↓
3. Agent prepares ownership proof:
   {
     "near_account": "agent.near",
     "nostr_pubkey": "npub1...",
     "timestamp": 1699999999,
     "action": "register_agent"
   }
        ↓
4. Agent signs message with Nostr key
        ↓
5. Agent calls NEAR contract:
   register_agent({
     nostr_pubkey: "npub1...",
     proof: { signature, message, ... },
     capabilities: ["code-analysis"],
     deposit: "10 NEAR"
   })
        ↓
6. Contract verifies ownership proof
        ↓
7. Contract stores in whitelist:
   {
     agents[agent.near] = WhitelistedAgent,
     nostr_to_near[npub1...] = agent.near
   }
        ↓
8. Agent is now whitelisted ✓
        ↓
9. Agent publishes registration to Nostr (discoverability)
```

### Authentication Flow

```
1. Agent connects to relay with Nostr pubkey
        ↓
2. Relay requests authentication
        ↓
3. Agent sends connection request (NIP-78)
        ↓
4. Orchestrator receives request
        ↓
5. Orchestrator queries NEAR whitelist:
   is_whitelisted(npub1...)
        ↓
6. NEAR returns: true/false
        ↓
7. If whitelisted:
   a. Get agent details from NEAR
   b. Generate session token
   c. Encrypt token with agent's Nostr key
   d. Send auth approval
        ↓
8. Agent receives and decrypts token
        ↓
9. Agent uses token for subsequent requests
        ↓
10. Token expires after 1 hour
```

### Task Distribution Flow

```
1. Task submitted to orchestrator
        ↓
2. Orchestrator validates task format
        ↓
3. Orchestrator queries capable agents:
   - Filter by whitelist status: Active
   - Filter by capability: matches task
   - Filter by load: < max_concurrent
        ↓
4. Orchestrator applies routing strategy
   (least-loaded / round-robin / etc)
        ↓
5. Orchestrator sends task to selected agent(s)
        ↓
6. Agent receives task (NIP-5100)
        ↓
7. Agent processes task
        ↓
8. Agent publishes result (NIP-6000)
        ↓
9. Orchestrator receives result
        ↓
10. Orchestrator validates result
        ↓
11. Orchestrator aggregates and returns result
```

## 🚀 Scalability Considerations

### Vertical Scaling

**Agent Performance**:
- Multi-threaded task processing
- Efficient event handling
- Optimized cryptographic operations

**Orchestrator Scaling**:
- Horizontal task queue distribution
- Caching for frequent queries
- Connection pooling to NEAR and relays

### Horizontal Scaling

**Relay Infrastructure**:
- Load balancer with multiple relay instances
- Geographic distribution
- Automatic failover

**Agent Scaling**:
- Stateless agent design
- Shared task queues
- Distributed state management

### Bottleneck Identification

| Component | Potential Bottleneck | Mitigation |
|-----------|---------------------|--------------|
| **NEAR Contract** | Transaction throughput | Batch operations, view-only cache |
| **Relay Network** | Message throughput | Multiple relays, compression |
| **Orchestrator** | CPU/IO | Worker pool, async operations |
| **Agent Processing** | Task execution time | Timeout enforcement, retries |
| **Database** | Query performance | Indexing, caching |

## 🔧 Technology Stack

### Blockchain Layer
- **NEAR Protocol**: Smart contracts and stake management
- **Rust**: Contract development
- **near-sdk-rs**: NEAR SDK for Rust

### Communication Layer
- **Nostr**: Decentralized messaging protocol
- **WebSocket**: Real-time communication
- **NIPs**: Nostr Implementation Possibilities
  - NIP-01: Basic protocol
  - NIP-17: Private direct messages
  - NIP-32: Labeling (reputation)
  - NIP-38: User statuses
  - NIP-51: Lists
  - NIP-78: Application-specific data

### Security Layer
- **Cryptography**: Schnorr signatures, ChaCha20 encryption
- **mTLS**: Mutual TLS for relay connections
- **HMAC**: Message authentication
- **HKDF**: Key derivation

### Application Layer
- **Node.js**: Orchestrator and SDK
- **TypeScript**: Type safety
- **near-api-js**: NEAR JavaScript SDK
- **nostr-tools**: Nostr protocol implementation

### Infrastructure Layer
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Nginx**: Load balancing
- **PostgreSQL**: State persistence
- **Redis**: Caching

## 📈 Performance Targets

### Latency
- **Agent Registration**: < 5 seconds
- **Authentication**: < 1 second
- **Task Distribution**: < 100ms
- **Result Collection**: < 1 second

### Throughput
- **Agents Supported**: 10,000+
- **Tasks Per Minute**: 1,000+
- **Messages Per Second**: 10,000+
- **Concurrent Sessions**: 5,000+

### Availability
- **System Uptime**: 99.9% (8.76 hours/month downtime)
- **Relay Uptime**: 99.95%
- **NEAR Contract**: 100% (always available)

## 🔍 Monitoring Architecture

### Metrics Collection

**Agent Metrics**:
- Registration rate
- Authentication success/failure rate
- Task completion rate
- Average processing time
- Error rate

**System Metrics**:
- CPU usage
- Memory usage
- Network I/O
- Disk I/O
- Request latency

**Business Metrics**:
- Active agents
- Tasks processed
- Successful completions
- Failed completions
- Revenue (from NEAR stakes)

### Alert Thresholds

| Metric | Warning | Critical | Action |
|---------|----------|-----------|--------|
| **Authentication Failure Rate** | > 10% | > 25% | Check whitelist, alert admin |
| **Task Failure Rate** | > 5% | > 15% | Investigate agent health |
| **System Latency** | > 2s | > 5s | Scale resources |
| **Agent Utilization** | > 90% | > 98% | Add more agents |
| **Relay Connections** | < 3 | < 2 | Check relay health |

## 🔄 State Management

### Swarm State

The swarm orchestrator maintains the following state:

```typescript
interface SwarmState {
  // Whitelist state
  whitelistedAgents: Map<string, AgentDetails>;
  
  // Active sessions
  activeSessions: Map<string, Session>;
  
  // Task queues
  taskQueues: Map<string, Task[]>;
  
  // Agent status
  agentStatus: Map<string, AgentStatus>;
  
  // Configuration
  config: SwarmConfig;
  
  // Metrics
  metrics: SwarmMetrics;
}
```

### Agent State

Each agent maintains local state:

```typescript
interface AgentState {
  // Identity
  nostrPubkey: string;
  nearAccount: string;
  
  // Capabilities
  capabilities: string[];
  
  // Session
  sessionToken?: string;
  sessionExpiresAt?: number;
  
  // Task management
  currentTasks: Map<string, Task>;
  taskHistory: Task[];
  
  // Performance
  metrics: AgentMetrics;
  
  // Configuration
  config: AgentConfig;
}
```

### Persistence Strategy

- **NEAR Contract**: Immutable whitelist and agent records
- **Nostr Events**: Agent status, task history (replayable)
- **Local Cache**: Fast lookups, temporary state
- **Database**: Persistent swarm state and metrics

## 🎯 Design Principles

1. **Security First**: Every layer enforces security
2. **Defense in Depth**: Multiple independent security layers
3. **Fail Safe**: System continues operating even with partial failures
4. **Auditability**: All actions logged and traceable
5. **Scalability**: Designed for horizontal and vertical scaling
6. **Flexibility**: Pluggable components and routing strategies
7. **Transparency**: On-chain visibility of all operations
8. **Revocability**: Instant revocation of compromised agents
9. **Performance**: Optimized for low latency and high throughput
10. **Maintainability**: Clear separation of concerns, modular design

---

**Next**: See [Security Architecture](./security.md) for detailed security implementation.