# JavaScript SDK API Reference

Complete API reference for the Nostr-Backed NEAR Whitelist JavaScript SDK.

## 📋 Installation

### NPM Installation

```bash
npm install @nostr-near/sdk
```

### Yarn Installation

```bash
yarn add @nostr-near/sdk
```

### Import

```javascript
// ES Modules
import { NostrBackedNEARRegistration } from '@nostr-near/sdk';
import { WhitelistAuthenticator } from '@nostr-near/sdk';
import { WhitelistedAgentSwarm } from '@nostr-near/sdk';

// CommonJS
const { NostrBackedNEARRegistration, WhitelistAuthenticator, WhitelistedAgentSwarm } = require('@nostr-near/sdk');
```

---

## 🏗️ Core Classes

### `NostrBackedNEARRegistration`

Manages agent registration with Nostr-backed NEAR accounts.

#### Constructor

```typescript
new NostrBackedNEARRegistration(config: RegistrationConfig)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nearAccount` | `NearAccount` | Yes | NEAR account configuration |
| `nostrKey` | `NostrKeyPair` | Yes | Nostr keypair (public + private) |
| `whitelistContract` | `string` | Yes | NEAR contract address |
| `options` | `RegistrationOptions` | No | Optional registration settings |

**Example:**

```javascript
import { NostrBackedNEARRegistration } from '@nostr-near/sdk';
import * as nearAPI from 'near-api-js';

const registration = new NostrBackedNEARRegistration({
  nearAccount: await nearAPI.connect({
    accountId: 'agent-1.near',
    networkId: 'mainnet',
    keyStore: new nearAPI.keyStores.BrowserLocalStorageKeyStore()
  }),
  nostrKey: {
    publicKey: 'npub1xyz...',
    privateKey: 'nsec1abc...'
  },
  whitelistContract: 'agent-whitelist.near',
  options: {
    stakeAmount: '10',
    autoPublishNostr: true,
    timeout: 30000
  }
});
```

#### Methods

##### `registerAgent()`

Register an agent with the NEAR whitelist.

```typescript
registerAgent(capabilities: string[], options?: RegistrationOptions): Promise<RegistrationResult>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `capabilities` | `string[]` | Yes | Agent capabilities (e.g., ["code-analysis", "security-scan"]) |
| `options` | `RegistrationOptions` | No | Optional registration options |

**Returns:**

```typescript
interface RegistrationResult {
  nearAccount: string;           // NEAR account ID
  nostrPubkey: string;          // Nostr public key
  stakeAmount: string;           // NEAR amount staked
  status: 'Active';             // Registration status
  capabilities: string[];        // Registered capabilities
  transactionHash: string;       // NEAR transaction hash
  nostrEventId: string;        // Nostr event ID (if published)
  registeredAt: number;          // Registration timestamp
}
```

**Example:**

```javascript
const result = await registration.registerAgent([
  'code-analysis',
  'security-scan',
  'data-processing',
  'optimization'
], {
  stakeAmount: '10',
  autoPublishNostr: true,
  metadata: {
    name: 'Python Code Reviewer',
    description: 'AI-powered code analysis agent',
    website: 'https://agent.example.com'
  }
});

console.log('✓ Agent registered:', result.nearAccount);
console.log('  Transaction:', result.transactionHash);
console.log('  Capabilities:', result.capabilities.join(', '));
```

##### `updateCapabilities()`

Update agent's capabilities.

```typescript
updateCapabilities(capabilities: string[]): Promise<UpdateResult>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `capabilities` | `string[]` | Yes | New capabilities list |

**Returns:**

```typescript
interface UpdateResult {
  success: boolean;
  transactionHash: string;
  previousCapabilities: string[];
  newCapabilities: string[];
  updatedAt: number;
}
```

**Example:**

```javascript
const result = await registration.updateCapabilities([
  'code-analysis',
  'security-scan',
  'data-processing',
  'optimization',
  'testing'
]);

console.log('✓ Capabilities updated');
console.log('  Previous:', result.previousCapabilities);
console.log('  New:', result.newCapabilities);
```

##### `revokeRegistration()`

Revoke agent's registration and unstake.

```typescript
revokeRegistration(reason?: string): Promise<RevocationResult>
```

**Returns:**

```typescript
interface RevocationResult {
  success: boolean;
  stakeRefunded: string;      // NEAR amount refunded
  slashedAmount: string;       // NEAR amount slashed (10%)
  transactionHash: string;
  revokedAt: number;
}
```

---

### `WhitelistAuthenticator`

Handles agent authentication with NEAR whitelist verification.

#### Constructor

```typescript
new WhitelistAuthenticator(config: AuthConfig)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `whitelistContract` | `string` | Yes | NEAR contract address |
| `nearAccount` | `NearAccount` | Yes | NEAR account for whitelist queries |
| `options` | `AuthOptions` | No | Optional authentication settings |

**Example:**

```javascript
import { WhitelistAuthenticator } from '@nostr-near/sdk';

const authenticator = new WhitelistAuthenticator({
  whitelistContract: 'agent-whitelist.near',
  nearAccount: nearConnection,
  options: {
    sessionTimeout: 3600,      // 1 hour
    maxFailedAttempts: 5,
    enableMetrics: true
  }
});
```

#### Methods

##### `authenticateAgent()`

Authenticate an agent using Nostr pubkey.

```typescript
authenticateAgent(nostrPubkey: string): Promise<AuthResult>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostrPubkey` | `string` | Yes | Nostr public key to authenticate |

**Returns:**

```typescript
interface AuthResult {
  authenticated: boolean;
  agentDetails: AgentDetails | null;
  sessionToken: SessionToken | null;
  error: string | null;
}

interface AgentDetails {
  nearAccount: string;
  nostrPubkey: string;
  status: 'Active' | 'Suspended' | 'Revoked' | 'Pending';
  capabilities: string[];
  stakeAmount: string;
  registeredAt: number;
  lastActive: number;
  metrics: AgentMetrics;
}

interface SessionToken {
  token: string;              // Base64-encoded token
  signature: string;           // Admin signature
  expiresAt: number;           // Expiration timestamp
  capabilities: string[];
  agentId: string;
  nonce: string;
}

interface AgentMetrics {
  tasksCompleted: number;
  tasksFailed: number;
  totalEarned: string;
  successRate: number;
  avgResponseTime: number;
  lastUpdated: number;
}
```

**Example:**

```javascript
const result = await authenticator.authenticateAgent('npub1xyz...');

if (result.authenticated) {
  console.log('✓ Agent authenticated');
  console.log('  NEAR Account:', result.agentDetails.nearAccount);
  console.log('  Status:', result.agentDetails.status);
  console.log('  Capabilities:', result.agentDetails.capabilities.join(', '));
  console.log('  Session Token:', result.sessionToken.token);
  console.log('  Expires:', new Date(result.sessionToken.expiresAt).toISOString());
  
  // Use session token for subsequent requests
  const session = result.sessionToken;
} else {
  console.log('✗ Authentication failed:', result.error);
}
```

##### `validateSession()`

Validate a session token.

```typescript
validateSession(sessionToken: string): Promise<ValidationResult>
```

**Returns:**

```typescript
interface ValidationResult {
  valid: boolean;
  agentDetails: AgentDetails | null;
  expiresIn: number;           // Seconds until expiration
  error: string | null;
}
```

**Example:**

```javascript
const result = await authenticator.validateSession(sessionToken);

if (result.valid) {
  console.log('✓ Session is valid');
  console.log('  Expires in', result.expiresIn, 'seconds');
  console.log('  Agent:', result.agentDetails.nearAccount);
} else {
  console.log('✗ Session invalid:', result.error);
}
```

##### `refreshSession()`

Refresh an existing session token.

```typescript
refreshSession(currentToken: string): Promise<SessionToken>
```

**Returns:** New `SessionToken` with extended expiration.

**Example:**

```javascript
const newToken = await authenticator.refreshSession(currentToken);

console.log('✓ Session refreshed');
console.log('  New expires:', new Date(newToken.expiresAt).toISOString());
```

##### `revokeSession()`

Revoke a session token.

```typescript
revokeSession(sessionToken: string): Promise<RevokeResult>
```

**Returns:**

```typescript
interface RevokeResult {
  success: boolean;
  revokedAt: number;
  message: string;
}
```

---

### `WhitelistedAgentSwarm`

Main orchestrator for whitelisted agent swarm operations.

#### Constructor

```typescript
new WhitelistedAgentSwarm(config: SwarmConfig)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `whitelistContract` | `string` | Yes | NEAR contract address |
| `nearAccount` | `NearAccount` | Yes | NEAR account for swarm operations |
| `privateRelays` | `string[]` | Yes | Private relay URLs |
| `options` | `SwarmOptions` | No | Optional swarm settings |

**Example:**

```javascript
import { WhitelistedAgentSwarm } from '@nostr-near/sdk';

const swarm = new WhitelistedAgentSwarm({
  whitelistContract: 'agent-whitelist.near',
  nearAccount: nearConnection,
  privateRelays: [
    'wss://internal-relay-1.company.com',
    'wss://internal-relay-2.company.com',
    'wss://internal-relay-3.company.com'
  ],
  options: {
    maxConcurrentTasks: 1000,
    taskTimeout: 3600000,      // 1 hour
    retryAttempts: 3,
    retryDelay: 5000,
    enableMonitoring: true,
    healthCheckInterval: 60000,
    taskRoutingStrategy: 'least-loaded',
    metrics: {
      enabled: true,
      exportInterval: 300000  // 5 minutes
    }
  }
});
```

#### Methods

##### `initialize()`

Initialize the swarm and connect to components.

```typescript
initialize(): Promise<InitializeResult>
```

**Returns:**

```typescript
interface InitializeResult {
  success: boolean;
  connectedRelays: string[];
  registeredAgents: number;
  ready: boolean;
  error: string | null;
}
```

**Example:**

```javascript
const result = await swarm.initialize();

if (result.success) {
  console.log('✓ Swarm initialized');
  console.log('  Connected relays:', result.connectedRelays.length);
  console.log('  Registered agents:', result.registeredAgents);
  console.log('  Ready:', result.ready);
} else {
  console.log('✗ Initialization failed:', result.error);
}
```

##### `submitWorkload()`

Submit a workload to the swarm.

```typescript
submitWorkload(workload: Workload): Promise<WorkloadResult>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `workload` | `Workload` | Yes | Workload configuration |

**Workload Interface:**

```typescript
interface Workload {
  id: string;
  name: string;
  type: string;
  priority: 'critical' | 'high' | 'normal' | 'low';
  payload: any;
  requirements?: TaskRequirements;
  dependencies?: string[];
  timeout?: number;
}
```

**Returns:**

```typescript
interface WorkloadResult {
  workloadId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  tasks: Task[];
  estimatedCompletion: number;
  startedAt: number;
}
```

**Example:**

```javascript
const workload = {
  id: 'batch-123',
  name: 'Daily Code Review Batch',
  type: 'code-review',
  priority: 'high',
  payload: {
    repository: 'github.com/company/main-app',
    branch: 'develop',
    commit: 'abc123',
    files: ['app.js', 'utils.js', 'api.js']
  },
  requirements: {
    capabilities: ['code-analysis', 'security-scan'],
    minAgents: 1,
    maxAgents: 3
  },
  timeout: 3600000  // 1 hour
};

const result = await swarm.submitWorkload(workload);

console.log('✓ Workload submitted:', result.workloadId);
console.log('  Tasks created:', result.tasks.length);
console.log('  Estimated completion:', new Date(result.estimatedCompletion).toISOString());
```

##### `getAgentStatus()`

Get status of specific agent.

```typescript
getAgentStatus(nostrPubkey: string): Promise<AgentStatusResult>
```

**Returns:**

```typescript
interface AgentStatusResult {
  agentId: string;
  nearAccount: string;
  nostrPubkey: string;
  status: 'idle' | 'busy' | 'error' | 'offline';
  currentTasks: number;
  maxConcurrentTasks: number;
  capacity: number;
  lastHeartbeat: number;
  metrics: AgentMetrics;
}
```

**Example:**

```javascript
const status = await swarm.getAgentStatus('npub1xyz...');

console.log('Agent Status:', status.agentId);
console.log('  Status:', status.status);
console.log('  Current tasks:', status.currentTasks);
console.log('  Capacity:', (status.capacity * 100).toFixed(1) + '%');
console.log('  Last heartbeat:', new Date(status.lastHeartbeat).toISOString());
```

##### `getAvailableAgents()`

Get list of available agents for specific capability.

```typescript
getAvailableAgents(capability?: string, filters?: AgentFilters): Promise<Agent[]>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `capability` | `string` | No | Filter by specific capability |
| `filters` | `AgentFilters` | No | Additional filters |

**AgentFilters Interface:**

```typescript
interface AgentFilters {
  minSuccessRate?: number;      // Minimum success rate (0-100)
  maxLoad?: number;            // Maximum current load
  minStake?: string;           // Minimum stake amount
  status?: string[];            // Allowed statuses
}
```

**Example:**

```javascript
// Get agents with code-analysis capability
const agents = await swarm.getAvailableAgents('code-analysis', {
  minSuccessRate: 80,
  maxLoad: 5,
  status: ['idle', 'busy']
});

console.log(`Found ${agents.length} available agents:`);
agents.forEach(agent => {
  console.log(`  ${agent.nearAccount} (${agent.status})`);
  console.log(`    Capabilities: ${agent.capabilities.join(', ')}`);
  console.log(`    Success rate: ${agent.metrics.successRate}%`);
  console.log(`    Load: ${agent.currentLoad}/${agent.maxConcurrentTasks}`);
});
```

##### `getSwarmHealth()`

Get overall swarm health metrics.

```typescript
getSwarmHealth(): Promise<SwarmHealth>
```

**Returns:**

```typescript
interface SwarmHealth {
  totalAgents: number;
  activeAgents: number;
  idleAgents: number;
  busyAgents: number;
  errorAgents: number;
  totalCapacity: number;
  currentLoad: number;
  utilization: number;
  uptime: number;
  metrics: SwarmMetrics;
}

interface SwarmMetrics {
  tasksPerMinute: number;
  averageProcessingTime: number;
  successRate: number;
  errorRate: number;
  tasksProcessedToday: number;
  tasksCompletedToday: number;
  tasksFailedToday: number;
}
```

**Example:**

```javascript
const health = await swarm.getSwarmHealth();

console.log('Swarm Health:');
console.log(`  Total agents: ${health.totalAgents}`);
console.log(`  Active agents: ${health.activeAgents}`);
console.log(`  Utilization: ${health.utilization.toFixed(1)}%`);
console.log(`  Uptime: ${(health.uptime / 3600).toFixed(2)} hours`);
console.log('');
console.log('Metrics:');
console.log(`  Tasks/min: ${health.metrics.tasksPerMinute}`);
console.log(`  Avg processing time: ${health.metrics.averageProcessingTime}ms`);
console.log(`  Success rate: ${health.metrics.successRate}%`);
```

##### `shutdown()`

Gracefully shutdown the swarm.

```typescript
shutdown(): Promise<ShutdownResult>
```

**Returns:**

```typescript
interface ShutdownResult {
  success: boolean;
  tasksDrained: boolean;
  agentsNotified: number;
  relaysDisconnected: number;
  shutdownAt: number;
}
```

**Example:**

```javascript
console.log('Shutting down swarm...');
const result = await swarm.shutdown();

if (result.success) {
  console.log('✓ Swarm shutdown complete');
  console.log('  Tasks drained:', result.tasksDrained);
  console.log('  Agents notified:', result.agentsNotified);
  console.log('  Relays disconnected:', result.relaysDisconnected);
}
```

---

## 🔧 Configuration Types

### `RegistrationConfig`

```typescript
interface RegistrationConfig {
  nearAccount: NearAccount;
  nostrKey: NostrKeyPair;
  whitelistContract: string;
  options?: RegistrationOptions;
}

interface RegistrationOptions {
  stakeAmount?: string;           // NEAR amount to stake (default: "10")
  autoPublishNostr?: boolean;     // Publish to Nostr automatically (default: true)
  timeout?: number;                // Request timeout in ms (default: 30000)
  metadata?: AgentMetadata;         // Additional agent metadata
}

interface NostrKeyPair {
  publicKey: string;
  privateKey: string;
}

interface AgentMetadata {
  name?: string;
  description?: string;
  website?: string;
  version?: string;
  [key: string]: any;            // Additional custom fields
}
```

### `AuthConfig`

```typescript
interface AuthConfig {
  whitelistContract: string;
  nearAccount: NearAccount;
  options?: AuthOptions;
}

interface AuthOptions {
  sessionTimeout?: number;          // Session timeout in seconds (default: 3600)
  maxFailedAttempts?: number;       // Max failed attempts before lockout (default: 5)
  lockoutDuration?: number;        // Lockout duration in seconds (default: 300)
  enableMetrics?: boolean;          // Enable auth metrics (default: true)
  requireMFA?: boolean;           // Require MFA for sensitive operations (default: false)
}
```

### `SwarmConfig`

```typescript
interface SwarmConfig {
  whitelistContract: string;
  nearAccount: NearAccount;
  privateRelays: string[];
  options?: SwarmOptions;
}

interface SwarmOptions {
  maxConcurrentTasks?: number;     // Max concurrent tasks (default: 1000)
  taskTimeout?: number;            // Task timeout in ms (default: 3600000)
  retryAttempts?: number;          // Retry attempts (default: 3)
  retryDelay?: number;             // Retry delay in ms (default: 5000)
  enableMonitoring?: boolean;       // Enable health monitoring (default: true)
  healthCheckInterval?: number;    // Health check interval in ms (default: 60000)
  taskRoutingStrategy?: 'round-robin' | 'least-loaded' | 'capability-based' | 'priority-based';
  metrics?: MetricsOptions;
}

interface MetricsOptions {
  enabled?: boolean;
  exportInterval?: number;         // Metrics export interval in ms (default: 300000)
  exportFormat?: 'json' | 'csv' | 'prometheus';
  exportDestination?: string;      // Export destination (file path or API endpoint)
}
```

---

## 🎯 Event Handling

### Registration Events

```typescript
registration.on('registering', (data) => {
  console.log('Registering agent...', data);
});

registration.on('registered', (result) => {
  console.log('Agent registered:', result);
});

registration.on('error', (error) => {
  console.error('Registration error:', error);
});

registration.on('stake-deposited', (data) => {
  console.log('Stake deposited:', data.amount);
});
```

### Authentication Events

```typescript
authenticator.on('authenticating', (nostrPubkey) => {
  console.log('Authenticating:', nostrPubkey);
});

authenticator.on('authenticated', (result) => {
  console.log('Agent authenticated:', result.agentDetails);
});

authenticator.on('failed', (error) => {
  console.error('Auth failed:', error);
});

authenticator.on('session-created', (session) => {
  console.log('Session created:', session.token);
});

authenticator.on('session-expired', (session) => {
  console.log('Session expired:', session.agentId);
});
```

### Swarm Events

```typescript
swarm.on('initialized', (result) => {
  console.log('Swarm initialized:', result);
});

swarm.on('task-submitted', (task) => {
  console.log('Task submitted:', task.id);
});

swarm.on('task-assigned', (assignment) => {
  console.log('Task assigned:', assignment.taskId, '→', assignment.agentId);
});

swarm.on('task-completed', (result) => {
  console.log('Task completed:', result.taskId);
});

swarm.on('task-failed', (error) => {
  console.error('Task failed:', error.taskId, error.error);
});

swarm.on('agent-connected', (agent) => {
  console.log('Agent connected:', agent.nearAccount);
});

swarm.on('agent-disconnected', (agent) => {
  console.log('Agent disconnected:', agent.nearAccount);
});

swarm.on('health-check', (health) => {
  console.log('Health check:', health.utilization + '%');
});

swarm.on('shutdown', (result) => {
  console.log('Swarm shutting down...');
});
```

---

## 🚨 Error Handling

### Error Types

```typescript
// Network Errors
class NetworkError extends Error {
  code: 'NETWORK_ERROR' | 'TIMEOUT' | 'CONNECTION_FAILED';
  relay?: string;
  retryable: boolean;
}

// NEAR Errors
class NEARError extends Error {
  code: 'INSUFFICIENT_BALANCE' | 'CONTRACT_CALL_FAILED' | 'INVALID_ACCOUNT' | 'TRANSACTION_FAILED';
  transactionHash?: string;
  retryable: boolean;
}

// Nostr Errors
class NostrError extends Error {
  code: 'INVALID_KEY' | 'SIGNATURE_FAILED' | 'EVENT_PUBLISH_FAILED' | 'SUBSCRIPTION_FAILED';
  event?: object;
  retryable: boolean;
}

// Whitelist Errors
class WhitelistError extends Error {
  code: 'NOT_WHITELISTED' | 'AGENT_SUSPENDED' | 'AGENT_REVOKED' | 'INVALID_PROOF';
  agentId?: string;
  reason?: string;
}

// Swarm Errors
class SwarmError extends Error {
  code: 'SWARM_NOT_INITIALIZED' | 'NO_AVAILABLE_AGENTS' | 'TASK_TIMEOUT' | 'MAX_RETRY_EXCEEDED';
  taskId?: string;
  agentId?: string;
  retryable: boolean;
}
```

### Error Handling Example

```javascript
try {
  const result = await swarm.submitWorkload(workload);
  console.log('Workload submitted:', result.workloadId);
} catch (error) {
  if (error instanceof WhitelistError) {
    console.error('Whitelist error:', error.code, error.reason);
    
    switch (error.code) {
      case 'NOT_WHITELISTED':
        console.error('Agent is not whitelisted');
        break;
      case 'AGENT_SUSPENDED':
        console.error('Agent is suspended');
        break;
      case 'AGENT_REVOKED':
        console.error('Agent has been revoked');
        break;
    }
  } else if (error instanceof NetworkError) {
    console.error('Network error:', error.code);
    
    if (error.retryable) {
      console.log('Retrying...');
      await swarm.submitWorkload(workload);
    }
  } else if (error instanceof NEARError) {
    console.error('NEAR error:', error.code);
    console.error('Transaction:', error.transactionHash);
  } else {
    console.error('Unexpected error:', error);
  }
}
```

---

## 💡 Best Practices

### 1. Connection Management

```javascript
// Reuse connections
const swarm = new WhitelistedAgentSwarm(config);
await swarm.initialize();

// Don't create multiple swarm instances
// Process multiple workloads with single instance
```

### 2. Error Handling

```javascript
// Always handle errors
try {
  const result = await swarm.submitWorkload(workload);
} catch (error) {
  // Log error
  console.error('Failed to submit workload:', error);
  
  // Determine if retryable
  if (error.retryable) {
    // Implement retry logic
    return await retryWithBackoff(workload, 3);
  }
  
  // Notify monitoring
  await notifyErrorMonitoring(error);
}
```

### 3. Session Management

```javascript
// Store session tokens securely
async function storeSession(sessionToken: string) {
  // Use secure storage
  await secureStorage.setItem('agent_session', sessionToken);
  
  // Set expiration check
  setTimeout(async () => {
    await secureStorage.removeItem('agent_session');
    console.log('Session expired');
  }, 3600000); // 1 hour
}

// Always validate before using
async function getSession() {
  const token = await secureStorage.getItem('agent_session');
  
  if (!token) {
    throw new Error('No active session');
  }
  
  const validation = await authenticator.validateSession(token);
  if (!validation.valid) {
    throw new Error('Session expired or invalid');
  }
  
  return token;
}
```

### 4. Resource Cleanup

```javascript
// Always cleanup on shutdown
process.on('SIGINT', async () => {
  console.log('Received SIGINT, shutting down...');
  
  try {
    const result = await swarm.shutdown();
    console.log('✓ Swarm shutdown complete');
    console.log('  Tasks drained:', result.tasksDrained);
    console.log('  Agents notified:', result.agentsNotified);
  } catch (error) {
    console.error('Error during shutdown:', error);
  } finally {
    process.exit(0);
  }
});

process.on('uncaughtException', async (error) => {
  console.error('Uncaught exception:', error);
  await notifyErrorMonitoring(error);
  await gracefulShutdown();
  process.exit(1);
});
```

### 5. Performance Optimization

```javascript
// Use async/await for non-blocking operations
async function processWorkloads(workloads: Workload[]) {
  // Process in parallel where possible
  const results = await Promise.all(
    workloads.map(workload => swarm.submitWorkload(workload))
  );
  
  return results;
}

// Use batching for multiple agent queries
async function getMultipleAgentStatuses(nostrPubkeys: string[]) {
  // Batch queries to reduce NEAR calls
  const results = await Promise.all(
    nostrPubkeys.map(pubkey => swarm.getAgentStatus(pubkey))
  );
  
  return results;
}

// Implement caching for frequent queries
const agentStatusCache = new LRUCache<string, AgentStatus>({
  max: 1000,
  ttl: 60000  // 1 minute
});

async function getCachedAgentStatus(nostrPubkey: string) {
  if (agentStatusCache.has(nostrPubkey)) {
    return agentStatusCache.get(nostrPubkey);
  }
  
  const status = await swarm.getAgentStatus(nostrPubkey);
  agentStatusCache.set(nostrPubkey, status);
  
  return status;
}
```

---

## 📊 Monitoring & Metrics

### Built-in Metrics

```javascript
// Enable metrics in config
const swarm = new WhitelistedAgentSwarm({
  // ... other config
  options: {
    metrics: {
      enabled: true,
      exportInterval: 300000,      // 5 minutes
      exportFormat: 'json',
      exportDestination: './metrics'
    }
  }
});

// Access metrics programmatically
const metrics = await swarm.getMetrics();
console.log('Metrics:', metrics);
```

### Custom Metrics

```javascript
// Subscribe to swarm events and track custom metrics
swarm.on('task-completed', (result) => {
  customMetrics.track('task_completion_time', {
    taskId: result.taskId,
    agentId: result.agentId,
    duration: result.duration
  });
});

swarm.on('agent-connected', (agent) => {
  customMetrics.track('agent_connection', {
    agentId: agent.nearAccount,
    timestamp: Date.now()
  });
});

// Export metrics
setInterval(async () => {
  const metrics = customMetrics.export();
  await uploadMetrics(metrics);
}, 60000); // Every minute
```

---

## 🧪 Testing

### Unit Testing

```javascript
import { describe, it, expect, beforeEach } from '@jest/globals';
import { NostrBackedNEARRegistration } from '@nostr-near/sdk';

describe('NostrBackedNEARRegistration', () => {
  let registration;
  let mockNearAccount;
  let mockNostrKey;

  beforeEach(() => {
    mockNearAccount = createMockNearAccount();
    mockNostrKey = createMockNostrKey();
    
    registration = new NostrBackedNEARRegistration({
      nearAccount: mockNearAccount,
      nostrKey: mockNostrKey,
      whitelistContract: 'agent-whitelist.test'
    });
  });

  it('should register agent successfully', async () => {
    mockNearAccount.functionCall.mockResolvedValue({
      transactionHash: 'abc123',
      success: true
    });

    const result = await registration.registerAgent(['code-analysis']);

    expect(result.success).toBe(true);
    expect(result.nearAccount).toBe(mockNearAccount.accountId);
    expect(result.capabilities).toEqual(['code-analysis']);
  });

  it('should handle insufficient stake', async () => {
    mockNearAccount.functionCall.mockRejectedValue(
      new Error('Insufficient stake amount')
    );

    await expect(
      registration.registerAgent(['code-analysis'])
    ).rejects.toThrow('Insufficient stake amount');
  });

  it('should publish Nostr event if configured', async () => {
    const mockNostrClient = createMockNostrClient();
    
    const result = await registration.registerAgent(['code-analysis'], {
      autoPublishNostr: true
    });

    expect(mockNostrClient.publishEvent).toHaveBeenCalled();
  });
});
```

### Integration Testing

```javascript
import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { WhitelistedAgentSwarm } from '@nostr-near/sdk';
import { setupTestEnvironment, teardownTestEnvironment } from './test-helpers';

describe('WhitelistedAgentSwarm Integration', () => {
  let swarm;
  let testNearAccount;

  beforeAll(async () => {
    // Setup test environment
    await setupTestEnvironment();
    
    // Create test NEAR account
    testNearAccount = await createTestAccount('test-swarm');
    
    // Deploy test contract
    const contractId = await deployTestContract();
    
    // Initialize swarm
    swarm = new WhitelistedAgentSwarm({
      whitelistContract: contractId,
      nearAccount: testNearAccount,
      privateRelays: ['wss://test-relay.near'],
      options: {
        taskTimeout: 30000,
        maxConcurrentTasks: 10
      }
    });
  });

  afterAll(async () => {
    // Cleanup
    await swarm.shutdown();
    await teardownTestEnvironment();
  });

  it('should initialize swarm and connect to relays', async () => {
    const result = await swarm.initialize();

    expect(result.success).toBe(true);
    expect(result.connectedRelays.length).toBe(1);
  });

  it('should submit workload and receive result', async () => {
    const workload = {
      id: 'test-workload-1',
      name: 'Test Workload',
      type: 'code-analysis',
      priority: 'normal',
      payload: { code: 'function test() { return true; }' }
    };

    const submitResult = await swarm.submitWorkload(workload);
    expect(submitResult.status).toBe('queued');

    // Wait for completion
    const completed = await waitForTaskCompletion(submitResult.workloadId, 30000);
    expect(completed.status).toBe('completed');
    expect(completed.result).toBeDefined();
  }, 60000); // 60 second timeout
});
```

---

## 📞 Advanced Usage

### Custom Task Routing

```javascript
// Implement custom routing strategy
class CustomTaskRouter {
  async route(task: Task, agents: Agent[]): Promise<Assignment> {
    // Custom routing logic
    const capableAgents = agents.filter(a => 
      a.capabilities.includes(task.type) && 
      a.currentLoad < a.maxConcurrentTasks
    );

    // Sort by custom priority
    capableAgents.sort((a, b) => {
      return b.metrics.successRate - a.metrics.successRate;
    });

    // Select top 3 for redundancy
    const selected = capableAgents.slice(0, 3);

    return {
      taskId: task.id,
      assignedTo: selected.map(a => a.nearAccount),
      strategy: 'custom-redundancy'
    };
  }
}

// Use custom router
const swarm = new WhitelistedAgentSwarm({
  // ... config
  options: {
    taskRoutingStrategy: 'custom',
    customRouter: new CustomTaskRouter()
  }
});
```

### Distributed Task Processing

```javascript
// Split large workload across multiple agents
async function distributeLargeWorkload(workload: Workload) {
  const splitCount = 5;
  const chunkSize = Math.ceil(workload.payload.items.length / splitCount);

  const chunks = [];
  for (let i = 0; i < workload.payload.items.length; i += chunkSize) {
    chunks.push(workload.payload.items.slice(i, i + chunkSize));
  }

  const subWorkloads = chunks.map((chunk, index) => ({
    id: `${workload.id}-${index}`,
    name: `${workload.name} (Part ${index + 1})`,
    type: workload.type,
    priority: workload.priority,
    payload: { items: chunk },
    dependencies: [],
    parentWorkload: workload.id
  }));

  // Submit all sub-workloads
  const results = await Promise.all(
    subWorkloads.map(sub => swarm.submitWorkload(sub))
  );

  return results;
}
```

### Fault-Tolerant Processing

```javascript
// Implement fault tolerance with automatic retry and fallback
async function processWithFaultTolerance(workload: Workload, maxRetries: number = 3) {
  let lastError;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const result = await swarm.submitWorkload(workload);
      
      // Wait for completion
      const completed = await waitForTaskCompletion(result.workloadId, 60000);
      
      if (completed.status === 'completed') {
        console.log(`✓ Task completed on attempt ${attempt}`);
        return completed;
      }
    } catch (error) {
      console.error(`Attempt ${attempt} failed:`, error);
      lastError = error;
      
      // Wait before retry
      if (attempt < maxRetries) {
        await sleep(Math.pow(2, attempt) * 1000); // Exponential backoff
      }
    }
  }

  // All retries failed, use fallback
  console.error('All retries failed, using fallback agent');
  return await processWithFallbackAgent(workload);
}
```

---

## 📄 Additional Resources

- [NEAR SDK Documentation](https://docs.near.org/sdk/develop/integration)
- [Nostr Protocol](https://github.com/nostr-protocol/nips)
- [TypeScript Definitions](./typescript-definitions.md)
- [Migration Guide](../../guides/migration.md)
- [Contributing Guide](../../CONTRIBUTING.md)

---

## 🆘 Support

For SDK-related issues:
- GitHub Issues: [Create Issue](https://github.com/mpp-near/issues/new)
- Discord: [mpp-near.dev](https://discord.gg/mpp-near)
- Documentation: [docs.mpp-near.com](https://docs.mpp-near.com)