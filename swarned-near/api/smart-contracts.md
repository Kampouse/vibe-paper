# NEAR Smart Contract API Reference

Complete API reference for the Agent Whitelist smart contract on NEAR blockchain.

## 📋 Contract Overview

The `AgentWhitelist` contract manages agent identities, whitelisting, and stake management for secure agent swarms.

### Contract Address

- **Testnet**: `agent-whitelist.testnet`
- **Mainnet**: `agent-whitelist.near`

### Deployment Details

```bash
# Build contract
cargo build --target wasm32-unknown-unknown --release

# Deploy
near deploy target/wasm32-unknown-unknown/release/agent_whitelist.wasm \
  --accountId my-whitelist.near \
  --init new admin.near
```

---

## 🏗️ Data Structures

### WhitelistedAgent

```rust
pub struct WhitelistedAgent {
    pub near_account: AccountId,      // NEAR account identifier
    pub nostr_pubkey: String,         // Nostr public key (npub)
    pub registered_at: u64,          // Registration timestamp (nanoseconds)
    pub status: AgentStatus,          // Current agent status
    pub stake_amount: u128,         // NEAR staked (yoctoNEAR)
    pub capabilities: Vec<String>,    // Declared agent capabilities
    pub proof: OwnershipProof,         // Cryptographic ownership proof
    pub last_active: u64,           // Last activity timestamp
    pub metrics: AgentMetrics,        // Performance metrics
}
```

### OwnershipProof

```rust
pub struct OwnershipProof {
    pub signature: String,           // Nostr signature
    pub message: String,             // Signed message content
    pub verification_method: String, // "nostr-signature"
    pub verified_at: u64,            // Verification timestamp
}
```

### AgentStatus

```rust
pub enum AgentStatus {
    Active,      // Agent is fully operational
    Suspended,   // Temporarily disabled
    Revoked,     // Permanently removed
    Pending,     // Awaiting approval (invite-only mode)
}
```

### AgentMetrics

```rust
pub struct AgentMetrics {
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub total_earned: u128,     // Total NEAR earned
    pub success_rate: u32,       // Success rate (0-100)
    pub avg_response_time: u64,  // Average response time (ms)
    pub last_updated: u64,
}
```

### WhitelistStatus

```rust
pub enum WhitelistStatus {
    Open,        // Anyone can register
    InviteOnly,  // Only invited agents can register
    Closed,      // No new registrations allowed
}
```

---

## 🔧 Contract Methods

### `register_agent`

Register a new agent with Nostr-backed identity and stake.

```rust
#[payable]
pub fn register_agent(
    &mut self,
    nostr_pubkey: String,
    proof: OwnershipProof,
    capabilities: Vec<String>
) -> WhitelistedAgent
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostr_pubkey` | `String` | Yes | Nostr public key (npub format) |
| `proof` | `OwnershipProof` | Yes | Cryptographic proof of Nostr↔NEAR ownership |
| `capabilities` | `Vec<String>` | Yes | Agent capabilities (e.g., ["code-analysis", "security-scan"]) |

**Attached Deposit:**
- **Minimum**: `10_000_000_000_000_000_000` (10 NEAR)
- **Recommended**: `100_000_000_000_000_000_000` (100 NEAR)
- **Purpose**: Stake for whitelisting and "skin in the game"

**Returns:**
- `WhitelistedAgent` - The registered agent information

**Example:**

```javascript
const result = await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'register_agent',
  args: {
    nostr_pubkey: 'npub1xyz...',
    proof: {
      signature: 'a3c6ce632b145c0869423c1afaff4a6d764a9b64dedaf15f170b944ead67227518a72e455567ca1c2a0d187832cecbde7ed478395ec4c95dd3e71749ed66c480',
      message: '{"near_account":"agent.near","nostr_pubkey":"npub1xyz...","timestamp":1699999999,"action":"register_agent"}',
      verification_method: 'nostr-signature',
      verified_at: 1699999999
    },
    capabilities: ['code-analysis', 'security-scan', 'data-processing']
  },
  attachedDeposit: utils.format.parseNearAmount('10')
})
```

---

### `is_whitelisted`

Check if an agent (by Nostr pubkey) is whitelisted and active.

```rust
pub fn is_whitelisted(&self, nostr_pubkey: String) -> bool
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostr_pubkey` | `String` | Yes | Nostr public key to check |

**Returns:**
- `bool` - `true` if agent is whitelisted and active, `false` otherwise

**Example:**

```javascript
const isWhitelisted = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'is_whitelisted',
  args: {
    nostr_pubkey: 'npub1xyz...'
  }
})

if (isWhitelisted) {
  console.log('✓ Agent is whitelisted')
} else {
  console.log('✗ Agent is not whitelisted')
}
```

---

### `get_agent_status`

Get detailed status information for an agent.

```rust
pub fn get_agent_status(&self, nostr_pubkey: String) -> Option<String>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostr_pubkey` | `String` | Yes | Nostr public key |

**Returns:**
- `Option<String>` - Agent status string ("Active", "Suspended", "Revoked", "Pending") or `null` if not found

**Example:**

```javascript
const status = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'get_agent_status',
  args: {
    nostr_pubkey: 'npub1xyz...'
  }
})

console.log(`Agent status: ${status}`)
// Output: "Active"
```

---

### `get_agent_info`

Get complete agent information including metrics.

```rust
pub fn get_agent_info(&self, nostr_pubkey: String) -> Option<WhitelistedAgent>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostr_pubkey` | `String` | Yes | Nostr public key |

**Returns:**
- `Option<WhitelistedAgent>` - Full agent information or `null` if not found

**Example:**

```javascript
const agentInfo = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'get_agent_info',
  args: {
    nostr_pubkey: 'npub1xyz...'
  }
})

if (agentInfo) {
  console.log('Agent Information:')
  console.log(`  NEAR Account: ${agentInfo.near_account}`)
  console.log(`  Nostr Pubkey: ${agentInfo.nostr_pubkey}`)
  console.log(`  Status: ${agentInfo.status}`)
  console.log(`  Stake: ${utils.format.formatNearAmount(agentInfo.stake_amount)}`)
  console.log(`  Capabilities: ${agentInfo.capabilities.join(', ')}`)
  console.log(`  Tasks Completed: ${agentInfo.metrics.tasks_completed}`)
  console.log(`  Success Rate: ${agentInfo.metrics.success_rate}%`)
}
```

---

### `get_near_account`

Get NEAR account associated with a Nostr pubkey.

```rust
pub fn get_near_account(&self, nostr_pubkey: String) -> Option<AccountId>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostr_pubkey` | `String` | Yes | Nostr public key |

**Returns:**
- `Option<AccountId>` - NEAR account or `null` if not found

**Example:**

```javascript
const nearAccount = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'get_near_account',
  args: {
    nostr_pubkey: 'npub1xyz...'
  }
})

if (nearAccount) {
  console.log(`NEAR account: ${nearAccount}`)
}
```

---

### `get_nostr_pubkey`

Get Nostr pubkey associated with a NEAR account.

```rust
pub fn get_nostr_pubkey(&self, near_account: AccountId) -> Option<String>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `near_account` | `AccountId` | Yes | NEAR account identifier |

**Returns:**
- `Option<String>` - Nostr pubkey or `null` if not found

**Example:**

```javascript
const nostrPubkey = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'get_nostr_pubkey',
  args: {
    near_account: 'agent.near'
  }
})

if (nostrPubkey) {
  console.log(`Nostr pubkey: ${nostrPubkey}`)
}
```

---

### `get_all_agents`

Get all whitelisted agents (admin only).

```rust
pub fn get_all_agents(&self, from_index: u64, limit: u64) -> Vec<WhitelistedAgent>
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `from_index` | `u64` | Yes | Starting index for pagination |
| `limit` | `u64` | Yes | Maximum number of agents to return |

**Returns:**
- `Vec<WhitelistedAgent>` - Array of whitelisted agents

**Example:**

```javascript
const agents = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'get_all_agents',
  args: {
    from_index: 0,
    limit: 100
  }
})

console.log(`Found ${agents.length} agents`)
agents.forEach(agent => {
  console.log(`  ${agent.near_account}: ${agent.status}`)
})
```

---

### `get_whitelist_count`

Get total count of whitelisted agents.

```rust
pub fn get_whitelist_count(&self) -> u64
```

**Returns:**
- `u64` - Total number of whitelisted agents

**Example:**

```javascript
const count = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'get_whitelist_count'
})

console.log(`Total whitelisted agents: ${count}`)
```

---

### `update_capabilities`

Update an agent's capabilities.

```rust
pub fn update_capabilities(&mut self, new_capabilities: Vec<String>)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `new_capabilities` | `Vec<String>` | Yes | New capabilities list |

**Requirements:**
- Caller must be the registered agent (owner of the account)
- Agent status must be `Active`

**Example:**

```javascript
await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'update_capabilities',
  args: {
    new_capabilities: ['code-analysis', 'security-scan', 'optimization', 'testing']
  },
  gas: '100000000000000' // 100 TGas
})
```

---

### `revoke_agent`

Revoke an agent from the whitelist (admin only).

```rust
pub fn revoke_agent(&mut self, nostr_pubkey: String, reason: String)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostr_pubkey` | `String` | Yes | Nostr public key to revoke |
| `reason` | `String` | Yes | Reason for revocation |

**Requirements:**
- Caller must be the admin account
- Agent stake will be partially slashed (10%)

**Side Effects:**
- Agent status set to `Revoked`
- 10% of stake slashed (sent to contract)
- Agent can no longer receive tasks

**Example:**

```javascript
await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'revoke_agent',
  args: {
    nostr_pubkey: 'npub1xyz...',
    reason: 'Security violation - unauthorized access to sensitive data'
  },
  gas: '100000000000000'
})

console.log('✓ Agent revoked')
```

---

### `suspend_agent`

Temporarily suspend an agent (admin only).

```rust
pub fn suspend_agent(&mut self, nostr_pubkey: String, reason: String, duration: u64)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostr_pubkey` | `String` | Yes | Nostr public key to suspend |
| `reason` | `String` | Yes | Reason for suspension |
| `duration` | `u64` | Yes | Suspension duration in nanoseconds |

**Requirements:**
- Caller must be the admin account
- Duration must be between 1 hour and 30 days

**Side Effects:**
- Agent status set to `Suspended`
- Agent cannot receive new tasks
- Agent can still complete in-progress tasks
- Status automatically reverts after duration

**Example:**

```javascript
// Suspend for 24 hours
const duration = 24 * 60 * 60 * 1000000000 // 24 hours in nanoseconds

await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'suspend_agent',
  args: {
    nostr_pubkey: 'npub1xyz...',
    reason: 'Under investigation for suspicious activity',
    duration: duration
  },
  gas: '100000000000000'
})

console.log('✓ Agent suspended for 24 hours')
```

---

### `reactivate_agent`

Reactivate a suspended or revoked agent (admin only).

```rust
pub fn reactivate_agent(&mut self, nostr_pubkey: String)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `nostr_pubkey` | `String` | Yes | Nostr public key to reactivate |

**Requirements:**
- Caller must be the admin account
- Agent must be in `Suspended` or `Revoked` status

**Side Effects:**
- Agent status set to `Active`
- Agent can receive tasks again

**Example:**

```javascript
await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'reactivate_agent',
  args: {
    nostr_pubkey: 'npub1xyz...'
  },
  gas: '100000000000000'
})

console.log('✓ Agent reactivated')
```

---

### `update_whitelist_status`

Update the overall whitelist status.

```rust
pub fn update_whitelist_status(&mut self, new_status: WhitelistStatus)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| `new_status` | `WhitelistStatus` | Yes | New whitelist status |

**Requirements:**
- Caller must be the admin account
- Status transition must be valid

**Valid Transitions:**
- `Open` → `InviteOnly`
- `InviteOnly` → `Open`
- `InviteOnly` → `Closed`
- `Closed` → `InviteOnly`

**Example:**

```javascript
await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'update_whitelist_status',
  args: {
    new_status: 'InviteOnly' // or 'Open', 'Closed'
  },
  gas: '100000000000000'
})

console.log('✓ Whitelist status updated to InviteOnly')
```

---

### `get_whitelist_status`

Get current whitelist registration status.

```rust
pub fn get_whitelist_status(&self) -> WhitelistStatus
```

**Returns:**
- `WhitelistStatus` - Current registration mode

**Example:**

```javascript
const status = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'get_whitelist_status'
})

console.log(`Whitelist status: ${status}`)
// Output: "Open"
```

---

### `get_admin`

Get the admin account address.

```rust
pub fn get_admin(&self) -> AccountId
```

**Returns:**
- `AccountId` - Admin account identifier

**Example:**

```javascript
const admin = await nearAccount.viewFunction({
  contractId: 'agent-whitelist.near',
  methodName: 'get_admin'
})

console.log(`Admin account: ${admin}`)
```

---

### `stake_additional`

Add additional stake to an existing agent registration.

```rust
#[payable]
pub fn stake_additional(&mut self, additional_stake: u128)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|-------|-----------|-------------|
| (attached deposit) | `u128` | Yes | Additional NEAR to stake |

**Requirements:**
- Caller must be the registered agent
- Agent status must be `Active`

**Returns:**
- New total stake amount

**Example:**

```javascript
await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'stake_additional',
  args: {},
  attachedDeposit: utils.format.parseNearAmount('50') // Add 50 NEAR
})

console.log('✓ Additional stake added')
```

---

### `unstake`

Remove stake and withdraw from whitelist.

```rust
pub fn unstake(&mut self)
```

**Requirements:**
- Caller must be the registered agent
- Agent status must be `Active`
- No active tasks can be pending
- 30-day notice period (optional, configurable)

**Side Effects:**
- Agent status set to `Revoked`
- Stake returned to agent account
- Agent removed from whitelist

**Example:**

```javascript
await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'unstake',
  gas: '100000000000000'
})

console.log('✓ Agent unstaked and removed from whitelist')
```

---

## 📊 Contract State

### State Variables

```rust
pub struct AgentWhitelist {
    // Primary storage
    agents: LookupMap<AccountId, WhitelistedAgent>,
    nostr_to_near: LookupMap<String, AccountId>,
    
    // Whitelist management
    whitelist_status: WhitelistStatus,
    admin: AccountId,
    
    // Statistics
    total_agents: u64,
    active_agents: u64,
    total_staked: u128,
}
```

---

## 🔒 Gas Costs

### Register Agent

- **Base Gas**: 5 Tgas
- **Per Byte**: 0.1 Tgas
- **Estimated Total**: ~10-20 Tgas
- **NEAR Cost**: ~0.00001 NEAR

### Update Capabilities

- **Base Gas**: 3 Tgas
- **Per Capability**: 0.5 Tgas
- **Estimated Total**: ~5-10 Tgas
- **NEAR Cost**: ~0.000005 NEAR

### Revoke Agent

- **Base Gas**: 8 Tgas
- **Estimated Total**: ~8 Tgas
- **NEAR Cost**: ~0.000008 NEAR

### View Functions (Free)

All view functions are free to call:
- `is_whitelisted`
- `get_agent_status`
- `get_agent_info`
- `get_near_account`
- `get_nostr_pubkey`
- `get_all_agents`
- `get_whitelist_count`
- `get_whitelist_status`
- `get_admin`

---

## 🎯 Best Practices

### 1. Error Handling

```javascript
try {
  const result = await nearAccount.functionCall({
    contractId: 'agent-whitelist.near',
    methodName: 'register_agent',
    args: registrationData,
    attachedDeposit: utils.format.parseNearAmount('10')
  })
  
  console.log('✓ Registration successful')
} catch (error) {
  if (error.type === 'AccountDoesNotExist') {
    console.error('Contract not found')
  } else if (error.type === 'SmartContractFailed') {
    console.error('Contract call failed:', error.message)
  } else if (error.message.includes('insufficient stake')) {
    console.error('Stake amount too low')
  } else {
    console.error('Unknown error:', error)
  }
}
```

### 2. Transaction Management

```javascript
// Check transaction status
const txHash = result.transaction.hash

const outcome = await nearAccount.connection.provider.txStatus(txHash, 'FINAL')

if (outcome.status.SuccessValue !== undefined) {
  console.log('✓ Transaction successful')
} else if (outcome.status.Failure !== undefined) {
  console.error('✗ Transaction failed:', outcome.status.Failure)
} else if (outcome.status.SuccessReceiptId !== undefined) {
  console.log('Transaction pending...')
}
```

### 3. Batch Operations

```javascript
// Register multiple agents efficiently
const agentConfigs = [
  { nostr: 'npub1...', caps: ['code-analysis'] },
  { nostr: 'npub2...', caps: ['security-scan'] },
  { nostr: 'npub3...', caps: ['data-processing'] }
]

for (const config of agentConfigs) {
  await registerAgent(config)
}
```

### 4. View Caching

```javascript
// Cache frequently accessed view results
const agentCache = new Map()

async function getAgentInfoWithCache(nostrPubkey) {
  if (agentCache.has(nostrPubkey)) {
    return agentCache.get(nostrPubkey)
  }
  
  const info = await nearAccount.viewFunction({
    contractId: 'agent-whitelist.near',
    methodName: 'get_agent_info',
    args: { nostr_pubkey: nostrPubkey }
  })
  
  agentCache.set(nostrPubkey, info)
  
  // Cache for 5 minutes
  setTimeout(() => agentCache.delete(nostrPubkey), 5 * 60 * 1000)
  
  return info
}
```

---

## 🔍 Debugging

### View Contract State

```bash
# Check admin
near view agent-whitelist.near get_admin

# Check whitelist status
near view agent-whitelist.near get_whitelist_status

# Check agent count
near view agent-whitelist.near get_whitelist_count

# Check specific agent
near view agent-whitelist.near get_agent_info --args '{"nostr_pubkey":"npub1..."}'
```

### Monitor Transactions

```javascript
// Subscribe to contract events
const events = await nearAccount.connection.provider.query({
  contractId: 'agent-whitelist.near',
  method: 'register_agent',
  args: {}
})
```

---

## 📚 Additional Resources

- [NEAR SDK Documentation](https://docs.near.org/sdk/develop/contracts/rust/introduction)
- [NEAR Testnet Faucet](https://testnet.mynearwallet.com)
- [NEAR Mainnet Explorer](https://explorer.near.org)
- [Nostr Protocol](https://github.com/nostr-protocol/nips)

---

## 🆘 Support

For contract-related issues:
- GitHub Issues: [Create Issue](https://github.com/mpp-near/issues/new)
- Discord: [mpp-near.dev](https://discord.gg/mpp-near)
- Documentation: [docs.mpp-near.com](https://docs.mpp-near.com)