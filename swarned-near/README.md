# Nostr-Backed NEAR Whitelist System

A secure, cryptographically verified whitelist system for agent swarms that combines Nostr identities with NEAR on-chain accounts.

## 🎯 Overview

The Nostr-Backed NEAR Whitelist System provides a production-grade security layer for agent swarms by:

1. **Binding Nostr identities to NEAR accounts** using cryptographic proofs
2. **Managing whitelisted agents on-chain** via NEAR smart contracts
3. **Enforcing strict access control** through stake-based registration
4. **Providing revocable, auditable agent identities**

This architecture enables agents to use Nostr for communication while maintaining immutable, verifiable identity on the NEAR blockchain.

## ✨ Key Features

- **Cryptographic Identity Binding**: Nostr pubkeys cryptographically linked to NEAR accounts
- **On-Chain Whitelist**: Immutable agent registry on NEAR blockchain
- **Stake-Based Security**: Agents must stake NEAR to register (skin-in-the-game)
- **Revocable Access**: Admin can revoke agents instantly if needed
- **Bidirectional Mapping**: Fast lookups between Nostr and NEAR identities
- **Capability System**: Agents declare capabilities for task routing
- **Transparent & Auditable**: All registrations visible on-chain
- **Token-Based Authentication**: Secure session management with expiration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │              NOSTR LAYER                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│  │  │Agent A  │  │Agent B  │  │Agent C  │        │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘        │   │
│  │       │ Nostr      │Nostr      │Nostr           │   │
│  │       │ Pubkey     │Pubkey     │Pubkey          │   │
│  └───────┼────────────┼────────────┼────────────────┘   │
│          │            │            │                        │
│          │ Ownership  │ Ownership  │ Ownership               │
│          │ Proof      │ Proof      │ Proof                  │
│          └────────────┴────────────┴────────────────┘   │
│                           │                             │
│                           ▼                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │              NEAR LAYER                        │   │
│  │                                                   │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │  Agent Whitelist Smart Contract      │    │   │
│  │  │                                  │    │   │
│  │  │  Whitelisted Agents:              │    │   │
│  │  │  • near-abc123 → agent-1.npub   │    │   │
│  │  │  • near-def456 → agent-2.npub   │    │   │
│  │  │  • near-ghi789 → agent-3.npub   │    │   │ │
│  │  │                                  │    │   │
│  │  │  Stakes: 10 NEAR per agent      │    │   │
│  │  │  Capabilities: Declared per agent │    │   │
│  │  │  Status: Active/Suspended/Revoked │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  │                                                   │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Core Concepts

### Nostr Identity
Each agent has a Nostr public key (npub) that serves as their communication identity on the Nostr network.

### NEAR Account
Each agent must have a NEAR blockchain account that provides:
- Immutable identity on-chain
- Financial stake for registration
- Whitelist entry management

### Ownership Proof
A cryptographic signature proving ownership of both Nostr and NEAR identities:
```json
{
  "signature": "nostr-signature-hex",
  "message": "{\"near_account\":\"agent.near\",\"nostr_pubkey\":\"npub1...\",\"timestamp\":1699999999}",
  "verification_method": "nostr-signature"
}
```

### Whitelist Status
Agents can be in one of these states:
- **Active**: Fully operational, can receive tasks
- **Suspended**: Temporarily disabled, still registered
- **Revoked**: Permanently removed, stake slashed
- **Pending**: Awaiting approval (if in invite-only mode)

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- NEAR CLI
- Nostr keys for each agent
- NEAR account with minimum 10 NEAR

### 1. Deploy Whitelist Contract

```bash
# Build the contract
cargo build --target wasm32-unknown-unknown --release

# Deploy to NEAR testnet
near deploy agent-whitelist.wasm --accountId my-whitelist.near --init new my-admin.near

# Verify deployment
near view my-whitelist.near get_admin
```

### 2. Register an Agent

```javascript
import { NostrBackedNEARRegistration } from '@nostr-near/sdk';

const registration = new NostrBackedNEARRegistration({
  nearAccount: nearAccount,
  nostrKey: nostrKeyPair,
  whitelistContract: 'my-whitelist.near'
});

// Register with 10 NEAR stake
await registration.registerAgent([
  'code-analysis',
  'security-scan',
  'data-processing'
]);
```

### 3. Authenticate Agent

```javascript
import { WhitelistAuthenticator } from '@nostr-near/sdk';

const authenticator = new WhitelistAuthenticator({
  whitelistContract: 'my-whitelist.near',
  nearAccount: nearAccount
});

// Authenticate with Nostr pubkey
const authResult = await authenticator.authenticateAgent(nostrPubkey);

if (authResult.authenticated) {
  console.log('✓ Agent authenticated and whitelisted');
  console.log('Session token:', authResult.sessionToken);
  console.log('Capabilities:', authResult.agentDetails.capabilities);
}
```

### 4. Initialize Swarm

```javascript
import { WhitelistedAgentSwarm } from '@nostr-near/sdk';

const swarm = new WhitelistedAgentSwarm({
  whitelistContract: 'my-whitelist.near',
  nearAccount: orchestratorNearAccount,
  privateRelays: [
    'wss://internal-relay.company.com'
  ]
});

await swarm.initialize();
console.log('✓ Whitelisted swarm is running');
```

## 📚 Documentation Structure

```
nostr-near-whitelist/
├── README.md                    # This file
├── architecture/
│   ├── overview.md             # Detailed architecture
│   ├── security.md             # Security model
│   └── data-flow.md           # Data flow diagrams
├── guides/
│   ├── getting-started.md       # Quick start guide
│   ├── agent-registration.md    # Registering agents
│   ├── authentication.md        # Authentication flow
│   └── task-distribution.md   # Distributing tasks
├── api/
│   ├── smart-contracts.md       # NEAR contract reference
│   ├── javascript-sdk.md       # JavaScript SDK API
│   └── nostr-events.md        # Nostr event kinds
├── deployment/
│   ├── smart-contract.md       # Deploying contracts
│   ├── relay-setup.md          # Setting up relays
│   └── swarm-setup.md         # Setting up swarm
├── security/
│   ├── threat-model.md         # Security threats
│   ├── best-practices.md      # Security best practices
│   └── incident-response.md   # Incident response
└── examples/
    ├── basic-usage.md         # Basic examples
    ├── advanced-usage.md      # Advanced patterns
    └── troubleshooting.md      # Common issues
```

## 🔒 Security Model

### Defense in Depth

1. **Network Layer**: Private relays with mTLS
2. **Identity Layer**: Cryptographic Nostr↔NEAR binding
3. **Access Layer**: NEAR whitelist with stake requirements
4. **Session Layer**: Time-limited authentication tokens
5. **Application Layer**: Capability-based authorization

### Threat Mitigation

| Threat | Mitigation |
|---------|-------------|
| Identity Spoofing | Cryptographic proof required |
| Unauthorized Access | NEAR whitelist enforcement |
| Sybil Attacks | 10 NEAR stake per agent |
| Replay Attacks | Timestamp validation |
| Privilege Escalation | Capability-based routing |
| Data Exfiltration | Network segmentation |
| Insider Threats | Audit logging & revocation |

## 📊 System Requirements

### Minimum

- **NEAR Network**: Testnet or Mainnet
- **Stake Amount**: 10 NEAR per agent
- **Relay Infrastructure**: 1+ private relays
- **Admin Account**: For whitelist management

### Recommended

- **NEAR Network**: Mainnet for production
- **Stake Amount**: 100 NEAR per agent (higher trust)
- **Relay Infrastructure**: 3+ private relays (redundancy)
- **Admin Account**: Multi-sig contract
- **Monitoring**: Real-time health monitoring

## 🔧 Configuration

### Environment Variables

```bash
# NEAR Configuration
NEAR_NETWORK=mainnet
NEAR_WALLET=~/.near-credentials/mainnet
NEAR_CONTRACT_ID=agent-whitelist.near

# Nostr Configuration
NOSTR_PRIVATE_KEY=nsec1...
NOSTR_RELAYS=wss://relay1.near,wss://relay2.near

# Swarm Configuration
SWARM_ADMIN_ACCOUNT=admin.near
SWARM_STAKE_AMOUNT=10
SWARM_SESSION_TIMEOUT=3600
```

## 📈 Monitoring & Metrics

### Key Metrics

- **Whitelisted Agents**: Total active agents
- **Agent Utilization**: % of agents actively processing
- **Authentication Rate**: Successful vs failed attempts
- **Task Distribution**: Tasks per minute
- **Error Rate**: Failed task completions
- **Stake Pool**: Total NEAR staked

### Health Checks

```bash
# Check contract status
near view agent-whitelist.near get_agent_status --accountId agent.near

# Check agent count
near view agent-whitelist.near get_whitelist_count

# Monitor authentication logs
tail -f /var/log/swarm/auth.log
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](../../LICENSE) for details.

## 🆘 Support

- **Documentation**: [docs.mpp-near.com](https://docs.mpp-near.com)
- **Issues**: [GitHub Issues](https://github.com/mpp-near/issues)
- **Discord**: [mpp-near.dev](https://discord.gg/mpp-near)

## 🗺️ Roadmap

- [ ] Multi-admin whitelist management
- [ ] Tiered stake levels
- [ ] Automated reputation scoring
- [ ] Agent group management
- [ ] Cross-chain whitelist support
- [ ] GUI for whitelist management

---

**Next Steps:**
- Read [Getting Started](./guides/getting-started.md)
- Review [Architecture Overview](./architecture/overview.md)
- Check [Security Best Practices](./security/best-practices.md)