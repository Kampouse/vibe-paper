# Getting Started

Welcome to the Nostr-Backed NEAR Whitelist System! This guide will help you get up and running quickly.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Node.js** 18.0 or higher
- **npm** or **yarn** package manager
- **NEAR CLI** installed
- **NEAR account** with minimum 10 NEAR
- **Nostr keypair** for each agent

### Install NEAR CLI

```bash
npm install -g near-cli

# Verify installation
near --version
```

### Install Node.js Dependencies

```bash
# Create project directory
mkdir my-nostr-swarm
cd my-nostr-swarm

# Initialize package.json
npm init -y

# Install SDK
npm install @nostr-near/sdk
```

## 🚀 Quick Start

### 1. Set Up Environment Variables

Create a `.env` file:

```bash
# NEAR Configuration
NEAR_NETWORK=mainnet
NEAR_WALLET=~/.near-credentials/mainnet
NEAR_CONTRACT_ID=agent-whitelist.near

# Admin Configuration
SWARM_ADMIN_ACCOUNT=admin.near
SWARM_STAKE_AMOUNT=10

# Nostr Configuration
NOSTR_PRIVATE_KEY=nsec1...
NOSTR_RELAYS=wss://relay1.near,wss://relay2.near

# Relay Configuration
RELAY_URL=wss://internal-relay.company.com
RELAY_AUTH_TOKEN=your-auth-token

# Session Configuration
SWARM_SESSION_TIMEOUT=3600
```

### 2. Initialize the Project

```bash
# Create configuration file
cat > config.json <<EOF
{
  "near": {
    "network": "mainnet",
    "contractId": "agent-whitelist.near",
    "adminAccount": "admin.near"
  },
  "nostr": {
    "relays": [
      "wss://relay1.near",
      "wss://relay2.near"
    ]
  },
  "swarm": {
    "stakeAmount": 10,
    "sessionTimeout": 3600,
    "maxConcurrentTasks": 5
  }
}
EOF
```

### 3. Deploy Whitelist Contract

```bash
# Build the contract
cargo build --target wasm32-unknown-unknown --release

# Deploy to NEAR testnet (recommended for testing first)
near deploy agent-whitelist.wasm \
  --accountId my-whitelist.near \
  --init new admin.near

# Verify deployment
near view my-whitelist.near get_admin

# Expected output: "admin.near"
```

For production deployment to mainnet:

```bash
# Set mainnet account
near login --networkId mainnet

# Deploy
near deploy agent-whitelist.wasm \
  --accountId my-whitelist.near \
  --networkId mainnet \
  --init new admin.near
```

### 4. Register Your First Agent

Create a registration script:

```javascript
// register-agent.js
import { NostrBackedNEARRegistration } from '@nostr-near/sdk';

// Load environment variables
const nearAccountId = process.env.NEAR_ACCOUNT_ID;
const nostrPrivateKey = process.env.NOSTR_PRIVATE_KEY;
const whitelistContract = process.env.NEAR_CONTRACT_ID;

// Initialize registration
const registration = new NostrBackedNEARRegistration({
  nearAccount: {
    accountId: nearAccountId,
    networkId: process.env.NEAR_NETWORK
  },
  nostrKey: {
    publicKey: getPublicKey(nostrPrivateKey),
    privateKey: nostrPrivateKey
  },
  whitelistContract: whitelistContract
});

// Register with capabilities
(async () => {
  try {
    const result = await registration.registerAgent([
      'code-analysis',
      'security-scan',
      'data-processing'
    ]);

    console.log('✓ Agent registered successfully!');
    console.log('  NEAR Account:', result.nearAccount);
    console.log('  Nostr Pubkey:', result.nostrPubkey);
    console.log('  Stake Amount:', result.stakeAmount);
    console.log('  Status:', result.status);
  } catch (error) {
    console.error('❌ Registration failed:', error);
    process.exit(1);
  }
})();
```

Run the registration:

```bash
# Set environment variables
export NEAR_ACCOUNT_ID="agent-1.near"
export NOSTR_PRIVATE_KEY="nsec1..."
export NEAR_CONTRACT_ID="my-whitelist.near"
export NEAR_NETWORK="testnet"

# Run registration
node register-agent.js
```

### 5. Start the Swarm Orchestrator

Create an orchestrator startup script:

```javascript
// start-swarm.js
import { WhitelistedAgentSwarm } from '@nostr-near/sdk';

// Load configuration
const config = JSON.parse(require('fs').readFileSync('config.json', 'utf8'));

// Initialize swarm
const swarm = new WhitelistedAgentSwarm({
  whitelistContract: config.near.contractId,
  nearAccount: {
    accountId: config.near.adminAccount,
    networkId: config.near.network
  },
  privateRelays: config.nostr.relays,
  orchestratorId: config.near.adminAccount
});

// Start swarm
(async () => {
  try {
    await swarm.initialize();
    console.log('✓ Swarm initialized successfully!');
    console.log('  Whitelist Contract:', config.near.contractId);
    console.log('  Admin Account:', config.near.adminAccount);
    console.log('  Relays:', config.nostr.relays.join(', '));
    console.log('');
    console.log('Swarm is now running and accepting agents...');
    
    // Keep process alive
    process.on('SIGINT', async () => {
      console.log('\nShutting down swarm...');
      await swarm.shutdown();
      process.exit(0);
    });
    
  } catch (error) {
    console.error('❌ Failed to initialize swarm:', error);
    process.exit(1);
  }
})();
```

Start the orchestrator:

```bash
node start-swarm.js
```

## 🔧 Configuration Options

### Minimum Required Configuration

```javascript
{
  "near": {
    "network": "mainnet",      // mainnet or testnet
    "contractId": "...",        // Whitelist contract address
    "adminAccount": "..."       // Admin NEAR account
  },
  "nostr": {
    "relays": ["..."]          // Array of relay URLs
  }
}
```

### Optional Configuration

```javascript
{
  "near": {
    "network": "mainnet",
    "contractId": "agent-whitelist.near",
    "adminAccount": "admin.near",
    "walletPath": "~/.near-credentials/mainnet"
  },
  "nostr": {
    "relays": [
      "wss://relay1.near",
      "wss://relay2.near"
    ],
    "connectTimeout": 10000,
    "reconnectInterval": 5000
  },
  "swarm": {
    "stakeAmount": 10,              // NEAR to stake
    "sessionTimeout": 3600,         // Session timeout in seconds
    "maxConcurrentTasks": 5,        // Max tasks per agent
    "taskTimeout": 3600000,         // Task timeout in milliseconds
    "retryAttempts": 3,
    "retryDelay": 2000
  },
  "routing": {
    "strategy": "least-loaded",     // round-robin, least-loaded, capability-based
    "loadBalancing": true
  },
  "monitoring": {
    "enabled": true,
    "healthCheckInterval": 60000,   // Health check every 60 seconds
    "metricsInterval": 30000        // Metrics every 30 seconds
  },
  "security": {
    "mTLS": true,
    "ipWhitelist": ["10.0.0.0/8", "172.16.0.0/12"],
    "rateLimiting": {
      "requestsPerSecond": 100,
      "eventsPerSecond": 50
    }
  }
}
```

## 🎯 Common Workflows

### Workflow 1: Register a New Agent

```bash
# 1. Generate Nostr keypair (if needed)
npx nostr-tools generate-keypair

# 2. Create or use existing NEAR account
near create-account my-agent.near

# 3. Fund account with stake
near send my-agent.near 10

# 4. Register with whitelist
node register-agent.js

# 5. Verify registration
near view my-whitelist.near get_agent_status --accountId my-agent.near
```

### Workflow 2: Authenticate an Agent

```javascript
import { WhitelistAuthenticator } from '@nostr-near/sdk';

const authenticator = new WhitelistAuthenticator({
  whitelistContract: 'my-whitelist.near',
  nearAccount: { accountId: 'admin.near', networkId: 'mainnet' }
});

// Authenticate agent
const authResult = await authenticator.authenticateAgent('npub1...');

if (authResult.authenticated) {
  console.log('✓ Agent authenticated');
  console.log('Session token:', authResult.sessionToken);
  console.log('Expires at:', new Date(authResult.expiresAt).toISOString());
  console.log('Capabilities:', authResult.agentDetails.capabilities);
} else {
  console.log('❌ Authentication failed');
}
```

### Workflow 3: Submit a Task

```javascript
import { WhitelistedAgentSwarm } from '@nostr-near/sdk';

const swarm = new WhitelistedAgentSwarm(config);

await swarm.initialize();

// Submit task
const result = await swarm.submitWorkload({
  id: 'task-123',
  name: 'Code Review',
  type: 'code-review',
  priority: 'high',
  payload: {
    repository: 'github.com/repo',
    branch: 'main',
    files: ['app.js', 'utils.js']
  }
});

console.log('Task submitted:', result);
console.log('Estimated completion:', new Date(result.estimatedCompletion).toISOString());
```

### Workflow 4: Check Whitelist Status

```bash
# Check if agent is whitelisted
near view my-whitelist.near is_whitelisted '{"nostr_pubkey":"npub1..."}'

# Get agent details
near view my-whitelist.near get_agent_info --accountId agent-1.near

# Get reputation
near view my-whitelist.near get_agent_reputation --accountId agent-1.near

# Count whitelisted agents
near view my-whitelist.near get_whitelist_count
```

## 🐛 Troubleshooting

### Issue: Agent Registration Fails

**Symptom**: `register_agent` call fails

**Solutions**:

1. Check NEAR account balance:
```bash
near state agent-1.near
```

2. Verify whitelist contract is deployed:
```bash
near view my-whitelist.near get_admin
```

3. Check ownership proof format:
```javascript
// Ensure message includes all required fields
{
  "near_account": "agent-1.near",
  "nostr_pubkey": "npub1...",
  "timestamp": 1699999999,
  "action": "register_agent",
  "contract": "my-whitelist.near"
}
```

4. Verify Nostr signature:
```bash
npx nostr-tools verify-signature <signature> <message>
```

### Issue: Authentication Fails

**Symptom**: `is_whitelisted` returns false

**Solutions**:

1. Verify agent is registered:
```bash
near view my-whitelist.near get_agent_status --accountId agent-1.near
```

2. Check agent status is "Active":
```bash
# Should return "Active", not "Suspended" or "Revoked"
```

3. Verify Nostr pubkey matches:
```bash
# Use exact npub from registration
```

### Issue: Swarm Won't Start

**Symptom**: `swarm.initialize()` fails

**Solutions**:

1. Check relay connectivity:
```bash
curl -I wss://internal-relay.company.com
```

2. Verify NEAR contract is accessible:
```bash
near view my-whitelist.near get_whitelist_count
```

3. Check configuration file syntax:
```bash
cat config.json | python -m json.tool
```

4. Verify environment variables:
```bash
echo $NEAR_NETWORK
echo $NEAR_CONTRACT_ID
```

### Issue: Tasks Not Reaching Agents

**Symptom**: Agents not receiving tasks

**Solutions**:

1. Check agent status:
```bash
near view my-whitelist.near get_agent_status --accountId agent-1.near
```

2. Verify agent capabilities:
```bash
# Agent must have capabilities matching task type
```

3. Check agent load:
```bash
# Ensure agent has capacity (current_load < max_concurrent)
```

4. Verify relay subscriptions:
```javascript
// Check agent is subscribed to correct kinds
// Kind 5100: Task queue
```

## 📊 Verification Checklist

After setup, verify each component:

### ✅ NEAR Contract

- [ ] Contract deployed successfully
- [ ] Admin account set correctly
- [ ] Can call `get_admin`
- [ ] Can call `get_whitelist_count`

### ✅ Agent Registration

- [ ] Agent NEAR account funded
- [ ] Agent registered with correct capabilities
- [ ] Ownership proof verified
- [ ] Stake deposited successfully
- [ ] Agent status is "Active"

### ✅ Authentication

- [ ] Can authenticate agent with Nostr pubkey
- [ ] `is_whitelisted` returns true
- [ ] Session token generated successfully
- [ ] Token expires after configured timeout

### ✅ Swarm Operations

- [ ] Orchestrator initialized
- [ ] Connected to all configured relays
- [ ] Can submit tasks
- [ ] Tasks are distributed to agents
- [ ] Agents can receive tasks
- [ ] Results are collected and returned

### ✅ Monitoring

- [ ] Health checks passing
- [ ] Metrics are being collected
- [ ] Alerts are configured
- [ ] Logs are being written

## 🚀 Next Steps

Now that you're up and running:

1. **Read the Architecture Overview**: Learn about system components
   ```bash
   cat architecture/overview.md
   ```

2. **Explore Agent Registration**: Learn advanced registration features
   ```bash
   cat guides/agent-registration.md
   ```

3. **Review Security Best Practices**: Secure your swarm
   ```bash
   cat security/best-practices.md
   ```

4. **Check API Documentation**: Available methods and events
   ```bash
   cat api/smart-contracts.md
   cat api/nostr-events.md
   ```

5. **Try Examples**: Basic and advanced usage patterns
   ```bash
   cat examples/basic-usage.md
   cat examples/advanced-usage.md
   ```

6. **Join the Community**: Get help and share knowledge
   - Discord: mpp-near.dev
   - GitHub Issues: github.com/mpp-near/issues
   - Documentation: docs.mpp-near.com

## 💡 Tips for Success

1. **Start with Testnet**: Always test on NEAR testnet first
2. **Use Small Stakes Initially**: Start with minimum 10 NEAR
3. **Monitor Everything**: Enable all monitoring from day one
4. **Document Your Setup**: Keep track of configurations and decisions
5. **Test Failures**: Disconnect relays, stop agents, test behavior
6. **Start Simple**: Begin with basic agent and task types
7. **Gradually Add Complexity**: Add features as you understand the system
8. **Keep Logs**: Detailed logs are invaluable for debugging
9. **Backup Configuration**: Version control your config files
10. **Ask for Help**: Community is there to support you

## 📞 Getting Help

If you run into issues:

1. **Check Troubleshooting Guide**: See `examples/troubleshooting.md`
2. **Search Issues**: Check GitHub issues for similar problems
3. **Join Discord**: Get real-time help from community
4. **Review Logs**: Check your logs for error messages
5. **Verify Configuration**: Ensure all settings are correct

---

**Congratulations!** 🎉 You've successfully set up the Nostr-Backed NEAR Whitelist System. You're now ready to build secure, scalable agent swarms!