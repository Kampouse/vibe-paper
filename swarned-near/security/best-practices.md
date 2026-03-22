# Security Best Practices

Comprehensive security guidelines for deploying and operating a Nostr-Backed NEAR Whitelist system.

## 🔒 Security Philosophy

This system implements a **defense-in-depth** approach with multiple independent security layers. No single layer failure should compromise the entire system.

### Core Security Principles

1. **Zero Trust**: Verify everything, trust nothing by default
2. **Least Privilege**: Grant minimum required access only
3. **Fail Safe**: System remains secure even with partial failures
4. **Auditability**: All actions logged and traceable
5. **Revocability**: Instant revocation of compromised identities
6. **Defense in Depth**: Multiple independent security layers
7. **Security by Design**: Security built into system from ground up

---

## 🌐 Layer 1: Network Security

### 1.1 Private Relay Infrastructure

**Critical Requirement**: All communication must occur over private, authenticated relays.

#### Relay Deployment Checklist

```bash
# 1. Deploy relays in isolated network segment
# 2. Configure mTLS (mutual TLS) for all relay connections
# 3. Enable IP whitelisting (only allow internal network ranges)
# 4. Configure rate limiting per IP and per connection
# 5. Enable DDoS protection
# 6. Set up monitoring and alerting
# 7. Regular security updates and patching
```

#### mTLS Configuration

```nginx
# Nginx configuration for mTLS
server {
    listen 443 ssl;
    server_name internal-relay.company.com;
    
    # SSL certificates
    ssl_certificate /etc/ssl/relay.crt;
    ssl_certificate_key /etc/ssl/relay.key;
    
    # Require client certificates
    ssl_client_certificate /etc/ssl/ca.crt;
    ssl_verify_client on;
    ssl_verify_depth 2;
    
    # Only allow whitelisted clients
    allow 10.0.0.0/8;
    allow 172.16.0.0/12;
    allow 192.168.0.0/16;
    deny all;
    
    # WebSocket upgrade
    location / {
        proxy_pass http://localhost:7000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### IP Whitelisting

```bash
# Configure firewall rules (ufw example)
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow only internal network ranges
sudo ufw allow from 10.0.0.0/8 to any port 443
sudo ufw allow from 172.16.0.0/12 to any port 443
sudo ufw allow from 192.168.0.0/16 to any port 443

# Block all other traffic
sudo ufw enable
sudo ufw reload
```

### 1.2 Rate Limiting

#### Per-IP Rate Limits

```yaml
# Relay configuration
rate_limits:
  per_ip:
    connections_per_minute: 60
    events_per_second: 50
    bandwidth_per_second: "5MB"
  
  per_agent:
    events_per_minute: 1000
    bandwidth_per_minute: "100MB"
  
  global:
    total_connections: 1000
    events_per_second: 1000
    bandwidth_per_second: "1GB"
```

#### DDoS Protection

```yaml
ddos_protection:
  enabled: true
  
  thresholds:
    connections_per_second: 100
    events_per_second: 1000
  
  mitigation:
    rate_limiting:
      enabled: true
      duration: 300  # 5 minutes
    ip_banning:
      enabled: true
      duration: 3600  # 1 hour
    captcha:
      enabled: false  # Internal system, not needed
```

### 1.3 Network Segmentation

```
┌─────────────────────────────────────────────────────────┐
│                  NETWORK ARCHITECTURE                  │
├─────────────────────────────────────────────────────────┤
│                                                              │
│  DMZ (Public)                                              │
│  ┌───────────────┐                                          │
│  │ Load Balancer │                                          │
│  └───────────────┘                                          │
│         │                                                    │
│         ▼                                                    │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │              FIREWALL                                 │   │
│  │  • Block all inbound except specific ports         │   │
│  │  • Block all outbound except specific dests    │   │
│  │  • Intrusion detection and prevention        │   │
│  └───────────────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  INTERNAL NETWORK (10.0.0.0/8)                         │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │              APPLICATION ZONE                         │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │   │
│  │  │Relay 1    │  │Relay 2    │  │Relay 3    │        │   │
│  │  │Primary     │  │Secondary   │  │Backup      │        │   │
│  │  └────────────┘  └────────────┘  └────────────┘        │   │
│  │                                                       │   │
│  │  ┌────────────┐  ┌────────────┐                      │   │
│  │  │Orchestrator│  │Database    │                      │   │
│  │  └────────────┘  └────────────┘                      │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Layer 2: Identity Security

### 2.1 Nostr Key Management

#### Key Generation Best Practices

```javascript
// Use cryptographically secure random number generator
import { generateKeyPair } from '@noble/curves';

// Generate new keypair with strong randomness
const keyPair = generateKeyPair();

// Never reuse keys across different environments
// Never share private keys (nsec) in unencrypted form
// Store private keys only in secure storage
// Rotate keys regularly (recommended: every 90 days)
```

#### Secure Key Storage

```javascript
// Hardware Security Module (HSM) - Best option
const hsmStorage = {
  type: 'hsm',
  provider: 'aws-kms' | 'gcp-kms' | 'azure-keyvault',
  keyId: 'alias/agent-private-key',
  encryption: 'aes-256-gcm'
};

// Encrypted Storage - Good option
const encryptedStorage = {
  type: 'encrypted-file',
  path: '~/.nostr/encrypted-keys',
  encryption: 'xchacha20-poly1305',
  passphrase: process.env.KEY_ENCRYPTION_PASSPHRASE
};

// Memory-Only (Runtime) - For ephemeral sessions
const runtimeStorage = {
  type: 'memory',
  ttl: 3600000,  // 1 hour
  secure: true,
  noSwapping: true  // Prevent swap to disk
};
```

#### Key Rotation Procedure

```javascript
// 1. Generate new keypair
const newKeyPair = generateKeyPair();

// 2. Create transition period (both keys valid for 30 days)
const transitionPeriod = 30 * 24 * 60 * 60 * 1000000000;

// 3. Publish key transition event
const transitionEvent = {
  kind: 30078,
  tags: [
    ["d", "key-transition"],
    ["old_pubkey", oldPublicKey],
    ["new_pubkey", newKeyPair.publicKey],
    ["transition_start", String(Date.now())],
    ["transition_end", String(Date.now() + transitionPeriod)]
  ],
  content: JSON.stringify({
    old_pubkey: oldPublicKey,
    new_pubkey: newKeyPair.publicKey,
    transition_start: Date.now(),
    transition_end: Date.now() + transitionPeriod,
    reason: "routine_key_rotation"
  })
};

// 4. Update NEAR whitelist with new pubkey
await nearAccount.functionCall({
  contractId: 'agent-whitelist.near',
  methodName: 'update_pubkey',
  args: {
    old_pubkey: oldPublicKey,
    new_pubkey: newKeyPair.publicKey,
    proof: await generateOwnershipProof(newKeyPair)
  }
});

// 5. Monitor for old key usage (alert on suspicious activity)
```

### 2.2 Ownership Proof Security

#### Robust Ownership Proof Structure

```javascript
interface OwnershipProof {
  // Core proof
  signature: string;           // Schnorr signature (64 bytes hex)
  message: string;             // Signed message (JSON)
  verification_method: "nostr-schnorr";
  verified_at: number;          // Unix timestamp (nanoseconds)
  
  // Additional security fields
  nonce: string;               // Random value prevents replay attacks
  expires_at: number;           // Proof expiration (e.g., 1 hour)
  device_fingerprint?: string;  // Device fingerprint
  ip_address?: string;          // IP address at time of signing
  user_agent?: string;          // User agent hash
}

// Generate secure ownership proof
async function generateSecureOwnershipProof(
  nostrKeyPair: NostrKeyPair,
  nearAccountId: string
): Promise<OwnershipProof> {
  
  // 1. Generate nonce (cryptographically secure)
  const nonce = crypto.randomBytes(32).toString('hex');
  
  // 2. Create timestamp
  const timestamp = Date.now();
  
  // 3. Get device fingerprint
  const deviceFingerprint = await getDeviceFingerprint();
  
  // 4. Create signed message
  const message = JSON.stringify({
    near_account: nearAccountId,
    nostr_pubkey: nostrKeyPair.publicKey,
    timestamp: timestamp,
    nonce: nonce,
    action: 'register_agent',
    contract: process.env.NEAR_CONTRACT_ID,
    expires_at: timestamp + 3600000  // 1 hour
  });
  
  // 5. Sign with Nostr private key
  const signature = await signSchnorr(nostrKeyPair.privateKey, message);
  
  return {
    signature,
    message,
    verification_method: 'nostr-schnorr',
    verified_at: timestamp,
    nonce,
    expires_at: timestamp + 3600000,
    device_fingerprint: deviceFingerprint,
    ip_address: await getPublicIP(),
    user_agent: hashUserAgent(navigator.userAgent)
  };
}
```

#### Ownership Proof Verification

```rust
// NEAR contract verification
pub fn verify_ownership(
    &self,
    near_account: AccountId,
    nostr_pubkey: String,
    proof: OwnershipProof
) -> bool {
    // 1. Check expiration
    let now = env::block_timestamp();
    assert!(now <= proof.expires_at, "Ownership proof expired");
    
    // 2. Verify message format
    let signed_data: serde_json::Value = 
        serde_json::from_str(&proof.message).unwrap();
    
    // 3. Verify all required fields present
    assert!(signed_data.get("near_account").is_some());
    assert!(signed_data.get("nostr_pubkey").is_some());
    assert!(signed_data.get("timestamp").is_some());
    assert!(signed_data.get("nonce").is_some());
    
    // 4. Verify message content matches expected values
    let data_near = signed_data.get("near_account")
        .unwrap().as_str().unwrap();
    let data_nostr = signed_data.get("nostr_pubkey")
        .unwrap().as_str().unwrap();
    
    assert!(data_near == near_account.as_str(), "NEAR account mismatch");
    assert!(data_nostr == nostr_pubkey, "Nostr pubkey mismatch");
    
    // 5. Verify signature (use Nostr schnorr verification)
    let is_valid_signature = verify_nostr_schnorr(
        &proof.signature,
        &proof.message,
        &nostr_pubkey
    );
    
    assert!(is_valid_signature, "Invalid Nostr signature");
    
    // 6. Verify nonce not reused (check recent proofs)
    if let Some(previous) = self.nonce_store.get(&proof.nonce) {
        assert!(false, "Nonce already used - replay attack detected");
    }
    
    // 7. Store nonce to prevent reuse
    self.nonce_store.insert(&proof.nonce, &now);
    
    true
}
```

### 2.3 NEAR Account Security

#### Secure Account Management

```javascript
// Use hardware wallet for admin accounts
const adminWallet = {
  type: 'ledger',
  derivationPath: "m/44'/397'/0'/0'/0",  // BIP-44 path
  requireConfirmations: true,
  allowNetwork: ['mainnet']  // No testnet
};

// Use multi-sig for critical operations
const multiSig = {
  threshold: 2,  // 2 of 3 required
  signers: [
    'admin-1.near',
    'admin-2.near',
    'admin-3.near'
  ],
  contract: 'multi-sig-agent-whitelist.near'
};

// Separate accounts by role
const roleAccounts = {
  deployer: 'deployer.near',      // Only for deployments
  admin: 'admin.near',           // Only for admin operations
  operations: 'ops.near',         // For monitoring/maintenance
  auditor: 'auditor.near',      // Read-only for audits
  treasury: 'treasury.near'       // Holds stake funds
};
```

#### Access Key Management

```javascript
// NEVER use full access keys for automated agents
// Instead, use limited access keys (function call auth)

const limitedAccessKey = {
  account_id: 'agent-service.near',
  public_key: 'ed25519...',
  allowance: {
    receiver_id: 'agent-whitelist.near',
    method_names: ['is_whitelisted', 'get_agent_info'],  // Read-only
    allowance: '200000000000000000000000',  // 0.2 NEAR
    period: '86400'  // 1 day
  }
};

// For admin operations, require wallet signature
async function adminOperation(methodName, args) {
  // This requires hardware wallet signature
  const signedTx = await wallet.signTransaction({
    receiverId: 'agent-whitelist.near',
    actions: [{
      type: 'FunctionCall',
      methodName: methodName,
      args: JSON.stringify(args),
      gas: '100000000000000'
    }]
  });
  
  return await nearAccount.connection.sendTransaction(signedTx);
}
```

---

## 🔐 Layer 3: Authentication & Authorization

### 3.1 Multi-Factor Authentication (MFA)

#### Implementing MFA for Sensitive Operations

```javascript
// MFA configuration
const mfaConfig = {
  // Require MFA for admin operations
  requireForAdmin: true,
  
  // Require MFA for agent revocation
  requireForRevoke: true,
  
  // Require MFA for whitelist changes
  requireForWhitelistChanges: true,
  
  // MFA methods
  methods: [
    'totp',              // Time-based one-time password
    'webauthn',           // Hardware security key
    'sms'                // SMS backup (less secure)
  ],
  
  // TOTP configuration
  totp: {
    issuer: 'Nostr-Near-Swarm',
    algorithm: 'SHA256',
    digits: 6,
    period: 30,
    secretEncryption: 'aes-256-gcm'
  }
};

// MFA verification flow
async function verifyMFA(userId, totpCode) {
  // 1. Rate limit MFA attempts
  const attempts = await getMFAAttempts(userId);
  if (attempts >= 5) {
    // Lock account for 15 minutes
    await lockAccount(userId, 900);
    throw new Error('Too many MFA attempts, account locked');
  }
  
  // 2. Verify TOTP code (with time window)
  const isValid = await verifyTOTP(userId, totpCode, {
    window: 2,  // Allow codes from +/- 1 time period
    skew: 1   // Allow 1 time step skew
  });
  
  // 3. Record attempt
  await recordMFAAttempt(userId, isValid);
  
  if (!isValid) {
    throw new Error('Invalid MFA code');
  }
  
  return true;
}
```

### 3.2 Session Management

#### Secure Session Token Structure

```typescript
interface SecureSessionToken {
  // Token data
  token: string;               // Base64-encoded session data
  signature: string;            // Admin signature
  expires_at: number;           // Expiration timestamp
  
  // Session metadata
  session_id: string;           // Unique session identifier
  agent_id: string;            // NEAR account ID
  nostr_pubkey: string;         // Nostr public key
  
  // Security fields
  issued_at: number;            // Issuance timestamp
  issued_from: string;          // Issuer IP address
  issued_device: string;        // Issuer device fingerprint
  capabilities: string[];       // Granted capabilities
  
  // Constraints
  max_concurrent_tasks: number; // Task limit for this session
  allowed_relays: string[];     // Specific relays this session can use
  ip_restriction: string;        // IP address restriction (if any)
  
  // Revocation
  revoked: boolean;              // Whether token has been revoked
  revoked_at: number | null;    // Revocation timestamp
  revoke_reason: string | null;  // Reason for revocation
}
```

#### Session Generation with Security

```javascript
async function generateSecureSession(
  agentInfo: AgentDetails,
  mfaVerified: boolean
): Promise<SecureSessionToken> {
  
  // 1. Generate session-specific secrets
  const sessionId = crypto.randomBytes(16).toString('hex');
  const nonce = crypto.randomBytes(16).toString('hex');
  
  // 2. Create token data with security constraints
  const tokenData = {
    session_id: sessionId,
    agent_id: agentInfo.near_account,
    nostr_pubkey: agentInfo.nostr_pubkey,
    issued_at: Date.now(),
    expires_at: Date.now() + 3600000,  // 1 hour
    capabilities: agentInfo.capabilities,
    
    // Security fields
    nonce: nonce,
    mfa_verified: mfaVerified,
    device_fingerprint: await getDeviceFingerprint(),
    ip_restriction: agentInfo.ip_whitelist || null,
    
    // Constraints
    max_concurrent_tasks: agentInfo.max_concurrent_tasks,
    allowed_relays: agentInfo.allowed_relays,
    
    // Add entropy
    random_salt: crypto.randomBytes(8).toString('hex')
  };
  
  // 3. Sign with admin key
  const tokenString = JSON.stringify(tokenData);
  const signature = await adminKey.sign(tokenString);
  
  // 4. Encode token
  const token = Buffer.from(tokenString).toString('base64');
  
  return {
    token,
    signature,
    session_id: sessionId,
    ...tokenData
  };
}
```

#### Session Validation

```javascript
async function validateSession(
  sessionToken: string,
  signature: string
): Promise<ValidationResult> {
  
  // 1. Decode token
  const tokenData = JSON.parse(
    Buffer.from(sessionToken, 'base64').toString('utf8')
  );
  
  // 2. Check expiration
  if (Date.now() > tokenData.expires_at) {
    return { valid: false, error: 'Session expired' };
  }
  
  // 3. Check revocation status
  if (tokenData.revoked) {
    return { valid: false, error: 'Session revoked' };
  }
  
  // 4. Verify signature
  const isSignatureValid = await verifyAdminSignature(
    tokenData,
    signature
  );
  
  if (!isSignatureValid) {
    return { valid: false, error: 'Invalid signature' };
  }
  
  // 5. Check IP restriction
  if (tokenData.ip_restriction) {
    const currentIP = await getClientIP();
    const allowedIPs = tokenData.ip_restriction.split(',');
    
    if (!allowedIPs.includes(currentIP)) {
      return { valid: false, error: 'IP address not allowed' };
    }
  }
  
  // 6. Check device fingerprint
  const currentFingerprint = await getDeviceFingerprint();
  if (tokenData.device_fingerprint !== currentFingerprint) {
    return { valid: false, error: 'Device mismatch' };
  }
  
  // 7. Verify nonce not reused
  const nonceUsed = await checkNonceUsed(tokenData.nonce);
  if (nonceUsed) {
    return { valid: false, error: 'Replay attack detected' };
  }
  
  return {
    valid: true,
    session: tokenData
  };
}
```

### 3.3 Role-Based Access Control (RBAC)

#### Define Roles and Permissions

```typescript
// Role definitions
enum AgentRole {
  ADMIN = 'admin',              // Full control
  OPERATOR = 'operator',        // Operations and monitoring
  AUDITOR = 'auditor',          // Read-only access
  AGENT = 'agent',              // Regular agent
  SERVICE = 'service'            // Service account (orchestrator)
}

// Permission definitions
enum Permission {
  // Admin permissions
  WHITELIST_MANAGE = 'whitelist:manage',
  WHITELIST_REVOKE = 'whitelist:revoke',
  CONFIG_UPDATE = 'config:update',
  
  // Operator permissions
  AGENT_SUSPEND = 'agent:suspend',
  AGENT_MONITOR = 'agent:monitor',
  METRICS_VIEW = 'metrics:view',
  
  // Auditor permissions
  LOGS_VIEW = 'logs:view',
  AGENT_STATUS_VIEW = 'agent:status:view',
  WHITELIST_VIEW = 'whitelist:view',
  
  // Agent permissions
  TASK_RECEIVE = 'task:receive',
  TASK_SUBMIT = 'task:submit',
  CAPABILITY_DECLARE = 'capability:declare'
}

// Role-permission mapping
const rolePermissions = {
  [AgentRole.ADMIN]: [
    Permission.WHITELIST_MANAGE,
    Permission.WHITELIST_REVOKE,
    Permission.CONFIG_UPDATE,
    Permission.AGENT_SUSPEND,
    Permission.AGENT_MONITOR,
    Permission.METRICS_VIEW,
    Permission.LOGS_VIEW,
    Permission.AGENT_STATUS_VIEW,
    Permission.WHITELIST_VIEW
  ],
  
  [AgentRole.OPERATOR]: [
    Permission.AGENT_SUSPEND,
    Permission.AGENT_MONITOR,
    Permission.METRICS_VIEW
  ],
  
  [AgentRole.AUDITOR]: [
    Permission.LOGS_VIEW,
    Permission.AGENT_STATUS_VIEW,
    Permission.WHITELIST_VIEW
  ],
  
  [AgentRole.AGENT]: [
    Permission.TASK_RECEIVE,
    Permission.TASK_SUBMIT,
    Permission.CAPABILITY_DECLARE
  ]
};
```

#### RBAC Enforcement

```javascript
class RBACEnforcer {
  constructor(config) {
    this.roleAssignments = new Map();
    this.permissionCache = new Map();
  }
  
  // Assign role to agent
  async assignRole(agentId: string, role: AgentRole) {
    const assignment = {
      role,
      assigned_at: Date.now(),
      assigned_by: this.currentAdmin,
      expires_at: null
    };
    
    this.roleAssignments.set(agentId, assignment);
    await this.persistAssignment(agentId, assignment);
    
    // Clear permission cache
    this.permissionCache.delete(agentId);
  }
  
  // Check if agent has permission
  async hasPermission(agentId: string, permission: Permission): Promise<boolean> {
    // Check cache first
    if (this.permissionCache.has(agentId)) {
      const permissions = this.permissionCache.get(agentId);
      return permissions.includes(permission);
    }
    
    // Get agent's role
    const assignment = this.roleAssignments.get(agentId);
    if (!assignment) {
      return false;  // No role = no permissions
    }
    
    // Check if assignment is expired
    if (assignment.expires_at && Date.now() > assignment.expires_at) {
      return false;
    }
    
    // Get permissions for role
    const rolePermissions = rolePermissions[assignment.role];
    const hasPermission = rolePermissions.includes(permission);
    
    // Cache result (5 minute TTL)
    this.permissionCache.set(agentId, rolePermissions);
    setTimeout(() => this.permissionCache.delete(agentId), 300000);
    
    return hasPermission;
  }
  
  // Require permission (throws if not authorized)
  async requirePermission(agentId: string, permission: Permission, action: string) {
    const hasPerms = await this.hasPermission(agentId, permission);
    
    if (!hasPerms) {
      // Log security violation
      await this.logSecurityViolation({
        agent: agentId,
        action: action,
        required_permission: permission,
        timestamp: Date.now()
      });
      
      // Throw error
      throw new Error(`Permission denied: ${permission} required for ${action}`);
    }
  }
}
```

---

## 🔒 Layer 4: Data Security

### 4.1 Encryption Requirements

#### Mandatory Encryption for Sensitive Data

```javascript
const encryptionRequirements = {
  // Communication encryption
  communication: {
    algorithm: 'xchacha20-poly1305',
    key_derivation: 'HKDF-SHA256',
    key_length: 256,
    nonce_length: 192,
    required_for: [
      'task_details',
      'agent_credentials',
      'financial_data',
      'admin_operations'
    ]
  },
  
  // Storage encryption
  storage: {
    algorithm: 'aes-256-gcm',
    key_derivation: 'PBKDF2',
    iterations: 100000,
    salt_length: 128,
    required_for: [
      'agent_private_keys',
      'session_tokens',
      'whitelist_data'
    ]
  },
  
  // In-transit encryption
  transit: {
    protocol: 'TLS 1.3',
    cipher_suites: [
      'TLS_AES_256_GCM_SHA384',
      'TLS_CHACHA20_POLY1305_SHA256'
    ],
    key_exchange: 'ECDHE',
    required_for: 'all_relay_communications'
  }
};
```

#### Encryption Implementation

```javascript
class SecureEncryptor {
  constructor() {
    this.keyCache = new Map();
  }
  
  // Encrypt data
  async encrypt(
    data: any,
    recipientPubkey: string,
    options: EncryptOptions = {}
  ): Promise<EncryptedData> {
    
    // 1. Derive shared secret using ECDH
    const sharedSecret = await this.deriveSharedSecret(
      this.privateKey,
      recipientPubkey
    );
    
    // 2. Derive encryption key using HKDF
    const encryptionKey = await hkdfSha256(
      sharedSecret,
      options.salt || crypto.randomBytes(16),
      'encryption',
      32  // 256 bits
    );
    
    // 3. Generate random nonce
    const nonce = crypto.randomBytes(12);
    
    // 4. Serialize data
    const plaintext = JSON.stringify(data);
    const plaintextBytes = Buffer.from(plaintext, 'utf8');
    
    // 5. Encrypt using XChaCha20-Poly1305
    const ciphertext = await xchacha20Poly1305Encrypt(
      plaintextBytes,
      encryptionKey,
      nonce
    );
    
    // 6. Create encrypted data structure
    const encryptedData: EncryptedData = {
      algorithm: 'xchacha20-poly1305',
      nonce: nonce.toString('hex'),
      ciphertext: ciphertext.toString('base64'),
      public_key: this.publicKey,
      recipient_key: recipientPubkey,
      timestamp: Date.now(),
      version: 1
    };
    
    return encryptedData;
  }
  
  // Decrypt data
  async decrypt(
    encryptedData: EncryptedData,
    senderPubkey: string
  ): Promise<any> {
    
    // 1. Derive shared secret
    const sharedSecret = await this.deriveSharedSecret(
      this.privateKey,
      senderPubkey
    );
    
    // 2. Derive decryption key using HKDF
    const decryptionKey = await hkdfSha256(
      sharedSecret,
      encryptedData.salt,
      'decryption',
      32
    );
    
    // 3. Parse encrypted data
    const nonce = Buffer.from(encryptedData.nonce, 'hex');
    const ciphertext = Buffer.from(encryptedData.ciphertext, 'base64');
    
    // 4. Decrypt using XChaCha20-Poly1305
    const plaintext = await xchacha20Poly1305Decrypt(
      ciphertext,
      decryptionKey,
      nonce
    );
    
    // 5. Parse plaintext
    const data = JSON.parse(plaintext.toString('utf8'));
    
    return data;
  }
  
  // Derive shared secret using ECDH (secp256k1)
  async deriveSharedSecret(
    privateKey: string,
    publicKey: string
  ): Promise<Buffer> {
    
    const privKey = secp256k1.keyFromPrivate(privateKey);
    const pubKey = secp256k1.keyFromPublic(publicKey);
    
    // ECDH: shared_secret = privKey * pubKey (X coordinate only)
    const sharedPoint = privKey.pubKey.mul(pubKey.privKey);
    const sharedSecret = sharedPoint.x.toArray();
    
    return Buffer.from(sharedSecret);
  }
}
```

### 4.2 Data Validation & Sanitization

#### Input Validation

```javascript
class InputValidator {
  // Validate Nostr pubkey
  static validateNostrPubkey(pubkey: string): ValidationResult {
    // Must be 64 hex characters
    if (!/^[a-f0-9]{64}$/.test(pubkey)) {
      return { valid: false, error: 'Invalid Nostr pubkey format' };
    }
    
    // Must start with npub prefix (optional, depends on context)
    if (!pubkey.startsWith('npub1')) {
      return { valid: false, error: 'Invalid Nostr pubkey prefix' };
    }
    
    return { valid: true };
  }
  
  // Validate NEAR account
  static validateNEARAccount(account: string): ValidationResult {
    // Must end with .near
    if (!account.endsWith('.near')) {
      return { valid: false, error: 'Invalid NEAR account format' };
    }
    
    // Must be 32-64 characters
    if (account.length < 32 || account.length > 64) {
      return { valid: false, error: 'Invalid NEAR account length' };
    }
    
    // Only lowercase letters, numbers, and hyphens
    if (!/^[a-z0-9-]+$/.test(account)) {
      return { valid: false, error: 'Invalid NEAR account characters' };
    }
    
    return { valid: true };
  }
  
  // Validate capabilities
  static validateCapabilities(capabilities: string[]): ValidationResult {
    const allowedCapabilities = [
      'code-analysis',
      'security-scan',
      'data-processing',
      'optimization',
      'testing',
      'documentation'
    ];
    
    if (capabilities.length === 0) {
      return { valid: false, error: 'At least one capability required' };
    }
    
    if (capabilities.length > 10) {
      return { valid: false, error: 'Maximum 10 capabilities allowed' };
    }
    
    for (const cap of capabilities) {
      if (!allowedCapabilities.includes(cap)) {
        return { valid: false, error: `Invalid capability: ${cap}` };
      }
    }
    
    return { valid: true };
  }
  
  // Sanitize JSON input
  static sanitizeJSON(input: any): any {
    if (typeof input !== 'object') {
      throw new Error('Input must be object');
    }
    
    const sanitized = JSON.parse(JSON.stringify(input));
    
    // Remove dangerous properties
    const dangerousProps = ['__proto__', '__define__', 'constructor'];
    dangerousProps.forEach(prop => delete sanitized[prop]);
    
    // Validate string lengths
    const maxLength = 10000;
    Object.keys(sanitized).forEach(key => {
      if (typeof sanitized[key] === 'string') {
        if (sanitized[key].length > maxLength) {
          throw new Error(`String too long: ${key}`);
        }
      }
    });
    
    return sanitized;
  }
}
```

#### Output Encoding

```javascript
class OutputEncoder {
  // Prevent XSS by encoding output
  static encodeHTML(data: any): string {
    if (typeof data !== 'string') {
      data = JSON.stringify(data);
    }
    
    // Escape HTML special characters
    return data
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  
  // JSON with safe serialization
  static toJSON(data: any): string {
    return JSON.stringify(data, null, 2);  // Pretty print with 2-space indent
  }
  
  // Base64 encode binary data
  static toBase64(data: Buffer): string {
    return data.toString('base64url');  // URL-safe base64
  }
  
  // Hex encode binary data
  static toHex(data: Buffer): string {
    return data.toString('hex');
  }
}
```

### 4.3 Secure File Handling

#### File Upload Security

```javascript
class SecureFileHandler {
  async uploadFile(
    file: File,
    options: UploadOptions
  ): Promise<UploadResult> {
    
    // 1. Validate file type
    const allowedTypes = [
      'application/json',
      'text/plain',
      'application/x-yaml',
      'application/zip'
    ];
    
    if (!allowedTypes.includes(file.type)) {
      throw new Error(`Invalid file type: ${file.type}`);
    }
    
    // 2. Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;  // 10MB
    if (file.size > maxSize) {
      throw new Error('File too large: max 10MB allowed');
    }
    
    // 3. Scan file for malware (if service available)
    if (options.virusScan) {
      const scanResult = await this.virusScan(file);
      if (!scanResult.clean) {
        throw new Error('File contains malware');
      }
    }
    
    // 4. Calculate file hash
    const fileHash = await this.calculateSHA256(file);
    
    // 5. Encrypt file if sensitive
    if (options.encrypt) {
      const encryptedFile = await this.encryptFile(file);
      
      // Upload encrypted file
      const uploadResult = await this.uploadToStorage(encryptedFile);
      
      return {
        ...uploadResult,
        original_hash: fileHash,
        encryption: 'aes-256-gcm',
        encrypted: true
      };
    }
    
    // 6. Upload file
    const uploadResult = await this.uploadToStorage(file);
    
    return {
      ...uploadResult,
      file_hash: fileHash,
      encryption: false
    };
  }
  
  async calculateSHA256(file: File): Promise<string> {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const buffer = e.target.result;
        const hashBuffer = await crypto.subtle.digest(
          'SHA-256',
          buffer
        );
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        resolve(hashHex);
      };
      reader.readAsArrayBuffer(file);
    });
  }
}
```

---

## 🔒 Layer 5: Smart Contract Security

### 5.1 Contract Security Best Practices

#### Access Control

```rust
// Only admin can call sensitive methods
#[near_bindgen]
impl AgentWhitelist {
    #[payable]
    pub fn revoke_agent(
        &mut self,
        nostr_pubkey: String,
        reason: String
    ) {
        // Validate caller is admin
        let caller = env::predecessor_account_id();
        assert_eq!(caller, self.admin, "Only admin can revoke agents");
        
        // Additional validation
        self._validate_revoke_request(&nostr_pubkey);
        
        // Execute revocation
        self._execute_revoke(&nostr_pubkey, &reason);
    }
    
    fn _validate_revoke_request(&self, nostr_pubkey: &str) {
        // Check agent exists
        assert!(self.agents.contains_key(&nostr_pubkey), 
                 "Agent not found");
        
        // Check agent is not admin
        assert_neq!(self.near_to_near.get(&nostr_pubkey), 
                     Some(&self.admin), 
                     "Cannot revoke admin");
        
        // Validate reason length
        assert!(reason.len() <= 500, 
                 "Reason too long");
    }
}
```

#### Stake Management

```rust
// Secure stake handling with minimums and maximums
#[near_bindgen]
impl AgentWhitelist {
    #[payable]
    pub fn register_agent(
        &mut self,
        nostr_pubkey: String,
        proof: OwnershipProof,
        capabilities: Vec<String>
    ) {
        let caller = env::predecessor_account_id();
        let deposit = env::attached_deposit();
        
        // Minimum stake requirement
        const MIN_STAKE: u128 = 10_000_000_000_000_000_000;  // 10 NEAR
        assert!(deposit >= MIN_STAKE, 
                 "Stake amount too low");
        
        // Maximum stake limit (prevent over-staking)
        const MAX_STAKE: u128 = 1000_000_000_000_000_000_000;  // 1000 NEAR
        assert!(deposit <= MAX_STAKE, 
                 "Stake amount exceeds maximum");
        
        // Verify ownership proof
        assert!(self._verify_ownership_proof(&caller, &nostr_pubkey, &proof),
                 "Invalid ownership proof");
        
        // Validate proof freshness
        let now = env::block_timestamp();
        let proof_age = now - proof.verified_at;
        const MAX_PROOF_AGE: u64 = 360000000000;  // 1 hour in nanoseconds
        assert!(proof_age <= MAX_PROOF_AGE, 
                 "Ownership proof too old");
        
        // Validate capabilities
        assert!(!capabilities.is_empty(), "At least one capability required");
        assert!(capabilities.len() <= 10, "Maximum 10 capabilities allowed");
        
        // Create agent record
        let agent = WhitelistedAgent {
            near_account: caller.clone(),
            nostr_pubkey,
            registered_at: now,
            status: AgentStatus::Active,
            stake_amount: deposit,
            capabilities,
            proof,
            last_active: now,
            metrics: AgentMetrics::default()
        };
        
        // Store in whitelist
        self.agents.insert(&caller, &agent);
        self.nostr_to_near.insert(&nostr_pubkey, &caller);
        
        // Update total stake counter
        self.total_staked += deposit;
        
        env::log_str(&format!(
            "Agent registered: {} (Nostr: {}), Stake: {}",
            caller, nostr_pubkey, deposit
        ));
    }
    
    fn _verify_ownership_proof(
        &self,
        caller: &AccountId,
        nostr_pubkey: &str,
        proof: &OwnershipProof
    ) -> bool {
        // Verify message contains correct fields
        let signed_data: serde_json::Value = 
            serde_json::from_str(&proof.message).unwrap();
        
        // Verify near_account matches
        if let Some(data_near) = signed_data.get("near_account") {
            assert_eq!(data_near.as_str().unwrap(), caller.as_str(),
                       "NEAR account mismatch in proof");
        }
        
        // Verify nostr_pubkey matches
        if let Some(data_nostr) = signed_data.get("nostr_pubkey") {
            assert_eq!(data_nostr.as_str().unwrap(), nostr_pubkey,
                       "Nostr pubkey mismatch in proof");
        }
        
        // Verify timestamp is recent
        if let Some(data_timestamp) = signed_data.get("timestamp") {
            let timestamp: u64 = data_timestamp.as_u64().unwrap();
            let now = env::block_timestamp();
            let age = now - timestamp;
            const MAX_AGE: u64 = 360000000000;  // 1 hour
            assert!(age <= MAX_AGE, "Proof too old");
        }
        
        // Verify action is correct
        if let Some(data_action) = signed_data.get("action") {
            assert_eq!(data_action.as_str().unwrap(), "register_agent",
                       "Invalid action in proof");
        }
        
        // Verify signature (would integrate Nostr schnorr verification)
        // This is a placeholder - actual implementation would verify
        // the Schnorr signature against the message
        true
    }
}
```

#### Slashing Mechanism

```rust
// Secure slashing with clear rules
#[near_bindgen]
impl AgentWhitelist {
    pub fn revoke_agent(
        &mut self,
        nostr_pubkey: String,
        reason: String
    ) {
        let caller = env::predecessor_account_id();
        let mut agent = self.agents.get(&nostr_pubkey).unwrap();
        
        // Slash calculation
        let slash_amount = agent.stake_amount / 10;  // 10% slash
        
        // Maximum slash cap (never slash more than 50%)
        let max_slash = agent.stake_amount / 2;
        if slash_amount > max_slash {
            slash_amount = max_slash;
        }
        
        // Send slashed amount to treasury
        Promise::new(self.treasury.clone())
            .transfer(slash_amount);
        
        // Update agent status
        agent.status = AgentStatus::Revoked;
        
        // Store in whitelist
        self.agents.insert(&nostr_pubkey, &agent);
        
        // Log slashing
        env::log_str(&format!(
            "Agent {} slashed {} (10%), Remaining refunded to treasury: {}",
            nostr_pubkey, slash_amount, agent.stake_amount - slash_amount
        ));
        
        // Store slash record for audit
        let slash_record = SlashRecord {
            agent_nostr: nostr_pubkey.clone(),
            near_account: caller.clone(),
            slashed_amount: slash_amount,
            reason: reason.clone(),
            slashed_at: env::block_timestamp(),
            slashed_by: caller.clone()
        };
        
        self.slash_history.insert(&format!("{}:{}", nostr_pubkey, env::block_timestamp()), 
                                &slash_record);
    }
    
    // Slash for inactivity (automated)
    pub fn check_and_slash_inactive_agents(&mut self) {
        let now = env::block_timestamp();
        const INACTIVE_THRESHOLD: u64 = 86400000000 * 30;  // 30 days
        
        for (nostr_pubkey, mut agent) in self.agents.iter_mut() {
            // Skip already revoked agents
            if matches!(agent.status, AgentStatus::Active) {
                continue;
            }
            
            // Check last activity
            let inactive_duration = now - agent.last_active;
            
            if inactive_duration > INACTIVE_THRESHOLD {
                // Slash and revoke inactive agent
                let slash_amount = agent.stake_amount;  // 100% slash for inactivity
                Promise::new(self.treasury.clone())
                    .transfer(slash_amount);
                
                agent.status = AgentStatus::Revoked;
                self.agents.insert(&nostr_pubkey, &agent);
                
                env::log_str(&format!(
                    "Agent {} slashed due to inactivity ({} days)",
                    nostr_pubkey, inactive_duration / 86400000000
                ));
            }
        }
    }
}
```

### 5.2 Contract Upgrade Security

#### Pauseable Contract Pattern

```rust
// Allow pausing of critical operations
#[near_bindgen]
impl AgentWhitelist {
    pub is_paused(&self) -> bool {
        self.paused
    }
    
    pub fn pause(&mut self) {
        let caller = env::predecessor_account_id();
        assert_eq!(caller, self.admin, "Only admin can pause");
        self.paused = true;
        
        env::log_str("Contract paused");
    }
    
    pub fn unpause(&mut self) {
        let caller = env::predecessor_account_id();
        assert_eq!(caller, self.admin, "Only admin can unpause");
        self.paused = false;
        
        env::log_str("Contract unpaused");
    }
    
    // Wrap all state-changing methods
    #[payable]
    pub fn register_agent(
        &mut self,
        nostr_pubkey: String,
        proof: OwnershipProof,
        capabilities: Vec<String>
    ) {
        assert!(!self.is_paused(), "Contract is paused");
        
        // ... rest of method
    }
}
```

#### Upgrade Governance

```rust
// Time-delayed upgrade pattern
#[near_bindgen]
impl AgentWhitelist {
    pub fn propose_upgrade(&mut self, new_code_hash: String) {
        let caller = env::predecessor_account_id();
        assert_eq!(caller, self.admin, "Only admin can propose upgrades");
        
        let proposal = UpgradeProposal {
            proposed_by: caller.clone(),
            new_code_hash,
            proposed_at: env::block_timestamp(),
            status: ProposalStatus::Pending,
            votes_for: 0,
            votes_against: 0
        };
        
        self.upgrade_proposals.insert(&new_code_hash, &proposal);
        
        env::log_str(&format!("Upgrade proposed: {}", new_code_hash));
    }
    
    pub fn vote_upgrade(
        &mut self,
        proposal_hash: String,
        vote: bool
    ) {
        let caller = env::predecessor_account_id();
        let mut proposal = self.upgrade_proposals.get(&proposal_hash).unwrap();
        
        // Check caller is admin
        assert!(self.is_admin(&caller), "Only admins can vote");
        
        // Check proposal is pending
        assert!(matches!(proposal.status, ProposalStatus::Pending),
                 "Proposal not pending");
        
        // Record vote
        if vote {
            proposal.votes_for += 1;
        } else {
            proposal.votes_against += 1;
        }
        
        self.upgrade_proposals.insert(&proposal_hash, &proposal);
        
        // Check if vote threshold met (2/3 majority)
        let total_votes = proposal.votes_for + proposal.votes_against;
        let required_votes = (self.admin_count * 2) / 3 + 1;
        
        if total_votes >= required_votes {
            proposal.status = ProposalStatus::Approved;
            self.upgrade_proposals.insert(&proposal_hash, &proposal);
            
            // Implement upgrade
            self._implement_upgrade(proposal.new_code_hash);
        }
        
        env::log_str(&format!("Vote recorded for {}: {}", proposal_hash, vote));
    }
    
    fn _implement_upgrade(&mut self, new_code_hash: String) {
        // Apply upgrade
        self.upgrade_code_hash = new_code_hash;
        self.upgrade_implemented_at = env::block_timestamp();
        
        env::log_str(&format!("Upgrade implemented: {}", new_code_hash));
    }
}
```

---

## 🔒 Layer 6: Agent Security

### 6.1 Agent Sandboxing

#### Docker Container Security

```dockerfile
# Secure agent container
FROM node:18-alpine

# Non-root user
RUN addgroup -g agent && \
    adduser -D -G agent -u 1000 agent

# Install security updates
RUN apk update && apk upgrade && \
    apk add --no-cache dumb-init gosu

# Set working directory
WORKDIR /app

# Copy application
COPY --chown=agent:agent package*.json ./
RUN npm ci --only=production

# Copy application
COPY --chown=agent:agent . .

# Set filesystem permissions
RUN chown -R agent:agent /app && \
    chmod -R 755 /app && \
    chmod -R +x /app/node_modules/.bin

# Remove unnecessary dependencies
RUN apk del git

# Security settings
ENV NODE_ENV=production
ENV NODE_OPTIONS="--max-old-space-size=4096"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node healthcheck.js || exit 1

# Non-root entrypoint
ENTRYPOINT ["dumb-init", "--"]
CMD ["gosu", "agent", "node", "index.js"]
```

#### Resource Limits

```yaml
# Kubernetes resource limits
apiVersion: v1
kind: Pod
metadata:
  name: agent-pod
  labels:
    app: agent
spec:
  securityContext:
    # Non-root
    runAsUser: 1000
    runAsGroup: 1000
    # Read-only root filesystem
    readOnlyRootFilesystem: true
    # Drop all capabilities
    capabilities:
      drop:
        - ALL
  containers:
    - name: agent
      image: agent:latest
      resources:
        limits:
          cpu: "1"
          memory: "1Gi"
        requests:
          cpu: "500m"
          memory: "512Mi"
      securityContext:
        # No privilege escalation
        allowPrivilegeEscalation: false
        # Read-only root filesystem
        readOnlyRootFilesystem: true
        # Seccomp profile
        seccompProfile:
          type: RuntimeDefault
```

### 6.2 Agent Credential Security

#### Secure Credential Management

```javascript
class AgentCredentialManager {
  constructor(config) {
    this.credentials = new Map();
    this.rotationSchedule = new Map();
  }
  
  // Generate and store credentials securely
  async generateCredentials(agentId: string): Promise<Credentials> {
    
    // 1. Generate strong random credentials
    const password = crypto.randomBytes(32).toString('base64');
    const apiKey = crypto.randomBytes(32).toString('hex');
    
    // 2. Hash password
    const passwordHash = await crypto.hash('sha256', password);
    
    // 3. Store in secure storage (encrypted)
    const encryptedCredentials = {
      agent_id: agentId,
      password_hash: passwordHash,
      api_key: apiKey,
      created_at: Date.now(),
      expires_at: Date.now() + 86400000 * 30  // 30 days
    };
    
    await this.secureStorage.store(
      `credentials:${agentId}`,
      JSON.stringify(encryptedCredentials),
      { encryption: true }
    );
    
    // 4. Schedule rotation
    await this.scheduleRotation(agentId, 86400000 * 7);  // 7 days
    
    return {
      agent_id: agentId,
      api_key: apiKey,
      // Never return plain password
      password_hash: passwordHash,
      expires_at: Date.now() + 86400000 * 30
    };
  }
  
  // Validate credentials
  async validateCredentials(
    agentId: string,
    providedHash: string
  ): Promise<boolean> {
    
    // Get stored credentials
    const encrypted = await this.secureStorage.get(`credentials:${agentId}`);
    const credentials = JSON.parse(await this.decrypt(encrypted));
    
    // Check expiration
    if (Date.now() > credentials.expires_at) {
      return false;
    }
    
    // Check password hash
    if (credentials.password_hash !== providedHash) {
      return false;
    }
    
    return true;
  }
}
```

### 6.3 Agent Network Security

#### Outbound Connection Security

```javascript
class AgentNetworkSecurity {
  constructor(config) {
    this.allowedEndpoints = new Set(config.allowedEndpoints);
    this.blockedDomains = new Set(config.blockedDomains);
  }
  
  // Validate endpoint before connecting
  validateEndpoint(url: string): boolean {
    try {
      const parsed = new URL(url);
      
      // Check protocol
      if (!['https:', 'wss:'].includes(parsed.protocol)) {
        console.error(`Insecure protocol: ${parsed.protocol}`);
        return false;
      }
      
      // Check domain
      const domain = parsed.hostname;
      
      // Block known malicious domains
      if (this.blockedDomains.has(domain)) {
        console.error(`Blocked domain: ${domain}`);
        return false;
      }
      
      // Check if domain is whitelisted
      if (!this.allowedEndpoints.has(domain)) {
        console.error(`Domain not whitelisted: ${domain}`);
        return false;
      }
      
      // Check for IP addresses in hostname
      if (this.isIPAddress(domain)) {
        console.error(`Direct IP address not allowed: ${domain}`);
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('Invalid URL:', error);
      return false;
    }
  }
  
  // Secure WebSocket connection
  async connectSecurely(relayUrl: string): Promise<WebSocket> {
    
    // Validate endpoint
    if (!this.validateEndpoint(relayUrl)) {
      throw new Error('Invalid endpoint');
    }
    
    // Create WebSocket with security options
    const ws = new WebSocket(relayUrl, {
      // Protocol version
      protocolVersion: 13,
      
      // Connection timeout
      handshakeTimeout: 10000,
      
      // TLS configuration
      rejectUnauthorized: true,
      
      // Headers (for authentication)
      headers: {
        'User-Agent': 'Nostr-Agent/1.0.0'
      },
      
      // Origin check
      origin: this.config.allowedOrigin
    });
    
    return ws;
  }
  
  isIP(value: string): boolean {
    const ipPattern = /^(\d{1,3}\.){3}\d{1,3}$/;
    return ipPattern.test(value);
  }
}
```

---

## 🔒 Layer 7: Operational Security

### 7.1 Monitoring & Alerting

#### Security Metrics to Monitor

```javascript
const securityMetrics = {
  // Authentication metrics
  authentication: {
    success_rate: { target: 0.95, alert_below: 0.85 },
    failure_rate: { target: 0.05, alert_above: 0.15 },
    mfa_failure_rate: { target: 0.01, alert_above: 0.05 }
  },
  
  // Agent behavior metrics
  agent_behavior: {
    abnormal_task_failure_rate: { threshold: 0.10 },
    suspicious_activity_score: { threshold: 0.7 },
    inactivity_period: { threshold: 86400000 * 7 },  // 7 days
    multiple_login_attempts: { threshold: 5 }
  },
  
  // Network metrics
  network: {
    unusual_ip_connections: { threshold: 3 },
    connection_time_anomalies: { threshold: 2 },  // Standard deviations
    bandwidth_usage: { alert_above: 0.8 }  // 80% of capacity
  },
  
  // Contract metrics
  contract: {
    unusual_revoke_rate: { threshold: 0.05 },
    failed_registration_attempts: { threshold: 10 },
    stake_anomalies: { threshold: 0.5 }  // Standard deviations
  },
  
  // Data metrics
  data: {
    unauthorized_access_attempts: { threshold: 10 },
    unusual_file_uploads: { threshold: 0.1 },
    data_exfiltration_indicators: { threshold: 0.05 }
  }
};

// Security monitoring service
class SecurityMonitor {
  constructor(config) {
    this.metrics = new Map();
    this.alerts = new Map();
    this.alertingRules = securityMetrics;
  }
  
  async monitorMetrics() {
    const currentMetrics = await this.collectMetrics();
    
    // Check each metric category
    for (const [category, metrics] of Object.entries(currentMetrics)) {
      for (const [metric, value] of Object.entries(metrics)) {
        const rule = this.alertingRules[category]?.[metric];
        
        if (rule) {
          const alert = this.evaluateRule(rule, value);
          
          if (alert) {
            await this.triggerSecurityAlert(category, metric, value, alert);
          }
        }
      }
    }
  }
  
  evaluateRule(rule: any, value: number): Alert | null {
    if (rule.alert_below && value < rule.alert_below) {
      return {
        severity: 'warning',
        message: `${value} is below threshold ${rule.alert_below}`
      };
    }
    
    if (rule.alert_above && value > rule.alert_above) {
      return {
        severity: 'critical',
        message: `${value} exceeds threshold ${rule.alert_above}`
      };
    }
    
    if (rule.threshold && value > rule.threshold) {
      return {
        severity: 'warning',
        message: `${value} exceeds threshold ${rule.threshold}`
      };
    }
    
    return null;
  }
  
  async triggerSecurityAlert(
    category: string,
    metric: string,
    value: any,
    alert: Alert
  ) {
    const securityAlert: SecurityAlert = {
      timestamp: Date.now(),
      category,
      metric,
      value,
      severity: alert.severity,
      message: alert.message,
      source: 'security-monitor'
    };
    
    // Log alert
    await this.logSecurityAlert(securityAlert);
    
    // Send notifications
    await this.sendNotifications(securityAlert);
    
    // Check if immediate action required
    if (alert.severity === 'critical') {
      await this.triggerAutomaticResponse(securityAlert);
    }
  }
}
```

### 7.2 Incident Response

#### Security Incident Response Playbook

```javascript
const incidentResponse = {
  // Phase 1: Detection & Identification (0-15 minutes)
  detection: {
    indicators: [
      'Multiple failed authentication attempts from single IP',
      'Agent making unusual number of API calls',
      'Sudden spike in task failure rates',
      'Unauthorized access attempts to protected resources',
      'Unusual data access patterns'
    ],
    
    actions: [
      'Verify incident through multiple sources',
      'Assess scope and impact',
      'Identify affected systems and agents',
      'Document initial findings',
      'Notify incident response team'
    ],
    
    owner: 'Security Lead',
    timeline: '15 minutes'
  },
  
  // Phase 2: Containment (15-60 minutes)
  containment: {
    strategies: [
      'Isolate affected agents (suspend from whitelist)',
      'Block malicious IP addresses',
      'Revoke compromised credentials',
      'Stop affected services if necessary',
      'Increase monitoring and logging'
    ],
    
    actions: [
      'Suspend affected agents from whitelist',
      'Revoke active sessions for affected agents',
      'Block source IP addresses at firewall',
      'Change admin credentials',
      'Enable additional rate limiting',
      'Enable audit logging for all actions'
    ],
    
    owner: 'Security Lead',
    timeline: '45 minutes'
  },
  
  // Phase 3: Eradication (1-4 hours)
  eradication: {
    strategies: [
      'Identify root cause',
      'Patch vulnerabilities',
      'Remove malware/backdoors',
      'Update all affected systems',
      'Verify no compromise remains'
    ],
    
    actions: [
      'Perform forensic analysis on compromised systems',
      'Identify attack vector and root cause',
      'Patch security vulnerabilities',
      'Update agent software/firmware',
      'Update relay software',
      'Update NEAR contract if compromised',
      'Scan for persistence mechanisms',
      'Verify no additional backdoors',
      'Change all passwords/credentials',
      'Review and update security policies'
    ],
    
    owner: 'Engineering Lead',
    timeline: '2 hours'
  },
  
  // Phase 4: Recovery (2-6 hours)
  recovery: {
    strategies: [
      'Restore systems from clean backups',
      'Validate system integrity',
      'Gradually restore service',
      'Monitor for re-infection'
    ],
    
    actions: [
      'Restore agent systems from known good state',
      'Restore relay systems from backups',
      'Verify NEAR contract state',
      'Re-enable suspended agents (after verification)',
      'Re-establish network connections',
      'Validate all systems are functioning',
      'Conduct security audit before full restoration',
      'Monitor for 24-48 hours post-restoration'
    ],
    
    owner: 'Operations Lead',
    timeline: '4 hours'
  },
  
  // Phase 5: Lessons Learned (1-2 weeks)
  lessons_learned: {
    deliverables: [
      'Detailed incident report',
      'Root cause analysis',
      'Timeline of events',
      'Impact assessment',
      'Recommendations for prevention',
      'Updated security policies',
      'Team training on lessons learned'
    ],
    
    actions: [
      'Conduct post-mortem analysis',
      'Document full incident timeline',
      'Identify process gaps',
      'Identify technical gaps',
      'Develop corrective actions',
      'Update security policies and procedures',
      'Provide security awareness training',
      'Update incident response playbooks',
      'Implement security improvements',
      'Schedule regular security reviews'
    ],
    
    owner: 'Security Lead',
    timeline: '1 week'
  },
  
  // Severity Classification
  severity: {
    critical: {
      examples: ['Complete system compromise', 'Data breach', 'Ransomware'],
      response_time: '< 1 hour',
      escalation: 'immediate',
      notification_channels: ['pager', 'sms', 'phone', 'slack', 'email']
    },
    
    high: {
      examples: ['Suspicious activity', 'Unauthorized access attempt'],
      response_time: '< 4 hours',
      escalation: 'within 1 hour',
      notification_channels: ['slack', 'email', 'phone']
    },
    
    medium: {
      examples: ['Minor security incident', 'Policy violation'],
      response_time: '< 24 hours',
      escalation: 'within 4 hours',
      notification_channels: ['slack', 'email']
    },
    
    low: {
      examples: ['Near miss', 'Failed security control'],
      response_time: '< 1 week',
      escalation: 'within 24 hours',
      notification_channels: ['email', 'slack']
    }
  }
};
```

#### Automated Incident Response

```javascript
class AutoIncidentResponse {
  constructor(config) {
    this.responseRules = new Map();
    this.securityEvents = [];
  }
  
  // Define automated response rules
  defineRule(ruleName: string, rule: ResponseRule) {
    this.responseRules.set(ruleName, rule);
  }
  
  async handleSecurityEvent(event: SecurityEvent): Promise<ResponseAction> {
    console.log(`Processing security event: ${event.type}`);
    
    // Determine response based on event severity
    switch (event.severity) {
      case 'critical':
        return await this.handleCriticalEvent(event);
      
      case 'high':
        return await this.handleHighEvent(event);
      
      case 'medium':
        return await this.handleMediumEvent(event);
      
      case 'low':
        return await this.handleLowEvent(event);
      
      default:
        return { action: 'ignore', reason: 'Unknown severity' };
    }
  }
  
  async handleCriticalEvent(event: SecurityEvent): Promise<ResponseAction> {
    console.log(`CRITICAL EVENT: ${event.type}`);
    
    // Immediate containment
    const actions = [];
    
    if (event.type === 'compromised_agent') {
      // 1. Immediately suspend agent from whitelist
      await this.suspendAgent(event.agentId, 'Security incident - automatic');
      actions.push({ action: 'suspend_agent', agentId: event.agentId });
      
      // 2. Revoke all active sessions
      await this.revokeAllSessions(event.agentId);
      actions.push({ action: 'revoke_sessions', agentId: event.agentId });
      
      // 3. Block IP address
      await this.blockIPAddress(event.ipAddress);
      actions.push({ action: 'block_ip', ip: event.ipAddress });
      
      // 4. Escalate to security team
      await this.escalateToSecurityTeam(event);
      actions.push({ action: 'escalate', to: 'security-team' });
      
      // 5. Start incident response procedures
      await this.initiateIncidentResponse(event);
      actions.push({ action: 'initiate_irp' });
    }
    
    return {
      action: 'executed',
      actions,
      automated: true
    };
  }
  
  async suspendAgent(agentId: string, reason: string) {
    await this.nearContract.functionCall({
      contractId: this.config.whitelistContract,
      methodName: 'suspend_agent',
      args: {
        nostr_pubkey: agentId,
        reason: reason,
        duration: String(7 * 24 * 60 * 60 * 1000000000)  // 7 days
      },
      gas: '100000000000000'
    });
    
    // Publish suspension notice
    await this.publishAgentNotice(agentId, {
      type: 'suspended',
      reason,
      automated: true,
      timestamp: Date.now()
    });
  }
  
  async revokeAllSessions(agentId: string) {
    const sessions = await this.getActiveSessions(agentId);
    
    for (const session of sessions) {
      await this.nearContract.functionCall({
        contractId: this.config.whitelistContract,
        methodName: 'revoke_sessions',
        args: {
          nostr_pubkey: agentId,
          session_ids: sessions.map(s => s.session_id)
        },
        gas: '100000000000000'
      });
    }
  }
  
  async escalateToSecurityTeam(event: SecurityEvent) {
    const incident = {
      ...event,
      escalated_at: Date.now(),
      escalated_by: 'automated-response-system',
      priority: this.getSeverityPriority(event.severity)
    };
    
    // Send notifications
    await this.sendPagerAlert(incident);
    await this.sendSlackAlert(incident);
    await this.sendEmailAlert(incident);
    
    // Log escalation
    await this.logSecurityIncident(incident);
  }
}
```

### 7.3 Backup & Disaster Recovery

#### Secure Backup Strategy

```javascript
class SecureBackupManager {
  constructor(config) {
    this.backupSchedule = config.backupSchedule || {
        agent_state: { interval: 3600000 },  // Every hour
        contract_state: { interval: 86400000 },  // Every day
        system_config: { interval: 604800000 }  // Every week
        logs: { interval: 3600000 }  // Every hour
      };
    
    this.backupLocations = config.backupLocations || {
      primary: 's3://primary-backups.near',
      secondary: 's3://secondary-backups.near',
      local: '/var/secure/backups',
      offsite: true
    };
  }
  
  // Perform secure backup
  async performBackup(backupType: string): Promise<BackupResult> {
    const timestamp = Date.now();
    
    // 1. Collect data to backup
    let data;
    switch (backupType) {
      case 'agent_state':
        data = await this.collectAgentStates();
        break;
      case 'contract_state':
        data = await this.collectContractStates();
        break;
      case 'logs':
        data = await this.collectLogs();
        break;
      default:
        throw new Error(`Unknown backup type: ${backupType}`);
    }
    
    // 2. Encrypt backup
    const encryptedBackup = await this.encryptBackup(data);
    
    // 3. Create backup manifest
    const manifest = {
      backup_type: backupType,
      timestamp,
      version: '1.0',
      data_size: encryptedBackup.length,
      encryption: 'aes-256-gcm',
      checksum: await this.calculateChecksum(encryptedBackup),
      agent_count: data.agents.length
    };
    
    // 4. Store in multiple locations (3-2-1 rule)
    const storageResults = [];
    
    try {
      const primary = await this.storeBackup(
        this.backupLocations.primary,
        encryptedBackup,
        manifest
      );
      storageResults.push({ location: 'primary', success: primary });
    } catch (error) {
      console.error('Primary backup failed:', error);
    }
    
    try {
      const secondary = await this.storeBackup(
        this.backupLocations.secondary,
        encryptedBackup,
        manifest
      );
      storageResults.push({ location: 'secondary', success: secondary });
    } catch (error) {
      console.error('Secondary backup failed:', error);
    }
    
    try {
      const local = await this.storeBackup(
        this.backupLocations.local,
        encryptedBackup,
        manifest
      );
      storageResults.push({ location: 'local', success: local });
    } catch (error) {
      console.error('Local backup failed:', error);
    }
    
    // 5. Verify at least one backup succeeded
    const successCount = storageResults.filter(r => r.success).length;
    if (successCount === 0) {
      throw new Error('All backup locations failed');
    }
    
    // 6. Log backup result
    await this.logBackupResult({
      backup_type: backupType,
      timestamp,
      storage_results: storageResults,
      success_count: successCount,
      manifest
    });
    
    return {
      success: successCount > 0,
      storage_results,
      manifest,
      timestamp
    };
  }
  
  // Restore from backup
  async restoreFromBackup(
    backupId: string,
    restoreOptions: RestoreOptions
  ): Promise<RestoreResult> {
    
    // 1. Retrieve backup
    const backup = await this.retrieveBackup(backupId);
    if (!backup) {
      throw new Error('Backup not found');
    }
    
    // 2. Verify backup manifest
    if (!this.verifyManifest(backup.manifest)) {
      throw new Error('Backup manifest verification failed');
    }
    
    // 3. Decrypt backup
    const decryptedData = await this.decryptBackup(backup.encryptedBackup);
    
    // 4. Validate backup data
    const validation = await this.validateBackupData(decryptedData);
    if (!validation.valid) {
      throw new Error(`Backup validation failed: ${validation.errors.join(', ')}`);
    }
    
    // 5. Create restore point (snapshot)
    if (restoreOptions.createSnapshot) {
      await this.createRestoreSnapshot(backupId);
    }
    
    // 6. Execute restore
    const restoreResult = await this.executeRestore(
      decryptedData,
      restoreOptions
    );
    
    // 7. Verify restore
    const verification = await this.verifyRestore(restoreResult);
    if (!verification.success) {
      throw new Error('Restore verification failed');
    }
    
    // 8. Log restore
    await this.logRestoreResult({
      backup_id: backupId,
      restore_result: restoreResult,
      verification
    });
    
    return restoreResult;
  }
}
```

---

## 🔒 Layer 8: Compliance & Auditing

### 8.1 Audit Logging

#### Comprehensive Audit Trail

```javascript
class AuditLogger {
  constructor(config) {
    this.auditLog = new Map();
    this.immutableLog = [];
    this.logStorage = config.auditLogStorage;
  }
  
  // Log all security-relevant events
  async logSecurityEvent(event: SecurityEvent): Promise<void> {
    const auditEntry = {
      id: generateUUID(),
      timestamp: Date.now(),
      event_type: event.type,
      severity: event.severity,
      source: event.source || 'system',
      
      // Actor information
      actor: {
        agent_id: event.agentId,
        nostr_pubkey: event.nostrPubkey,
        near_account: event.nearAccount,
        ip_address: event.ipAddress,
        user_agent: event.userAgent,
        session_id: event.sessionId
      },
      
      // Event details
      details: event.details,
      
      // Outcome
      outcome: event.outcome,
      action_taken: event.actionTaken,
      additional_context: event.context
      
    };
    
    // Store in immutable log
    this.immutableLog.push(auditEntry);
    
    // Store in persistent storage
    await this.logStorage.append(auditEntry);
    
    // Send to SIEM (Security Information and Event Management)
    if (event.severity === 'critical' || event.severity === 'high') {
      await this.sendToSIEM(auditEntry);
    }
  }
  
  // Generate audit report
  async generateAuditReport(
    startDate: Date,
    endDate: Date
  ): Promise<AuditReport> {
    
    // 1. Query immutable log for period
    const events = this.immutableLog.filter(e =>
      e.timestamp >= startDate.getTime() &&
      e.timestamp <= endDate.getTime()
    );
    
    // 2. Categorize events
    const categorized = {
      authentication: events.filter(e => e.event_type.startsWith('auth')),
      authorization: events.filter(e => e.event_type.startsWith('authz')),
      suspicious: events.filter(e => e.severity === 'high' || e.severity === 'critical'),
      operational: events.filter(e => e.event_type.startsWith('ops'))
    };
    
    // 3. Calculate metrics
    const metrics = {
      total_events: events.length,
      critical_events: events.filter(e => e.severity === 'critical').length,
      high_events: events.filter(e => e.severity === 'high').length,
      unique_agents: new Set(events.map(e => e.actor.agent_id)).size,
      unique_ips: new Set(events.map(e => e.actor.ip_address)).size,
      failure_rate: events.filter(e => e.outcome === 'failure').length / events.length
    };
    
    // 4. Generate recommendations
    const recommendations = this.generateRecommendations(metrics, categorized);
    
    return {
      report_id: generateUUID(),
      period: { startDate, endDate },
      metrics,
      categorized_events: categorized,
      recommendations,
      generated_at: Date.now()
    };
  }
}
```

### 8.2 Regular Security Audits

#### Audit Schedule

```yaml
# Security audit schedule
security_audits:
  # Daily audits
  daily:
    - name: "Authentication Audit"
      scope: [authentication_logs, failed_attempts, mfa_usage]
      automation: true
      alert_on_failure: true
    
    - name: "Agent Health Audit"
      scope: [agent_status, task_completion_rates, error_rates]
      automation: true
      alert_thresholds:
        task_failure_rate: 0.15
        error_rate: 0.10
  
  # Weekly audits
  weekly:
    - name: "Access Control Audit"
      scope: [role_assignments, permission_changes, unauthorized_access_attempts]
      automation: true
      review_period: 7  # Last 7 days
    
    - name: "Whitelist Audit"
      scope: [agent_registrations, revocations, status_changes]
      automation: true
      reviewer: security_team
  
  # Monthly audits
  monthly:
    - name: "Comprehensive Security Audit"
      scope: [all_logs, contracts, configurations, access_controls]
      automation: false
      reviewer: external_auditor
      report_to: [management, compliance]
    
    - name: "Penetration Testing"
      scope: [network_perimeter, application_security, contract_security]
      automation: false
      provider: security_firm
      report_to: [cto, ciso]
```

---

## ✅ Security Checklist

### Pre-Deployment Checklist

```markdown
## Network Security
- [ ] Private relay infrastructure deployed
- [ ] mTLS configured for all relay connections
- [ ] IP whitelisting configured and tested
- [ ] Network segmentation implemented
- [ ] DDoS protection enabled
- [ ] Rate limiting configured
- [ ] Firewall rules verified

## Identity Security
- [ ] Nostr key generation process documented
- [ ] Secure key storage mechanism implemented
- [ ] Key rotation schedule defined
- [ ] Ownership proof verification implemented
- [ ] NEAR account security configured
- [ ] Multi-sig for critical operations
- [ ] Hardware wallet for admin accounts

## Authentication & Authorization
- [ ] MFA implemented for admin operations
- [ ] Session token security implemented
- [ ] RBAC roles and permissions defined
- [ ] Access control enforcement active
- [ ] Session timeout configured
- [ ] Revocation mechanisms implemented

## Data Security
- [ ] Encryption requirements defined
- [ ] Encryption libraries integrated
- [ ] Data validation and sanitization implemented
- [ ] Secure file handling implemented
- [ ] Key management procedures documented
- [ ] Data classification implemented

## Smart Contract Security
- [ ] Access control implemented
- [ ] Stake management security reviewed
- [ ] Slashing mechanism implemented
- [ ] Contract upgrade security planned
- [ ] Pause/resume capabilities added
- [ ] Code review completed

## Agent Security
- [ ] Agent sandboxing implemented
- [ ] Resource limits configured
- [ ] Credential security implemented
- [ ] Network security configured
- [ ] Monitoring enabled

## Monitoring & Alerting
- [ ] Security metrics defined
- [ ] Alert thresholds configured
- [ ] Notification channels set up
- [ ] Automated response rules defined
- [ ] Incident response playbook created

## Operational Security
- [ ] Backup strategy defined
- [ ] Backup automation configured
- [ ] Recovery procedures documented
- [ ] Audit logging implemented
- [ ] Security audit schedule defined
- [ ] Incident response procedures documented

## Documentation & Training
- [ ] All security procedures documented
- [ ] Security team trained
- [ ] Incident response drills conducted
- [ ] Security awareness training provided
- [ ] Escalation procedures defined
```

### Operational Security Checklist

```markdown
## Daily
- [ ] Review authentication logs
- [ ] Monitor agent health
- [ ] Check for unusual activity
- [ ] Review backup completion
- [ ] Verify security metrics

## Weekly
- [ ] Review security alerts
- [ ] Analyze incident trends
- [ ] Update security policies if needed
- [ ] Conduct threat assessment
- [ ] Review access logs
- [ ] Test backup recovery

## Monthly
- [ ] Comprehensive security audit
- [ ] Penetration testing
- [ ] Review and update security policies
- [ ] Conduct security training
- [ ] Review incident response effectiveness
- [ ] Update threat models
- [ ] Review compliance requirements

## Quarterly
- [ ] Security architecture review
- [ ] Third-party security assessment
- [ ] Business continuity plan review
- [ ] Disaster recovery drill
- [ ] Update security roadmap
- [ ] Executive security review

## Annually
- [ ] Comprehensive security assessment
- [ ] Red team exercise
- [ ] Compliance audit
- [ ] Security program review
- [ [ ] Update security strategy
```

---

## 🚨 Security Incident Response Procedures

### Immediate Actions (0-15 Minutes)

```markdown
## Detection
1. Identify the incident
2. Determine scope and impact
3. Classify severity
4. Notify incident response team

## Containment
1. Isolate affected systems
2. Block malicious IPs/accounts
3. Revoke compromised credentials
4. Increase monitoring
5. Preserve evidence

## Eradication
1. Identify root cause
2. Patch vulnerabilities
3. Remove malware
4. Update systems
5. Verify no compromise

## Recovery
1. Restore from backups
2. Validate integrity
3. Gradually restore service
4. Monitor for re-infection
5. Conduct security audit

## Post-Incident
1. Document full timeline
2. Root cause analysis
3. Lessons learned
4. Update procedures
5. Provide training
```

---

## 📚 Additional Resources

### Security Documentation
- [OWASP Top 10](https://owasp.org/www-project-top-ten)
- [Nostr Security Considerations](https://github.com/nostr-protocol/nips/blob/master/17.md)
- [NEAR Security Best Practices](https://docs.near.org/docs/develop/contracts/security/)
- [Smart Contract Security](https://docs.soliditylang.org/en/develop/security-considerations)

### Tools & Services
- [Cryptography Libraries](https://github.com/paulmillr/noble)
- [Penetration Testing Tools](https://www.owasp.org/index.php/Category:Vulnerability_Scanning_Tools)
- [Security Monitoring](https://www.splunk.com/en_us/software/security-monitoring)

### Community Resources
- [Nostr Security Research](https://github.com/nostr-protocol/nips)
- [NEAR Security Audits](https://near.org/security/)
- [Agent Security Best Practices](https://github.com/mpp-near/security)

---

## 🎯 Summary

This security best practices document provides a comprehensive framework for securing your Nostr-Backed NEAR Whitelist system. Key takeaways:

1. **Defense in Depth**: Multiple independent security layers
2. **Zero Trust**: Verify everything, trust nothing
3. **Automated Response**: Quick containment and recovery
4. **Continuous Monitoring**: Real-time threat detection
5. **Regular Audits**: Proactive security reviews
6. **Incident Response**: Clear procedures and escalation
7. **Compliance**: Adhere to security standards
8. **Training**: Continuous security awareness

**Remember**: Security is an ongoing process, not a one-time setup. Regular reviews, updates, and improvements are essential for maintaining a secure system.

---

**Next Steps**:
1. Review [Threat Model](./threat-model.md) for detailed threat analysis
2. Implement [Incident Response](./incident-response.md) procedures
3. Set up [Monitoring](../guides/getting-started.md) and alerting
4. Conduct regular security audits
5. Stay informed about new security research and threats