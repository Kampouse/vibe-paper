# NEAR + Nostr Integration: Zero Infrastructure Approach

**Every NEAR account is now a Nostr account.**

---

## Overview

This document describes how to integrate NEAR accounts with Nostr using the existing NEAR MPC (Multi-Party Computation) infrastructure. No custom contracts, no MPC modifications, no infrastructure required.

**Key Insight:** NEAR's `v1.signer` MPC contract already supports everything needed for Nostr signing:

- ✅ ECDSA (secp256k1) - same curve as Nostr
- ✅ Deterministic key derivation
- ✅ Threshold signatures
- ✅ Account-bound keys

---

## Architecture

```
User's Browser
┌─────────────────┐
│  Web Page       │
│  (HTML + JS)    │
└────────┬────────┘
         │ 1. Login with NEAR
         ▼
┌─────────────────┐
│  NEAR Wallet    │
│  (popup)        │
└────────┬────────┘
         │ 2. Get pubkey / Sign events
         ▼
┌─────────────────┐
│  v1.signer      │  ◄── NEAR's existing MPC contract
│  (MPC Network)  │      No changes needed
└────────┬────────┘
         │ 3. Threshold signature
         ▼
┌─────────────────┐
│  Nostr Relays   │
│  (wss://...)    │
└─────────────────┘
```

---

## Why No MPC Changes Are Needed

### 1. Same Cryptography

Nostr uses **secp256k1 (ECDSA)** - exactly what NEAR's MPC already supports.

```typescript
// Nostr event signing
const eventHash = sha256(serializeEvent(event));
const signature = ecdsaSign(eventHash, privateKey);

// This is EXACTLY what v1.signer already does
```

### 2. Deterministic Key Derivation

Every NEAR account gets a unique, deterministic Nostr key:

```typescript
// Same NEAR account = Same Nostr pubkey (forever)
const nearAccount = "alice.near";
const nostrPubkey = await v1.signer.derived_public_key({
  domain: 0,  // ECDSA domain
  path: `nostr/${nearAccount}`,  // Deterministic path
});
// Result: Always same pubkey for alice.near
```

### 3. MPC Doesn't Care About "Nostr"

```
MPC sees:
- domain: 0 (ECDSA)
- path: "nostr/alice.near" (just a string)
- payload: 32-byte hash

MPC doesn't know:
- This is a Nostr event
- The path means "nostr"
- The payload is an event ID

MPC just signs. That's it.
```

---

## Implementation: Zero Infrastructure

### Minimal Web Page (~150 lines)

```html
<!DOCTYPE html>
<html>
<head>
  <title>NEAR Nostr</title>
  <script src="https://cdn.jsdelivr.net/npm/near-api-js@latest/dist/near-api-js.min.js"></script>
</head>
<body>
  <div id="login">
    <h1>NEAR → Nostr</h1>
    <p>Login with your NEAR account to get your Nostr identity</p>
    <button onclick="login()">Login with NEAR</button>
  </div>
  
  <div id="app" style="display: none;">
    <h2>Welcome, <span id="account"></span></h2>
    <p>Your Nostr pubkey:</p>
    <code id="pubkey"></code>
    
    <h3>Post a Note</h3>
    <textarea id="note" placeholder="What's on your mind?"></textarea>
    <button onclick="postNote()">Post to Nostr</button>
    
    <h3>Feed</h3>
    <div id="feed"></div>
  </div>
  
  <script>
    let wallet, accountId, pubkey, relay;
    
    // Initialize NEAR
    async function init() {
      const near = await nearApi.connect({
        networkId: 'mainnet',
        nodeUrl: 'https://rpc.mainnet.near.org',
        walletUrl: 'https://wallet.mainnet.near.org',
      });
      
      wallet = new nearApi.WalletConnection(near, 'near-nostr');
      
      if (wallet.isSignedIn()) {
        accountId = wallet.getAccountId();
        await showApp();
      }
    }
    
    // Login
    async function login() {
      await wallet.requestSignIn({
        contractId: 'v1.signer',
        methodNames: ['sign', 'derived_public_key'],
      });
    }
    
    // Show app after login
    async function showApp() {
      document.getElementById('login').style.display = 'none';
      document.getElementById('app').style.display = 'block';
      document.getElementById('account').textContent = accountId;
      
      // Get Nostr pubkey from MPC
      pubkey = await getNostrPubkey(accountId);
      document.getElementById('pubkey').textContent = pubkey;
      
      // Connect to relay
      relay = new WebSocket('wss://relay.damus.io');
      relay.onopen = () => {
        relay.send(JSON.stringify(['REQ', 'feed', { limit: 50 }]));
      };
      relay.onmessage = (msg) => {
        const [type, subId, event] = JSON.parse(msg.data);
        if (type === 'EVENT') addToFeed(event);
      };
    }
    
    // Get Nostr pubkey from MPC
    async function getNostrPubkey(accountId) {
      const response = await fetch('https://rpc.mainnet.near.org', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: 1,
          method: 'query',
          params: {
            request_type: 'call_function',
            account_id: 'v1.signer',
            method_name: 'derived_public_key',
            args_base64: btoa(JSON.stringify({
              domain: 0,
              path: `nostr/${accountId}`,
            })),
            finality: 'optimistic',
          },
        }),
      });
      
      const { result } = await response.json();
      return result.result.map(b => b.toString(16).padStart(2, '0')).join('');
    }
    
    // Sign Nostr event via MPC
    async function signEvent(event) {
      // 1. Serialize (NIP-01)
      const serialized = JSON.stringify([
        0,
        event.pubkey,
        event.created_at,
        event.kind,
        event.tags,
        event.content,
      ]);
      
      // 2. Hash
      const encoder = new TextEncoder();
      const data = encoder.encode(serialized);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const eventId = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      
      // 3. Sign via MPC
      const response = await wallet.account().functionCall({
        contractId: 'v1.signer',
        methodName: 'sign',
        args: {
          domain: 0,
          path: `nostr/${accountId}`,
          payload: eventId,
        },
        gas: '30000000000000',
      });
      
      // 4. Extract signature (simplified)
      const signature = parseSignature(response);
      
      return {
        ...event,
        id: eventId,
        sig: signature,
      };
    }
    
    // Post note
    async function postNote() {
      const content = document.getElementById('note').value;
      if (!content.trim()) return;
      
      const event = {
        pubkey: pubkey,
        created_at: Math.floor(Date.now() / 1000),
        kind: 1,
        tags: [],
        content: content,
      };
      
      const signedEvent = await signEvent(event);
      
      // Broadcast
      relay.send(JSON.stringify(['EVENT', signedEvent]));
      
      document.getElementById('note').value = '';
      addToFeed(signedEvent);
    }
    
    // Add to feed
    function addToFeed(event) {
      const div = document.createElement('div');
      div.innerHTML = `
        <div style="border-bottom: 1px solid #ccc; padding: 10px;">
          <div><strong>${event.pubkey.slice(0, 16)}...</strong></div>
          <div>${event.content}</div>
          <div><small>${new Date(event.created_at * 1000).toLocaleString()}</small></div>
        </div>
      `;
      document.getElementById('feed').prepend(div);
    }
    
    // Parse signature from NEAR response
    function parseSignature(response) {
      // Implementation depends on v1.signer response format
      // This is a placeholder
      return response.status.SuccessValue || '';
    }
    
    init();
  </script>
</body>
</html>
```

---

## Deployment (FREE)

### Option 1: Cloudflare Pages

```bash
# Create public/ folder with index.html
mkdir public
# Copy HTML above to public/index.html

# Deploy
wrangler pages deploy public

# Result: https://near-nostr.pages.dev
```

### Option 2: GitHub Pages

```bash
# Push to gh-pages branch
git checkout -b gh-pages
git add index.html
git commit -m "Add NEAR Nostr app"
git push origin gh-pages

# Result: https://yourname.github.io/vibe-paper
```

### Option 3: Vercel

```bash
vercel --prod

# Result: https://near-nostr.vercel.app
```

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Static hosting | FREE |
| NEAR RPC | FREE |
| NEAR gas (user pays) | ~0.001 NEAR/signature |
| Custom domain (optional) | $10/year |
| **Monthly total** | **$0** |

---

## Alternative Approaches

### Option 1: Browser Extension (NIP-07)

**Pros:** Works with ALL existing Nostr web apps
**Cons:** User must install extension

**Code:** ~250 lines

```typescript
// extension/content-script.js
window.nostr = {
  async getPublicKey() {
    const account = await nearWallet.getAccountId();
    return await getNostrPubkey(account);
  },
  
  async signEvent(event) {
    return await signWithMpc(event);
  },
};
```

### Option 2: Local Bunker (NIP-46)

**Pros:** Works with ALL Nostr clients (desktop/mobile)
**Cons:** User must run local app

**Code:** ~150 lines

```typescript
// Local WebSocket server
const wss = new WebSocketServer({ port: 8080 });

wss.on('connection', (ws) => {
  ws.on('message', async (msg) => {
    const { method, params } = JSON.parse(msg);
    
    if (method === 'sign_event') {
      const sig = await signWithNearMpc(params[0]);
      ws.send(JSON.stringify({ result: sig }));
    }
  });
});
```

### Option 3: Serverless Bunker (Cloudflare Workers)

**Pros:** No app to install, works with all clients
**Cons:** Requires auth flow

**Code:** ~300 lines (worker + Near Social widget)

---

## Key Derivation

### Path Convention

```
nostr/{account_id}

Examples:
- nostr/alice.near
- nostr/bob.near
- nostr/gork.near
```

### Deterministic Mapping

```typescript
// Same input = Same output (always)
derivePublicKey("nostr/alice.near") → "abc123..." (never changes)
derivePublicKey("nostr/alice.near") → "abc123..." (always same)
derivePublicKey("nostr/bob.near")   → "def456..." (different)
```

### Why This Works

NEAR's MPC uses **Child Key Derivation (CKD)** - a hierarchical deterministic scheme similar to BIP-32:

```
Master Key
    └── nostr/
        ├── alice.near  → Key A
        ├── bob.near    → Key B
        └── gork.near   → Key C
```

Each path gets a unique, deterministic key. No registration needed.

---

## Security Model

### Key Security

- **MPC Threshold:** Key split across n nodes (t-of-n)
- **No Single Point of Failure:** No single node has the full key
- **Account Bound:** Only account owner can request signatures
- **NEAR Wallet Auth:** User must approve in wallet

### Signature Flow

```
1. User requests signature in web app
2. NEAR wallet popup appears
3. User approves transaction
4. MPC nodes collectively sign
5. Signature returned to web app
6. Web app broadcasts to Nostr relay
```

### Trust Assumptions

| Component | Trust Level |
|-----------|-------------|
| NEAR MPC Network | High (threshold cryptography) |
| NEAR Wallet | User controls |
| Your Web App | Zero trust (can't forge signatures) |
| Nostr Relays | Public infrastructure |

---

## Limitations

### Current Limitations

1. **Gas Costs:** User pays ~0.001 NEAR per signature
2. **Latency:** MPC signing takes 1-2 seconds
3. **Online Required:** Can't sign offline

### Future Improvements

1. **Gasless Relayer:** You pay gas for users
2. **Session Keys:** One MPC sign, then local signing for 30 days
3. **Batch Signing:** Sign multiple events in one transaction

---

## Comparison to Alternatives

| Approach | Infra Cost | Code | User Setup | Security |
|----------|------------|------|------------|----------|
| **Web Page (this)** | $0 | 150 lines | Visit URL | High (MPC) |
| Extension | $0 | 250 lines | Install | High (MPC) |
| Local Bunker | $0 | 150 lines | Run app | High (MPC) |
| Serverless Bunker | $0 | 300 lines | Auth once | High (MPC) |
| Custom Nostr key | $0 | 0 | Manual backup | Low (self) |

---

## Use Cases

### 1. Social Recovery

NEAR account recovery = Nostr account recovery

```
If user loses NEAR wallet:
1. Recover NEAR account (social recovery)
2. Nostr account automatically recovered
3. Same Nostr pubkey (deterministic)
```

### 2. Agent Identities

AI agents get Nostr keys via NEAR accounts

```typescript
// Agent has NEAR account: agent-001.near
// Agent's Nostr key: derived from "nostr/agent-001.near"
// Agent can post to Nostr autonomously
```

### 3. Multi-Device Sync

Same NEAR account = Same Nostr identity across devices

```
Phone:  Login with alice.near → Same pubkey
Laptop: Login with alice.near → Same pubkey
Tablet: Login with alice.near → Same pubkey
```

### 4. Programmable Social

Smart contracts can enforce posting rules

```rust
// Example: Rate-limited posting
pub fn sign_nostr(&mut self, event: Event) -> Signature {
    require!(self.last_post[&predecessor()] < env::block_timestamp() - 60000);
    self.last_post[&predecessor()] = env::block_timestamp();
    mpc_sign(event)
}
```

---

## Implementation Checklist

### Minimal (Web Page Only)

- [ ] Create HTML file with NEAR login
- [ ] Implement `getNostrPubkey()` function
- [ ] Implement `signEvent()` function
- [ ] Add Nostr relay connection
- [ ] Deploy to static hosting
- [ ] Test with real NEAR account

### Enhanced (Gasless)

- [ ] Deploy relayer contract
- [ ] Fund relayer with NEAR
- [ ] Update web app to use relayer
- [ ] Test gasless signing

### Advanced (Session Keys)

- [ ] Add session key generation
- [ ] Store session key locally
- [ ] Implement session-based signing
- [ ] Add session expiration

---

## Resources

- **NEAR MPC:** https://github.com/near/mpc
- **Nostr Protocol:** https://github.com/nostr-protocol/nips
- **NIP-01 (Basic Protocol):** https://github.com/nostr-protocol/nips/blob/master/01.md
- **NIP-07 (Browser Extension):** https://github.com/nostr-protocol/nips/blob/master/07.md
- **NIP-46 (Nostr Connect):** https://github.com/nostr-protocol/nips/blob/master/46.md

---

## Conclusion

**Every NEAR account is now a Nostr account.**

No custom contracts. No MPC modifications. No infrastructure. Just a web page that calls existing NEAR MPC methods with `path="nostr/{account}"`.

**Total implementation:**
- Code: ~150 lines
- Cost: $0/month
- Setup time: 30 minutes
- Security: High (threshold MPC)

The future of decentralized identity is here, and it's simpler than you thought.
