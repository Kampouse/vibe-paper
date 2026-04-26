# vibe-paper

> **This is not real.** Nothing in this repo is production code or even fully working implementations. These are exploratory ideas — concepts that stick to the wall, inspired by research papers about AI memory systems.

## What is this?

A collection of speculative ideas exploring what could be done with AI for memory. These are rough concepts, not polished solutions.

## Contents

### [Compact Context Model (CCM)](./compact-context-model.md)
A speculative architecture for handling long-term context in LLMs through graph-based decision extraction rather than raw token storage.

**Key idea:** Extract the *decisional structure* of conversations, discard the rest. Potentially reduce token usage by 80-90% while preserving what actually matters.

**Status:** Concept paper only. No implementation.

### [NEAR + Nostr Integration](./near-nostr-integration.md)
A zero-infrastructure approach to giving every NEAR account a Nostr identity using existing MPC infrastructure.

**Key idea:** NEAR's v1.signer MPC contract already supports everything needed for Nostr signing. No custom contracts, no MPC modifications, just a web page.

**Status:** Production-ready concept. ~150 lines of code, $0/month hosting.

## Background

These ideas emerged from exploring research on:
- Long-term memory for AI systems
- Context compression techniques
- Graph-based knowledge representation
- Agent learning systems

## Reality Check

- ❌ No working code
- ❌ No benchmarks
- ❌ No rigorous evaluation
- ✅ Ideas that seemed interesting enough to write down
- ✅ Starting points for actual research

## Philosophy

Sometimes you need to throw ideas at the wall and see what sticks. This is the wall.

---

### [Standing Intentions](./standing-intentions/paper.md)
An architecture for autonomous LLM agents that maintain persistent internal state through homeostatic drives, act on standing intentions without external prompting, and improve through evolving inspectable skill artifacts — all without modifying the model or its runtime.

**Key idea:** Current agents are reactive stateless functions. We formalize the drive loop (continuous internal monitoring), standing intentions (persistent commitments that don't consume working memory), three-tier execution (route decisions to the cheapest capable processor), and skill artifact evolution (the meta-agent closes the loop by revising guidance based on grounded evaluator feedback). Fixed model, fixed runtime, evolving artifacts.

**Status:** Concept paper with formalization. No implementation. No benchmarks.

---

### [Red-Black Fractal Reasoning](./red-black-fractal-reasoning/paper.md)
A hypothesis for structuring LLM agent context as a self-balancing red-black tree, where node colors encode epistemic status (verified observation vs. speculative inference) and tree invariants enforce reasoning discipline. Inspired by FractalBench's finding that models achieve 76% syntactic but only 4% semantic correctness on recursive tasks.

**Key idea:** The gap between pattern matching and recursive abstraction is a context architecture problem, not a model capability problem. Red-black tree invariants → epistemic constraints → grounded reasoning.

**Status:** Hypothesis only. No implementation. No benchmarks.

---

*Last updated: 2026-04-24*
