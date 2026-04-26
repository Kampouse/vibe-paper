# Standing Intentions: Autonomous Agent Behavior Through Evolving Skill Artifacts Without Model Modification

**Status:** Hypothesis + architecture. No implementation. No benchmarks.

---

## Abstract

Current LLM agents are reactive stateless functions: they wait for external input, process it, produce output, and go idle. This is fundamentally unlike biological agents, which maintain continuous internal monitoring and self-trigger action based on their own state. We present an architecture where an agent maintains persistent internal state through homeostatic drives, acts on standing intentions without external prompting, and improves through evolving inspectable skill artifacts — all without modifying the model or its runtime. We formalize the drive loop as a computational model grounded in Hull's drive reduction theory (1943) and Bratman's theory of intention (1987), define standing intentions as a persistent commitment data structure, and propose a three-tier execution model that minimizes expensive LLM calls while preserving autonomous decision-making. The architecture separates concerns into an immutable kernel (drive loop, rule engine, execution engine), a fixed model, a grounded evaluator, and an evolving artifact layer. We argue that self-improvement through skill artifact evolution — rather than weight modification — is the correct abstraction level for safe, inspectable, and practical autonomous agent behavior.

---

## 1. Introduction

### 1.1 Every LLM Agent is a Stateless Function

Consider the dominant agent architectures in 2025-2026:

| Pattern | Trigger | Internal State | Self-Generated Action |
|---------|---------|----------------|----------------------|
| Request-response | External prompt | None | No |
| Cron-scheduled | Fixed clock | None | No (schedule is external) |
| Event-driven | External webhook | Minimal | No |
| RAG-augmented | External prompt | Retrieval-dependent | No |
| Tool-using agent | External prompt | Session-only | No |

All of these share a fundamental property: the agent does nothing until something external happens. The agent is a tool you pick up and put down. It never acts on its own initiative.

This is not how biological agents work. A biological agent maintains continuous internal monitoring — heart rate, blood sugar, body temperature, hunger, thirst — and self-triggers action when internal variables drift from equilibrium. No external prompt is needed. The organism *wants* to eat because its glucose is low, not because someone asked it to.

### 1.2 The Cost of Reactivity

Reactive agents pay two costs:

**Opportunity cost.** An agent that only acts when prompted misses time-sensitive opportunities. A trading agent that checks positions only when asked will miss a flash crash. A monitoring agent that runs only on schedule will miss an anomaly that happens between ticks.

**Cognitive cost.** Every reactive invocation starts from zero context. The agent must reconstruct what it was doing, what has changed, and what matters — each time. There is no continuity of purpose across invocations.

### 1.3 Self-Improvement at the Wrong Level

Recent work on self-improving agents typically modifies one of:
- **Model weights** (RLHF, fine-tuning, constitutional AI)
- **Prompt templates** (system message engineering)
- **Retrieval indices** (RAG corpus updates)

Weight modification is expensive, opaque, and risky. It changes the model globally — every capability, not just the one being improved. It requires retraining infrastructure, validation sets, and rollback mechanisms. And it can introduce regressions in capabilities the system already had.

The metasurface inverse design work (Huang et al., 2026) demonstrated a compelling alternative: keep the model fixed, keep the evaluator fixed, and evolve only the skill artifact — a structured guidance document that the coding agent reads before each attempt. Their results: task success improved from 38% to 74%, average attempts dropped from 4.10 to 2.30, all without a single gradient step.

We argue this is not a special case for metasurface design. It is the general pattern.

### 1.4 Our Contribution

We present an architecture for autonomous LLM agents with four properties:

1. **Autonomous triggering** — the agent acts from internal state, not external prompts
2. **Persistent commitments** — standing intentions maintain goals across invocations without consuming working memory
3. **Economical execution** — a three-tier model routes decisions to the cheapest capable processor
4. **Safe self-improvement** — the meta-agent evolves inspectable skill artifacts, never model weights or runtime code

The model stays fixed. The runtime stays fixed. Only the artifact layer evolves.

---

## 2. Related Work

### 2.1 Cognitive Science Foundations

Our architecture draws on six results from cognitive science:

| Concept | Source | Agent Mapping |
|---------|--------|---------------|
| Homeostatic drives | Hull, 1943 | Internal variables that decay and trigger action |
| Standing intention | Bratman, 1987 | Persistent background goals, non-occurrent until context-matched |
| Prospective memory | Einstein & McDaniel, 2005 | Spontaneous retrieval of future intentions without prompts |
| Belief-Desire-Intention | Rao & Georgeff, 1991 | Hierarchical intention structure with commitment |
| Self-Determination Theory | Deci & Ryan, 1985 | Intrinsic motivation through autonomy, competence, relatedness |
| Metacognition | Flavell, 1979 | Meta-agent monitoring and revising its own operation |
| Zeigarnik effect | Zeigarnik, 1927 | Open drives stay elevated until resolved |

BDI architectures have been explored in traditional AI (Rao & Georgeff, 1991; Wooldridge, 2000) but were largely abandoned with the rise of deep learning. We argue the concepts remain valid — they just needed a substrate (the LLM) that can handle the "desire" and "intention" components with sufficient generality.

### 2.2 Self-Evolving Agent Frameworks

Huang et al. (2026) introduced a self-evolving agentic framework for metasurface inverse design. Their architecture uses a coding agent, a deterministic physics evaluator, and a meta-agent that updates a SKILL.md guidance file based on rollout logs. The model is fixed. Only the skill artifact evolves. Their results demonstrate that explicit, inspectable guidance evolution can match or exceed fine-tuning approaches for domain-specific tasks.

Our work extends this pattern in three ways:
- We formalize the *triggering* mechanism (homeostatic drives) so the agent decides when to act
- We formalize the *commitment* mechanism (standing intentions) so the agent maintains goals across invocations
- We add the *routing* mechanism (three-tier execution) so the agent uses the cheapest capable processor

### 2.3 Autonomous Agent Architectures

Recent work on autonomous agents (AutoGPT, BabyAGI, CrewAI) attempts continuous operation but typically relies on:
- Fixed loops with no internal state modulation
- External goal decomposition (user sets the goal, agent pursues it)
- No homeostatic regulation (the agent either runs or stops, with no gradation)

These systems are *persistent* but not *autonomous* in the biological sense. They don't self-trigger based on internal state. They execute a plan until it's done or they run out of context.

### 2.4 Red-Black Problem Trees

The red-black fractal reasoning hypothesis (Kampouse, 2026) proposes that the model should not see the orchestration structure — the tree exists in the agent's runtime, controlling what gets generated, executed, retried, and composed. Each node is an independent generate-execute cycle with its own narrow context.

We adopt this principle: the drive loop is the orchestration layer, and the LLM agent operates on narrow, focused sub-problems. The model never sees the full drive state or intention hierarchy — it receives a specific task with specific context when the drive loop wakes it.

---

## 3. Architecture

### 3.1 Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  TIER 1: DRIVE LOOP (always on, microseconds)                       │
│    │                                                                 │
│    ├─ CLEAR-CUT? ──→ RULE ENGINE (microseconds)                     │
│    │                  executes hard limits, stop-losses, breakers    │
│    │                                                                 │
│    └─ AMBIGUOUS? ──→ wake LLM AGENT (seconds)                       │
│                       receives focused task from drive context      │
│                       │                                              │
│                       ├─ STRATEGIC DECISION ──→ parameterize         │
│                       │   EXECUTION ENGINE (microseconds)            │
│                       │   order routing, slicing, fill monitoring    │
│                       │                                              │
│                       └─ NO ACTION ──→ log, reset drive, sleep      │
│                                                                      │
│  TIER 2: META-AGENT (periodic / end-of-cycle)                       │
│    reviews outcomes, revises skills + drive thresholds + rules       │
│                                                                      │
│  IMMUTABLE KERNEL: drive loop, rule engine, execution engine        │
│  FIXED MODEL: LLM weights never change                              │
│  GROUNDED EVALUATOR: deterministic outcome scoring                   │
│  EVOLVING ARTIFACTS: SKILL.md, drive thresholds, rule table         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 The Drive Loop

The drive loop is a lightweight always-on process that monitors internal state variables against thresholds. No LLM. No reasoning. Just comparisons.

```python
class DriveLoop:
    def __init__(self, drives: list[Drive], rule_engine: RuleEngine):
        self.drives = drives
        self.rule_engine = rule_engine
        self.base_interval = 5.0  # seconds at rest

    def run(self):
        while alive:
            for drive in self.drives:
                if drive.check_threshold():
                    if self.rule_engine.has_clear_action(drive):
                        self.rule_engine.execute(drive)        # microseconds
                    else:
                        intent = self._compose_intent(drive)  # wake LLM
                        self.intent_queue.push(intent)

            # Variable tick rate: faster when drives are near threshold
            near_count = sum(1 for d in self.drives if d.near_threshold())
            interval = self.base_interval / (1 + near_count * 3)
            sleep(interval)
```

**Variable tick rate** is critical. Like a heart rate — 60 bpm at rest, 120 when exerting. When all drives are at equilibrium, the loop ticks slowly (saving compute). When drives approach threshold, the loop accelerates (responding faster to emerging situations).

#### Formal Definition

A drive is a tuple:

```
D = (v, τ, δ, f_decay, f_urgency)
```

Where:
- `v` — current value (scalar, updated by environment or agent actions)
- `τ` — threshold (scalar, action trigger point)
- `δ` — decay function: `v(t+1) = v(t) - δ(t)` (value drifts toward triggering)
- `f_urgency` — urgency function: `u = f_urgency(v, τ)` (how far past threshold)
- Associated standing intention: what to do when triggered

The drive fires when `v(t) crosses τ`. The urgency scales with distance from threshold. Multiple drives can fire simultaneously, creating compound intents.

#### Drive Interaction

Drives interact through two mechanisms:

1. **Priority queue** — drives are sorted by urgency. Highest urgency drives get dispatched first. When the LLM is woken, it sees the compound intent from all firing drives, ordered by priority.

2. **Suppression** — one drive can suppress another. For example, a `critical_error` drive can suppress all other drives, forcing the agent to focus on recovery. This maps to biological hormone cascades where one signal (adrenaline) suppresses others (digestion).

Open question: what interaction topology minimizes incoherence? Biology uses dozens of homeostatic variables with complex interactions. Software agents likely need fewer (5-15) with simpler topologies.

### 3.3 Standing Intentions

Standing intentions are persistent background commitments that structure behavior without consuming working memory. They are the agent's equivalent of "maintain portfolio within risk bounds" — a goal that persists whether or not you're actively thinking about it.

#### Bratman's Distinction

- **Occurrent intention** — actively in mind right now. "Sell NVDA within 30 minutes." Consumes working memory (tokens).
- **Standing intention** — latent but always present. "Maintain portfolio within risk bounds." Does not consume working memory until activated.

Standing intentions have four properties:

| Property | Description |
|----------|-------------|
| **Persistent** | Does not decay because you're not thinking about it |
| **Non-occurrent** | Does not consume tokens until triggered |
| **Context-activated** | Surfaces when the situation calls for it |
| **Hierarchical** | Root intentions decompose into specific sub-intentions |

#### Data Structure

```python
@dataclass
class StandingIntention:
    id: str
    description: str              # "maintain portfolio within risk bounds"
    activation_conditions: list   # drive triggers that activate this intention
    priority: float               # resolution order when multiple intentions activate
    children: list                # sub-intentions
    associated_skill: str | None  # link to skill artifact
    last_activated: float | None  # timestamp
    last_resolved: float | None   # timestamp
    resolution_criteria: list     # conditions that mark this intention as satisfied
```

The intention hierarchy is a tree:

```
root: "generate risk-adjusted returns"
├─ "maintain portfolio within risk bounds"
│   ├─ "position concentration < 0.65 per asset"
│   └─ "correlation risk < 0.60"
├─ "execute when edge exists"
│   ├─ "act on signals above 0.80 conviction"
│   └─ "skip noise, avoid overtrading"
├─ "protect capital"
│   ├─ "cut losses at 15% drawdown"
│   └─ "reduce size after 3 consecutive losses"
└─ "learn and adapt"
    ├─ "review strategy performance daily"
    └─ "evolve entry/exit rules from trade log"
```

When a drive fires, the drive loop traverses the intention tree to find matching activation conditions. The matching intention (with its children) becomes the task context for the LLM agent. The model never sees the full tree — only the relevant subtree.

### 3.4 Three-Tier Execution Model

```
TIER 1: DRIVE LOOP     →  microseconds, always on, no LLM
TIER 2: RULE ENGINE    →  microseconds, deterministic, no LLM
TIER 3: LLM AGENT      →  seconds, on-demand, woken by drive loop
TIER 4: EXECUTION ENGINE → microseconds, parameterized by LLM decisions
```

Each tier operates at its natural timescale and cost:

| Tier | Latency | Cost | When Used |
|------|---------|------|-----------|
| Drive loop | ~μs | Negligible | Always running, checking thresholds |
| Rule engine | ~μs | Negligible | Clear-cut threshold breaches with known actions |
| LLM agent | ~seconds | High (tokens + compute) | Ambiguous situations, novel contexts, cross-drive conflicts |
| Execution engine | ~μs | Negligible | Carrying out LLM decisions (order routing, etc.) |

**The rule engine is the key efficiency gain.** Most drive firings have clear-cut responses. A stop-loss trigger doesn't need an LLM to decide what to do — it needs to sell. A data freshness trigger doesn't need reasoning — it needs to fetch. The rule engine handles these in microseconds.

The LLM is reserved for situations where:
- No clear-cut rule exists (ambiguous)
- Multiple drives conflict (need weighing)
- The situation is novel (no prior pattern)
- Cross-drive interaction matters (system-level reasoning)

### 3.5 The Meta-Agent

The meta-agent is a periodic process (end-of-cycle, daily, or after N drive activations) that reviews outcomes and evolves the artifact layer.

```
Input:  outcome logs + drive activation history + current SKILL.md
Output: revised SKILL.md + adjusted drive thresholds + new rules

Process:
1. Read trade log / action log since last review
2. Identify patterns: successes, failures, near-misses
3. Compare to current skills — what worked, what didn't
4. Propose skill revisions with justification
5. Propose threshold adjustments with justification
6. Propose new rules for patterns now consistent enough to automate
7. Validate proposals against held-out data
8. Apply bounded revisions
```

**What the meta-agent can modify:**
- SKILL.md content (tactics, heuristics, domain knowledge)
- Drive thresholds (within bounded ranges)
- Rule engine entries (add new clear-cut rules)

**What the meta-agent cannot modify:**
- Model weights (immutable)
- Drive loop code (immutable)
- Execution engine code (immutable)
- Core safety constraints (immutable)

This is the "safer self-improvement" pattern: the agent gets better at using its tools without rewriting its tools.

### 3.6 Skill Artifacts

Skill artifacts are structured documents that encode learned tactics, domain knowledge, and procedural guidance. They are:

- **Inspectable** — human-readable markdown
- **Revertible** — version-controlled, can roll back
- **Bounded** — changes are limited in scope per cycle
- **Grounded** — every revision references specific outcome data

```markdown
# SKILL.md — Trading Agent

## Entry Tactics
- Lower signal threshold to 0.75 for high-liquidity pairs (validated on 47 trades, win rate +12%)
- Pre-emptive rebalance when concentration > 0.65 (prevented 3 drawdowns > 5%)

## Risk Management
- VWAP execution in low-volume windows causes slippage spike — avoid 9:30-9:45 AM
- After 3 consecutive losses, reduce position size by 50% before re-evaluating

## Failure Fingerprints
- Pattern: buying breakout on low volume → 78% failure rate → require volume confirmation
- Pattern: holding through earnings → 62% loss rate → exit before earnings unless conviction > 0.90
```

Each entry traces back to specific outcome data. The skill is not opinion — it is compressed experience.

---

## 4. Formalization

### 4.1 Drive Dynamics

We model drive state as a continuous-time stochastic process. For drive $i$ with current value $v_i$ and threshold $\tau_i$:

$$\frac{dv_i}{dt} = -\delta_i(t) + \eta_i(t)$$

Where $\delta_i(t)$ is the deterministic decay rate and $\eta_i(t) \sim \mathcal{N}(0, \sigma_i^2)$ is environmental noise.

The drive fires at the first hitting time:

$$T_i = \inf\{t : v_i(t) \geq \tau_i\}$$

Urgency at firing:

$$u_i = f_{\text{urgency}}(v_i(T_i), \tau_i) = \frac{v_i(T_i) - \tau_i}{\tau_i}$$

When the agent acts on drive $i$ and resolves the trigger condition, the drive resets:

$$v_i \leftarrow v_i^{\text{reset}}$$

The **Zeigarnik effect** is modeled as a persistent elevation: if a drive fires but is not resolved (agent chose not to act or action failed), the reset value is higher than equilibrium:

$$v_i^{\text{reset, unresolved}} > v_i^{\text{reset, resolved}}$$

This ensures unresolved drives re-trigger faster, creating persistent attention to open problems.

### 4.2 Intention Resolution

When multiple drives fire simultaneously, the intention resolver produces a compound intent:

$$I = \text{resolve}(\{(D_i, u_i) : v_i \geq \tau_i\})$$

Resolution strategy: priority queue by urgency, with suppression rules. Let $S$ be the suppression relation: $D_i \gg D_j$ means drive $i$ suppresses drive $j$.

$$I = \text{sort}_u(\{D_i : v_i \geq \tau_i \land \nexists D_j : D_j \gg D_i\})$$

The compound intent maps to the intention tree — each firing drive activates its linked standing intention (and children), producing a focused task context for the LLM.

### 4.3 Three-Tier Cost Model

Expected cost per time step:

$$C_{\text{total}} = C_{\text{drive}} + p_{\text{rule}} \cdot C_{\text{rule}} + p_{\text{llm}} \cdot C_{\text{llm}} + p_{\text{exec}} \cdot C_{\text{exec}}$$

Where:
- $C_{\text{drive}} \approx 0$ (threshold comparisons)
- $p_{\text{rule}} = P(\text{drive fires} \cap \text{clear-cut rule exists})$
- $C_{\text{rule}} \approx 0$ (deterministic execution)
- $p_{\text{llm}} = P(\text{drive fires} \cap \text{no clear-cut rule})$
- $C_{\text{llm}} \gg 0$ (token cost + latency)
- $p_{\text{exec}} = P(\text{LLM produced actionable decision})$
- $C_{\text{exec}} \approx 0$ (parameterized execution)

The efficiency thesis: as the rule engine accumulates more patterns (from meta-agent evolution), $p_{\text{rule}}$ increases and $p_{\text{llm}}$ decreases. The system gets cheaper over time as it learns which situations have clear-cut responses.

### 4.4 Skill Evolution Dynamics

The meta-agent's revision process is modeled as a bounded optimization:

$$\text{SKILL}_{t+1} = \text{SKILL}_t + \Delta \text{SKILL}_t$$

Subject to:
$$|\Delta \text{SKILL}_t| \leq \epsilon_{\text{max}}$$
$$\text{validate}(\text{SKILL}_{t+1}, \mathcal{D}_{\text{held-out}}) \geq \theta_{\text{min}}$$

Where $\epsilon_{\text{max}}$ bounds the revision size per cycle and $\mathcal{D}_{\text{held-out}}$ is a held-out validation set. This prevents the meta-agent from making large, unrecoverable changes based on noisy feedback.

For drive threshold revision:

$$|\tau_i^{t+1} - \tau_i^t| \leq \Delta\tau_{\text{max}}$$

This prevents threshold drift in noisy environments (e.g., adversarial markets).

---

## 5. Comparison to Existing Approaches

### 5.1 Architectural Comparison

| Property | Reactive Agent | Cron Agent | RAG Agent | Always-On RL | **Drive Loop Agent** |
|----------|---------------|------------|-----------|--------------|---------------------|
| Self-triggered | No | No (clock) | No | Yes (loop) | **Yes (internal state)** |
| Persistent goals | No | No | Weak (retrieval) | No | **Yes (standing intentions)** |
| Internal state | None | None | Retrieval-augmented | Memory DB | **Homeostatic drives** |
| Self-improvement | None | None | None | Weight update | **Artifact evolution** |
| Inspectable changes | N/A | N/A | N/A | No (weights) | **Yes (skill files)** |
| LLM call efficiency | N/A (always LLM) | N/A (always LLM) | N/A (always LLM) | Always LLM | **Tiered (most bypass LLM)** |

### 5.2 The Key Distinctions

**vs. Always-On RL (weight modification):**
- Weight changes are global — modifying one capability can regress another
- Weight changes are opaque — you cannot read a weight and understand what it represents
- Weight changes are expensive — require training infrastructure and validation
- Weight changes are hard to revert — need checkpoint management

**vs. Skill artifact evolution:**
- Skill changes are local — only the specific tactic is modified
- Skill changes are inspectable — you can read SKILL.md and understand every entry
- Skill changes are cheap — text editing, no training infrastructure
- Skill changes are trivially revertible — version control

The tradeoff: skill evolution is bounded by the model's capability ceiling. You cannot evolve skills that require capabilities the model doesn't have. Weight updates can, in principle, add new capabilities. But the metasurface results (38% → 74%) suggest that for most practical domains, the model's ceiling is not the bottleneck — the guidance is.

### 5.3 The Meta-Evaluation Question

The metasurface paper's evaluator is deterministic (physics simulation). Trading feedback is noisy (adversarial, non-stationary). This matters for meta-agent safety.

Noisy feedback creates two failure modes:
1. **Overreaction** — a single bad outcome causes threshold/revision that makes things worse
2. **Underreaction** — noise drowns signal, meta-agent never converges

Mitigations:
- Require structural patterns over many outcomes, not single-event reactions
- Bounded revision ranges per cycle
- Validation on held-out data before applying
- Human approval gate for revisions above a confidence threshold
- A losing trade doesn't mean the skill is wrong. A winning trade doesn't mean it's right.

---

## 6. Experimental Design

### 6.1 Proposed Experiments

We propose three experiments of increasing complexity:

**Experiment 1: Drive Loop Efficiency**

Domain: code generation with test suites (clear pass/fail evaluator)

| Agent | Task Completion | LLM Calls / Hour | Autonomous Actions / Hour |
|-------|----------------|-------------------|---------------------------|
| Reactive (request-response) | baseline | measured | 0 |
| Cron (fixed 5-min schedule) | measured | measured | measured |
| Drive loop (this work) | measured | measured | measured |

Hypothesis: drive loop agent achieves higher task completion with fewer LLM calls per hour due to rule engine offloading.

**Experiment 2: Skill Evolution Effectiveness**

Domain: same as Experiment 1, over extended runs

| Agent | Cycle 1 Completion | Cycle 10 Completion | Adaptation Speed |
|-------|-------------------|---------------------|-----------------|
| No skill evolution | baseline | baseline | N/A |
| Skill evolution (this work) | baseline | measured | measured |

Hypothesis: skill evolution produces measurable improvement over cycles without model modification.

**Experiment 3: Standing Intentions vs. RAG Retrieval**

Domain: multi-step tasks with persistent goals

| Agent | Goal Retention | Context Efficiency | Decision Quality |
|-------|---------------|-------------------|-----------------|
| RAG retrieval | measured | measured | measured |
| Standing intentions (this work) | measured | measured | measured |

Hypothesis: standing intentions provide better goal retention with lower context cost because they are structured commitments, not similarity-based retrieval.

### 6.2 Metrics

- **Task completion rate** — primary effectiveness metric
- **LLM calls per unit time** — efficiency metric
- **Autonomous action frequency** — self-triggered actions per hour (the key differentiator)
- **Adaptation speed** — cycles to reach target performance after domain shift
- **Skill artifact quality** — human evaluation of evolved skills
- **Drive coherence** — measure of drive interaction stability (no oscillation, no starvation)

### 6.3 Domains with Grounded Evaluators

The architecture requires a deterministic or semi-deterministic evaluator for the meta-agent to learn from. Suitable domains:

| Domain | Evaluator | Noise Level |
|--------|-----------|-------------|
| Code generation | Test suites | Low |
| Metasurface design | Physics simulation | Zero |
| Trading simulation | Backtester | Medium (but bounded with simulation) |
| Math problem solving | Solution verification | Zero |
| API integration testing | Endpoint responses | Low |

---

## 7. Discussion

### 7.1 The Right Level of Abstraction

We argue that most practical agent improvement happens at the artifact layer, not the weight layer. The model's capability ceiling is high — current frontier models can reason about most domains given the right context. The bottleneck is not capability, it's guidance.

Skill artifact evolution works because:
1. The model already knows how to do most things — it just doesn't know *when* and *which variant*
2. Skills encode *when* (trigger conditions) and *which variant* (tactics for specific situations)
3. This is exactly what the metasurface meta-agent provides: context-dependent guidance

### 7.2 Autonomous Action and Safety

Self-triggering agents raise safety concerns. A drive loop that fires incorrectly could burn compute, make bad trades, or take harmful actions.

Mitigations:
- **Circuit breakers at the drive loop level** — max activations per time window, max LLM calls per cycle
- **Immutable kernel** — the drive loop cannot modify its own safety constraints
- **Human approval gates** — high-impact actions require confirmation
- **Auditable provenance** — every action traces to a drive firing, an intention activation, and a skill entry

### 7.3 Scalability and Multi-Agent Composition

Open questions:
1. How many drives before the system becomes incoherent? Biological systems have dozens. Software agents likely need 5-15.
2. Can one agent's drive output be another agent's input? This would enable multi-agent coordination without explicit message passing — just shared drive state.
3. Does the drive loop generalize across domains, or is it domain-specific? The structure (homeostatic variables with thresholds) is general, but the specific drives are domain-specific.

### 7.4 Limitations

- The architecture is bounded by the underlying model's capability. Evolving skills cannot add capabilities the model doesn't have.
- The meta-agent's effectiveness depends on the evaluator's quality. Noisy or adversarial feedback will produce poor skill evolution.
- The formal drive dynamics are simplified. Real-world environments are more complex than the hitting-time model suggests.
- No implementation exists yet. All claims are theoretical.

### 7.5 Future Work

- Implement the drive loop in NullClaw (Zig) as the always-on process
- Benchmark against reactive, cron, and RAG baselines on code generation tasks
- Investigate drive interaction topologies (additive, multiplicative, suppression networks)
- Explore whether the meta-agent can learn *new drives* (not just tune existing ones)
- Apply the architecture to autonomous on-chain agents (NEAR protocol) as a concrete deployment target

---

## 8. Conclusion

We have presented an architecture for autonomous LLM agents that maintains persistent internal state through homeostatic drives, acts on standing intentions without external prompting, and improves through evolving inspectable skill artifacts. The architecture is grounded in cognitive science (Hull, 1943; Bratman, 1987), validated conceptually by recent empirical results (Huang et al., 2026), and designed for practical deployment with bounded self-improvement.

The key insight is that self-improvement at the artifact layer — rather than the weight layer — is the correct abstraction for most practical domains. The model is a capable generalist. What it needs is not more capability, but better context-dependent guidance. Skill artifacts provide that guidance in a form that is inspectable, revertible, and safe to evolve.

The agent that checks its own state, decides to act, and writes a better guide for itself next time — that is the pattern.

---

## References

1. Bratman, M. E. (1987). *Intention, Plans, and Practical Reason*. Harvard University Press.
2. Deci, E. L., & Ryan, R. M. (1985). *Intrinsic Motivation and Self-Determination in Human Behavior*. Plenum.
3. Einstein, G. O., & McDaniel, M. A. (2005). Prospective memory: Multiple retrieval processes. *Current Directions in Psychological Science*, 14(6), 286-290.
4. Flavell, J. H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10), 906-911.
5. Huang, Y., et al. (2026). A Self-Evolving Agentic Framework for Metasurface Inverse Design. *arXiv:2604.01480*.
6. Hull, C. L. (1943). *Principles of Behavior*. Appleton-Century-Crofts.
7. Kampouse, J. (2026). Red-Black Problem Trees: Fractal Orchestration for LLM Agent Decomposition. *vibe-paper repository*.
8. Ondras, J., & Šuppa, B. (2025). FractalBench: Benchmarking Fractal Program Synthesis. *arXiv*.
9. Rao, A. S., & Georgeff, M. P. (1991). Modeling rational agents within a BDI-architecture. *KR*, 91, 473-484.
10. Wooldridge, M. (2000). Reasoning about rational agents. *MIT Press*.
11. Zeigarnik, B. (1927). Über das Behalten von erledigten und unerledigten Handlungen. *Psychologische Forschung*, 9, 1-85.

---

*Last updated: 2026-04-26*
*Status: Concept paper. No implementation. No benchmarks.*
