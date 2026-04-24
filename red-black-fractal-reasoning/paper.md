# Fractal Reasoning via Red-Black Tree Context Architecture: Self-Balancing Recursive Abstraction for LLM Agents

## Abstract

Current LLM agents process context as a flat, linear sequence of messages — fundamentally incompatible with the recursive structure of complex reasoning. We introduce the **Red-Black Fractal Context (RBFC)** architecture, which restructures agent context as a self-balancing red-black tree where node colors encode epistemic status (verified observation vs. speculative inference) and tree invariants enforce reasoning discipline. Drawing on evidence that multimodal models achieve 76% syntactic but only 4% semantic correctness on recursive program synthesis tasks (Ondras & Šuppa, 2025), we argue that the gap between pattern matching and recursive abstraction is fundamentally a context architecture problem, not a model capability problem. RBFC externalizes branching state, enforces ground-before-speculate ordering, and provides automatic O(log n) context balancing — making recursive reasoning tractable regardless of context length. We formalize the mapping from red-black tree invariants to epistemic constraints, propose a tree-rotation-based context compaction algorithm, and outline implementation paths for existing agent frameworks.

---

## 1. Introduction

### 1.1 The Recursive Reasoning Gap

Recent work on visual-mathematical reasoning reveals a striking asymmetry in LLM capabilities. FractalBench (Ondras & Šuppa, 2025) evaluated four leading multimodal models on synthesizing fractal programs from images — tasks defined by Iterated Function Systems (IFS) with as few as 2-8 contraction mappings. Results:

| Capability | Performance |
|-----------|-------------|
| Syntactic code generation (code runs) | 76.1% |
| Semantic correctness (reproduces fractal) | 4.2% |
| Geometric transformations (Koch curves) | 17-21% |
| Branching recursion (tree fractals) | <2% |

The models can compose local operations but cannot infer generative rules — they produce *something* recursive but not the *right* recursion. This failure is not about intelligence; it is about architecture. The models are asked to perform recursive abstraction while operating on a context that is fundamentally linear.

### 1.2 The Flat Context Problem

Every mainstream agent framework — LangChain, AutoGPT, Claude, GPT, nullclaw, ironclaw — represents context as an ordered list:

```
[system, user, assistant+tool, assistant+tool, ..., user]
```

This is a list. A sequence. Linear. It has no concept of branching, nesting, or self-similar sub-structure. Yet the reasoning tasks we care about — debugging, system design, mathematical proof, multi-step planning — are inherently recursive. They decompose into sub-problems that have the same structure as the whole.

When an agent spawns sub-tasks (tool calls, delegated agents, parallel execution), the results are flattened back into the linear stream. The recursive structure of the computation is destroyed before the model ever sees it. The model must reconstruct what was a tree from what is now a list.

### 1.3 Our Hypothesis

**If context is structured as a self-balancing red-black tree — where node color encodes epistemic status and tree invariants enforce reasoning discipline — then LLM agents will exhibit significantly improved recursive reasoning, particularly on tasks requiring branching abstraction.**

This is not a model change. It is a context architecture change. The model stays the same. What changes is the shape of the information it receives.

---

## 2. Background & Related Work

### 2.1 Fractal Thinking in AI

Fractals are objects defined by self-similarity across scales. An IFS with contraction maps {f₁, f₂, ..., fₖ} generates a fractal F where F = ∪fᵢ(F). The entire structure is contained in the mapping rules — infinite complexity from finite specification.

This property makes fractals ideal probes for recursive abstraction: to reproduce a fractal, you must infer the generative rule, not copy the visible pattern. FractalBench exploited this to show that current models can recognize self-similarity but cannot infer precise IFS parameters.

### 2.2 Structured Context Approaches

| Approach | Structure | Recursion Support | Self-Balancing |
|----------|-----------|-------------------|----------------|
| Linear context (standard) | Flat list | None | N/A |
| Hierarchical summarization | Tree of summaries | Single path | No |
| Graph RAG | Arbitrary graph | Via traversal | No |
| Constraint Composite Graphs | DAG of decisions | Dependency chains | No |
| **RBFC (ours)** | **Red-black tree** | **Intrinsic** | **Yes** |

### 2.3 Red-Black Trees

Red-black trees are self-balancing binary search trees with five invariants that guarantee O(log n) operations:

1. Every node is either red or black
2. The root is black
3. Every leaf (NIL) is black
4. If a node is red, both its children are black
5. Every path from a node to its descendant NIL nodes contains the same number of black nodes

These invariants ensure the tree height is at most 2·log₂(n+1), providing guaranteed logarithmic balance regardless of insertion order. This property has been used in operating systems (Linux completely fair scheduler, CFS), databases (MySQL InnoDB), and language runtimes (Java TreeMap) — but never, to our knowledge, as a reasoning architecture.

---

## 3. The Red-Black Fractal Context

### 3.1 Core Mapping: Tree Invariants → Epistemic Constraints

We map each red-black tree invariant to a specific reasoning discipline:

**Invariant 1: Every node is RED or BLACK.**

→ Every reasoning step has an epistemic status:
- **BLACK (verified):** Grounded in observation. Tool output, user input, confirmed fact, measurement, test result. The model *saw* this. High confidence.
- **RED (speculative):** Generated by the model. Hypothesis, prediction, plan, inferred rule. The model *believes* this but has not verified it. Uncertain.

**Invariant 2: The root is BLACK.**

→ The initial user request is ground truth. It is the verified input from which all reasoning grows. The model does not speculate about what the user wants — it accepts the request as given.

**Invariant 3: Every leaf (NIL) is BLACK.**

→ Unanswered questions, unknowns, dead ends, and "I don't know" states are all grounded in honest ignorance — not speculation. When a branch of reasoning terminates without resolution, it terminates at a verified BLACK dead end, not a dangling RED hypothesis.

**Invariant 4: RED nodes must have BLACK children.**

→ Speculation must be immediately followed by verification. A hypothesis (RED) cannot spawn further hypotheses (RED) — it must first produce observations (BLACK) through tool calls, measurements, or user confirmation. No chains of ungrounded reasoning. No speculation built on speculation.

**Invariant 5: Equal black-height on all root-to-leaf paths.**

→ Every line of reasoning, regardless of how deep or how many branches it explores, must be anchored by the same number of verified facts. You cannot go arbitrarily deep without accumulating evidence. Shallow branches that lack grounding are rotated to promote verified nodes, maintaining balance.

### 3.2 Why This Matters

The FractalBench results map directly onto these invariants:

| FractalBench Finding | RBFC Explanation |
|---------------------|-------------------|
| 76% code runs but 4% is correct | Models generate plausible RED nodes (speculative code) without sufficient BLACK children (verification against reference) |
| Koch curves succeed at 17-21% | Geometric transforms are local operations — easily verified within a single BLACK-RED-BLACK path |
| Tree fractals fail at <2% | Branching recursion requires maintaining multiple independent RED→BLACK verification paths simultaneously — exactly what invariant #4 enforces |
| CoT hurts fractal tasks | Verbal reasoning chains RED→RED→RED (speculation on speculation), violating invariant #4 |

### 3.3 Tree Structure

```
                         BLACK: User request
                        /                    \
              RED: Hypothesis A          RED: Hypothesis B
              /              \            /              \
     BLACK: Tool result    BLACK: Obs    BLACK: Tool    BLACK: Obs
         /        \                      /        \
    RED: Next   RED: Next            RED: Refine  RED: Refine
    step       step                  hypothesis   hypothesis
    /    \      /    \                /    \        /    \
  BLK   BLK   BLK   BLK           BLK   BLK    BLK   BLK
  tool  tool  tool  tool           tool  tool   tool  tool
```

Each subtree is itself a valid red-black tree. Every level follows the same rules. This is self-similarity — the defining property of a fractal.

---

## 4. Formal Definition

### 4.1 Context Node

```python
@dataclass
class ContextNode:
    content: str                    # The actual message/content
    role: str                       # "user", "assistant", "tool", "system"
    color: Literal["red", "black"]  # Epistemic status
    
    # Tree structure
    parent: Optional["ContextNode"] = None
    left: Optional["ContextNode"] = None
    right: Optional["ContextNode"] = None
    
    # Metadata
    black_height: int = 0           # Number of BLACK nodes to NIL
    depth: int = 0                  # Depth from root
    timestamp: float = 0.0          # When this node was created
    evidence_refs: List[str] = field(default_factory=list)  # IDs of BLACK nodes that ground this
    
    def is_verified(self) -> bool:
        return self.color == "black"
    
    def is_speculative(self) -> bool:
        return self.color == "red"
    
    def verify(self, evidence: str) -> "ContextNode":
        """Convert a RED hypothesis to BLACK once evidence arrives."""
        assert self.color == "red"
        self.color = "black"
        self.evidence_refs.append(evidence)
        return self
```

### 4.2 Invariant Validation

```python
class RBFCValidator:
    """Enforces red-black invariants on the context tree."""
    
    def validate(self, root: ContextNode) -> List[str]:
        violations = []
        
        # Invariant 2: Root must be black
        if root and root.color == "red":
            violations.append("ROOT_NOT_BLACK")
        
        # Invariant 4: Red nodes must have black children
        self._check_red_children(root, violations)
        
        # Invariant 5: Equal black-height on all paths
        heights = self._collect_black_heights(root)
        if len(set(heights)) > 1:
            violations.append(f"BLACK_HEIGHT_MISMATCH: {set(heights)}")
        
        return violations
    
    def _check_red_children(self, node, violations):
        if node is None:
            return
        if node.color == "red":
            if node.left and node.left.color == "red":
                violations.append(f"RED_RED_EDGE: {node.content[:30]}... -> left")
            if node.right and node.right.color == "red":
                violations.append(f"RED_RED_EDGE: {node.content[:30]}... -> right")
        self._check_red_children(node.left, violations)
        self._check_red_children(node.right, violations)
    
    def _collect_black_heights(self, node, current_height=0):
        if node is None:
            return [current_height]
        h = current_height + (1 if node.color == "black" else 0)
        return self._collect_black_heights(node.left, h) + \
               self._collect_black_heights(node.right, h)
```

### 4.3 Insertion with Rebalancing

When a new reasoning step is added to context, it follows standard red-black insertion with a domain-specific twist:

```python
class RBFCTree:
    def insert(self, content: str, role: str, 
               color: str = "red", parent_id: str = None) -> ContextNode:
        """Insert a new reasoning step, maintaining all invariants."""
        
        node = ContextNode(
            content=content, 
            role=role, 
            color=color,
            timestamp=time.time()
        )
        
        # Standard BST insertion
        self._bst_insert(node, parent_id)
        
        # Rebalance to maintain invariants
        self._rebalance(node)
        
        return node
    
    def _rebalance(self, node):
        """Apply rotations to restore red-black invariants.
        
        In the reasoning context, rotation means:
        - Promote a verified (BLACK) fact to anchor a deep branch
        - Demote a speculative (RED) hypothesis that lacks grounding
        - Restructure the reasoning tree to maintain balance
        """
        while node.parent and node.parent.color == "red":
            # Standard red-black rotation cases
            # Case 1: Uncle is red → recolor
            # Case 2: Uncle is black, node is inner child → rotate
            # Case 3: Uncle is black, node is outer child → rotate + recolor
            self._apply_rotation(node)
        
        # Ensure root stays black (Invariant 2)
        self.root.color = "black"
    
    def _apply_rotation(self, node):
        """
        Reasoning rotation: when a branch becomes too deep 
        relative to others (too many RED nodes without BLACK grounding),
        restructure to promote verified evidence.
        
        This is the key mechanism that prevents context from growing 
        unboundedly along speculative paths.
        """
        # ... standard RB rotation logic ...
        # See CLRS Chapter 13 for algorithmic details
        pass
```

---

## 5. Context Compaction via Tree Rotation

### 5.1 The Problem with Linear Compaction

Current context compaction strategies (summarization, sliding window, token truncation) operate on the flat message list. They decide *what to keep* based on recency, relevance scores, or token budgets. This is lossy and structure-agnostic — a 50-turn reasoning chain might lose its middle, breaking the logical arc.

### 5.2 Rotation-Based Compaction

RBFC provides a *principled* alternative. Instead of discarding messages, we restructure the tree. Red-black rotations are the mechanism:

**Left rotation (promote right child):**
When a right-heavy branch has accumulated too many speculative nodes (RED), rotate to promote a verified subtree (BLACK) as the new anchor.

```
Before rotation:          After rotation:
    B (evidence)              D (promoted)
     \                         /
      D (speculation)         B
     / \                       \
    C   E                     C
   (spec) (verified)        (now demoted)
```

**Right rotation (promote left child):** Mirror operation.

**The insight:** rotation doesn't delete information — it *restructures* it. A deep speculative branch is collapsed by promoting its most grounded subtree. The information is preserved in a more balanced configuration.

### 5.3 Black-Height as Evidence Density

The black-height of a subtree measures how many verified facts anchor it. During compaction:

```python
def compact_context(tree: RBFCTree, max_nodes: int) -> RBFCTree:
    """Compact context by rotating to promote evidence-dense subtrees."""
    
    while tree.node_count > max_nodes:
        # Find the shallowest BLACK node with the highest black-height
        # This is the most evidence-dense anchor point
        target = find_best_rotation_target(tree)
        
        if target:
            # Rotate to flatten the subtree under this anchor
            # Verified facts (BLACK) are preserved
            # Speculative branches (RED) without grounding are pruned
            tree.rotate(target)
        else:
            # No good rotation target — fall back to pruning
            # the least-grounded RED subtree
            prune_weakest_branch(tree)
    
    return tree
```

### 5.4 Comparison with Existing Approaches

| Method | What it discards | Structure preserved? | Balancing |
|--------|-----------------|---------------------|-----------|
| Sliding window | Oldest messages | No | None |
| Summarization | Detail, nuance | Partially | None |
| Token truncation | Whatever exceeds limit | No | None |
| Graph RAG | Unretrieved nodes | Yes (graph) | No |
| **RBFC rotation** | **Ungrounded speculation only** | **Yes (tree)** | **O(log n) guaranteed** |

---

## 6. Integration with Agent Loops

### 6.1 The Predict-Act-Observe Cycle as IFS

The standard agent loop is:

```
PREDICT → ACT → OBSERVE → (repeat or stop)
```

This is an Iterated Function System. Each iteration applies the same transformation (predict-act-observe) to the current state. The attractor of this IFS is the final answer.

In RBFC, each iteration becomes a subtree:

```
Iteration 1:
  BLACK: user request
  └── RED: plan (hypothesis)
      └── BLACK: tool result (observation)
          └── RED: next plan (hypothesis)
              └── BLACK: tool result (observation)

Iteration 2 (deeper, same structure):
  └── RED: refined plan (hypothesis)
      ├── BLACK: tool result A (branch 1)
      │   └── RED: sub-plan A.1
      │       └── BLACK: tool result A.1
      └── BLACK: tool result B (branch 2)
          └── RED: sub-plan B.1
              └── BLACK: tool result B.1
```

Each iteration is a self-similar copy of the predict-act-observe pattern. The tree *is* the fractal. The agent doesn't just execute a loop — it grows a fractal of reasoning.

### 6.2 Enforcement in the System Prompt

```python
REASONING_PROTOCOL = """
You are operating within a Red-Black Fractal Context.

RULES:
1. Every claim you make is either VERIFIED (black) or SPECULATIVE (red).
2. Verified claims are grounded in: tool output, user input, code execution, 
   measurements, or direct observation.
3. Speculative claims are: hypotheses, predictions, plans, inferences.
4. CRITICAL: Every speculation (red) must be followed by verification (black) 
   before further speculation. You cannot chain predictions.
5. If you are unsure, say "I don't know" — this is a black leaf (verified 
   ignorance), which is valid. Do not guess.

OUTPUT FORMAT:
- [B] for verified/black statements
- [R] for speculative/red statements
- [?] when you need to verify before continuing

Example valid reasoning:
[R] I hypothesize the bug is in the allocator
[?] Let me check — running: grep -n "alloc" src/main.zig
[B] Found 3 allocation sites at lines 45, 89, 142
[R] Line 142 uses the arena allocator which may be exhausted
[?] Let me verify — running: zig build
[B] Build succeeds, but runtime crash at arena exhaustion
[B] CONFIRMED: bug is in arena allocator at line 142

Example INVALID reasoning (do NOT do this):
[R] The bug is probably in the allocator
[R] It's probably an OOM issue
[R] We should switch to heap allocation
^^^ VIOLATION: three consecutive speculative statements without verification
"""
```

### 6.3 Implementation in Existing Frameworks

```python
class RBFCAgentWrapper:
    """Wraps any LLM agent with red-black fractal context."""
    
    def __init__(self, base_agent, max_context_nodes=128):
        self.agent = base_agent
        self.tree = RBFCTree()
        self.max_nodes = max_context_nodes
    
    def chat(self, user_message: str) -> str:
        # 1. Insert user message as BLACK (ground truth)
        self.tree.insert(user_message, "user", color="black")
        
        # 2. Flatten tree to linear context for the LLM
        context = self._tree_to_context()
        
        # 3. Get model response
        response = self.agent.generate(context)
        
        # 4. Parse response for RED/BLACK annotations
        statements = self._parse_annotations(response)
        
        # 5. Insert into tree, enforcing invariants
        for stmt in statements:
            if stmt.color == "red":
                # Speculation — insert as RED
                node = self.tree.insert(stmt.content, "assistant", color="red")
            elif stmt.color == "black":
                # Verification — insert as BLACK
                node = self.tree.insert(stmt.content, "assistant", color="black")
                # Check if this verifies any pending RED parent
                self._ground_pending_hypotheses(node)
        
        # 6. Validate invariants
        violations = RBFCValidator().validate(self.tree.root)
        if violations:
            # Reject response, ask model to fix violations
            return self._request_revision(violations)
        
        # 7. Compact if needed
        if self.tree.node_count > self.max_nodes:
            self.tree = compact_context(self.tree, self.max_nodes)
        
        return response
    
    def _tree_to_context(self) -> str:
        """Flatten the RB tree back to a linear context for the LLM.
        
        Key insight: the flattening order encodes the tree structure.
        We use in-order traversal, which for a BST produces sorted output.
        For RBFC, we use depth-annotated in-order traversal that preserves
        the reasoning hierarchy.
        """
        lines = []
        self._inorder_traverse(self.tree.root, lines, depth=0)
        return "\n".join(lines)
    
    def _inorder_traverse(self, node, lines, depth):
        if node is None:
            return
        prefix = "  " * depth
        marker = "[B]" if node.color == "black" else "[R]"
        role = node.role.upper()
        lines.append(f"{prefix}{marker} [{role}] {node.content}")
        self._inorder_traverse(node.left, lines, depth + 1)
        self._inorder_traverse(node.right, lines, depth + 1)
```

---

## 7. Why This Should Work: Theoretical Arguments

### 7.1 Fractal Self-Similarity Provides Compositional Generalization

A fractal is defined by rules that apply identically at every scale. In RBFC, the predict-verify cycle has the same structure at every depth:

- Depth 0: User asks question → Agent hypothesizes → Agent verifies
- Depth 1: Sub-question → Sub-hypothesis → Sub-verification
- Depth n: ... same pattern ...

This means a model that learns to reason well at one depth should transfer to all depths. The reasoning pattern is *self-similar* — it's a fractal. Current flat contexts don't exhibit this property because depth information is lost during flattening.

### 7.2 Invariant Enforcement Prevents Reasoning Failure Modes

The specific failures observed in FractalBench map to invariant violations:

**Failure: Models produce valid code that draws the wrong fractal.**
→ The model generates a plausible hypothesis (RED) but never verifies against the reference (no BLACK child). The RED node is accepted because there's no mechanism to reject ungrounded speculation.

**Failure: Tree fractals fail at <2%.**
→ Branching requires maintaining N independent verification paths. In a flat context, the model loses track of which branches have been verified. The tree structure makes branching explicit — each branch is a subtree with its own black-height.

**Failure: Chain-of-thought hurts fractal tasks.**
→ CoT produces RED→RED→RED chains (speculation on speculation), violating invariant #4. RBFC rejects such chains at insertion time.

### 7.3 Self-Balancing Provides Bounded Reasoning

The red-black balance guarantee ensures that no reasoning path exceeds 2·log₂(n) in depth. For a context of 128 nodes, the maximum reasoning depth is 14. For 1024 nodes, it's 20. This is a hard upper bound — not a soft heuristic.

This matters because: (a) it prevents the model from going on unbounded speculative spirals, (b) it ensures roughly equal evidence density across all branches, and (c) it makes context compaction deterministic and structure-preserving.

### 7.4 Rotation as Meaning-Preserving Compression

Standard compaction (summarization, truncation) is lossy. Information is destroyed. Tree rotation is different — it restructures without destroying. A rotation changes the *shape* of the tree but preserves the *content* and the *invariants*.

In reasoning terms: rotation promotes the most evidence-grounded subtree and demotes speculative branches. The verified facts (BLACK nodes) survive. The ungrounded hypotheses (RED nodes without BLACK children) are the ones that get pruned. This is the right thing to throw away.

---

## 8. Proposed Experiments

### 8.1 Experiment 1: FractalBench with RBFC Context

**Hypothesis:** Restructuring the context as a red-black tree with epistemic annotations will improve visual correctness on FractalBench from 4.2% baseline to >15%.

**Setup:**
- Same 12 fractals, same models, same evaluation (IoU > 0.95)
- Replace flat context with RBFC-structured context
- System prompt enforces [B]/[R] annotations
- Invariant violations trigger revision requests

**Metrics:**
- Visual correctness (primary)
- Code execution rate
- Average invariant violations per response
- Context compaction ratio

### 8.2 Experiment 2: Branching Recursion Recovery

**Hypothesis:** RBFC will improve tree fractal accuracy from <2% to >10% by making branching structure explicit in context.

**Setup:**
- Focus on the 4 tree fractal types
- Compare flat context vs. RBFC vs. RBFC + explicit branching annotations
- Measure whether the model produces actual branching recursion vs. iterative approximation

### 8.3 Experiment 3: Multi-Step Agent Tasks

**Hypothesis:** Agents using RBFC context will complete multi-step debugging and planning tasks with fewer iterations and less context waste.

**Setup:**
- Standard agent benchmarks (SWE-bench, HumanEval, planning tasks)
- Measure: iterations to completion, total tokens used, success rate
- Compare flat context vs. RBFC with rotation-based compaction

### 8.4 Experiment 4: Context Compaction Quality

**Hypothesis:** Rotation-based compaction preserves more decision-critical information than summarization at equivalent compression ratios.

**Setup:**
- Take 100 agent sessions with 50+ turns each
- Compress to 25%, 10%, 5% of original size using: (a) summarization, (b) sliding window, (c) RBFC rotation
- Measure: decision recall, constraint coverage, answer quality on retrospective questions

---

## 9. Potential Counterarguments & Limitations

### 9.1 "Models Can't Maintain Tree Structure in Attention"

Valid concern. Flattening a tree back to a linear sequence for the LLM loses structural information. However:
- Depth-annotated traversal preserves hierarchy (indentation + markers)
- The [B]/[R] annotations are a lightweight encoding that fits within existing token budgets
- The tree structure is primarily for the *system* (compaction, validation) — the model sees an annotated linear context, not a raw tree

### 9.2 "Red-Black Invariants Are Too Strict for Creative Reasoning"

The invariants prevent chains of speculation, which might suppress creative exploration. Mitigations:
- Allow configurable "speculation budgets" — N consecutive RED nodes before requiring BLACK
- Separate "exploration mode" (relaxed invariants) from "verification mode" (strict invariants)
- The model can still explore — it just needs to ground each step before going deeper

### 9.3 "This Is Just Prompt Engineering"

Partially true. The [B]/[R] annotations are a form of structured prompting. But the *tree structure* and *rotation-based compaction* are architectural changes that go beyond prompting. The key insight is that context *shape* matters independently of context *content*.

### 9.4 "Overhead of Tree Maintenance"

Red-black insertion and rotation are O(log n) — negligible compared to LLM inference time. The validation step adds a single pass over the tree, also O(n). Total overhead: minimal.

### 9.5 "The Model Might Not Respect the Annotations"

The system can enforce compliance:
- Reject responses with invariant violations and request revision
- Strip [B] labels from ungrounded claims (demote RED→check)
- Use structured output / tool calls to make annotations machine-parseable

---

## 10. Connection to Existing Work

### 10.1 Iterated Function Systems

RBFC is, quite literally, an IFS applied to reasoning. The red-black invariants are the contraction maps. Each iteration of the agent loop applies the same transformation (predict-verify) to produce a self-similar structure. The tree's attractor is the final verified answer.

### 10.2 Scientific Method as Fractal

The RBFC cycle mirrors the scientific method:
1. **Hypothesize** (RED) — propose a rule
2. **Experiment** (tool call) — test the rule
3. **Observe** (BLACK) — record the result
4. **Revise** (rotation) — restructure if wrong

This cycle repeats at every level of abstraction — from individual tool calls to entire agent sessions. The fractal nature of the scientific method has been noted before (Gleick, 1987; Bak, 1996) but never formalized as a computational architecture.

### 10.3 Duality with Constraint Composite Graphs

The CCM paper in this repository proposes DAG-based decision extraction. RBFC is complementary:
- CCM captures *what was decided* (content)
- RBFC captures *how reasoning is structured* (process)
- Together, a CCG (content) embedded in an RBFC (process) would provide both semantic and structural reasoning support

---

## 11. Implementation Roadmap

### Phase 1: Annotations Only (Week 1-2)
- Add [B]/[R] annotation system to agent system prompt
- Parse annotations from model responses
- Log invariant violations (read-only, no enforcement)
- Measure impact on FractalBench-style tasks

### Phase 2: Tree Structure (Week 3-4)
- Implement RBFCTree data structure
- Build context as tree instead of flat list
- Flatten with depth annotations for LLM consumption
- Enforce invariants at insertion time

### Phase 3: Rotation-Based Compaction (Week 5-6)
- Implement rotation logic
- Replace summarization with rotation-based compaction
- Benchmark against existing compaction strategies

### Phase 4: Full Evaluation (Week 7-8)
- Run all four proposed experiments
- Analyze failure modes and invariant violation patterns
- Write up results

---

## 12. Conclusion

We propose the Red-Black Fractal Context — a self-balancing tree architecture for LLM agent reasoning that enforces epistemic discipline through red-black tree invariants. The core insight is that the gap between pattern matching and recursive abstraction (76% vs. 4% on FractalBench) is fundamentally a context architecture problem: models are asked to perform recursive reasoning on linear context.

By structuring context as a red-black tree where BLACK = verified observation and RED = speculative inference, we:
1. **Make branching explicit** — each reasoning branch is a subtree, not a flattened sequence
2. **Enforce ground-before-speculate** — invariant #4 prevents chains of ungrounded reasoning
3. **Guarantee bounded depth** — self-balancing ensures O(log n) reasoning paths
4. **Provide principled compaction** — rotation preserves verified facts while pruning ungrounded speculation

The fractal connection is not metaphorical. Each subtree of an RBFC is itself a valid red-black tree — the same rules apply at every node, at every depth. This self-similarity is exactly the property that fractals exploit to generate infinite complexity from finite rules. If reasoning has fractal structure, then reasoning architecture should too.

---

## References

- Ondras, J., & Šuppa, M. (2025). FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis. *arXiv:2511.06522v1*.
- Cormen, T. H., et al. (2009). Introduction to Algorithms (3rd ed.). MIT Press. Chapter 13: Red-Black Trees.
- Barnsley, M. F. (2014). Fractals Everywhere. Dover Publications.
- Gleick, J. (1987). Chaos: Making a New Science. Viking.
- Bak, P. (1996). How Nature Works: The Science of Self-Organized Criticality. Copernicus.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.

---

*Status: Hypothesis paper. No implementation. No benchmarks. Ideas thrown at the wall.*

*Last updated: 2026-04-24*
