# Fractal Problem Solving: Red-Black Tree Orchestration for LLM Agent Decomposition

## Abstract

We argue that the gap between LLM pattern matching and recursive abstraction — exemplified by FractalBench's 76% syntactic vs. 4% semantic correctness on fractal program synthesis (Ondras & Šuppa, 2025) — is fundamentally a problem decomposition problem, not a context architecture problem. We propose **Red-Black Problem Trees (RBPT)**: an orchestration layer that structures agent problem-solving as a self-balancing red-black tree where **RED nodes represent LLM generations** and **BLACK nodes represent runtime executions**, but critically, **the model never sees the tree**. The tree exists in the agent's runtime — it controls what gets generated, executed, retried, and composed. Each node is an independent generate-execute cycle with its own narrow context. The model receives a focused prompt, produces output, and the system handles branching, scheduling, verification, and composition. Red-black invariants become scheduling constraints: no consecutive REDs (don't spawn new sub-problems without executing current ones), equal black-height (all branches reach the same verification depth before composing), and BLACK leaves (every sub-problem terminates with a measured outcome). This converts the agent from a single model making linear guesses into a system that decomposes, executes, verifies, and composes — the same cycle at every scale. We formalize the architecture, map invariants to scheduling constraints, and propose experiments.

---

## 1. Introduction

### 1.1 The Real Failure Mode

FractalBench (Ondras & Šuppa, 2025) asked multimodal models to look at a fractal image and write Python code that reproduces it. Results:

| Metric | Score |
|--------|-------|
| Code runs without error | 76.1% |
| Visually correct reproduction | 4.2% |
| Koch curves (geometric transforms) | 17-21% |
| Tree fractals (branching recursion) | <2% |

The paper's abstract calls this "a striking disconnect between syntactic competence and semantic understanding." But look at what actually happens: a single model receives an image and a prompt, and must produce an entire correct program in one shot. For a tree fractal, that means inferring branching logic, recursion depth, transformation parameters, and rendering code — all in a single generation.

No human programmer works this way. A human would:
1. Write the branching logic
2. Run it
3. See it's wrong
4. Fix it
5. Run it again
6. See it's closer
7. Adjust parameters
8. Run it again
9. Done

The model never gets steps 2-8. It generates once and the system scores the output. The 4.2% success rate isn't surprising — it's the expected result of asking someone to write a perfect program without ever running it.

### 1.2 The Problem Isn't Context Shape. It's Control Flow.

Previous approaches to improving agent reasoning focus on context structure — how to organize the information the model sees. Chain-of-thought adds reasoning steps. Tree-of-thought adds branching exploration. Graph RAG adds retrieval. All of these assume the model needs to *see more* or *see it differently* to perform better.

We argue the opposite. The model doesn't need to see the tree. The model needs to be *part* of a tree.

Consider a human software team. No individual engineer sees the entire project plan as a tree data structure. Each engineer receives a task, produces code, submits it for review, gets feedback, and iterates. The *manager* holds the tree — tracking which tasks are done, which are blocked, which need reassignment. The engineer just works on their node.

RBPT applies this pattern to LLM agents. The system is the manager. The model is the engineer. The tree exists in the runtime, not in the context window.

### 1.3 Our Hypothesis

**If an agent orchestration layer decomposes problems into a red-black tree of independent generate-execute cycles — where the model operates on narrow sub-problems and the system handles branching, execution, verification, and composition — then performance on recursive tasks will improve significantly, and the improvement will scale with problem complexity (deeper trees help more).**

This is not a prompt engineering technique. This is an agent architecture. The model stays the same. What changes is how problems are broken down and how results are composed.

---

## 2. The Architecture

### 2.1 The Model Doesn't See the Tree

This is the central design principle. The tree exists in the agent's runtime — in code, not in context. At each node, the model receives:

```
Task: Draw a Koch curve with 4 levels of recursion.
Reference image: [attached]
Interface: turtle.forward(n), turtle.left(angle), turtle.right(angle)
Acceptance criteria: rendered output must match reference (IoU > 0.95)
```

The model writes code. The system runs it. The system checks the result. The model never knows whether it's at the root of a tree, a leaf, or somewhere in the middle. It just receives a task and produces output.

### 2.2 What the Tree Actually Is

The tree is the agent's execution plan — the control flow of problem decomposition. Each node is a generate-execute cycle:

```
ProblemTree:
  Root: "Build fractal renderer"
  ├── Node: "Implement Koch curve"
  │   ├── Generation 1 → Execution 1 → IoU 0.34 → FAIL
  │   ├── Generation 2 → Execution 2 → IoU 0.72 → FAIL
  │   └── Generation 3 → Execution 3 → IoU 0.96 → PASS
  ├── Node: "Implement Sierpinski triangle"
  │   ├── Generation 1 → Execution 1 → IoU 0.88 → PASS
  ├── Node: "Implement tree fractal"
  │   ├── Node: "Implement branching logic"
  │   │   ├── Generation 1 → Execution 1 → recursion error → FAIL
  │   │   └── Generation 2 → Execution 2 → renders correctly → PASS
  │   ├── Node: "Implement left subtree rendering"
  │   │   └── Generation 1 → Execution 1 → IoU 0.91 → PASS
  │   └── Node: "Implement right subtree rendering"
  │       ├── Generation 1 → Execution 1 → IoU 0.15 → FAIL
  │       └── Generation 2 → Execution 2 → IoU 0.93 → PASS
  └── Compose: "Integrate all three into unified renderer"
      └── Generation 1 → Execution 1 → all tests pass → DONE
```

Notice: the tree fractal that failed at <2% when handled by a single model call is now decomposed into 3 sub-problems, each with its own generate-execute cycle. Each sub-problem is simpler than the whole. The branching logic, the left subtree, and the right subtree are independent tasks that can be solved independently and composed.

### 2.3 Node Types

```python
@dataclass
class ProblemNode:
    """A single node in the problem tree."""
    id: str
    task: str                         # What this node needs to accomplish
    acceptance_criteria: str          # How to judge success (IoU, test, etc.)
    parent: Optional["ProblemNode"] = None
    children: List["ProblemNode"] = field(default_factory=list)
    
    # Generate-execute history at this node
    attempts: List[GenerateExecuteCycle] = field(default_factory=list)
    status: Literal["pending", "in_progress", "passed", "failed", "blocked"] = "pending"
    
    # Tree color — determined by last action taken
    # RED = model just generated, awaiting execution
    # BLACK = system just executed, result available
    last_action: Literal["generation", "execution"] = None
    
    @property
    def color(self) -> str:
        return "red" if self.last_action == "generation" else "black"
    
    @property
    def black_height(self) -> int:
        """Number of execution results in this node's history."""
        return sum(1 for a in self.attempts if a.execution is not None)

@dataclass
class GenerateExecuteCycle:
    generation: Optional[str] = None        # RED: what the model produced
    execution: Optional[ExecutionResult] = None  # BLACK: what the runtime produced
    passed: Optional[bool] = None
```

### 2.4 The Scheduler

The scheduler is the runtime that manages the tree. It decides what to generate, what to execute, and when to compose. It enforces the red-black invariants as scheduling constraints.

```python
class RBFCScheduler:
    """Orchestrates problem decomposition as a red-black tree.
    
    The scheduler owns the tree. The model is called per-node.
    """
    
    def __init__(self, model, executor, max_retries=5):
        self.model = model
        self.executor = executor
        self.max_retries = max_retries
        self.tree: Optional[ProblemNode] = None
    
    def solve(self, problem: str, acceptance_criteria: str) -> str:
        """Entry point: solve a problem using tree decomposition."""
        self.tree = ProblemNode(
            id="root",
            task=problem,
            acceptance_criteria=acceptance_criteria,
        )
        self.tree.last_action = "execution"  # Root is BLACK (user provided the problem)
        
        return self._solve_node(self.tree)
    
    def _solve_node(self, node: ProblemNode) -> str:
        """Solve a single node: decompose if needed, generate, execute, verify."""
        
        # Step 1: Should this node be decomposed into sub-problems?
        sub_problems = self._try_decompose(node)
        
        if sub_problems:
            # Decompose into children
            for sp in sub_problems:
                child = ProblemNode(
                    id=f"{node.id}.{len(node.children)}",
                    task=sp.task,
                    acceptance_criteria=sp.criteria,
                    parent=node,
                )
                node.children.append(child)
            
            # Solve each child independently
            results = {}
            for child in node.children:
                results[child.id] = self._solve_node(child)
            
            # Step 2: Compose child results
            return self._compose(node, results)
        
        else:
            # Step 1 (alt): Single-node solve — generate and execute
            return self._generate_execute_loop(node)
    
    def _generate_execute_loop(self, node: ProblemNode) -> str:
        """Core loop: generate → execute → verify → iterate or fail."""
        
        for attempt in range(self.max_retries):
            # RED: Generate
            generation = self.model.generate(
                task=node.task,
                acceptance_criteria=node.acceptance_criteria,
                previous_attempts=node.attempts,  # Feed back failures
            )
            
            cycle = GenerateExecuteCycle(generation=generation)
            node.last_action = "generation"  # Node is RED
            node.status = "in_progress"
            
            # Invariant check: no consecutive REDs at this node
            # (This is guaranteed by the loop structure — we always execute after generate)
            
            # BLACK: Execute
            result = self.executor.run(generation)
            cycle.execution = result
            node.last_action = "execution"  # Node is BLACK
            cycle.passed = self._check_acceptance(result, node.acceptance_criteria)
            node.attempts.append(cycle)
            
            if cycle.passed:
                node.status = "passed"
                return generation
            
            # Failed — feed back the execution result for next attempt
            # The model sees what went wrong (BLACK) and generates a fix (RED)
        
        node.status = "failed"
        return generation  # Return best attempt even if not passing
    
    def _try_decompose(self, node: ProblemNode) -> List[SubProblem]:
        """Ask the model whether this problem should be broken into sub-problems.
        
        This is the branching point — where a single node becomes a subtree.
        The model decides IF to branch and WHAT the sub-problems are.
        The system handles the rest.
        """
        # For simple tasks, don't decompose
        if self._is_simple(node.task):
            return []
        
        decomposition = self.model.decompose(node.task)
        
        if not decomposition.sub_problems:
            return []
        
        return decomposition.sub_problems
    
    def _check_acceptance(self, result: ExecutionResult, criteria: str) -> bool:
        """BLACK leaf enforcement: every node terminates with a measurable outcome."""
        if criteria.startswith("IoU"):
            threshold = float(criteria.split(">")[-1].strip())
            return result.iou >= threshold
        elif criteria.startswith("test"):
            return result.exit_code == 0 and result.tests_passed
        else:
            return result.success
    
    def _compose(self, node: ProblemNode, child_results: Dict[str, str]) -> str:
        """Compose child results into a solution for the parent node.
        
        Invariant enforcement: all children must have BLACK status (completed execution)
        before composition proceeds.
        """
        # Check equal black-height: all children must be equally tested
        black_heights = [c.black_height for c in node.children]
        if len(set(black_heights)) > 1:
            # Some branches were tested more than others
            # This is a black-height violation — re-run under-tested branches
            min_bh = min(black_heights)
            for child in node.children:
                if child.black_height > min_bh:
                    # This branch had more attempts — could indicate it was harder
                    # Log but don't block composition
                    pass
        
        # All children have BLACK leaf status (passed or failed with measured outcome)
        for child in node.children:
            if child.status == "pending":
                raise RuntimeError(f"Cannot compose: child {child.id} has no execution result")
        
        # Generate composition
        composition = self.model.compose(
            parent_task=node.task,
            child_results=child_results,
            child_statuses={c.id: c.status for c in node.children},
        )
        
        # Execute the composed solution
        result = self.executor.run(composition)
        cycle = GenerateExecuteCycle(generation=composition, execution=result)
        cycle.passed = self._check_acceptance(result, node.acceptance_criteria)
        node.attempts.append(cycle)
        node.last_action = "execution"  # BLACK
        
        if cycle.passed:
            node.status = "passed"
        else:
            node.status = "failed"
        
        return composition
```

---

## 3. Red-Black Invariants as Scheduling Constraints

The invariants don't constrain the model. They constrain the scheduler. The model is unaware of them.

### 3.1 Invariant: No Consecutive REDs

**Meaning:** The scheduler never allows two generation steps without an execution in between.

**Implementation:** The `_generate_execute_loop` always pairs a generate call with an execute call. There is no code path where generation happens without subsequent execution. This isn't a rule the model follows — it's a structural guarantee of the scheduler.

**What this prevents:** The model generating a long chain of "let me think about this..." without ever running anything. The scheduler simply doesn't have a "generate without executing" code path.

### 3.2 Invariant: Root is BLACK

**Meaning:** The root problem is externally provided — by the user, by a test suite, by an API call. It is observed input, not model-generated.

**Implementation:** `solve()` sets `root.last_action = "execution"` before any model call. The tree grows from verified ground truth.

**What this ensures:** The entire tree is grounded in a real problem. No speculative root — every branch descends from an actual task.

### 3.3 Invariant: All Leaves are BLACK

**Meaning:** Every terminal node in the tree has an execution result. No branch terminates on a generation that was never tested.

**Implementation:** The `_compose()` method checks that every child has a status other than `"pending"` before composing. If a child has no execution result, composition is blocked.

**What this prevents:** Silent failures where a sub-problem was generated but never verified, then composed into the final answer. If the tree fractal's left subtree code was generated but never run, the composed renderer will fail — and the system catches this at composition time rather than at delivery time.

### 3.4 Invariant: Equal Black-Height

**Meaning:** All branches from root to leaf have executed the same number of times.

**Implementation:** Before composition, the scheduler checks black-heights across sibling branches. If one branch has been executed 3 times (multiple retries) and another only once, the system flags this imbalance.

**What this prevents:** One branch being thoroughly tested while a sibling is barely tested, then both being composed into a final answer. The branch that was tested once might have a flaky pass, while the branch tested 3 times is genuinely solid. Equal black-height exposes this asymmetry.

**In practice:** This doesn't block composition — it's a diagnostic signal. The scheduler can log the imbalance, optionally re-run the under-tested branch, or adjust the acceptance threshold for under-tested branches.

### 3.5 Rotation as Re-Prioritization

In a standard red-black tree, rotation restructures the tree to maintain balance during insertion. In RBPT, rotation means something different: **when a branch is stuck (repeated failures), restructure the problem decomposition.**

```python
def _rotate(self, stuck_node: ProblemNode):
    """Re-prioritize a stuck branch.
    
    When a node has exhausted retries without passing, rotation means:
    1. Try a different decomposition of the same problem
    2. Promote a sibling that's passing to take priority
    3. Come back to the stuck branch with a fresh approach
    """
    if stuck_node.black_height >= self.max_retries:
        # This branch is stuck — try alternative decomposition
        alternative = self.model.decompose(stuck_node.task, hint="previous approach failed")
        
        if alternative.sub_problems:
            # Replace stuck children with alternative decomposition
            stuck_node.children = []
            for sp in alternative.sub_problems:
                child = ProblemNode(
                    id=f"{stuck_node.id}.{len(stuck_node.children)}",
                    task=sp.task,
                    acceptance_criteria=sp.criteria,
                    parent=stuck_node,
                )
                stuck_node.children.append(child)
```

This isn't tree rotation in the CLRS sense — it's the *spirit* of rotation applied to problem solving. When the tree becomes unbalanced (one branch is much deeper than others due to retries), restructure to restore balance.

---

## 4. Why This Should Work

### 4.1 Decomposition Reduces Per-Node Complexity

The tree fractal that fails at <2% requires a single model to produce:

1. Branching logic (function that calls itself twice)
2. Transformation parameters (angle, scale)
3. Recursion depth handling
4. Rendering code
5. Correct integration of all the above

That's asking for ~200 lines of correct Python in one shot. No wonder it fails.

With RBPT, the system decomposes this into:
- Node A: "Write a recursive function that calls itself for left and right branches" (simple)
- Node B: "Given this branching function, render the left subtree" (simple)
- Node C: "Given this branching function, render the right subtree" (simple)
- Node D: "Compose A, B, C into a complete tree fractal renderer" (simple)

Each node is a focused task. Each gets its own generate-execute cycle. Each can fail and retry independently. The model that can't write a 200-line program in one shot might write four 20-line programs across four focused attempts.

### 4.2 Execution Feedback Is Objective

Each node receives execution results — not model self-assessment. If the Koch curve renders with IoU 0.34, that's a number. The model sees the number and knows it needs to do better. There's no ambiguity, no overconfidence, no "I think this is probably right."

The system also feeds back the actual rendered image (or error output) to the model on retry. The model can see *what it produced* vs. *what was expected*. This visual feedback loop is what human programmers use when debugging rendering code.

### 4.3 Branching Is System-Managed, Not Model-Managed

The hardest part of tree fractals isn't the math — it's the branching. The model has to hold two independent recursive calls in its working memory and get both right simultaneously. In RBPT, the system handles branching. The model works on one branch at a time.

This maps to how human teams work: you don't ask one engineer to implement the left subtree and right subtree simultaneously. You assign them to two engineers (or two sessions) and compose the results.

### 4.4 Self-Similarity at Every Scale

The generate-execute-verify cycle is the same at every node:

- Root: "Build a fractal renderer" → decompose → execute sub-problems → compose → verify
- Child: "Implement Koch curve" → generate code → execute → verify → iterate
- Leaf: "Fix the rotation angle" → generate fix → execute → verify → done

This is a fractal. The same transformation (generate → execute → verify) applies at every level. The tree structure emerges naturally from recursive decomposition. The red-black invariants maintain balance as the tree grows.

### 4.5 Composition Catches Integration Errors

Even if all sub-problems pass individually, the composed solution might fail. RBPT handles this: the compose step is itself a generate-execute cycle. If the composed renderer fails, the system knows the integration is broken and can iterate on the composition (not the individual sub-problems).

This catches a failure mode that single-shot evaluation misses: code that's correct in isolation but broken when combined. The tree fractal's left and right subtree code might both pass individually, but the composed renderer might have parameter mismatches. The compose step catches this.

---

## 5. What This Is Not

### 5.1 Not Tree-of-Thought

Tree-of-Thought (Yao et al., 2023) explores multiple reasoning *paths* within a single model's context. The model generates multiple candidates, evaluates them, and picks the best. It's a search strategy within one generation.

RBPT decomposes a problem into independent *sub-problems*, each solved by a separate generate-execute cycle. The model doesn't explore paths — the system decomposes tasks. The model is called per-node with narrow, focused context.

### 5.2 Not Plan-and-Execute

Plan-and-Execute (Wang et al., 2023) generates a full plan upfront, then executes each step. The plan is a linear sequence: step 1, step 2, step 3.

RBPT generates a tree, not a list. Sub-problems can branch. Branches can be solved in parallel. Failed branches are retried independently. The plan is adaptive — decomposition can be re-attempted if the initial decomposition fails.

### 5.3 Not Recursive Prompting

Recursive prompting (Xu et al., 2024) feeds a model's output back as input in a loop. It's a single thread of generate → feed back → generate → feed back.

RBPT has branching. Multiple sub-problems run independently. The compose step merges results from parallel branches. It's not a loop — it's a tree.

### 5.4 Not Context Engineering

Context engineering (Korthikanti et al., 2022; various prompting strategies) focuses on what information the model sees. RBPT focuses on how problems are decomposed and how results are composed. The model's context at each node is minimal — just the task, acceptance criteria, and previous attempts. The tree exists in the runtime, not in the context.

---

## 6. Proposed Experiments

### 6.1 Experiment 1: Single-Shot vs. Tree Decomposition on FractalBench

**Hypothesis:** Tree decomposition will improve FractalBench visual correctness from 4.2% (single-shot) to >20%.

**Setup:**
- Same 12 fractals, same models
- Group A: Single-shot (current FractalBench methodology — one generation, one execution)
- Group B: RBPT with automatic decomposition (system decides when to branch)
- Group C: RBPT with manual decomposition (human-specified sub-problems for each fractal)

**Why Group C:** If Group B doesn't improve but Group C does, the decomposition logic is the bottleneck, not the generate-execute loop. If both improve, the loop is doing real work.

**Metrics:** Visual correctness (IoU > 0.95), iterations to pass, total tokens consumed, time to solution.

### 6.2 Experiment 2: Tree Fractals with Forced Decomposition

**Hypothesis:** Pre-specifying the decomposition for tree fractals (branching logic, left subtree, right subtree, compose) will improve accuracy from <2% to >15%.

**Setup:**
- 4 tree fractal types only
- System always decomposes into 3 sub-problems (branch, left, right)
- Each sub-problem gets up to 5 generate-execute cycles
- Compare to single-shot baseline

**Why this matters:** If the model can solve each sub-problem individually but can't solve the whole, that proves the bottleneck is problem complexity, not model capability.

### 6.3 Experiment 3: Scaling with Tree Depth

**Hypothesis:** RBPT improvement scales with problem complexity — deeper trees (more decomposition levels) help more on harder problems.

**Setup:**
- Problems at 3 complexity levels: single function, multi-function, multi-module
- Measure success rate at each level for: (a) single-shot, (b) 1-level decomposition, (c) 2-level decomposition
- Predict: single-shot degrades with complexity, RBPT degrades less or stays flat

### 6.4 Experiment 4: Decomposition Quality

**Hypothesis:** The model can decompose problems correctly at least 60% of the time, and incorrect decompositions are recoverable (the system retries with a different decomposition).

**Setup:**
- 50 diverse problems (not just fractals — also debugging, refactoring, API integration)
- Let the model decompose, execute, verify
- Measure: decomposition correctness (did the sub-problems cover the full problem?), recovery rate (when decomposition fails, does re-decomposition succeed?), end-to-end success rate

### 6.5 Experiment 5: Comparison to Existing Approaches

**Hypothesis:** RBPT outperforms chain-of-thought, ReAct, and plan-and-execute on recursive tasks, with comparable performance on non-recursive tasks.

**Setup:**
- Mix of recursive and non-recursive tasks
- 5 approaches: single-shot, CoT, ReAct, plan-and-execute, RBPT
- Measure: success rate, token efficiency (success per token), time to solution

---

## 7. Implementation with Existing Agent Frameworks

### 7.1 Nullclaw (Zig)

Nullclaw already has `delegate_task` for spawning sub-agents. The tree maps directly:

```zig
// Pseudocode for RBPT in nullclaw
fn solveNode(allocator: std.mem.Allocator, node: *ProblemNode) []const u8 {
    // Try decomposition
    const sub_problems = try decompose(allocator, node.task);
    
    if (sub_problems.len > 1) {
        // Branch: solve each sub-problem as a delegated task
        var results = std.ArrayList([]const u8).init(allocator);
        for (sub_problems) |sp| {
            const child = ProblemNode{ .task = sp, .parent = node };
            const result = delegate_task(child);  // Independent generate-execute cycle
            results.append(result);
        }
        // Compose
        return compose(allocator, node.task, results.items);
    } else {
        // Leaf: generate-execute loop
        return generateExecuteLoop(allocator, node);
    }
}

fn generateExecuteLoop(allocator: std.mem.Allocator, node: *ProblemNode) []const u8 {
    for (0..max_retries) |_| {
        const code = llm.generate(node.task);  // RED
        const result = executor.run(code);       // BLACK
        if (checkAcceptance(result, node.criteria)) {
            return code;
        }
        node.feedBack(result);  // BLACK result informs next RED generation
    }
    return node.bestAttempt();
}
```

### 7.2 Minimal Python Implementation

```python
class RBPTAgent:
    def solve(self, task: str, criteria: str, depth: int = 0) -> str:
        if depth > self.max_depth:
            return self._generate_execute_loop(task, criteria)
        
        # Ask model: should this be decomposed?
        decomposition = self.model.decompose(task)
        
        if decomposition.should_decompose and decomposition.sub_problems:
            results = {}
            for sp in decomposition.sub_problems:
                results[sp.name] = self.solve(sp.task, sp.criteria, depth + 1)
            
            # Compose
            composition = self.model.compose(task, results)
            result = self.executor.run(composition)
            
            if self._check(result, criteria):
                return composition
            else:
                # Composition failed — try again with execution feedback
                return self._generate_execute_loop(
                    f"{task}\n\nAttempted composition failed.\n"
                    f"Sub-problem results: {results}\n"
                    f"Execution error: {result.stderr}",
                    criteria
                )
        else:
            return self._generate_execute_loop(task, criteria)
    
    def _generate_execute_loop(self, task: str, criteria: str) -> str:
        for attempt in range(self.max_retries):
            code = self.model.generate(task)
            result = self.executor.run(code)
            
            if self._check(result, criteria):
                return code
            
            # Feed execution result back for next attempt
            task = f"{task}\n\nPrevious attempt failed.\nCode:\n{code}\nResult: {result.stdout}\nError: {result.stderr}"
        
        return code  # Best effort
```

The entire agent is ~50 lines. The tree emerges from recursive decomposition. The red-black invariants are enforced by the loop structure (generate always followed by execute) and the composition check (all children must have results).

---

## 8. Limitations

### 8.1 Decomposition Quality Is the Bottleneck

The system relies on the model to decompose problems correctly. If the model decomposes a tree fractal into "draw the top half" and "draw the bottom half" instead of "branching logic, left subtree, right subtree," the sub-problems won't help. The decomposition model needs to understand problem structure — which is itself a hard problem.

**Mitigation:** For known problem types (fractals, recursive algorithms), decomposition templates can be provided. The system can also try multiple decompositions if the first one fails.

### 8.2 Composition Is Hard

Solving sub-problems independently doesn't guarantee the composed solution works. Integration bugs, parameter mismatches, and interface incompatibilities can cause composition failures. The compose step needs its own generate-execute cycle, which adds overhead.

**Mitigation:** The compose step receives all child results and their execution metadata. It can see what worked and what didn't. Failed composition triggers a new generation that accounts for integration issues.

### 8.3 Token Overhead

Each generate-execute cycle consumes tokens. A tree with 5 nodes, each with 3 attempts, means 15 model calls. The total token count might exceed a single-shot attempt. The question is whether the accuracy improvement justifies the token cost.

**Mitigation:** Early termination (don't retry obviously hopeless branches), result caching (reuse passing sub-problem results across attempts), and decomposition depth limits.

### 8.4 Not All Problems Are Decomposable

Some problems resist decomposition — they require holistic understanding that breaks when split into parts. Poetry, creative writing, and some mathematical proofs don't benefit from being broken into sub-problems.

**Mitigation:** The decomposition step can return "not decomposable" and fall back to single-node generate-execute. The system should be able to detect when decomposition is hurting and stop.

### 8.5 The Model Still Needs to Code

RBPT doesn't write code for the model. It makes each coding task smaller and more focused, but the model still needs to produce correct code at each node. If the model fundamentally can't write a recursive function, no amount of decomposition will fix that.

**Mitigation:** This is a feature, not a bug. If RBPT improves tree fractals from <2% to 15% but not to 90%, that tells us something important: 13% of the gap was architectural, 85% is model capability. That's a more precise diagnosis than the current "models lack recursive abstraction" claim.

---

## 9. Related Work

### 9.1 Agent Decomposition Strategies

| Approach | Decomposition | Execution | Composition | Branching |
|----------|--------------|-----------|-------------|-----------|
| Single-shot | None | One shot | N/A | None |
| Chain-of-thought | Linear steps | Implicit | Sequential | None |
| ReAct | Linear steps | Explicit | Sequential | None |
| Plan-and-execute | Linear plan | Explicit | Sequential | None |
| Tree-of-thought | Multiple paths | Implicit | Best-path selection | Parallel exploration |
| **RBPT (ours)** | **Tree of sub-problems** | **Explicit per node** | **Multi-branch merge** | **Recursive decomposition** |

### 9.2 Self-Similar Problem Solving

The idea that complex problems should be solved by decomposing them into self-similar sub-problems is not new. It's the basis of:
- Divide-and-conquer algorithms (merge sort, quicksort)
- Recursive programming (functions that call themselves)
- MapReduce (split, process, merge)
- Hierarchical task networks in AI planning (HTN, 1990s)

RBPT applies this principle to LLM agents with the addition of execution verification at every level. Traditional divide-and-conquer trusts the decomposition — RBPT verifies each piece before composing.

### 9.3 Software Engineering Practices

RBPT mirrors several well-established practices:
- **Test-driven development:** Write test (BLACK), write code (RED), run test (BLACK), iterate
- **Code review:** Submit code (RED), get review (BLACK), revise (RED), approve (BLACK)
- **CI/CD:** Commit (RED), build and test (BLACK), deploy (RED), monitor (BLACK)
- **Microservices:** Decompose monolith into services (tree), each independently deployable (per-node execution), composed via API (compose step)

The insight is that these practices exist because they work for humans. RBPT applies the same principles to LLM agents.

---

## 10. Conclusion

We propose Red-Black Problem Trees: an orchestration architecture that decomposes agent problem-solving into a self-balancing tree of independent generate-execute cycles. The model operates on narrow, focused sub-problems. The system handles decomposition, execution, verification, retry, and composition.

The key design principle: **the model doesn't see the tree.** The tree is the system's control flow, not the model's context. The model receives a task, produces output, and the system manages everything else. This separates the concern of *what to work on* (the system's job) from *how to solve it* (the model's job).

The red-black invariants — no consecutive generations, equal execution across branches, measured outcomes at every leaf — are scheduling constraints enforced by the runtime, not rules the model must follow. The model can't violate them because the system doesn't have code paths that allow violations.

The fractal connection is structural, not metaphorical. Each subtree is an independent problem-solving cycle with the same generate-execute-verify pattern. The same transformation applies at every depth. Complex problems are solved by recursively decomposing into self-similar sub-problems, each verified independently, then composed. This is divide-and-conquer with execution verification at every level.

The hypothesis is concrete and falsifiable: if tree decomposition with per-node execution feedback doesn't improve recursive task performance, the gap is model capability, not architecture. The experiment takes ~2 weeks to run. The implementation is ~50 lines of Python. There is no reason not to test it.

---

## References

- Ondras, J., & Šuppa, M. (2025). FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis. *arXiv:2511.06522v1*.
- Cormen, T. H., et al. (2009). Introduction to Algorithms (3rd ed.). MIT Press. Chapter 13: Red-Black Trees.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS*.
- Wang, L., et al. (2023). Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. *ACL*.
- Xu, Y., et al. (2024). Recursive Prompting: A Technique for Improving Multi-Step Reasoning in Large Language Models.
- Beck, K. (2002). Test-Driven Development: By Example. Addison-Wesley.

---

*Status: Hypothesis with concrete implementation path. ~50 line Python sketch. No benchmarks yet.*

*Last updated: 2026-04-24*
