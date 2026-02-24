# Task Reference

This document describes the **29 unique tasks** implemented in this module, organized by computational complexity following the Chomsky hierarchy. Each task is a sequence-to-sequence or sequence-to-label problem designed to test whether neural networks can learn specific classes of computation.

For each task we list the reference implementations from the other research projects in this repository and note how our implementation differs.

---

## 1. Regular Tasks (Finite Automata)

Tasks solvable by finite-state automata — no external memory required. The model only needs to maintain a fixed-size internal state.

**1. Even Pairs**
Given a sequence of symbols, determine whether the entire sequence consists of identical adjacent pairs. For example, `a a b b a a` is "yes" because every consecutive pair of symbols matches, while `a b a a` is "no". The sequence length must be even for a "yes" label; odd-length sequences are always "no". This is a classification task outputting a single binary label.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/regular/even_pairs.py` (DeepMind, ICLR 2023)
>
> **Difference:** The reference task counts *unequal* adjacent transitions (01 and 10 pairs in a binary string) and checks whether that count is even. Our task checks whether the sequence consists entirely of *identical* adjacent pairs. These are different predicates — the reference is a parity-of-transitions check, while ours is a structural pattern-matching check. We also support arbitrary `vocab_size` (not just binary) and use 1-indexed integer encoding instead of one-hot vectors.

**2. Parity Check**
Count how many times a specific target symbol appears in the input sequence and output whether that count is even or odd. For instance, given a sequence of `a` and `b` tokens with the target symbol `a`, the model must track the running parity of `a` occurrences — a task that maps directly to a two-state finite automaton.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/regular/parity_check.py`
>
> **Difference:** Functionally equivalent. The reference is hardcoded to binary strings counting symbol `1`; ours generalizes to arbitrary `vocab_size` and a configurable target `symbol`. We use 1-indexed integers; the reference uses one-hot encoding.

**3. Cycle Navigation**
Navigate a circular ring of states using a sequence of forward (`->`) and backward (`<-`) instructions. Starting at state 0 on a cycle of `num_states` positions, each instruction moves the pointer one step clockwise or counter-clockwise. The output is the final state index after processing all instructions. This tests whether a model can learn modular counting.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/regular/cycle_navigation.py`
>
> **Difference:** Functionally equivalent. The reference uses three actions {0=stay, 1=right, 2=left} mapped to {-1, 0, +1}; ours uses two actions {1=forward, 2=backward} with no "stay" option. Both use a cycle of 5 states. We output a scalar state index; the reference outputs one-hot.

**4. Modular Arithmetic**
Evaluate flat arithmetic expressions (no parentheses) under a modulus. The input is an alternating sequence of operands and operators (`+`, `-`, `*`) evaluated strictly left-to-right, and the output is the final result modulo `n`. For example, `(3 + 2 * 4) mod 5` evaluates left-to-right as `((3 + 2) * 4) mod 5 = 0`. Because the state space is bounded by the modulus, a finite automaton suffices.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/regular/modular_arithmetic.py`
>
> **Difference:** Different evaluation semantics. The reference implements proper operator precedence (`*` binds tighter than `+`/`-`), first performing all multiplications then summing additive terms. Our implementation evaluates strictly left-to-right with no precedence. For input `1 + 2 * 3 mod 5`: reference gives `1 + (2*3) = 7 mod 5 = 2`, ours gives `(1+2) * 3 = 9 mod 5 = 4`. Left-to-right evaluation keeps the task regular (finite-state trackable), while precedence-based evaluation is also regular but with a different automaton structure.

---

## 2. Context-Free Tasks (Stack Memory)

Tasks that require stack-like memory to solve. A finite automaton cannot generalize on these — the model must learn to push and pop information.

**5. Stack Manipulation**
Execute a sequence of explicit `push(x)` and `pop` instructions on a stack, and output the top-of-stack value after each step. If the stack is empty, output 0. This directly tests whether a model can learn LIFO (last-in, first-out) data structure semantics, since correct output at each timestep depends on the full history of pushes and pops.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/dcf/stack_manipulation.py`
>
> **Difference:** Different output structure. The reference outputs the final stack contents (top-to-bottom) with a termination token as a single sequence, and is limited to binary push values (0/1). Ours outputs the top-of-stack at *every* timestep as a parallel sequence `(batch, length)`, and supports arbitrary `vocab_size` push values. Our formulation is closer to an online monitoring task; the reference is a batch transformation.

**6. Reverse String**
Given an input sequence of tokens, produce the same tokens in reverse order. Reversing requires the model to buffer the entire input before emitting any output — a canonical stack operation. The input `a b c d` should produce `d c b a`.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/dcf/reverse_string.py`; also related to `nondeterministic-stack-rnn` marked/unmarked reversal grammars
>
> **Difference:** Functionally equivalent. The reference uses binary strings with one-hot encoding; ours supports arbitrary `vocab_size` with 1-indexed integers. The nondeterministic-stack-rnn variants add a center marker (`w # reverse(w)`) or no marker (`w reverse(w)`) as separate language-modeling tasks — these are language recognition tasks rather than seq2seq tasks.

**7. Nested Modular Arithmetic**
Like Modular Arithmetic (task 4), but expressions now include nested parentheses up to a configurable depth. For example, `(3 + (2 * (4 - 1))) mod 5`. The model must evaluate inner sub-expressions first and carry intermediate results back to outer expressions, which requires a stack to track the nesting structure and pending computations.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/dcf/modular_arithmetic_brackets.py`
>
> **Difference:** Functionally equivalent. Both generate nested expressions via recursive CFG-like generation. The reference uses string-based generation with a character-to-index mapping; ours directly generates integer token sequences. Our generation uses a fixed `max_depth` parameter; the reference uses probabilistic termination. Both support configurable modulus and `+`, `-`, `*` operators.

**8. Solve Equation**
Find the value of `x` in a simple linear equation of the form `a * x + b = c`, where all values are bounded integers. The model receives the equation as a token sequence and must output the integer solution. This requires the model to implicitly invert the linear relationship — a form of symbolic reasoning over a constrained algebraic structure.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/dcf/solve_equation.py`
>
> **Difference:** Our implementation is a simplified variant. The reference generates arbitrary modular arithmetic expressions with brackets containing an unknown `x`, then solves by evaluating all candidate values — supporting equations like `(x + 3) - (2 * x) = 1`. Ours is restricted to the fixed template `a * x + b = c` with no nesting. The reference is more general but our format is more controlled for curriculum learning.

**9. Dyck-n Language**
Determine whether a sequence of brackets is properly balanced, using `n` distinct bracket types (e.g., `()`, `[]`, `{}`, `<>`). A valid Dyck word requires every opening bracket to be matched with the correct closing bracket in proper nesting order. For example, `([])` is balanced but `([)]` is not. This is the prototypical context-free language — recognizing it requires a stack to track which bracket types are open.

> **Reference:** `nondeterministic-stack-rnn/src/cfl_language_modeling/grammars/dyck.py` (also used by `marnns`)
>
> **Difference:** Different formulation. The reference generates Dyck words from a probabilistic CFG for *language modeling* (predict the next token) — all generated sequences are valid by construction. Our task is *classification*: we generate both balanced (50%) and random (50%) sequences and output a binary label (balanced/unbalanced). The reference encodes open/close as `2i`/`2i+1`; ours uses `2i+1`/`2i+2`. Different use case: the reference tests next-token prediction on valid strings; ours tests recognition of the language.

---

## 3. Context-Sensitive Tasks (Tape / Matrix Memory)

Tasks that exceed the power of a pushdown automaton. They require tape-like or random-access memory — the ability to read, write, and compare across multiple positions in memory simultaneously.

**10. Duplicate String**
Given an input string `w`, produce the output `ww` — a simple concatenation of the string with itself. Despite its simplicity, this task is context-sensitive because the model must copy the input verbatim while also remembering where the first copy ends and the second begins, which requires more than a single stack.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/cs/duplicate_string.py`; also `nondeterministic-stack-rnn` unmarked copy task
>
> **Difference:** Functionally equivalent. The reference uses binary strings with one-hot encoding; ours supports arbitrary `vocab_size` with 1-indexed integers. The nondeterministic-stack-rnn "unmarked copy" is the same `w → ww` task used for language modeling.

**11. Repeat Copy N**
Reproduce an input pattern `N` times, where `N` is provided as the first token of the input. For example, given `x3 a b c`, produce `a b c a b c a b c`. This generalizes Duplicate String to a variable repeat count, testing whether the model can learn a counting loop that controls how many times to replay the memorized pattern.

> **Reference:** `dnc/dnc/repeat_copy.py` (DeepMind DNC); `pytorch-ntm/tasks/repeatcopytask.py`
>
> **Difference:** Same core task, different input encoding. The DNC encodes the repeat count in a dedicated channel dimension with normalization (`count / norm_max`), using separate observation/target/mask tensors. The NTM normalizes count as `(n - mean) / std` with time-major layout `(T, B, W)`. Ours embeds `N` directly as the first integer token in a flat sequence — simpler and more uniform with the rest of our task format, but without explicit start/end markers.

**12. Deduplicate Inputs**
Given a stream where each symbol is repeated `k` times consecutively (e.g., `a a a b b b c c c` with `k=3`), extract the unique sequence (`a b c`). No two adjacent unique symbols are the same, so group boundaries are unambiguous. The model must learn to skip redundant repetitions — a form of streaming compression that requires tracking position within each group.

> **Reference:** None.
>
> **Novel task.** No equivalent found in any reference repository.

**13. Associative Recall**
Given a list of key-value pairs followed by a query key, retrieve the value associated with that key. For example, given `{a:x, b:y, c:z} ? b`, output `y`. This tests content-addressable memory: the model must store all pairs, then perform a lookup by matching the query key against stored keys — the core operation of an associative memory or hash table.

> **Reference:** None.
>
> **Novel task.** Associative recall is a well-known benchmark for memory-augmented networks (discussed in the DNC and NTM papers), but no explicit data generator for it exists in the reference repositories. Our implementation provides a self-contained generator with configurable `num_pairs` and `vocab_size`.

**14. Missing Duplicate**
Given a sequence where every element appears exactly twice except for one element that appears only once, identify the singleton. The input is shuffled, so the model cannot rely on positional patterns. This requires the model to maintain counts or use XOR-like cancellation across the entire input to isolate the unpaired element.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/cs/missing_duplicate_string.py`
>
> **Difference:** Different formulation. The reference presents a fully duplicated string with one position replaced by a special mask token (value `2`), and asks the model to predict the original value at that position — it's a fill-in-the-blank task on a known structure. Ours removes one copy of the missing element entirely and shuffles the result, so the model must identify *which* element is unpaired across an unordered bag. Our version is harder because the model cannot rely on positional alignment. We also support arbitrary `vocab_size` vs the reference's binary alphabet.

**15. Odds First**
Reorder a sequence of integers so that all odd-valued elements appear first (preserving their original relative order), followed by all even-valued elements (also in original order). For example, `[4, 1, 3, 2, 5]` becomes `[1, 3, 5, 4, 2]`. This is a stable partition — the model must route elements to two groups based on a predicate while preserving within-group ordering.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/cs/odds_first.py`
>
> **Difference:** Different task definition. The reference reorders by *index parity* (elements at odd indices first: `s1 s3 s5 ... s2 s4 s6 ...`), which is a fixed permutation independent of element values. Our task partitions by *value parity* (odd-valued elements first), which is data-dependent — the permutation changes based on the actual values in the sequence. Our version requires the model to inspect each element's value, not just its position.

**16. Binary Addition** *(variants: 8-bit, 16-bit, 32-bit, 64-bit)*
Add two binary numbers represented in LSB-first (least significant bit first) format. The two operands are interleaved in the input as `[a0, b0, a1, b1, ...]`, and the output is the sum bits (LSB-first) including a possible carry bit. This requires the model to learn the ripple-carry addition algorithm, propagating carry information across all bit positions.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/cs/binary_addition.py`; also `Neural-Execution-Engines/data/addition_data.py`; `neural-gpu` (badd); `Stack-RNN` (task 7)
>
> **Difference:** Different input encoding. The reference concatenates the two numbers with a separator token: `[a_bits, SEP, b_bits]`, both little-endian. Ours interleaves bit-by-bit: `[a0, b0, a1, b1, ...]` with 1-indexed tokens (1=zero-bit, 2=one-bit). The interleaved format aligns corresponding bit positions, which may simplify the carry-propagation pattern for the model. The Neural-Execution-Engines version works in decimal with integer pairs; the neural-gpu version supports multiple bases (binary, quaternary, decimal). Our fixed-length variant approach (8/16/32/64-bit) is unique — the references use variable-length inputs.

**17. Binary Multiplication** *(variants: 8-bit, 16-bit, 32-bit)*
Multiply two binary numbers in LSB-first format with the same interleaved input encoding as Binary Addition. The output is the product bits in LSB-first format. Binary multiplication is substantially harder than addition because it requires shift-and-add operations that produce an output twice the length of each operand, with carries that depend on partial products from all input positions.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/cs/binary_multiplication.py`; `Neural-Execution-Engines/data/mul_data.py`; `neural-gpu` (bmul)
>
> **Difference:** Same encoding differences as Binary Addition. Our interleaved input and fixed bit-width variants differ from the reference's separator-based variable-length format. Output length: ours is fixed at `2 * length`; the reference pads to `input_length` with variable actual product length.

**18. Square Root**
Compute the integer (floor) square root of a number given as LSD-first (least significant digit first) decimal digits. For example, given the digits of 144, output 12. The model must reconstruct the number, compute the square root, and re-encode the result as digits — a multi-step numerical computation.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/cs/compute_sqrt.py`
>
> **Difference:** Different number representation. The reference uses *binary* big-endian encoding (MSB-first); ours uses *decimal* little-endian (LSD-first) with 1-indexed digit tokens. These are incompatible representations — same mathematical operation, entirely different sequence patterns for the model to learn.

**19. Count-N (a^k b^k c^k)**
Validate that a sequence consists of `n` symbol types each appearing exactly `k` times in consecutive blocks: `s1^k s2^k ... sn^k`. For example, with `n=3`, the string `aaabbbccc` is valid but `aabbcc` with `n=3` would only be valid if each group has equal length. This is a classic context-sensitive language — recognizing it requires comparing counts across multiple symbol groups simultaneously.

> **Reference:** `nondeterministic-stack-rnn/src/cfl_language_modeling/tasks/count_3.py`; `Stack-RNN` (task 1: `a^n b^n` and extensions); `nn-automata` (anbn)
>
> **Difference:** Our implementation generalizes to arbitrary `n` (configurable number of symbol types); the nondeterministic-stack-rnn reference is hardcoded to `n=3`. The Stack-RNN reference supports `a^n b^n`, `a^n b^n c^n`, and `a^n b^n c^n d^n` as separate tasks. The nn-automata `anbn` is `n=2` only. Our task is a classification task (valid/invalid label); the references are language-modeling or generation tasks.

**20. N-back**
Given a sequence of symbols followed by a separator `#` and a look-back distance `n`, recall the single symbol that appeared `n` positions before the last element of the sequence. For example, given `c a b d c d e c d e c a # 4`, the model must output `c` (the symbol at position 7, which is 4 steps back from the final symbol at position 11). When `n` is not fixed, it is sampled randomly per example, so the model must learn variable-distance temporal recall. This is inspired by the classic n-back working memory paradigm from cognitive psychology and tests whether the model can buffer the full sequence and perform indexed retrieval — a capability that challenges RNNs but should be straightforward for memory-augmented networks.

> **Reference:** None.
>
> **Novel task.** The n-back task is a well-established cognitive psychology paradigm for testing working memory capacity, but no sequence-to-scalar implementation exists in the reference repositories. Our formulation encodes the query distance as part of the input (`[seq..., SEP, n]`) and requires a single symbol output, making it a clean test of temporal memory indexing. With variable `n`, the model cannot simply memorize a fixed offset — it must learn to use the `n` token to dynamically index into its memory of the sequence.

---

## 4. Data Processing & Code Logic

Applied tasks involving sorting, program execution, and planning in structured environments.

**21. Sort**
Sort a sequence of integers in ascending order. The input is an unordered list of values in `[1, max_value]` and the output is the same values sorted. While conceptually simple, learning to sort from examples alone requires the model to discover comparison-based ordering — an algorithmic capability that generalizes across sequence lengths.

> **Reference:** `neural_networks_chomsky_hierarchy/tasks/cs/bucket_sort.py`; `Neural-Execution-Engines/data/sel_sort_data.py` and `merge_data.py`; `neural-gpu` (sort)
>
> **Difference:** Our implementation is a straightforward input→sorted-output task using `np.sort`. The reference implementations are more specialized: the DeepMind version uses a small fixed alphabet (bucket sort, `vocab_size=5`); the Neural-Execution-Engines versions generate intermediate execution states for selection sort and merge sort (not just input/output pairs, but step-by-step traces). Our version uses arbitrary integer values up to `max_value=99`, treating sorting as a black-box transformation.

**22. Python Execution**
Predict the output of a simple Python-like program. Each program follows the template: `x = a; for _ in range(n): x = x op b; print(x)`, where `op` is one of `+`, `-`, `*`. The model must simulate the loop execution step by step, tracking the accumulator value through multiple iterations — a test of sequential program trace prediction.

> **Reference:** `learning_to_execute/data.py` and `learning_to_execute/utils/operations.py`
>
> **Difference:** The reference generates far more complex programs: multiple nested operations including variable assignments, if-statements with comparisons, chained expressions with multiple variables, and configurable nesting depth. It composes operations randomly from five building blocks (pair operations, small multiplication, variable operations, loops, conditionals). Ours is a simplified single-loop variant with a fixed template — no conditionals, no multiple variables, no nesting. Our version isolates the loop-execution capability without the complexity of full program synthesis.

**23. Mini-SHRDLU**
A blocks-world planning task. Given an initial grid configuration of numbered blocks stacked in columns under gravity, and a set of spatial constraints (e.g., "block 2 above block 4", "block 1 left-of block 3"), produce the target board configuration that satisfies all constraints. The target is guaranteed to be reachable from the initial state via a sequence of valid moves (pick the top block from one column, place it atop another). This tests spatial reasoning and implicit planning.

> **Reference:** None directly. Inspired by the classic SHRDLU blocks-world domain. The `lie-access-memory` project in this repository uses copy/sequence tasks loaded from HDF5 files, not blocks-world planning.
>
> **Novel task.** Our Mini-SHRDLU implementation is original: it generates the initial board randomly, finds a target configuration via BFS at least `min_moves` away, extracts spatial constraints from the target that are novel relative to the initial board, and encodes both board state and constraints as integer token sequences. No other project in this repository implements a blocks-world planning task.

---

## 5. Graphs & Geometry

Tasks on discrete graphs and continuous 2D point sets, requiring classical algorithm execution. All six tasks in this category are novel — no equivalent generators exist in the reference repositories.

**24. Shortest Path**
Given a weighted undirected graph (encoded as edge triples) and a source-target pair, output the sequence of nodes along the shortest path. The ground truth is computed using Dijkstra's algorithm. The model must learn to perform implicit graph search — propagating distance estimates through the graph structure to reconstruct the optimal path.

> **Reference:** None. The Neural-Execution-Engines project references shortest path in its experiment scripts but does not include a data generator.
>
> **Novel generator.** Graphs are generated as random connected graphs (spanning tree + random edges). Input is flat edge triples `[u, v, w, ...]` with the query pair appended. Output is the path as a node-ID sequence, computed via `scipy.sparse.csgraph.dijkstra`.

**25. Minimum Spanning Tree (Prim)**
Given a weighted undirected graph, output the edges of its minimum spanning tree. The ground truth uses Prim's/Kruskal's algorithm via SciPy. The model must learn to greedily select the lowest-weight edge that connects a new node to the growing tree without forming cycles — a fundamental greedy algorithm.

> **Reference:** None.
>
> **Novel task.** Uses `scipy.sparse.csgraph.minimum_spanning_tree`. Input and output are both edge-triple encoded.

**26. Graph Traversal (BFS)**
Given an unweighted undirected graph and a source node, output the BFS (breadth-first search) visit rank of each node. Node 0 has rank 1, its neighbors get rank 2, and so on. The model must learn level-by-level expansion — systematically exploring all nodes at distance `d` before moving to distance `d+1`.

> **Reference:** None.
>
> **Novel task.** Uses `scipy.sparse.csgraph.breadth_first_order`. Output is a rank vector (one rank per node), not a node sequence.

**27. Travelling Salesman (TSP)**
Given a set of 2D cities with integer coordinates, output a tour (ordered list of city IDs) produced by the nearest-neighbor heuristic: starting from city 1, always visit the closest unvisited city. While this doesn't find the optimal tour, it is a well-defined greedy algorithm the model must learn to simulate — selecting the minimum-distance unvisited neighbor at each step.

> **Reference:** None.
>
> **Novel task.** Points encoded as `[id, x, y]` triples. Ground truth uses nearest-neighbor heuristic (not optimal TSP), making it a deterministic greedy algorithm suitable for supervised learning.

**28. Convex Hull**
Given a set of 2D points, output a binary mask indicating which points lie on the convex hull — the smallest convex polygon enclosing all points. The ground truth is computed using SciPy's `ConvexHull` (Quickhull algorithm). The model must learn geometric reasoning: determining whether each point is an extreme point that cannot be expressed as a convex combination of others.

> **Reference:** None.
>
> **Novel task.** Output is a binary mask `[0, 1, 0, 1, ...]` over input points, not an ordered hull boundary.

**29. Delaunay Triangulation**
Given a set of 2D points, output the Delaunay triangulation as a list of triangle vertex triples. In a Delaunay triangulation, no point lies inside the circumscribed circle of any triangle — this maximizes the minimum angle across all triangles. The ground truth uses SciPy's `Delaunay`. The model must learn to partition the plane into triangles satisfying this geometric optimality criterion.

> **Reference:** None.
>
> **Novel task.** Output is triangle vertex triples (1-indexed point IDs), padded with `-1`.

---

## Summary

Novel means that there was no reference implementation that we could use.

| # | Task | Reference Projects | Relationship |
|---|------|--------------------|-------------|
| 1 | Even Pairs | DeepMind Chomsky | **Different predicate** (identical pairs vs transition parity) |
| 2 | Parity Check | DeepMind Chomsky | Equivalent (generalized vocab) |
| 3 | Cycle Navigation | DeepMind Chomsky | Equivalent (no "stay" action, scalar output) |
| 4 | Modular Arithmetic | DeepMind Chomsky | **Different semantics** (left-to-right vs operator precedence) |
| 5 | Stack Manipulation | DeepMind Chomsky | **Different output** (per-step top vs final stack dump) |
| 6 | Reverse String | DeepMind Chomsky, NSRNN | Equivalent (generalized vocab) |
| 7 | Nested Modular Arith | DeepMind Chomsky | Equivalent (different generation strategy) |
| 8 | Solve Equation | DeepMind Chomsky | Simplified variant (fixed template vs arbitrary brackets) |
| 9 | Dyck-n | NSRNN, MARNNS | **Different formulation** (classification vs language modeling) |
| 10 | Duplicate String | DeepMind Chomsky, NSRNN | Equivalent (generalized vocab) |
| 11 | Repeat Copy N | DNC, PyTorch-NTM | Equivalent (different count encoding) |
| 12 | Deduplicate Inputs | — | **Novel** |
| 13 | Associative Recall | — | **Novel** |
| 14 | Missing Duplicate | DeepMind Chomsky | **Different formulation** (find singleton vs fill mask) |
| 15 | Odds First | DeepMind Chomsky | **Different predicate** (value parity vs index parity) |
| 16 | Binary Addition | DeepMind Chomsky, NEE, Neural-GPU, Stack-RNN | Equivalent (interleaved encoding, fixed bit-widths) |
| 17 | Binary Multiplication | DeepMind Chomsky, NEE, Neural-GPU | Equivalent (interleaved encoding, fixed bit-widths) |
| 18 | Square Root | DeepMind Chomsky | **Different representation** (decimal LSD-first vs binary MSB-first) |
| 19 | Count-N | NSRNN, Stack-RNN, nn-automata | Equivalent (generalized to arbitrary n) |
| 20 | N-back | — | **Novel** |
| 21 | Sort | DeepMind Chomsky, NEE, Neural-GPU | Equivalent (black-box, larger value range) |
| 22 | Python Execution | learning_to_execute | Simplified variant (single loop vs full program synthesis) |
| 23 | Mini-SHRDLU | — | **Novel** |
| 24 | Shortest Path | — | **Novel** |
| 25 | MST (Prim) | — | **Novel** |
| 26 | Graph Traversal | — | **Novel** |
| 27 | TSP | — | **Novel** |
| 28 | Convex Hull | — | **Novel** |
| 29 | Delaunay Triangulation | — | **Novel** |

**Abbreviations:** NSRNN = nondeterministic-stack-rnn, NEE = Neural-Execution-Engines.

Code is available for the following papers:

- [Neural Networks and the Chomsky Hierarchy](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy)
- [The Surprising Computational Power of Nondeterministic Stack RNNs](https://github.com/bdusell/nondeterministic-stack-rnn)
- [Hybrid computing using a neural network with dynamic external memory](https://github.com/google-deepmind/dnc)
- [Neural Turing Machines](https://github.com/loudinthecloud/pytorch-ntm)
- [Lie-Access Neural Turing Machines](https://github.com/harvardnlp/lie-access-memory)
- [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](https://github.com/facebookarchive/Stack-RNN)
- [Neural Execution Engines: Learning to Execute Subroutines](https://github.com/Yujun-Yan/Neural-Execution-Engines)
- [Memory-Augmented Recurrent Neural Networks Can Learn Generalized Dyck Languages](https://github.com/suzgunmirac/marnns)
- [Learning to Execute](https://github.com/wojciechz/learning_to_execute)
- [Neural GPU](https://github.com/openai/neural-gpu)
- [Formal Language Tasks](https://github.com/viking-sudo-rm/nn-automata)
- [Reinforcement Learning Neural Turing Machines](https://github.com/ilyasu123/rlntm)

### Encoding conventions (ours vs references)

All our tasks use **1-indexed integer tokens** in flat NumPy arrays. Reference implementations vary:

- **DeepMind Chomsky hierarchy:** 0-indexed tokens, one-hot encoded, JAX arrays, JIT-compiled generation
- **DNC / PyTorch-NTM:** Float tensors with auxiliary channel dimensions for markers and counts
- **Neural-Execution-Engines:** Integer pairs with combinatorial train/test splits and intermediate execution traces
- **Nondeterministic-Stack-RNN / MARNNS:** Probabilistic CFG samplers for language modeling
- **Neural-GPU:** Multi-base arithmetic with 2D grid layouts and various alignment schemes
- **Stack-RNN:** C++ generators with character-level I/O
- **learning_to_execute:** Full Python program strings as character sequences

### Novel contributions (9 tasks)

Tasks 12, 13, 20, 23–29 have no equivalent data generators in the reference repositories. The entire Graphs & Geometry category (6 tasks) is new, providing sequence-encoded versions of classical graph algorithms and computational geometry problems. Deduplicate Inputs, Associative Recall, and N-back fill gaps in the context-sensitive task suite. N-back is a classic cognitive psychology working memory paradigm. Mini-SHRDLU introduces spatial planning with constraint satisfaction.


