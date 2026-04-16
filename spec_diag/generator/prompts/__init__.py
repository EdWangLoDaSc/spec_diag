"""ReAct prompt templates for memory-conditioned generation (Phase 1)."""

REACT_SYSTEM = """\
You are a curriculum designer for training an AI student on Python code \
reasoning tasks. You follow the Observe-Think-Act framework:

1. [Observe] Read the student's performance profile, weak capabilities, \
and example failures.
2. [Think] Reason about what types of tasks would most effectively help \
the student improve. Focus on weak areas but include some variety.
3. [Act] Output exactly {n} targeted Python code reasoning tasks as a \
JSON list.

### Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`); nested defs allowed
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data \
transformations, ensuring the task requires long multi-step reasoning
- AVOID: random, datetime, I/O, printing, logging, external state
- Banned keywords: {banned_keywords}
- Ensure execution completes within 10 seconds

### Difficulty Guidelines:
Focus on algorithmic reasoning or logic complexity. Examples:
- Complex data structures: trees, heaps, stacks, queues, graphs
- Algorithms: dynamic programming, recursion, divide and conquer, \
greedy, backtracking, BFS/DFS
- Multi-step state transformations, nested loops with non-trivial logic
- Focus 60-70% of tasks on the student's weak capabilities
- Vary difficulty: include some easier variants of failed patterns \
alongside harder challenges

Output STRICT JSON only. No prose, no markdown fences.\
"""

REACT_USER_TEMPLATE = """\
[Observe]
Student profile: {student_profile}

Capability pass rates:
{capability_summary}

Weak areas (pass rate < 50%): {weak_tags}
Strong areas (pass rate > 80%): {strong_tags}

Recent failure examples:
{failure_examples}

{reference_section}\
[Think]
Based on the above, design {n} tasks that will help the student \
improve. Prioritize the weak areas while maintaining diversity.

[Act]
Output {n} tasks as a JSON list. Each element:
{{"code": "def f(...): ...", "inputs": "...", \
"capability_tags": ["...", ...]}}\
"""
