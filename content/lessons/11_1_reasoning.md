<<SVG:reasoning_methods_ladder>>

*Figure: Reasoning methods spend more tokens or search for accuracy; PAL hands exact arithmetic to an interpreter.*

**Reasoning prompts trade tokens for accuracy: write the steps, vote over several chains, search over a tree of thoughts, or let a program do the exact computation.** Part of the gain is pattern retrieval, so prefer answers you can verify.

> [!compiler] Source note
> The brief's pointers for this chapter are partly wrong (for example NPTEL lecture 27 is KG completion). Real sources: lecture 23 (reasoning ladder), lecture 24 (constitutional AI), Lec_31 and Lec_32 slides (mechanistic view and interpretability). Long context, editing and self-evolution come from reading lists, so they are Compiler's Notes.

## The Problem: "Can It Reason?"

First attempt: ask for the answer. It pattern-matches and fails (the lecture's apples: the model says 27, the answer is 9). Fix: make it write the steps.

> [!lecture] The ladder (lecture 23)
> A standard few-shot prompt gives a question and an answer; the model answers 27 to the apples question because it pattern-matches. **Chain of thought (CoT)** writes the steps in the example (23 minus 20 is 3, plus 6 is 9) and the model produces steps and the right answer. In **zero-shot** form, just write "let's think step by step". **Self-consistency:** sample several chains and take the majority answer. **Tree of Thoughts** explores sub-steps with branching, scoring and backtracking; **Graph of Thoughts** adds reuse and aggregation.

> [!compiler] What these have in common
> Each step trades **tokens for accuracy**: more tokens of thinking, more search. Self-consistency picks $\hat a=\arg\max_a\sum_{i=1}^{n}\mathbb{1}[a_i=a]$. If five chains give $9,9,24,9,12$, the answer is 9. The hierarchy: *chain $\subset$ chains-with-vote $\subset$ tree (backtracking) $\subset$ graph (reuse and merge)*.

## Programs as Reasoning (PAL)

Natural-language arithmetic fails in a new way: each step is plausible but the *sum* is wrong. **Program-aided language models** make the chain **executable code** and let an interpreter compute.

:::analogy
A student who writes the setup of a word problem and hands the arithmetic to a calculator, instead of doing long multiplication in their head.
:::

```python
# What a model would write for: "Roger has 5 balls and buys 2 cans of 3 balls. How many now?"
generated = """
tennis_balls = 5
bought = 2 * 3
answer = tennis_balls + bought
"""
env = {}
exec(generated, {"__builtins__": {}}, env)       # toy sandbox; never exec untrusted code in real systems
print("PAL answer:", env["answer"])
```

```output
PAL answer: 11
```

> [!compiler] Is it reasoning or reciting?
> The reading list collects **skeptical** papers: *The Reversal Curse* (a model trained on "A is B" fails on "B is A"), *Reasoning or Reciting?* (performance drops on counterfactual variants of familiar tasks), *Alice in Wonderland* (simple common-sense tasks break strong models), and papers arguing LLMs **cannot plan** on their own.

> [!slides] A mechanistic view of CoT (Lec_32, Dutta et al.)
> In Llama-2 7B, CoT is implemented by **multiple parallel pathways** that collect answers from different parts of the input (generated context, question, few-shot context). There is a **functional rift in the middle of the model (the 16th decoder block)**: first-half heads move information between tokens and align representations (token mixing); second-half heads do the answer aggregation.

:::def
**Chain-of-thought prompting** elicits intermediate reasoning steps before the final answer. **Program-aided language modeling (PAL)** replaces the natural-language chain with executable code run by an interpreter.
:::

:::key
- CoT, self-consistency (vote), ToT (branch, score, backtrack), GoT (reuse, aggregate), PAL (code).
- Each trades tokens for accuracy; part of the gain is reciting.
- Prefer verifiable reasoning (executable checks).
:::

:::mistakes
- "CoT means the model reasons." It improves accuracy; part is pattern retrieval.
:::

:::exam
- *"Name ToT's three modules and what GoT adds."* (Tier 1)
:::

## Quick Check

:::q Q1 · What does self-consistency return for chains giving 9, 9, 24, 9, 12?
:::ans
9, by majority vote.
:::
:::

:::q Q2 · Why is PAL more reliable than CoT for arithmetic?
:::ans
The model only translates the problem to code; an interpreter performs the exact computation.
:::
:::
