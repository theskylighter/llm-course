**Self-evolving LLMs generate data, judge it with a verifier and fine-tune on the keepers, repeating the loop; everything hangs on the verifier.** STaR, ReST and constitutional AI are instances.

## The Problem: Humans Are the Bottleneck

Human data is limited and expensive, but the model can **generate data, judge it, and learn from the keepers**. The survey (from memory) organizes it as a cycle: *experience acquisition*, *experience refinement* (filter, critique, correct), *updating* (fine-tune), *evaluation*, repeat.

```text
for round in range(R):
    data = []
    for x in prompts:
        for y in sample(model, x, n=k):
            if verifier(x, y): data.append((x, y))     # the only source of signal
    model = finetune(model, data)
```

Everything hangs on the **verifier**: with none, the loop amplifies the model's own mistakes.

:::analogy
A student who solves practice problems, checks them against the answer key, and rewrites only the correct solutions into their notebook to study from. Without an answer key they would memorise their own errors.
:::

## STaR, ReST, Self-Instruct

> [!compiler] From the papers
> **STaR** (Self-Taught Reasoner): for each problem, sample a *rationale* and answer; **keep rationales whose answer is correct**; fine-tune on them; repeat. For problems it fails, **rationalization**: give the correct answer as a hint and ask for a rationale that reaches it, keep that. It bootstraps reasoning from a handful of rationale examples plus answer labels. **ReST** alternates *Grow* (generate data with the current policy) and *Improve* (filter by reward and fine-tune). **Self-Instruct** (lesson 6.2) generates instructions with filtering.

```python
import random
rng = random.Random(0)

def sample_rationale(problem, skill):
    """A toy model: it answers correctly with probability `skill`."""
    ok = rng.random() < skill
    return ("steps...", problem["answer"] if ok else problem["answer"] + 1)

problems = [{"q": i, "answer": i * 2} for i in range(200)]
skill = 0.4
for rnd in range(3):
    kept = [p for p in problems if sample_rationale(p, skill)[1] == p["answer"]]
    print(f"round {rnd}: skill={skill:.2f}, kept {len(kept)} of {len(problems)} rationales")
    skill = min(0.95, skill + 0.4 * len(kept) / len(problems))   # fine-tuning on keepers raises skill
```

```output
round 0: skill=0.40, kept 73 of 200 rationales
round 1: skill=0.55, kept 109 of 200 rationales
round 2: skill=0.76, kept 150 of 200 rationales
```

> [!lecture] Constitutional AI as self-evolution (lecture 24)
> Anthropic's constitutional AI uses **16 principles**. The model produces two responses $Y_1,Y_2$ to a harmful request; it is asked which is **less harmful** (more wise, ethical, polite, friendly) and **its own preference is fed back**, in the hope that the model **keeps refining itself over the iterations.**

> [!compiler] What the lecture leaves out (from the paper)
> The paper has **two phases**. *Supervised:* the model **critiques and revises its own** answers against a randomly drawn principle, and is fine-tuned on the revisions. *RL from AI feedback (RLAIF):* the model **labels preference pairs** against the principles, a preference model is trained on them, and RL follows (lesson 6.5). It replaces most human harm labels.

:::def
**Self-evolution** is an iterative loop in which a model generates candidate outputs, a verifier or critic filters them, and the model is fine-tuned on the accepted outputs.
:::

:::key
- Generate, verify, fine-tune, repeat; the verifier is the only source of signal.
- STaR: filter by correct answer, rationalize failures. ReST: Grow and Improve. Constitutional AI: self-critique (supervised) then RLAIF.
:::

:::mistakes
- "STaR needs no labels." It needs **answer labels** to filter; no rationale labels.
:::

:::exam
- *"Pseudo-code for STaR with rationalization and a round counter."* (Tier 3)
:::

## Quick Check

:::q Q1 · What does STaR do for problems it cannot solve?
:::ans
Rationalization: it supplies the correct answer as a hint, asks for a rationale reaching it, and keeps that rationale.
:::
:::

:::q Q2 · What are the two phases of constitutional AI?
:::ans
Supervised self-critique and revision against principles, then RL from AI feedback (RLAIF) using AI-labelled preference pairs.
:::
:::
