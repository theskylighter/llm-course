<<SVG:prompting_cot_sensitivity>>

*Figure: Prompts add task context without weight updates; CoT and voting can help, while paraphrases expose sensitivity.*

**Prompting replaces fine-tuning for large models, chain-of-thought prompts make the model write its steps, and sensitivity measures how much a model's answer changes when the prompt is rephrased.** Report sensitivity next to accuracy.

## The Problem: Fine-Tuning Needs Data and Compute

> [!lecture] A new paradigm
> Until 2019: **pre-train, fine-tune, predict**. The GPT-3 paper (2020) said the model is so big you need not fine-tune: just write a prompt, which is plain English instruction. They called this an *emerging property*, and they were surprised. Accuracy may be about 70-80% of a fine-tuned model, which is huge when you have no training data. Now: **pre-train, prompt, predict**.

A prompt = task description + delimiter + (optional) examples + the query. **Zero-shot** gives no examples, **one-shot** one, **few-shot** several. With 1,000 examples, *fine-tune* instead.

> [!lecture] Terminology (2020-21 models)
> Input $x$, output $y$. A **prompt template** has slots, "$x$ Overall, it was a $z$ movie". Three steps: *prompt addition*, *answer prediction*, and *answer-label mapping*. A **prefix prompt** puts the blank at the end (GPT); a **cloze prompt** allows the blank anywhere (BERT).

> [!lecture] The professor's own caveat
> TriviaQA questions are not unfamiliar: the model was trained on gigantic web data, so those facts were likely seen. Prompting then locates and triggers the right parameters. What would be surprising is a truly out-of-data question answered correctly. (This is the data-contamination worry.)

**Hard prompts are brittle.** Same fact, three templates: "$X$ is located in $Y$" gives 31%, "...in which city, country or state? $Y$" gives 19%, "...in which country? $Y$" gives 51%. That pushed the field toward **soft prompts** (Chapter 7).

## Chain of Thought and Its Relatives

> [!lecture] Chain of thought
> Standard few-shot fails on arithmetic word problems: one example "5 balls + 2 cans of 3 = 11", then "23 apples, use 20, buy 6" and the model says **27** (it pattern-matches). Put the *reasoning steps* in the example ("23 - 20 = 3, 3 + 6 = 9") and the model produces steps and the right answer, **9**. Zero-shot version: just write "Let's think step by step".

| Method | Idea | Needs |
|---|---|---|
| CoT | write reasoning steps | one chain |
| Self-consistency | sample many chains, **vote** on the answer | many chains + vote |
| Tree of Thoughts | explore sub-steps like branch and bound | **branching, scoring, backtracking** modules |
| Graph of Thoughts | thoughts connected, reused and **aggregated** | adds an aggregator |

:::analogy
Showing your working in a maths exam: the final answer is more often right when you write the steps, and if three independent attempts agree, you trust the answer more.
:::

```python
from collections import Counter
chains = [9, 9, 24, 9, 12]           # final answers from five sampled reasoning chains
vote = Counter(chains).most_common(1)[0]
print("self-consistency answer:", vote[0], "with", vote[1], "of", len(chains), "votes")
```

```output
self-consistency answer: 9 with 3 of 5 votes
```

## Prompt Sensitivity

The same question asked as "how much are you familiar with Buddhism?" versus "how much do you understand Buddhism?" can get a serious answer once and a joke the next time. So report **sensitivity** next to accuracy. Model A: 85% accuracy, 60% sensitivity. Model B: 75% accuracy, 20% sensitivity. The lecture prefers **B**: accuracy comparable, far more robust.

> [!lecture] POSIX (Prompt Sensitivity Index, the professor's lab with Adobe)
> Take a prompt and its **intent-preserving variants**. A robust model gives the same answer to all. The metric must respond to four things: **response diversity**, **response distribution entropy**, **semantic coherence**, and **variance of confidence**. The assumption: for two paraphrases the probability of any fixed response should be the same.

> [!compiler] The formula, and what the lab found
> The slide equation is an image. A reconstruction of Sheth et al. (verify before citing): $\text{POSIX}=\frac{1}{N(N-1)}\sum_{i\neq j}\frac{1}{L_{y_j}}\left|\log\frac{P(y_j\mid x_i)}{P(y_j\mid x_j)}\right|$. Findings: **no clear evidence that sensitivity falls with model size**; an instruction-tuned **chat model is not guaranteed** less sensitive than its base; few-shot examples usually **reduce** sensitivity.

:::def
**Chain-of-thought prompting** includes intermediate reasoning steps in the prompt examples (or the instruction "let's think step by step") so the model generates steps before the final answer. **Self-consistency** samples several chains and takes the majority answer.
:::

:::key
- Zero-, one-, few-shot; no parameter updates in prompting.
- Hard prompts are brittle (31%, 19%, 51% for three templates); report sensitivity with accuracy.
- CoT, self-consistency (vote), ToT (branch, score, backtrack), GoT (reuse, aggregate).
:::

:::mistakes
- "Few-shot prompts are fine-tuning." No gradient step, so the prompt must be re-sent on every call.
- "Bigger model means less prompt-sensitive." Not established.
:::

:::exam
- *"Model A: 85% accuracy, 60% sensitivity; Model B: 75%, 20%. Which would you prefer?"* B, for robustness.
- *"Explain CoT and self-consistency."*
:::

## Quick Check

:::q Q1 · What does ToT need beyond CoT?
:::ans
Branching, scoring and backtracking modules (each can be an LM).
:::
:::

:::q Q2 · Why is accuracy alone insufficient for comparing prompted models?
:::ans
Two models can have similar accuracy but very different robustness to paraphrasing; sensitivity (for example POSIX) measures that.
:::
:::
