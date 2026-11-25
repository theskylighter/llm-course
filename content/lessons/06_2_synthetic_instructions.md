<<SVG:synthetic_instruction_data>>

*Figure: FLAN broadens task phrasing; Self-Instruct generates candidates from seeds and filters near-duplicates before adding them to the pool.*

**Instruction data comes from reformatting benchmarks (FLAN), from many phrasings of one task, and from asking a model to write new tasks itself (Self-Instruct).** Quality matters more than quantity for style.

## The Problem: Human-Written Instructions Do Not Scale

> [!lecture] The problem
> Human-written instructions are tedious and do not scale beyond a point. Automate it with a pre-trained LM.

## FLAN: Template Diversity and Many Tasks

> [!lecture] Template diversity
> For one task there are many ways to phrase the instruction. For natural language inference you can write "based on the paragraph above, can we conclude that...", or "can we infer the following?", or "read the following and determine if the hypothesis can be inferred from the premise". Train on **all** these versions and the model generalizes better.

The scaling paper built **1,836 tasks**. From one dataset you can make several tasks: SQuAD is extractive QA, but you can build query generation (paragraph to question) or context generation (question to paragraph). Results: **more tasks and a bigger model both help**, and fine-tuning compute was only **0.2%, 0.6%, 1.6%** of pre-training compute for the three model sizes. Gains held across encoder-decoder (T5) and decoder-only (PaLM) models.

> [!slides] Human-crafted sources (12.2)
> **Flan-2021** turns NLP benchmarks into natural-language input-output pairs. **Super-NaturalInstructions** collects 1600+ tasks from NLP practitioners, creative modifications of existing tasks, and synthetic tasks.

## Self-Instruct

**Self-Instruct** starts from **175 seed tasks** (one instruction and one instance each) in a task pool. Repeat: (1) *generate instructions*: sample **8** tasks from the pool (6 human, 2 model-generated) as few-shot examples and ask GPT-3 for new ones; (2) *classify*: is it a classification task (finite output labels)?; (3) *generate instances*; (4) *filter* and add to the pool.

Why split classification from the rest? For classification the model favours popular labels, so the generated data is unbalanced. The fix is the **output-first** approach: give the label and ask for an input that matches it. Non-classification tasks use **input-first**.

| Filter (lecture) | Rule |
|---|---|
| Novelty | ROUGE-L similarity with every existing instruction must be $<0.7$ |
| Modality | drop instructions containing "image", "picture", "graph" |
| Instances | drop duplicates; drop same-input / different-output pairs |
| Heuristics | instruction neither too long nor too short |

:::analogy
A teacher with a few model exam questions asks a bright student to invent more, then throws away the ones that copy an existing question.
:::

```python
def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x == y else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]

def rouge_l(a, b):
    a, b = a.split(), b.split()
    l = lcs(a, b)
    if l == 0:
        return 0.0
    p, r = l / len(b), l / len(a)
    return 2 * p * r / (p + r)

pool = ["write a poem about the monsoon", "translate this sentence to hindi"]
for cand in ["write a poem about the monsoon rains", "list three uses of a hash table"]:
    sim = max(rouge_l(cand, p) for p in pool)
    print(f"{cand!r}: max ROUGE-L {sim:.2f} ->", "reject" if sim >= 0.7 else "keep")
```

```output
'write a poem about the monsoon rains': max ROUGE-L 0.92 -> reject
'list three uses of a hash table': max ROUGE-L 0.15 -> keep
```

> [!slides] Results and relatives (12.2)
> On unseen SuperNI tasks (ROUGE-L): GPT-3 **6.8**, GPT-3 + Self-Instruct **39.9**, InstructGPT-001 **40.8**. Relatives: **Evol-Instruct** (an LLM rewrites instructions: add constraints, deepen, increase reasoning, breadth), **Orca** (system instructions and chain-of-thought to enrich responses), **instruction back-translation**. Models: Flan-T5, Alpaca (LLaMA-7B on Self-Instruct data), WizardLM, Mistral-7B-OpenOrca.

## How Much Data Is Enough?

> [!slides] LIMA (12.2)
> *A model's knowledge and capabilities are learnt almost entirely during pre-training, while alignment teaches it which sub-distribution of formats to use when interacting with users.* A small number of examples should suffice.

> [!compiler] Do not confuse with the FLAN scaling result
> FLAN says *more tasks help* (zero-shot generalisation to new tasks); LIMA says *few high-quality examples suffice* for style and format. They do not contradict. The figure "about 1,000 examples" for LIMA is from the compiler's memory of the paper, not the slides.

:::def
**Self-Instruct** bootstraps instruction data from a small seed set by prompting a language model to generate new instructions and instances, filtering by novelty (ROUGE-L below 0.7) and quality, and adding survivors to the pool.
:::

:::key
- FLAN: many tasks, many templates; compute cost about 0.2% to 1.6% of pre-training.
- Self-Instruct: 175 seeds, 8 few-shot examples (6 human + 2 model), ROUGE-L below 0.7, output-first for classification.
- LIMA: few high-quality examples teach style; knowledge comes from pre-training.
:::

:::mistakes
- Confusing "more tasks help" (FLAN) with "few examples suffice" (LIMA).
:::

:::exam
- *"Why does Self-Instruct use output-first generation for classification tasks?"* (Tier 1)
- *"Pseudo-code the Self-Instruct ROUGE-L filter and a classification router."* (Tier 3)
:::

## Quick Check

:::q Q1 · What is the novelty rule in Self-Instruct?
:::ans
A new instruction is kept only if its ROUGE-L similarity with every existing instruction is below 0.7.
:::
:::

:::q Q2 · Why train on many phrasings of one task?
:::ans
It makes the model generalise to new phrasings and to unseen tasks described in words.
:::
:::
