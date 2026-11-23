**Prompt tuning learns a few vectors prepended to the input while the whole model stays frozen; prefix tuning does the same at every layer.** Both train a tiny fraction of the parameters.

## The Problem: Fine-Tuning Costs Memory and Storage

> [!slides] Opening
> *Transfer learning in the LLM era:* pre-training, then instruction tuning, then alignment. **In-context learning has mostly replaced fine-tuning in large models**, and it is very useful when we do not have direct access to the model, for instance through an API. But prompting has downsides, and full fine-tuning is challenging for two reasons: **hardware requirements** and **storage** (one full copy of the LLM per task).
> — *Lec_29 slides, Dinesh Raghu (IBM Research)*

Take a 65B-parameter model and a laptop-sized budget. Follow the ladder.

**Attempt 1: full fine-tuning for every task.** Training needs the weights, the gradients and the optimizer state. The paper behind Lec_30 says fine-tuning a 65B model needs **more than 780 GB**. A reconstruction of that number: 12 bytes per parameter = 2 (bf16 weights) + 2 (gradients) + 8 (two fp32 Adam moments), and $65\text{B}\times12\text{ B}=780$ GB. And you store a full copy per task.
**Attempt 2: do not train at all, use in-context learning.** Downsides: prompting generally performs worse than fine-tuning; sensitivity to wording and example order; unclear what the model learns (**even random labels work**, Min et al. 2022); the prompt is processed on every prediction.
**Attempt 3: freeze the model, train a tiny task-specific piece.** That is **PEFT**: fewer GPUs, smaller GPUs, **less overfitting and less catastrophic forgetting**, and less storage because most weights are shared.

| Method | What is trained | Stored per task |
|---|---|---|
| Full fine-tuning | all weights | whole model |
| Soft prompt tuning | a few input vectors | the vectors |
| Prefix tuning | prefix vectors at every layer | the prefixes |
| Adapters | small bottleneck modules | the modules |
| LoRA | low-rank $A,B$ per target matrix | $A,B$ |

## Soft Prompt Tuning

> [!lecture] Intuition (lecture 22)
> With a hard prompt you write English: "translate English to Hindi: ...". Now **prepend a set of parameters**, a vector whose size is a hyperparameter, before the task description. Say you have a T5. **All the model's parameters are frozen.** Only the prepended vector is unfrozen. Take a tiny dataset, compute the loss, back-propagate end to end down to the input, but update **only** the prepended embeddings. Don't mix this up with actual fine-tuning. For English-to-Hindi you have one learned prompt, for summarization another, for QA another. They are small vectors, so you can share them from one machine to another: **replicate the model, share the prompt parameters.**

> [!slides] Lec_29: what the plots show
> Results (Lester et al. 2021, T5): prompt tuning **performs poorly at smaller model sizes and on harder tasks**; increasing prompt length helps and **beyond about 20 tokens gains are marginal**; as the model grows, prompt tuning approaches full fine-tuning. **Initialization** matters for small models (sample words from the vocabulary or class labels beat random), but for big models random is good enough.

:::analogy
Instead of retraining a chef (the model) for every cuisine, you hand them a short coaching note (the soft prompt) for each cuisine. The chef stays the same; only the note changes.
:::

```python
import numpy as np
rng = np.random.default_rng(0)
d_model, p = 16, 4
W = rng.normal(size=(d_model, 3))               # a frozen "model": one linear map to 3 classes
P = rng.normal(size=(p, d_model)) * 0.1         # the soft prompt: the only trainable thing
x = rng.normal(size=(5, d_model))               # embedded task text
y = 2                                           # target class

def forward(P):
    seq = np.concatenate([P, x])                # prepend to the INPUT embeddings only
    logits = seq.mean(0) @ W
    e = np.exp(logits - logits.max()); return e / e.sum()

for step in range(200):
    pr = forward(P)
    grad_logits = pr.copy(); grad_logits[y] -= 1
    grad_seq_mean = W @ grad_logits             # backprop THROUGH the frozen model
    P -= 0.5 * np.tile(grad_seq_mean / (p + 5), (p, 1))
print("probability of target after tuning only P:", round(float(forward(P)[y]), 3))
print("trainable parameters:", P.size, "frozen parameters:", W.size)
```

```output
probability of target after tuning only P: 0.985
trainable parameters: 64 frozen parameters: 48
```

> [!lecture] The catch
> Soft prompts are **not interpretable**: you cannot tell what happened inside these vectors. That is why hard prompts are still the default in many contexts.

## Prefix Tuning

> [!lecture] Intuition (lecture 22)
> The same idea (2021, around the same time). The only difference: you add or prepend parameters not only at the input but at **every layer**. You increase the number of parameters, but it is okay because the size is not huge. When you fine-tune, you tune only these.

> [!slides] Lec_29: Li and Liang 2021
> A mapping $f_\theta$ sends each prefix token $p_i$ to a hidden state $h_i=f_\theta(p_i)$. **Directly optimizing it is unstable**, so reparameterize $f_\theta(p_i)=\text{MLP}_\theta(f'_\theta(p_i))$. With **0.1% of the parameters** it is comparable to full fine-tuning, and it **extrapolates better** to unseen topics.

> [!compiler] EndSem Q1(c): prompt tuning versus prefix tuning
> **Where** the trainable vectors live. Prompt tuning: only at the **input embedding layer**. Prefix tuning: trainable vectors at **every Transformer layer** (in the paper, as extra *key and value* entries that attention can read, with queries unchanged), generated through an MLP during training and then stored as plain vectors. Prefix tuning has more parameters and more expressiveness; prompt tuning is the lighter special case.

:::def
**Prompt tuning** trains a small set of continuous vectors prepended to the input embeddings while all model weights are frozen. **Prefix tuning** trains such vectors at every Transformer layer.
:::

:::key
- Full fine-tuning of 65B: more than 780 GB; PEFT trains a small piece and shares the frozen model.
- Prompt tuning: input layer only; poor on small models, matches full tuning at large size; length beyond about 20 gives marginal gains.
- Prefix tuning: every layer, MLP reparameterization; about 0.1% of parameters.
:::

:::mistakes
- "Prompt tuning equals prompt engineering." Engineering edits English; tuning **learns vectors** by gradient descent with the model frozen.
- "Prefix tuning is just a longer prompt." The vectors sit at every layer.
:::

:::exam
- *(EndSem Q1(c))* "Differentiate prompt tuning from prefix tuning." Where the trainable vectors are inserted.
:::

## Quick Check

:::q Q1 · What is updated during soft prompt tuning?
:::ans
Only the prepended prompt embeddings; all model weights stay frozen, though gradients flow through the model to reach them.
:::
:::

:::q Q2 · Why can soft prompts be shared easily across machines?
:::ans
They are small vectors per task; replicate the frozen model and ship only the prompt.
:::
:::
