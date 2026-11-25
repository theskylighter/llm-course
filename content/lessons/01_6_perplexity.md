<<SVG:perplexity_average_surprise>>

*Figure: Average held-out token log loss, then exponentiate to perplexity; compare models on the same test set and vocabulary.*

**Perplexity measures how surprised a language model is by held-out text, and lower is better.** It is the exponentiated cross-entropy, and it equals the inverse probability of the test set normalised by length.

## The Problem: How Do We Say One LM Beats Another?

**Extrinsic** evaluation plugs models A and B into a downstream task (machine translation, speech recognition) and compares scores. It is honest but slow.

**Intrinsic** evaluation asks one question: does the model assign higher probability to real sentences than to bad ones, on a **held-out test set disjoint from training**?

> [!lecture] Intuition
> Language is a stochastic process whose true distribution we never know. We approximate it with a model $M$ and measure the **cross-entropy** between the language and the model. Evaluate on test data only; otherwise you reward memorisation.

## The Metric

> [!slides] Perplexity
> $$PP(W)=P(w_1\dots w_n)^{-1/n}=\sqrt[n]{\prod_{i=1}^{n}\frac{1}{P(w_i\mid w_1\dots w_{i-1})}}\;\overset{\text{bigram}}{=}\;\sqrt[n]{\prod_i\frac{1}{P(w_i\mid w_{i-1})}}$$
> **Minimizing perplexity = maximizing probability.** Cross-entropy: $H(L,M)=-\lim_{n\to\infty}\tfrac1n\log P_M(w_1\dots w_n)$. Perplexity is the exponentiated cross-entropy.

> [!compiler] Precise statements
> - $PP=2^{H(L,M)}$ when $\log$ is base 2 (use $e^{H}$ for natural log). The "branching factor" reading is standard but is not on the slides: a model that is uniform over 10 choices at every step has $PP=10$.
> - Perplexities are comparable only on the **same test set and same vocabulary**.

:::analogy
Perplexity is the number of equally likely options the model is effectively choosing between at each word. $PP=50$ means the model is as unsure as if it had to pick uniformly among 50 words.
:::

Always use log-probabilities. Multiplying hundreds of small probabilities underflows to 0.

```python
import math

def perplexity(probs):
    """probs: the model probability of each test token."""
    log2_sum = sum(math.log2(p) for p in probs)
    return 2 ** (-log2_sum / len(probs))

print("uniform over 10 words :", round(perplexity([0.1] * 20), 2))
print("sure model (p=0.5)    :", round(perplexity([0.5] * 20), 2))
print("confident but wrong   :", round(perplexity([0.9, 0.9, 0.001, 0.9]), 2))
tiny = [1e-5] * 100
print("raw product underflows? ->", math.prod(tiny) == 0.0, "; log version:", round(perplexity(tiny)))
```

```output
uniform over 10 words : 10.0
sure model (p=0.5)    : 2.0
confident but wrong   : 6.09
raw product underflows? -> True ; log version: 100000
```

:::def
**Perplexity** of a test set $W=w_1\dots w_n$ under a model is $PP(W)=P(w_1\dots w_n)^{-1/n}=2^{H}$, where $H$ is the cross-entropy in bits per token. Lower perplexity means a better model.
:::

:::key
- $PP(W)=P(W)^{-1/n}$; lower is better.
- Uniform over $k$ choices gives $PP=k$.
- Use held-out data, the same vocabulary, and log-probabilities.
:::

:::mistakes
- Computing perplexity on training data.
- Comparing perplexities across different vocabularies or test sets.
- Multiplying raw probabilities instead of summing logs.
:::

:::exam
- *"Define perplexity. What does PP = 50 mean? Why is lower better?"* (short)
- *"List three limitations of N-gram LMs."* Sparsity, short fixed context, no generalisation to similar words.
:::

## Quick Check

:::q Q1 · A model assigns probability $1/8$ to every token of a test sentence. What is its perplexity?
:::ans
$PP=(1/8)^{-1}=8$: uniform over 8 choices.
:::
:::

:::q Q2 · Why must perplexity be computed on held-out text?
:::ans
On training text a model that memorises would score very low perplexity, rewarding memorisation rather than generalisation.
:::
:::
