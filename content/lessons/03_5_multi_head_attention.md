**Multi-head attention runs several attention functions in parallel on narrower slices, so one layer can look at several positions for several reasons.** The heads are concatenated and mixed back by an output matrix $W_O$.

## The Problem: One Softmax, One Focus

> [!slides] Motivation
> Given that we're fully depending on attention now, it could be beneficial to attend to more than one position. With a single head, the softmax is heavily influenced by a single value, and it is hard to say "I want two distinct elements, like the subject and the object". **Solution:** use multiple keys, queries and values for each time step.

> [!lecture] Intuition: the CNN analogy
> In a CNN you don't use one kernel on an image patch, you use *several*. The hope is that each kernel picks up a different aspect of that patch: one the ear, one the eye, one the tail. Here also, we use attention heads to fetch different information from the same token. One head may look at whether the token is a subject, another at whether it is a noun or pronoun, another at whether it is a named entity. **We do not know which head will learn which aspect**.

:::analogy
Several reviewers read the same paper, each with a different checklist (novelty, maths, writing). Their reports are concatenated and an editor ($W_O$) merges them into one decision.
:::

> [!slides] Definition
> Each head $j$ has its own projections $W_Q^{j},W_K^{j},W_V^{j}$:
> $$\operatorname{head}_j=\operatorname{Attention}(XW_Q^{j},XW_K^{j},XW_V^{j}),\qquad \operatorname{MHA}(X)=\operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)\,W_O$$
> The paper uses $h=8$ heads and $d_k=d_v=d_{\text{model}}/h=64$ (for $d_{\text{model}}=512$).

> [!lecture] The output projection
> Every head gives a vector per token. If there are two heads you have two vectors, so we **concatenate** them. But the next layer expects a vector of size $d$, not $2d$. So we pass the concatenation through a linear layer, the **output matrix** $W_O$.

> [!compiler] Lecture vs paper vs notebook
> The lecture sketches each head as producing a full $d$-dimensional vector, so $W_O\in\mathbb{R}^{hd\times d}$. In the paper and the notebook each head is **narrower**: $d_k=d_{\text{model}}/h$, so the concatenation is exactly $d_{\text{model}}$ wide and $W_O\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}$. Compute per head is then the same as one full-width head. The code asserts `embed_size % heads == 0` for this reason, and implements all heads at once by one big projection whose output is *reshaped* into $h$ pieces.

```python
import numpy as np
rng = np.random.default_rng(7)
n, d, h = 5, 8, 2
dk = d // h
X = rng.normal(size=(n, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.3 for _ in range(4))

def split(M):                       # (n, d) -> (h, n, dk)
    return M.reshape(n, h, dk).transpose(1, 0, 2)

Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
S = Q @ K.transpose(0, 2, 1) / np.sqrt(dk)           # (h, n, n)
A = np.exp(S - S.max(-1, keepdims=True)); A /= A.sum(-1, keepdims=True)
heads = A @ V                                        # (h, n, dk)
concat = heads.transpose(1, 0, 2).reshape(n, d)      # back to (n, d)
out = concat @ Wo
print("per-head attention:", A.shape, " output:", out.shape)
print("rows sum to 1:", np.allclose(A.sum(-1), 1))
```

```output
per-head attention: (2, 5, 5)  output: (5, 8)
rows sum to 1: True
```

:::def
**Multi-head attention** projects the input into $h$ sets of queries, keys and values of width $d_{\text{model}}/h$, applies scaled dot-product attention in each, concatenates the results and applies the output projection $W_O$.
:::

:::key
- $h$ heads, each of width $d_k=d_{\text{model}}/h$; the concatenation is $d_{\text{model}}$ wide.
- Same total compute as one full-width head, but several views.
- $W_O\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}$ mixes the heads.
:::

:::mistakes
- "The lecture says $W_O$ is $2d\times d$, the code has $d\times d$." The lecture sketches full-width heads; the paper uses narrow heads.
- Assuming we know which head learns what. We do not.
:::

:::exam
- *"Why multi-head attention?"* One softmax is dominated by one value; multiple heads attend to several positions and aspects.
- *"For $d_{\text{model}}=512$, $h=8$, what is $d_k$?"* 64.
:::

> **See it live.** Open an attention head and compare the 12 heads in the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (GPT-2 small, runs in your browser). Full walkthrough in [Lesson 5.5](/05_5_decoder_only).

## Quick Check

:::q Q1 · Why must $d_{\text{model}}$ be divisible by $h$?
:::ans
So that $d_k=d_{\text{model}}/h$ is an integer and the concatenated heads are exactly $d_{\text{model}}$ wide.
:::
:::

:::q Q2 · What does $W_O$ do?
:::ans
It projects the concatenated head outputs back to $d_{\text{model}}$ and mixes information across heads.
:::
:::
