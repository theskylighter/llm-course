**Self-attention lets every token play three roles at once: a query that asks, a key that is matched, and a value that is returned.** The output is a softmax-weighted sum of values.

## The Problem: Seq2Seq Attention Had Fixed Roles

In seq2seq attention the query was a decoder state, and keys and values were *the same thing* (the encoder states $h_i$). We want every token to attend to every other token in the same sequence, with separate learned roles.

> [!lecture] Intuition: the information-retrieval picture
> Think about a search engine. The user types a **query**. Every document has some metadata or keywords, its **key**. You match the query against all the keys, say with cosine similarity, and pick the best match. But you may not want to return the *entire* document, only the paragraph relevant to the query. What you return is the **value**. Query and key decide *where to look*; the value is *what you take away*.

:::analogy
Query = your search words. Key = each page's title and tags. Value = the paragraph you actually copy out. The score compares query and key; the answer is built from values.
:::

> [!slides] Definition: self-attention layer
> Given hidden states $\mathbf{h}_1,\dots,\mathbf{h}_n$ (one per token):
> $$\mathbf{q}_i = W_Q \mathbf{h}_i,\qquad \mathbf{k}_i = W_K \mathbf{h}_i,\qquad \mathbf{v}_i = W_V \mathbf{h}_i$$
> $$e_{\ell t} = \mathbf{q}_\ell^{\top}\mathbf{k}_t,\qquad \alpha_{\ell t} = \frac{\exp(e_{\ell t})}{\sum_{t'}\exp(e_{\ell t'})},\qquad \mathbf{a}_\ell = \sum_{t}\alpha_{\ell t}\,\mathbf{v}_t$$
> In matrix form, with $X\in\mathbb{R}^{n\times d}$ and $Q=XW_Q,\ K=XW_K,\ V=XW_V$:
> $$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

> [!lecture] Now remember...
> - The *same* $W_Q, W_K, W_V$ are used for **every token**: parameters are shared across tokens.
> - They are **not** shared across layers: layer 1 has its own, layer 2 has its own.
> - Each token's output can be computed independently, so on a GPU or TPU we do it as one matrix multiplication. That is the whole point.

## Step by Step

```python
import numpy as np
rng = np.random.default_rng(42)
n, d, dk = 3, 4, 2
X = rng.normal(size=(n, d))
Wq, Wk, Wv = (rng.normal(size=(d, dk)) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores = Q @ K.T / np.sqrt(dk)
A = np.exp(scores - scores.max(1, keepdims=True))
A /= A.sum(1, keepdims=True)
out = A @ V
print("shapes: Q", Q.shape, "scores", scores.shape, "out", out.shape)
print("each row of attention sums to 1:", np.allclose(A.sum(1), 1))
```

```output
shapes: Q (3, 2) scores (3, 3) out (3, 2)
each row of attention sums to 1: True
```

> [!compiler] Complexity
> With $n$ tokens every token attends to every token, so one layer costs $O(n^2 d)$ time and $O(n^2)$ memory for the score matrix, versus $O(n d^2)$ for an RNN layer. The compensation is **$O(1)$ sequential steps** instead of $O(n)$, and any two tokens are one hop apart. The $n^2$ term is what Chapter 4 attacks.

:::def
**Self-attention** maps hidden states $X$ to $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ where $Q=XW_Q$, $K=XW_K$, $V=XW_V$ are learned linear projections of the same sequence.
:::

:::key
- Query-key similarity decides where to look; values carry what is returned.
- $W_Q,W_K,W_V$ are shared across tokens but not across layers.
- The whole layer is a few matrix multiplications, so it is fully parallel.
- Cost: $O(n^2d)$ time, $O(n^2)$ memory.
:::

:::mistakes
- Thinking keys and values must be equal. In self-attention they are separate projections.
- Softmax is over the **keys** (each query's weights sum to 1), not over queries.
:::

:::exam
- *"Explain Q, K, V with the search-engine analogy and write the attention equation."*
- *"What is the time complexity of a self-attention layer?"* $O(n^2d)$.
:::

## Quick Check

:::q Q1 · Are $W_Q$, $W_K$, $W_V$ shared across tokens? Across layers?
:::ans
Shared across tokens within a layer: yes. Across layers: no, each layer has its own.
:::
:::

:::q Q2 · What shape is the score matrix for $n$ tokens?
:::ans
$n\times n$: entry $(\ell,t)$ is $\mathbf{q}_\ell^\top\mathbf{k}_t$.
:::
:::
