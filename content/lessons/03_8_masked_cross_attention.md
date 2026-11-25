<<SVG:causal_mask_cross_attention>>

*Figure: The causal mask blocks future target positions; cross-attention lets decoder queries read encoder states.*

**A decoder must not see the future, so we add $-\infty$ above the diagonal of the score matrix; cross-attention then links the decoder to the encoder with queries from one and keys and values from the other.** These two ideas turn self-attention into a translator.

## The Problem: Looking Ahead Is Cheating

> [!lecture] The problem
> In the encoder the source sentence is given, so each word may look at every other word. In the decoder the target is **not** given at inference: we generate one token, then the next based on it. So a decoder state must not see future states. *But at training time we do have the whole target sentence*, and if we let position $i$ look at position $i+1$ that is **cheating**, since at test time that information doesn't exist.

> [!lecture] How the mask works
> $QK^\top$ is an $n\times n$ matrix whose $(i,j)$ entry is $\mathbf{q}_i^\top\mathbf{k}_j$. Entry $q_1k_2$ is **not allowed**: key 2 is in the future of query 1. So we keep the **lower triangle** (including the diagonal) and kill the upper triangle. Before the softmax we replace the upper entries with $-\infty$; since $e^{-\infty}=0$, after the softmax those weights are exactly 0.

> [!slides] Masked attention
> $$\operatorname{MaskedAttn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V,\qquad M_{ij}=\begin{cases}0&j\le i\\-\infty&j>i\end{cases}$$

:::analogy
Writing an exam: when answering question 3 you may look at your answers to questions 1 and 2, but the paper from question 4 onward is covered with a blank sheet.
:::

> [!compiler] Two precision notes on the lecture's wording
> (1) The lecture says the mask is "multiplied" by $-\infty$. It is **added** (or applied with `masked_fill`). A literal product would give $0\times(-\infty)=\text{NaN}$. (2) It also describes $QK^\top$ as "element-wise multiplication"; it is an ordinary matrix product.

```python
import numpy as np
rng = np.random.default_rng(3)
n = 4
S = rng.normal(size=(n, n))
mask = np.triu(np.ones((n, n), dtype=bool), k=1)      # True above the diagonal
S_masked = np.where(mask, -np.inf, S)
A = np.exp(S_masked - S_masked.max(1, keepdims=True))
A /= A.sum(1, keepdims=True)
np.set_printoptions(precision=3, suppress=True)
print(A)
print("future weights are exactly zero:", bool((A[mask] == 0).all()))
```

```output
[[1.    0.    0.    0.   ]
 [0.441 0.559 0.    0.   ]
 [0.014 0.943 0.043 0.   ]
 [0.329 0.224 0.152 0.295]]
future weights are exactly zero: True
```

> [!compiler] Padding is a second, separate mask
> Batches contain sentences of different lengths, so shorter ones are padded. A **source mask** sets scores of `<pad>` keys to $-\infty$. The decoder therefore needs two things: causal AND non-pad.

## Cross-Attention

So far the encoder and decoder stacks are **not linked**. The missing link is cross-attention.

> [!lecture] Cross-attention
> It is the same attention concept, but now the **query comes from the decoder** and the **keys and values come from the encoder**. Take the decoder state after masked self-attention and produce $\mathbf{q}_1$ from it; produce $\mathbf{k}_j,\mathbf{v}_j$ from the final encoder outputs $H_j$ (with a new set of parameters). **Is it masked? No**: the source is fully visible. One point to remember: it is **not** layer-1-to-layer-1. *Every decoder layer attends to the last layer of the encoder.*

> [!slides] Block structure
> **Encoder block**: multi-head self-attention, Add & Norm, FFN, Add & Norm.
> **Decoder block**: *masked* multi-head self-attention, Add & Norm, **cross-attention** (Q from decoder, K/V from encoder), Add & Norm, FFN, Add & Norm.
> Stack $N$ of each; after the top decoder layer, a linear layer and softmax predict the next token.

:::def
**Masked (causal) self-attention** adds $-\infty$ to scores for key positions $j>i$ before the softmax so token $i$ attends only to positions $\le i$. **Cross-attention** takes queries from the decoder and keys and values from the final encoder output.
:::

:::key
- The mask is **added** before softmax: $-\infty$ above the diagonal gives exact zeros.
- Training uses the mask to process all positions in parallel without cheating.
- Cross-attention: Q from decoder, K and V from the encoder; not masked; every decoder layer uses the last encoder layer.
- Padding and causality are two separate masks.
:::

:::mistakes
- "The mask is multiplied by $-\infty$." It is added.
- "Cross-attention is masked." The source is fully visible.
- "Decoder layer $k$ attends to encoder layer $k$." It attends to the final encoder output.
:::

:::exam
- *"What is masked self-attention and why is it needed?"* Prevent lookups into the future; training would otherwise cheat.
- *"Which of Q, K, V come from the encoder in cross-attention?"* K and V.
:::

## Quick Check

:::q Q1 · Why $-\infty$ and not 0 in the masked positions?
:::ans
Adding 0 would leave the original scores, so the softmax would still give positive weight. $e^{-\infty}=0$ gives exact zero weight to the future.
:::
:::

:::q Q2 · Where do the keys and values of a decoder cross-attention come from?
:::ans
The final output of the encoder stack.
:::
:::
