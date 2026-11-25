<<SVG:transformer_remove_recurrence>>

*Figure: Self-attention removes the sequential dependency between token states, enabling parallel work—but positional information is needed to recover order.*

**If every state can already see every other state through attention, the recurrent links are only a speed bump; removing them gives parallelism but costs us the sense of order.** The rest of the chapter is the list of repairs.

## The Problem: Why Do We Still Need Recurrence?

> [!lecture] Opening
> In the last class we completed RNNs, LSTMs, GRUs and sequence-to-sequence models, and we introduced attention to fix the *bottleneck problem*. Now the question we ask is whether the attention we have seen so far is *all we need*. Attention partly fixes vanishing gradients and fully fixes the bottleneck, but it still leaves one problem: we access the tokens **sequentially**, so at time $t$ you must wait for time $t-1$. In this chapter we remove the recurrence and gradually build the Transformer.
> — *NPTEL Lecture 15*

Take an encoder-decoder RNN with attention. Every decoder state can already look at **all** encoder states directly. So the lecture asks: if a state can reach every other state through attention, **why do we still need the recurrent links** $h_{t-1}\to h_t$?

| Problem | Vanilla RNN | + Attention | Transformer |
|---|---|---|---|
| Bottleneck (one vector holds the input) | no | yes | yes |
| Vanishing gradient across time | no | partly | yes (short paths and residuals) |
| Parallel access to all tokens | no | no | yes |

> [!lecture] Intuition
> Because of the recurrent connections you are not able to access the tokens in parallel. If these had not been there, you could have accessed all the tokens in parallel. So we remove those dependencies, and now each state can access all the other states, and each of these operations is independent of the others. This is the premise behind self-attention.

## What Removing Recurrence Breaks

If the hidden states have no links between them, then *shuffling the tokens shuffles the outputs and changes nothing else*. The model has lost the notion of order.

:::analogy
A group of people each holding a card with one word. If everyone can see all cards but nobody knows who was standing where, they cannot tell "the man eats the cheese" from "the cheese eats the man".
:::

> [!slides] From Self-Attention to Transformers
> 1. **Positional encoding**: addresses the lack of sequence information
> 2. **Multi-head attention**: allows querying multiple positions at each layer
> 3. **Non-linearities**: so far each successive layer is linear in the previous one
> 4. **Masked decoding**: prevents attention lookups into the future
>
> We add two supporting pieces in Lecture 16: **residual connections** and **layer normalization**.

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))                  # four tokens, three features
scores = X @ X.T                             # every token against every token
A = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)
out = A @ X
perm = [2, 0, 3, 1]
out_perm = (lambda Y: (np.exp(Y @ Y.T) / np.exp(Y @ Y.T).sum(1, keepdims=True)) @ Y)(X[perm])
print("shuffled input gives the shuffled output:", np.allclose(out_perm, out[perm]))
```

```output
shuffled input gives the shuffled output: True
```

:::def
Self-attention without positional information is **permutation equivariant**: permuting the input tokens permutes the output rows in the same way, so word order is invisible to the model.
:::

:::key
- Recurrence forces sequential access; removing it gives parallelism and one-hop access between any two tokens.
- Without recurrence the model loses order, so positional information must be added.
- The four repairs: positional encoding, multi-head attention, non-linearity (FFN), masked decoding; plus residuals and layer norm.
:::

:::mistakes
- "Self-attention is order-aware." No: it is permutation-equivariant without positions. (MidSem 2024 Q3(D): "self-attention eliminates the need for positional encoding" is **false**.)
:::

:::exam
- *"Why do we need positional encoding in a Transformer?"* (2 marks)
- *True/False with reason:* self-attention eliminates the need for positional encoding.
:::

## Quick Check

:::q Q1 · What does attention without recurrence gain and what does it lose?
:::ans
It gains parallel access to all tokens and short gradient paths; it loses the notion of token order.
:::
:::

:::q Q2 · Which problems of the vanilla RNN does attention alone not fix?
:::ans
Parallel access (tokens are still processed sequentially) and, only partly, vanishing gradients.
:::
:::
