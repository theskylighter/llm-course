**Neural LMs replace count tables with vectors, but CNN windows are fixed, RNNs forget, and seq2seq squeezes the whole input into one vector; attention fixes the last problem.** This lesson is the bridge to the Transformer.

## The Problem: What N-grams Could Not Do

N-grams fail on sparsity and short context. A neural LM moves from discrete counts to **continuous vectors**: similar words get similar vectors, so unseen combinations still get probability. The recipe is always the same: embed the context, compute a hidden state, project to $\lvert V\rvert$, softmax, sample.

## CNN Window Model

> [!lecture] Window model (lecture 10)
> Take a window of 4 word embeddings of size 4 each, flatten into a $1\times16$ vector, multiply by a filter $W$ ($16\times8$) to get a hidden vector, then a linear layer $U$ ($8\times\lvert V\rvert$) and softmax. No sparsity and no giant bigram table. Three problems: (1) you must fix a **window size**, (2) $W$ **grows** with the window, (3) **no message passing**: each embedding only affects its own column of weights. CNNs cannot capture long-distance relations.

## RNN and LSTM

> [!compiler] Vanilla RNN (standard form, not from a slide)
> $$h_t=\sigma\big(W_h h_{t-1}+W_e x_t+b\big),\qquad P(w_{t+1}\mid w_{\le t})=\operatorname{softmax}(U h_t)$$
> Same weights at every step, so any context length works. Training with ground truth fed as the next input is **teacher forcing**.

> [!lecture] Vanishing and exploding gradient (lecture 11)
> The gradient of the loss at step 4 with respect to $h_1$ is a product of Jacobians. Small matrices give vanishing gradients, large ones give exploding. Example: "She printed her ___" where "tickets" appeared seven words earlier: the RNN cannot recall it.
>
> **Fix: gating (LSTM, GRU).** Keep a **cell state** $c_t$ (memory) and a **hidden state** $h_t$ (output). Gates are vectors with values in $[0,1]$: 1 lets information pass, 0 blocks it. LSTM has three gates: forget, input and output. It is read/write on a memory.

## The Bottleneck and Attention

In seq2seq the **last encoder state** must remember the whole input. That is the **bottleneck**. Fix: let every decoder state look at every encoder state directly.

> [!slides] Dot-product attention
> Decoder state $s_t$ is the query, encoder states $h_1\dots h_n$ are the values:
> $$e_t=[s_t^\top h_1,\dots,s_t^\top h_n],\qquad \alpha_t=\operatorname{softmax}(e_t),\qquad a_t=\sum_i\alpha_{t,i}\,h_i$$
> Concatenate $[a_t;s_t]$, apply a linear layer and non-linearity, then softmax over $\mathcal{V}$.

:::analogy
Translating a paragraph. The RNN reads it once and then translates from memory of its last sentence. Attention lets you glance back at the exact source word each time you write a target word.
:::

```python
import numpy as np
h = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # three encoder states
s = np.array([2.0, 0.0])                              # decoder state (query)
e = h @ s                                             # dot-product scores
alpha = np.exp(e) / np.exp(e).sum()                   # softmax
a = alpha @ h                                         # weighted sum
print("scores :", e)
print("weights:", np.round(alpha, 3))
print("context:", np.round(a, 3))
```

```output
scores : [2. 0. 2.]
weights: [0.468 0.063 0.468]
context: [0.937 0.532]
```

> [!lecture] Intuition
> The weights $\alpha_t$ change at each decoder step, and they tell you which source word helped which target word: attention gives **interpretability**. Vanilla attention adds **no new parameters** (scores are plain dot products). It solves the bottleneck and partly vanishing gradients.

> [!compiler] Forward link
> Chapter 3 replaces recurrence entirely: queries, keys and values come from learned projections of the same sequence, scaled by $\sqrt{d_k}$. Here, $h_i$ plays both key and value.

:::def
**Attention** computes a context vector as a weighted sum of values, with weights given by a softmax over query-key similarity scores.
:::

:::key
- CNN LMs need a fixed window. RNNs handle any length but forget (LSTM and GRU gates help). Attention removes the seq2seq bottleneck.
- Vanilla dot-product attention adds no parameters.
- Teacher forcing feeds the true previous token during training.
:::

:::mistakes
- "Attention adds parameters." Not in plain dot-product form.
- "RNNs cannot handle long sequences because of memory." The issue is vanishing gradients and the single state.
:::

:::exam
- *"Why does the seq2seq bottleneck motivate attention? Write the attention equations."*
- *"Three limitations of a CNN window LM."*
:::

## Quick Check

:::q Q1 · What is the bottleneck problem in encoder-decoder RNNs?
:::ans
The final encoder state must summarise the entire input sentence in one fixed-size vector.
:::
:::

:::q Q2 · What do LSTM gates take values in, and what does 0 mean?
:::ans
Values in $[0,1]$; 0 blocks information completely, 1 lets it pass.
:::
:::
