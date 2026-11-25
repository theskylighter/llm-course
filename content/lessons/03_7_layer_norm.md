<<SVG:layer_norm_per_token>>

*Figure: LayerNorm normalizes each token across its own features, independent of batch size.*

**Layer normalization standardizes each token's own feature vector, so it needs no batch statistics and behaves the same in training and inference.** That is why Transformers use it instead of batch normalization.

## The Problem: Features Live on Different Scales

> [!lecture] Intuition
> You are given two features for a student: **age** (18 to 25) and **tuition fee** (20K to 50K dollars). They live on totally different scales. **Normalization** (min-max) maps a feature to $[0,1]$: $x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}$. **Standardization** makes it have zero mean and unit standard deviation: $x'=\frac{x-\mu}{\sigma}$. In practice "normalization" is used for both.

Why normalise *inside* a deep network? After each layer so many operations have happened that the entries of a vector drift away from a well-behaved range. Benefits named in the lecture: it **stabilises and speeds up training**, keeps gradients from exploding or vanishing, and helps a bit against overfitting.

## Batch Normalization and Why It Is Wrong Here

> [!slides] Batch normalization
> BatchNorm takes **one feature across the batch** and standardises it:
> $$\hat{x}_{i}=\frac{x_{i}-\mu_{B,i}}{\sqrt{\sigma^2_{B,i}+\epsilon}},\qquad y_i=\gamma_i\hat{x}_i+\beta_i$$
> **Limitations:** with a small batch the sample mean and std are not representative; and it is less suited to sequence models, where lengths vary within a batch.

> [!lecture] Lecture
> Say a batch has sequences of size 4, 8 and 16. The fifth element exists in the second sequence but not the first, so how do you take the mean "across the batch" for that position? It doesn't work well. Also, BatchNorm needs different processing at training and at inference.

## Layer Normalization

> [!slides] Layer normalization
> For **one** input vector $\mathbf{x}\in\mathbb{R}^d$, normalise across its own $d$ features:
> $$\mu=\frac1d\sum_{j=1}^{d}x_j,\qquad \sigma^2=\frac1d\sum_{j=1}^{d}(x_j-\mu)^2,\qquad \operatorname{LN}(\mathbf{x})=\boldsymbol{\gamma}\odot\frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}}+\boldsymbol{\beta}$$
> Worked: $\mathbf{x}=(1,3,5,7)$ gives $\mu=4$ and $\sigma=\sqrt5\approx2.23$.

| | BatchNorm | LayerNorm |
|---|---|---|
| Normalises | each feature **across the batch** | each input **across its features** |
| Depends on batch size | yes | **no** |
| Train vs inference | different | **same operations** |
| Variable-length sequences | awkward | natural |

:::analogy
BatchNorm grades on a curve across the whole class. LayerNorm re-scales each student's own answer sheet by that sheet's own average and spread. A student's result no longer depends on who else is in the room.
:::

```python
import numpy as np
x = np.array([1.0, 3.0, 5.0, 7.0])
mu, sigma = x.mean(), x.std()
print("mu =", mu, " sigma =", round(float(sigma), 4), " sqrt(5) =", round(5 ** 0.5, 4))
print("normalised:", np.round((x - mu) / sigma, 4))

batch = np.array([[1.0, 3.0, 5.0, 7.0], [10.0, 0.0, 20.0, 30.0]])
ln = (batch - batch.mean(1, keepdims=True)) / batch.std(1, keepdims=True)
print("each row mean ~ 0:", np.allclose(ln.mean(1), 0), " each row std ~ 1:", np.allclose(ln.std(1), 1))
```

```output
mu = 4.0  sigma = 2.2361  sqrt(5) = 2.2361
normalised: [-1.3416 -0.4472  0.4472  1.3416]
each row mean ~ 0: True  each row std ~ 1: True
```

> [!compiler] Post-LN versus Pre-LN, and RMSNorm
> The original Transformer uses **post-LN**: $\operatorname{LN}(x+\operatorname{Sublayer}(x))$. Most modern LLMs use **pre-LN**, $x+\operatorname{Sublayer}(\operatorname{LN}(x))$ (it trains stably without warm-up), and many replace LayerNorm by **RMSNorm** (no mean subtraction).

:::def
**Layer normalization** normalises each input vector across its feature dimension to zero mean and unit variance, then applies learned scale $\gamma$ and shift $\beta$. It uses no batch statistics.
:::

:::key
- LN normalises across features of one token; BN across the batch for one feature.
- LN is independent of batch size and identical at train and test time.
- $\sigma$ in the lecture example is the population standard deviation: $\sqrt5\approx2.23$.
:::

:::mistakes
- "Layer norm is applied over the batch." No: over each token's feature vector.
- Using the sample ($d-1$) standard deviation when the lecture uses population.
:::

:::exam
- *"Why layer norm and not batch norm in Transformers?"* Variable-length sequences, small or varying batches, train/inference mismatch.
- *Numerical:* normalise $(1,3,5,7)$.
:::

## Quick Check

:::q Q1 · Compute $\mu$ and $\sigma$ for $(1,3,5,7)$.
:::ans
$\mu=4$; variance $=(9+1+1+9)/4=5$; $\sigma=\sqrt5\approx2.23$.
:::
:::

:::q Q2 · Name two problems of batch normalization for sequence models.
:::ans
Statistics are unreliable with small batches and awkward with variable-length sequences; it also behaves differently in training and inference.
:::
:::
