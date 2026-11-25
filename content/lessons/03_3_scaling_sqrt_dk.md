<<SVG:scaled_dot_product_variance>>

*Figure: Scaling by √dₖ restores unit score variance, keeping softmax from collapsing onto one position.*

**Dot products of $d_k$-dimensional vectors have variance $d_k$, so large dimensions push softmax into saturation; dividing by $\sqrt{d_k}$ restores unit variance.** That is the "scaled" in scaled dot-product attention.

## The Problem: Softmax Collapses on Large Scores

Suppose the components of $\mathbf{q}$ and $\mathbf{k}$ are independent with mean 0 and variance 1. Then $\mathbf{q}^\top\mathbf{k}=\sum_{j=1}^{d_k}q_jk_j$ has mean 0 and variance $d_k$, because each term has variance 1 and the terms are independent. For $d_k=64$ the scores have standard deviation 8.

When one score is 8 and the others are near 0, the softmax is almost one-hot, and its gradient is almost zero. Learning stalls.

> [!compiler] Why $\sqrt{d_k}$? (not derived in the lecture; the notebook uses it)
> Dividing by $\sqrt{d_k}$ restores unit variance: $\operatorname{Var}\!\big(\mathbf{q}^\top\mathbf{k}/\sqrt{d_k}\big)=1$.

:::analogy
Imagine a judging panel whose scores are multiplied by 8 before voting. Tiny differences become landslides, and the panel can no longer express "these two are about equal".
:::

```python
import numpy as np
rng = np.random.default_rng(0)
for dk in (4, 64, 1024):
    q = rng.normal(size=(20000, dk))
    k = rng.normal(size=(20000, dk))
    s = (q * k).sum(1)
    print(f"d_k={dk:5d}  var(raw)={s.var():8.1f}  var(scaled)={(s / np.sqrt(dk)).var():5.2f}")

scores = np.array([8.0, 1.0, 0.0, -1.0])
sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
print("softmax raw   :", np.round(sm(scores), 4))
print("softmax scaled:", np.round(sm(scores / 8), 4))
```

```output
d_k=    4  var(raw)=     4.0  var(scaled)= 1.00
d_k=   64  var(raw)=    64.5  var(scaled)= 1.01
d_k= 1024  var(raw)=  1023.5  var(scaled)= 1.00
softmax raw   : [9.986e-01 9.000e-04 3.000e-04 1.000e-04]
softmax scaled: [0.4741 0.1976 0.1744 0.1539]
```

:::def
**Scaled dot-product attention** divides the query-key scores by $\sqrt{d_k}$ before the softmax so that their variance stays near 1 regardless of $d_k$.
:::

:::key
- Raw score variance is $d_k$; scaled variance is 1.
- Without scaling, softmax saturates and gradients vanish.
:::

:::mistakes
- Dividing by $d_k$ instead of $\sqrt{d_k}$. The standard deviation, not the variance, scales as $\sqrt{d_k}$.
:::

:::exam
- *"Why is the dot product scaled by $\sqrt{d_k}$?"* State mean 0, variance $d_k$; saturation; fix.
:::

## Quick Check

:::q Q1 · For $d_k=64$, what is the standard deviation of unscaled scores?
:::ans
$\sqrt{64}=8$.
:::
:::

:::q Q2 · What happens to the softmax gradient when one score dominates?
:::ans
The output is nearly one-hot and the gradient nearly 0, so the model stops learning from that row.
:::
:::
