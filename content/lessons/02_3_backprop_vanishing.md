**Backpropagation applies the chain rule to values stored during the forward pass; in deep networks the gradient becomes a product of many small numbers and vanishes.** Residual connections, gating and attention are the later cures.

## The Problem: Thousands of Weights, One Loss

We have a loss $E$ and many weights. We need $\partial E/\partial w$ for every weight, cheaply.

> [!slides] Squared error and chain rule
> $E=\tfrac12\sum_k(y_k-t_k)^2$ (the $\tfrac12$ cancels the 2 on differentiation). Update: $w\leftarrow w-\eta\,\partial E/\partial w$ (SGD, learning rate $\eta$).
> - Output layer (linear output): $\dfrac{\partial E}{\partial w^{(2)}_{kj}}=(y_k-t_k)\,z_j$
> - Hidden layer, $h=\tanh$: $\dfrac{\partial E}{\partial w^{(1)}_{ji}}=\Big[\sum_k(y_k-t_k)\,w^{(2)}_{kj}\Big]\,\big(1-(z_j)^2\big)\,x_i$
>
> Chain: $\dfrac{\partial E}{\partial w^{(1)}_{ji}}=\dfrac{\partial E}{\partial z_j}\cdot\dfrac{\partial z_j}{\partial a_j}\cdot\dfrac{\partial a_j}{\partial w_{ji}}$. The forward pass stores $y,z,x$, so everything on the right is already known. That is the point of backprop.

:::analogy
Marking a group project: the final grade is wrong, so you pass blame backwards, each person's share depending on how much they contributed downstream.
:::

```python
import numpy as np
rng = np.random.default_rng(1)
W1, b1 = rng.normal(size=(3, 2)) * 0.5, np.zeros(3)
W2, b2 = rng.normal(size=(1, 3)) * 0.5, np.zeros(1)
x, t = np.array([0.5, -1.0]), np.array([0.7])

def forward(W1):
    z = np.tanh(W1 @ x + b1)
    return z, W2 @ z + b2

z, y = forward(W1)
d2 = y - t                              # dE/dy
d1 = (W2.T @ d2) * (1 - z ** 2)         # chain rule through tanh
dW1 = np.outer(d1, x)                   # analytic gradient

eps = 1e-6                              # numeric check
num = np.zeros_like(W1)
for i in range(3):
    for j in range(2):
        Wp = W1.copy(); Wp[i, j] += eps
        Wm = W1.copy(); Wm[i, j] -= eps
        num[i, j] = (0.5 * (forward(Wp)[1] - t) ** 2 - 0.5 * (forward(Wm)[1] - t) ** 2)[0] / (2 * eps)
print("max |analytic - numeric| < 1e-8:", bool(np.abs(dW1 - num).max() < 1e-8))
```

```output
max |analytic - numeric| < 1e-8: True
```

## Vanishing Gradient

> [!lecture] Vanishing gradient
> Now think of hundreds of layers. The gradient for a first-layer weight is a product of many small numbers, so it vanishes and the early layers hardly update. The same problem returns in RNNs. Remedies: other activations, skip/residual connections, gating, attention.

```python
for layers in (1, 5, 10, 20):
    print(layers, "sigmoid layers: gradient factor at most", 0.25 ** layers)
```

```output
1 sigmoid layers: gradient factor at most 0.25
5 sigmoid layers: gradient factor at most 0.0009765625
10 sigmoid layers: gradient factor at most 9.5367431640625e-07
20 sigmoid layers: gradient factor at most 9.094947017729282e-13
```

> [!lecture] Overfitting remedies (named in lecture 5)
> Early stopping, L1/L2 regularisation added to the loss, and **dropout** (randomly drop a fraction of neurons). Learning rate is a hyper-parameter and may decay with epoch.

:::def
**Backpropagation** computes the gradient of the loss with respect to every weight by applying the chain rule layer by layer from the output backwards, reusing activations saved in the forward pass. The **vanishing gradient problem** is the exponential shrinking of this gradient through many layers.
:::

:::key
- Backprop is the chain rule on stored forward values.
- Output layer: $(y_k-t_k)z_j$. Hidden tanh layer: add the $(1-z_j^2)$ factor and the back-propagated sum.
- Deep sigmoid networks: the product of derivatives (at most 0.25 each) vanishes.
- Remedies: residual connections, gating (LSTM), attention, better activations.
:::

:::mistakes
- Dropping the activation derivative from the hidden-layer gradient.
- Forgetting that biases follow the same pattern as weights.
:::

:::exam
- *"Derive $\partial E/\partial w^{(1)}_{ji}$ for a one-hidden-layer MLP."* (derivation)
- *"What is vanishing gradient? Give two remedies."*
:::

## Quick Check

:::q Q1 · Why is the forward pass important for backprop?
:::ans
It stores $x$, $z$ and $y$, which are exactly the quantities needed to evaluate the chain-rule factors.
:::
:::

:::q Q2 · How small can the gradient factor be after 10 sigmoid layers?
:::ans
At most $0.25^{10}\approx 9.5\times10^{-7}$, since each sigmoid derivative is at most 0.25.
:::
:::
