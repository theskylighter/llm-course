<<SVG:mlp_nonlinearity_layers>>

*Figure: Without a nonlinear activation, stacked layers collapse to one linear map; nonlinear activations add expressive power.*

**An MLP stacks layers of weighted sums followed by non-linear activations; without the non-linearity every layer collapses into one linear map.** Choosing the activation decides whether gradients survive.

## The Problem: Why Not Just Stack Linear Layers?

If $g(x)=x$, then layer 2 applied to layer 1 is $W_2(W_1\mathbf{x})=(W_2W_1)\mathbf{x}$: one linear map. A constant $g$ is worse, because its derivative is 0 and nothing learns. So the activation must be non-linear **and** have a useful derivative.

> [!slides] MLP notation (pay attention, the lecture says memorise it)
> $N$ inputs, $M$ hidden neurons per layer, $K$ outputs. **Depth** = number of layers, **width** = neurons per layer. Weight $w^{(l)}_{ji}$ connects neuron $i$ of the previous layer to neuron $j$ of layer $l$. Every neuron does two operations:
> $$a^{(1)}_j=\sum_{i=1}^{N}w^{(1)}_{ji}x_i+w^{(1)}_{j0}\quad\text{(pre-activation)},\qquad z^{(1)}_j=h\big(a^{(1)}_j\big)\quad\text{(post-activation)}$$

```python
import numpy as np
rng = np.random.default_rng(0)
W1, W2 = rng.normal(size=(4, 3)), rng.normal(size=(2, 4))
x = rng.normal(size=3)
two_layers = W2 @ (W1 @ x)           # no activation between the layers
one_layer = (W2 @ W1) @ x            # a single combined matrix
print("identical:", np.allclose(two_layers, one_layer))
print("with tanh in between they differ:", not np.allclose(W2 @ np.tanh(W1 @ x), one_layer))
```

```output
identical: True
with tanh in between they differ: True
```

## Activation Functions (High-Yield Table)

> [!slides] Activations, their derivatives, and what breaks
> | Name | Formula | Derivative | Problem or point |
> |---|---|---|---|
> | Sigmoid | $\sigma(x)=\frac1{1+e^{-x}}$ | $\sigma(1-\sigma)$ | outputs in $(0,1)$, not zero-centred, gradient tiny beyond $\lvert x\rvert\approx3$ |
> | tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $1-\tanh^2x$ | zero-centred, range $(-1,1)$, still saturates |
> | Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | | outputs a probability distribution |
> | ReLU | $\max(0,x)$ | 1 or 0 | **dead neuron**: negative input gives zero output and zero gradient |
> | Leaky ReLU | $x$ if $x\ge0$ else $0.01x$ | | fixes dead neurons |
> | GELU | $x\,\Phi(x)$ | | $\Phi$ = standard normal CDF |
> | GLU | $a\odot\sigma(Wa+b)$ | | a learned gate decides how much passes |
> | Swish / SiLU | $x\,\sigma(\beta x)$ | | $\beta=1$ near linear, $\beta=10$ near ReLU |
> | SwiGLU | $x\odot\mathrm{Swish}(Wx+b)$ | | Swish + GLU, used in modern FFNs |

```python
import numpy as np
x = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
sig = 1 / (1 + np.exp(-x))
print("sigmoid      :", np.round(sig, 4))
print("sigmoid deriv:", np.round(sig * (1 - sig), 4))
print("tanh deriv   :", np.round(1 - np.tanh(x) ** 2, 4))
print("relu         :", np.maximum(0, x))
e = np.exp(np.array([2.0, 1.0, 0.1]))
print("softmax      :", np.round(e / e.sum(), 3), "sum =", round(float((e / e.sum()).sum()), 3))
```

```output
sigmoid      : [0.0025 0.0474 0.5    0.9526 0.9975]
sigmoid deriv: [0.0025 0.0452 0.25   0.0452 0.0025]
tanh deriv   : [0.     0.0099 1.     0.0099 0.    ]
relu         : [0. 0. 0. 3. 6.]
softmax      : [0.659 0.242 0.099] sum = 1.0
```

> [!compiler] Correction to the transcript
> The lecture says the sigmoid derivative "is 0.5 at 0". $\sigma(0)=0.5$, but $\sigma'(0)=\sigma(0)(1-\sigma(0))=0.25$, its **maximum**. Keep 0.25 for numerical answers. The ASR also garbles "short max" for softmax.

:::def
An **activation function** is the non-linear function $h$ applied to a neuron's pre-activation $a$ to give its post-activation $z=h(a)$. Non-linearity is what lets stacked layers represent more than one linear map.
:::

:::key
- No non-linearity means the MLP collapses to one linear function.
- Sigmoid derivative peaks at 0.25 (at $x=0$); tanh derivative peaks at 1.
- ReLU dead neurons come from zero gradient for negative input.
- Modern FFNs use GELU or SwiGLU.
:::

:::mistakes
- Confusing sigmoid (squashes one number) with softmax (normalises a vector to sum 1).
- Confusing depth (layers) with width (neurons per layer).
- Mixing pre-activation $a$ with post-activation $z=h(a)$.
:::

:::exam
- *"Compare sigmoid, tanh and ReLU: range, derivative, problem."* Use a table.
- *"Write $a_j^{(l)}$ and $z_j^{(l)}$ for a 3-layer MLP with indexed weights."*
:::

## Quick Check

:::q Q1 · Why is the maximum of the sigmoid derivative 0.25?
:::ans
$\sigma'(x)=\sigma(1-\sigma)$ is largest when $\sigma=0.5$, giving $0.5\times0.5=0.25$.
:::
:::

:::q Q2 · What is a "dead" ReLU neuron?
:::ans
One whose pre-activation is negative for essentially all inputs: its output and gradient are both 0, so it stops learning.
:::
:::
