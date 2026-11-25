<<SVG:transformer_ffn_residual_block>>

*Figure: Attention passes information between tokens; the position-wise FFN processes it, with residual paths supporting gradients.*

**Self-attention only moves information around linearly; the position-wise feed-forward network adds non-linearity and acts as the model's memory, while residual connections keep gradients alive.** A Transformer block is attention plus FFN, each wrapped in add and norm.

## The Problem: Attention Is (Almost) Linear

> [!lecture] Self-attention is linear... really?
> Write the output of one head: $\mathbf{a}_\ell=\sum_t \alpha_{\ell t}\,W_V\mathbf{h}_t=W_V\sum_t\alpha_{\ell t}\mathbf{h}_t$. This is a weighted sum of the hidden states, then a linear projection. The *weights* $\alpha$ come out of a non-linear process (the softmax), but the **output is linear in the hidden states**. And we always want to add more non-linearity into the network.

> [!slides] Position-wise feed-forward network
> Alternate self-attention with a non-linearity: two linear transformations with a ReLU in between, applied to **each position separately**, same weights at every position but different weights in every layer.
> $$\operatorname{FFN}(\mathbf{x})=\max(0,\ \mathbf{x}W_1+\mathbf{b}_1)\,W_2+\mathbf{b}_2,\qquad W_1\in\mathbb{R}^{d_{\text{model}}\times d_{ff}},\ W_2\in\mathbb{R}^{d_{ff}\times d_{\text{model}}}$$
> In the paper $d_{\text{model}}=512$ and $d_{ff}=2048$ (**expansion factor 4**).

> [!lecture] Intuition: message passing versus processing
> Self-attention exchanges information between tokens; you can think of it as **fetching from memory** or **message passing**. The actual **processing** happens in the feed-forward network. Fetch from memory, process, fetch again, process again. The FFN is also considered the **internal memory** of the Transformer: look at how many parameters sit in $W_1$ and $W_2$. Why project **up** and then down? When you down-project first you lose information.

:::analogy
Attention is a meeting where everyone shares notes. The FFN is each person going back to their desk to think about what they heard.
:::

## Residual Connections

> [!lecture] Lecture
> Let $f$ be a layer. Instead of feeding $f(x)$ to the next layer, feed $f(x)+x$. This helps with the vanishing gradient problem because now there's a direct connection. It also helps convergence. We add one residual connection across self-attention and another across the feed-forward network.

$$\mathbf{y}=x+f(x)\ \Rightarrow\ \frac{\partial \mathbf{y}}{\partial x}=I+\frac{\partial f}{\partial x}$$

The identity term means the gradient always has a path with factor 1, whatever $f$ does.

## Where the Parameters Are

> [!compiler] Parameters per layer (no biases, $d=d_{\text{model}}$)
> Attention: $W_Q,W_K,W_V,W_O\to 4d^2$. FFN: $2\,d\,d_{ff}=8d^2$ when $d_{ff}=4d$. So an **encoder layer is about $12d^2$** and a decoder layer about $16d^2$ (one extra attention). The FFN holds about two thirds of every layer's parameters.

> [!compiler] Correction: "Transformer large has 12 layers"
> Lecture 15 says base = 6 layers, large = 12. In Vaswani et al. Table 3 *both* configurations have $N=6$ layers. **Base:** $d_{\text{model}}=512$, $d_{ff}=2048$, $h=8$. **Big:** $d_{\text{model}}=1024$, $d_{ff}=4096$, $h=16$. The 12-layer figure matches BERT-base.

```python
d, dff = 512, 2048
attn = 4 * d * d
ffn = 2 * d * dff
print("attention params:", attn)
print("ffn params      :", ffn)
print("encoder layer   :", attn + ffn, "= 12 d^2 =", 12 * d * d)
print("ffn share       :", round(ffn / (attn + ffn), 3))
print("decoder layer   :", 2 * attn + ffn, "= 16 d^2 =", 16 * d * d)
```

```output
attention params: 1048576
ffn params      : 2097152
encoder layer   : 3145728 = 12 d^2 = 3145728
ffn share       : 0.667
decoder layer   : 4194304 = 16 d^2 = 4194304
```

:::def
The **position-wise feed-forward network** is $\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$, applied independently to each position with the same weights, expanding to $d_{ff}$ (typically $4d_{\text{model}}$) and back.
:::

:::key
- Attention mixes tokens; the FFN processes each token and holds most parameters.
- Residual: $y=x+f(x)$, gradient $I+\partial f/\partial x$.
- Encoder layer $\approx12d^2$; decoder layer $\approx16d^2$.
- Modern FFNs use GELU or SwiGLU instead of ReLU.
:::

:::mistakes
- "Both base and large Transformer have different depths (6 and 12)." In the original paper both have $N=6$.
- Forgetting the FFN is applied per position, with shared weights across positions.
:::

:::exam
- *"Why is a feed-forward network needed after self-attention?"* Non-linearity and processing.
- *"Compute parameters of one encoder layer for $d=512$."* $12d^2=3{,}145{,}728$.
:::

> **See it live.** See the MLP block and the residual connection in the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (GPT-2 small, runs in your browser). Full walkthrough in [Lesson 5.5](/05_5_decoder_only).

## Quick Check

:::q Q1 · Why project up to $d_{ff}$ first and then down?
:::ans
Going to a higher-dimensional space first avoids the information loss that a down-projection first would cause.
:::
:::

:::q Q2 · What is $\partial y/\partial x$ for a residual block $y=x+f(x)$?
:::ans
$I+\partial f/\partial x$: the identity term gives the gradient a direct path.
:::
:::
