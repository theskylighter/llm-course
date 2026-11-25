<<SVG:lora_low_rank_update>>

*Figure: LoRA adds a trainable low-rank update in parallel with frozen weights, then merges that update after training.*

**Adapters insert small bottleneck modules in series, adding inference latency; LoRA learns a low-rank update $\Delta W=BA$ in parallel that can be merged into the weights, adding none.** LoRA starts with $B=0$ so the model is initially unchanged.

## The Problem: Soft Prompts Are Limited, Full Tuning Is Heavy

We want to modify the model's *internal* computation cheaply.

## Adapters

> [!slides] Lec_29: Houlsby et al. 2019
> A small **bottleneck** module inside every Transformer layer: reduce the $d$-dimensional feature to an $m$-dimensional vector and project back. The slide's arithmetic with $d=1024$, $m=24$: a full $1024\times1024$ matrix has $1{,}048{,}576$ parameters; the down- and up-projection $2\times(1024\times24)=49{,}152$ (about 4.7%). **$m$ sets the parameter-versus-performance trade-off.** About the same as fully fine-tuned BERT while training **3.6%** of the parameters. **Cost: inference overhead**, since extra layers in every block add latency.

```text
def adapter(h):                       # h: (L, d)
    z = nonlinearity(h @ W_down)      # W_down: (d, m),  m << d
    return h + z @ W_up               # W_up: (m, d); residual keeps the layer near identity
```

> [!compiler] Placement detail
> In Houlsby et al. there are **two** adapters per block (after attention and after the FFN). Adapters **add modules in series** (so they add latency), while LoRA adds a parallel update that can be folded into the weights.

## LoRA: Low-Rank Adaptation

Li et al. (2018): a model can be optimized in a **low-dimensional subspace**. The smallest dimension reaching 90% of full performance is the task's **intrinsic dimensionality**, and it turns out to be small, and smaller for larger pre-trained models.

> [!slides] The method (Hu et al. 2022)
> For a frozen weight $W\in\mathbb{R}^{d\times k}$, learn two low-rank matrices applied to the self-attention weights: $\Delta W=BA$ with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r\ll\min(d,k)$.
> **Initialization:** $B=0$ so that $\Delta W=BA=0$ at the start, **preserving the behaviour of the original model**; $A$ Gaussian so its values are neither too large nor biased in any direction.

$$h=Wx+\frac{\alpha}{r}\,BAx,\qquad \text{trainable params per matrix}=r(d+k)\ \text{versus}\ dk$$

:::analogy
Instead of repainting a whole house, you hang a thin transparent overlay on the wall. The overlay (two small matrices multiplied) adds the new look, and after approval you can paint it permanently onto the wall (merge).
:::

```python
import numpy as np
rng = np.random.default_rng(0)
d = k = 64
r, alpha = 4, 8
W = rng.normal(size=(d, k))
A = rng.normal(size=(r, k)) * 0.1
B = np.zeros((d, r))                       # B = 0: the model starts unchanged
x = rng.normal(size=k)
print("starts identical:", np.allclose(W @ x + (alpha / r) * B @ (A @ x), W @ x))
B = rng.normal(size=(d, r)) * 0.1          # pretend training changed B
h_lora = W @ x + (alpha / r) * B @ (A @ x)
W_merged = W + (alpha / r) * B @ A         # merge: zero extra latency
print("merged weights give the same output:", np.allclose(W_merged @ x, h_lora))
print("trainable:", r * (d + k), "of", d * k, f"= {r * (d + k) / (d * k):.1%}")
print("d=k=4096, r=8:", 8 * 8192, "of", 4096 * 4096, f"= {8 * 8192 / 4096 ** 2:.2%}")
```

```output
starts identical: True
merged weights give the same output: True
trainable: 512 of 4096 = 12.5%
d=k=4096, r=8: 65536 of 16777216 = 0.39%
```

> [!compiler] The three knobs
> **Rank $r$:** capacity of the update; small is enough. **$\alpha$:** the scale $\alpha/r$ on the update (from the LoRA paper; the slides do not define it). **Target modules:** the slide says "applied to the self-attention weights"; the paper's finding is that adapting $W_q$ and $W_v$ beats spending the whole budget on one matrix. For GPT-3 175B, rank 4 on $W_q$ and $W_v$ in all 96 layers is about 18.9M parameters, about 0.01% of 175B.

> [!slides] Extensions listed on the slide
> **QLoRA** (gradients through a 4-bit quantized model), **LongLoRA** (sparse local attention during fine-tuning), **LoRA+** (different learning rates for $A$ and $B$), **DyLoRA** (selects the rank without multiple training runs).

:::def
**LoRA** freezes the pre-trained weight $W$ and learns a low-rank update $\Delta W=BA$ with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r\ll\min(d,k)$, initialised with $B=0$; the update can be merged into $W$ after training.
:::

:::key
- Adapters: bottleneck $d\to m\to d$ in series, inference latency.
- LoRA: $\Delta W=BA$, $B=0$ init, parameters $r(d+k)$, mergeable, no latency.
- Rank, $\alpha$ and target modules are the knobs.
:::

:::mistakes
- "Adapters and LoRA both add latency." LoRA's $BA$ merges into $W$.
- "LoRA starts random, so the model is perturbed." $B=0$, so $\Delta W=0$ at step 0.
:::

:::exam
- *"Why is $B$ initialised to zero and $A$ Gaussian?"*
- *Numerical:* $d=k=4096$, $r=8$: 65,536 trainable versus 16.8M, 0.39% per matrix.
:::

## Quick Check

:::q Q1 · Parameters of a LoRA update for $W\in\mathbb{R}^{1024\times1024}$ with $r=8$?
:::ans
$r(d+k)=8\times2048=16{,}384$, versus $1{,}048{,}576$ for the full matrix (about 1.6%).
:::
:::

:::q Q2 · Why does LoRA add no inference latency while adapters do?
:::ans
$BA$ can be added into $W$ after training, leaving the same computation as the original model; adapters are extra layers executed in series.
:::
:::
