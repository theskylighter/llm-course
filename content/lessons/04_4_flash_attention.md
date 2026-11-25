<<SVG:flash_attention_tiles_online_softmax>>

*Figure: FlashAttention keeps tiles on-chip and updates softmax incrementally, avoiding full score/probability matrices in HBM.*

**FlashAttention returns exactly the same output as standard attention but never stores the $n\times n$ matrix in slow GPU memory; it tiles the computation and uses an online softmax.** The gain is in memory traffic, not FLOPs.

## The Problem: the Bottleneck Is Memory Traffic, Not Arithmetic

A GPU has two memories: **HBM** (large, tens of GB, relatively slow) and on-chip **SRAM** (a few tens of MB, roughly an order of magnitude faster). Standard attention:

```text
S = Q @ K.T / sqrt(d)        # write n x n matrix to HBM
P = softmax(S)               # read S, write P  (n x n again)
O = P @ V                    # read P, write O
```

The arithmetic is cheap on modern GPUs; the **round trips of the $n\times n$ matrices through HBM** dominate. Why not fuse the three lines? Softmax needs the maximum and the sum over a **whole row** before it can output anything. FlashAttention *tiles* the computation and makes softmax computable **incrementally**.

:::analogy
A cook with a tiny countertop (SRAM) and a big pantry (HBM). The slow way: carry every ingredient to the counter and back for each step. The fast way: bring a small batch, do all the steps on it, and only then return.
:::

## Online Softmax

Process the keys in blocks. For one query row keep: the row maximum $m$, the normaliser $\ell$, and an un-normalised output accumulator $\mathbf{o}$. When a new block with scores $\mathbf{s}^{(b)}$ and values $V^{(b)}$ arrives:

$$m'=\max\big(m,\max\mathbf{s}^{(b)}\big),\quad \ell'=e^{m-m'}\ell+\sum_j e^{s^{(b)}_j-m'},\quad \mathbf{o}'=e^{m-m'}\mathbf{o}+\sum_j e^{s^{(b)}_j-m'}\mathbf{v}^{(b)}_j$$

At the end the output is $\mathbf{o}/\ell$. The factor $e^{m-m'}$ rescales everything accumulated so far to the new maximum, so the result is identical to a full softmax.

```python
import numpy as np
rng = np.random.default_rng(0)
s = rng.normal(size=12) * 3          # one row of scores
V = rng.normal(size=(12, 4))
full = (np.exp(s - s.max()) / np.exp(s - s.max()).sum()) @ V

m, l, o = -np.inf, 0.0, np.zeros(4)
for b in range(0, 12, 4):            # three blocks of four
    sb, Vb = s[b:b + 4], V[b:b + 4]
    m_new = max(m, sb.max())
    scale = np.exp(m - m_new) if np.isfinite(m) else 0.0
    p = np.exp(sb - m_new)
    l = scale * l + p.sum()
    o = scale * o + p @ Vb
    m = m_new
online = o / l
print("max difference:", float(np.abs(full - online).max()) < 1e-12)
```

```output
max difference: True
```

## What Exactly Improved

- **Exact**, not approximate: the output equals standard attention.
- **FLOPs unchanged**, still $O(n^2d)$. The gain is in **HBM accesses**.
- **Memory $O(n)$** instead of $O(n^2)$, because full $S$ and $P$ are never stored.
- **Backward pass:** recomputes the tiles from $Q,K,V$ and the saved log-sum-exp $L_i=m+\log\ell$. More FLOPs, far less traffic, a net win (recomputation, as in gradient checkpointing).
- Helps **training and inference**, but is a different axis from the KV-cache. Single-token decode gains little from tiling a $1\times t$ row, which motivated *Flash-Decoding*.

> [!compiler] Exam pattern: MidSem 2024-25 MCQ Q5 (more than one option may be correct)
> (A) KV-cache reuses past keys/values: **true**. (B) FlashAttention cuts HBM-SRAM I/O: **true**. (D) Both reduce memory "by sharing attention weights across layers": **false**. **(C) is the trap**: FlashAttention helps training and inference, but it does **not** reduce *computational* complexity (FLOPs stay $O(n^2d)$). No answer key is in the repo; mention the FLOPs-versus-I/O distinction in any written version.

:::def
**FlashAttention** is an exact attention algorithm that computes attention in tiles held in on-chip SRAM, using an online softmax and recomputation in the backward pass, to reduce HBM reads and writes and memory from $O(n^2)$ to $O(n)$.
:::

:::key
- Exact, same FLOPs, far fewer HBM accesses, $O(n)$ memory.
- Online softmax keeps $m$, $\ell$, $\mathbf{o}$ and rescales by $e^{m-m'}$.
- KV-cache and FlashAttention are different axes: decoding cost versus attention-compute memory traffic.
:::

:::mistakes
- "FlashAttention is an approximation (like Linformer)." It is exact.
- "FlashAttention reduces FLOPs." It reduces memory traffic and memory use.
:::

:::exam
- *"Explain how FlashAttention reduces memory I/O."* Tiling, SRAM, online softmax, recomputation.
- *The MCQ above:* decide each option and justify C.
:::

## Quick Check

:::q Q1 · What are the three running quantities of the online softmax?
:::ans
The running maximum $m$, the normaliser $\ell$, and the un-normalised output accumulator $\mathbf{o}$.
:::
:::

:::q Q2 · Why does the backward pass recompute rather than store $P$?
:::ans
Storing the $n\times n$ matrix would need $O(n^2)$ memory and traffic; recomputing tiles from $Q,K,V$ and the saved log-sum-exp costs extra FLOPs but far less memory traffic.
:::
:::
