**Long context fails in three ways: positions never seen in training, memory that grows with length, and uneven use of the middle of the text; ALiBi, Position Interpolation and ring attention address the first two.** ALiBi adds a distance penalty to scores; PI squeezes new positions into the trained range.

## The Problem: "Can It Read a Whole Book?"

First attempt: raise the context limit. But positions never seen in training make the model break, and attention memory explodes.

> [!compiler] Three separate problems (the compiler's organisation)
> **(i) Position:** a model trained at length $L$ has never seen position $m>L$. **(ii) Memory and compute:** attention is $O(n^2)$ and the KV cache grows linearly (lesson 4.1: 524,288 bytes per token for that example, so **128k tokens = 64 GiB**, since $524{,}288\times131{,}072=2^{36}$). **(iii) Use:** even when the text fits, the model uses it unevenly (*lost in the middle*).

## ALiBi and the Exam Traps

> [!compiler] EndSem Q0(a): "ALiBi adds positional embeddings to token embeddings." False.
> ALiBi adds **no embeddings at all**. It adds a fixed, non-learned **penalty to the attention scores** proportional to the query-key distance: $\text{score}_{ij}=q_i^{\top}k_j-m_h\,(i-j)$, with a head-specific slope $m_h$. For $h=8$ heads the slopes are the geometric sequence $\tfrac12,\tfrac14,\dots,\tfrac1{256}$.

> [!compiler] EndSem Q0(c): "RoPE represents positions as complex numbers and tokens as pure rotations." False.
> It is the wrong way round: positions are rotations, tokens (the $q$, $k$ vectors) are the complex numbers being rotated. See lesson 3.10.

```python
slopes = [2 ** -(i + 1) for i in range(8)]
print("slopes for 8 heads:", slopes)
m = 2 ** -4                      # the slope 1/16 head
print("penalty at distance 32 for slope 1/16:", m * 32)
for h, mh in zip(range(3), slopes):
    print(f"head {h}: scores to tokens 0, 4, 16 positions back lose", [mh * d for d in (0, 4, 16)])
```

```output
slopes for 8 heads: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
penalty at distance 32 for slope 1/16: 2.0
head 0: scores to tokens 0, 4, 16 positions back lose [0.0, 2.0, 8.0]
head 1: scores to tokens 0, 4, 16 positions back lose [0.0, 1.0, 4.0]
head 2: scores to tokens 0, 4, 16 positions back lose [0.0, 0.5, 2.0]
```

## Extending RoPE: Position Interpolation

RoPE angles are $m\theta_i$ with $\theta_i=b^{-2i/d}$, $b=10000$. At $m>L$ the angles are out of the trained range and quality collapses. **Position Interpolation (PI):** instead of extrapolating, **squeeze** new positions into the trained range, $m\mapsto m\cdot\frac{L}{L'}$.

:::analogy
A school photo: if the class grows from 30 to 120 students you do not make the frame four times larger; you shrink everyone to fit the same frame.
:::

```python
L, Lnew = 4096, 16384
s = L / Lnew
print("scale s =", s, "-> position 10000 is treated as", 10000 * s)
print("trained at 8192, extended to 32768: position 20000 maps to", 20000 * 8192 / 32768)

def rope_angle(m, i, d, base=10000, scale=1.0):
    return (m * scale) * base ** (-2 * i / d)
print("angle at m=10000, i=0 with PI:", rope_angle(10000, 0, 64, scale=s))
```

```output
scale s = 0.25 -> position 10000 is treated as 2500.0
trained at 8192, extended to 32768: position 20000 maps to 5000.0
angle at m=10000, i=0 with PI: 2500.0
```

> [!compiler] Attention layouts and compression
> **Sliding window and sinks** (lesson 4.3; StreamingLLM keeps the first few "sink" tokens plus a window): constant memory, but the model cannot recall evicted text. **Sparse or dilated attention** (**LongNet**) approaches linear cost. **Ring attention** shards the sequence across devices, reducing **per-device memory**; the FLOPs stay $O(n^2)$. NTK-aware scaling and YaRN adjust the frequencies instead of scaling uniformly.

:::def
**Position Interpolation** extends the context of a RoPE model from $L$ to $L'$ by rescaling position indices $m\mapsto mL/L'$ so that all angles stay within the trained range, followed by brief fine-tuning.
:::

:::key
- Three problems: position, memory and compute, use.
- ALiBi = linear distance penalty on scores, slope per head, no embeddings. RoPE = rotate $q,k$.
- PI interpolates (squeezes), it does not extrapolate; ring attention reduces per-device memory, not FLOPs.
:::

:::mistakes
- "ALiBi is a positional embedding." It is a distance penalty.
- "Position Interpolation extrapolates." It interpolates.
- "Ring attention reduces FLOPs." It reduces per-device memory.
:::

:::exam
- *(EndSem Q0(a), Q0(c), 2 marks each)* the two True/False statements above.
- *"ALiBi with 8 heads: penalty for the slope 1/16 head at distance 32."* $2$.
:::

## Quick Check

:::q Q1 · Trained at 4096, extended to 16384 with PI: how is position 10,000 treated?
:::ans
$s=0.25$, so as position 2,500.
:::
:::

:::q Q2 · Why does ALiBi extrapolate to longer sequences better than a learned table?
:::ans
The penalty is a simple function of distance with no learned per-position parameters, so it is defined at any distance.
:::
:::
