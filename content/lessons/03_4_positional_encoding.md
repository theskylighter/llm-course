<<SVG:positional_encoding_order>>

*Figure: A position vector is added to each token embedding; sinusoidal frequencies encode order without changing vector width.*

**A good positional encoding gives every position a unique vector that is the same in every sequence; the sinusoid appears as the answer to the periodicity of binary numbers.** The professor teaches it as a ladder of failed attempts.

## The Problem: Self-Attention Cannot Tell Order

> [!slides] Motivation
> Self-attention processes all elements of a sequence in parallel without regard for order. *the sun rises in the east* and *rises in the sun the east* give the same bag of words. In natural language, order matters. **Solution:** explicitly add positional information to each token.

> [!lecture] Lecture
> *The man eats the cheese* versus *the cheese eats the man*: only the positions of "man" and "cheese" differ, but the meaning flips. Self-attention alone cannot tell these apart.

The position information is a vector $\mathbf{p}_t$ **of the same size** as the token embedding, and the two are *summed*: $\mathbf{x}_t=\mathbf{e}_t+\mathbf{p}_t$.

> [!lecture] Why sum, not concatenate?
> There is no concrete reason, and you can concatenate if you want. Summing keeps the vector size the same, so fewer parameters are needed. And the position information is added **only once, at the input** of the first encoder and first decoder layer. It is not re-added at layer 2.

## The Ladder of Attempts

**Attempt 1: use the index itself.** Position 0 gives $[0,0,0,0]$, position 1 gives $[1,1,1,1]$, position 3 gives $[3,3,3,3]$.

> [!lecture] Why it fails
> The maximum length is not known in advance. A 3-token sentence ends with $[3,3,3,3]$, while in a 30-token sentence $[3,3,3,3]$ is just the third word. The model cannot tell whether it has reached the end. The values grow without bound, so at test time a 5000-word sequence gives numbers the model never saw.

**Attempt 2: normalise by length.** Divide by the sequence length, so the last position is always $[1,1,1,1]$.

> [!lecture] Why it fails
> The end-of-sequence problem is solved, but now position 1 is $0.33$ in a 3-token sentence and $0.2$ in a 5-token one, though both denote *the first position*. We always assume that **a position should have a unique vector, the same in every sequence.**

**Attempt 3: binary numbers.** $0\to0000,\ 1\to0001,\ 2\to0010,\ 3\to0011,\dots$ Every position is now unique.

> [!lecture] Think about it carefully
> Adding $0/1$ vectors to fractional word embeddings makes little sense. But look at the structure. The least significant bit flips at *every* step, the next bit every 2 steps, the next every 4, the next every 8. **Each bit is a periodic signal with a different period.** And the best way to capture a periodic signal is a sinusoid.

:::analogy
A binary odometer: the last wheel spins fastest, the next one a tenth as fast. The sinusoidal encoding is the same odometer made smooth, with wheels of geometrically spaced speeds.
:::

> [!slides] Sinusoidal positional encoding (Vaswani et al., 2017)
> For $d_{\text{model}}=512$ the encoding is a 512-dimensional vector. With $pos$ the position and $i$ the pair index:
> $$PE_{(pos,\,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad PE_{(pos,\,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
> Small $i$ gives high frequency (changes fast with position). Large $i$ gives low frequency.

```python
import numpy as np

def sinusoidal_pe(max_pos, d):
    pe = np.zeros((max_pos, d))
    pos = np.arange(max_pos)[:, None]
    div = 10000 ** (2 * np.arange(d // 2) / d)
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)
    return pe

np.set_printoptions(precision=4, suppress=True)
print(sinusoidal_pe(3, 4))
```

```output
[[ 0.      1.      0.      1.    ]
 [ 0.8415  0.5403  0.01    1.    ]
 [ 0.9093 -0.4161  0.02    0.9998]]
```

> [!compiler] A worked example you can check by hand ($d_{\text{model}}=4$)
> Here $10000^{2i/4}$ is $1$ for $i=0$ and $100$ for $i=1$, so $PE_{pos}=[\sin pos,\ \cos pos,\ \sin(pos/100),\ \cos(pos/100)]$. Dims 0-1 move quickly (the high-frequency band); dims 2-3 barely move between neighbours.

> [!lecture] Why both frequencies?
> A single sinusoid $\sin(pos)$ has period about 6.28, so positions 0 and 6 get almost the same value. Now add the index $i$. Two **nearby** positions differ in the high-frequency band, and their low-frequency entries are almost identical. Two **far-apart** positions differ in both. So the model can read "near" versus "far" off the encoding.

> [!compiler] Precision on "same value at 0 and 6"
> $\sin 0=0$ but $\sin 6\approx-0.279$, so the collision is approximate, not exact. The full 64-dimensional encodings at positions 0 and 6 are far apart.

## Three Properties of a Good Encoding

> [!lecture] The three properties
> 1. **Monotonicity**: proximity decreases as positions move further apart.
> 2. **Translation invariance**: the proximity of $(x, x+m)$ equals that of $(y, y+m)$. Only the relative offset matters.
> 3. **Symmetry**: proximity of $x$ and $y$ equals proximity of $y$ and $x$.

> [!compiler] Why the sinusoid has translation invariance
> Group dimensions in pairs with $\omega_i=10000^{-2i/d}$. For a shift $k$, the angle-addition formulas show $PE_{pos+k}$ is a fixed linear function (a rotation depending only on $k$) of $PE_{pos}$. Further,
> $$PE_{pos}\!\cdot\!PE_{pos+k}=\sum_{i=0}^{d/2-1}\cos(\omega_i k),$$
> which depends **only on $k$**. It is a sum of cosines, so it decays with $k$ on average but is not strictly monotone.

```python
import numpy as np
def pe(max_pos, d):
    p = np.zeros((max_pos, d)); pos = np.arange(max_pos)[:, None]
    div = 10000 ** (2 * np.arange(d // 2) / d)
    p[:, 0::2] = np.sin(pos / div); p[:, 1::2] = np.cos(pos / div)
    return p
P = pe(64, 32)
print("dot(0,5) vs dot(20,25):", round(float(P[0] @ P[5]), 6), round(float(P[20] @ P[25]), 6))
print("symmetric:", np.isclose(P[3] @ P[9], P[9] @ P[3]))
print("distance from pos 0 to 1, 6, 30:", [round(float(np.linalg.norm(P[0] - P[k])), 2) for k in (1, 6, 30)])
```

```output
dot(0,5) vs dot(20,25): 11.777383 11.777383
symmetric: True
distance from pos 0 to 1, 6, 30: [1.17, 2.87, 4.1]
```

## Learned Positional Embeddings

> [!lecture] Lecture
> Later people said: let's treat the position embedding as a **trainable parameter**. For every position there is a trainable vector, the same across all sequences. Think of a dictionary from position to vector.

> [!slides] Trade-off
> A learned table **cannot generalize past its maximum length**, yet many models use it. The sinusoid is independent of sequence length and can in principle be evaluated at any position.

:::def
**Sinusoidal positional encoding** adds to each token embedding a fixed vector whose $2i$-th and $(2i+1)$-th entries are $\sin$ and $\cos$ of $pos/10000^{2i/d_{\text{model}}}$.
:::

:::key
- Index and normalised index both fail; binary is unique but discrete, and its bit-flips are periodic.
- Sinusoids give unique, bounded, length-independent encodings with a relative-offset structure.
- The encoding is **added once, at the input**.
- Learned tables cannot extend beyond their trained length.
:::

:::mistakes
- "Positions are added at every layer." Only at the input (RoPE is the exception).
- "Sine and cosine are for decoration." Together they make $PE_{pos+k}$ a linear (rotation) function of $PE_{pos}$.
:::

:::exam
- *"Why does the raw index fail as a positional encoding?"* Unbounded, and the same vector means different positions relative to length.
- *"Write the sinusoidal formula and list three properties of a good PE."*
:::

## Quick Check

:::q Q1 · Why does dividing the index by the sequence length fail?
:::ans
The same absolute position (for example the first word) gets different values in sentences of different length, but a position should have the same unique vector in every sequence.
:::
:::

:::q Q2 · In the sinusoid, which dimensions change fastest with position?
:::ans
Small $i$ (the first dimensions): high frequency.
:::
:::

:::q Q3 · Why is the dot product $PE_{pos}\cdot PE_{pos+k}$ useful?
:::ans
It equals $\sum_i\cos(\omega_ik)$, which depends only on the offset $k$: translation invariance and symmetry.
:::
:::
