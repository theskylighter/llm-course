**Multi-query attention shares one key/value head across all query heads, and grouped-query attention shares one per group; both shrink the KV-cache by $h/G$ at little quality cost.** Only $W_K$ and $W_V$ get smaller.

## The Problem: the Cache Formula Has a Knob

The cache formula has an obvious knob: $h_{kv}$, the number of distinct K/V heads. In standard multi-head attention (MHA) it equals the number of query heads $h$. **Does each query head really need its own private key and value?**

**Attempt 1: Multi-Query Attention (MQA).** Keep $h$ query heads but share **one** K and one V head across all of them (Shazeer, 2019). The cache shrinks by a factor of $h$. For the 7B example: 2 GiB falls to 64 MiB per sequence.

**Why it is not the end of the story.** One key/value head forces every query head to retrieve from the same representation. Quality drops and training can become unstable.

**Attempt 2: Grouped-Query Attention (GQA).** Split the $h$ query heads into $G$ groups; each group shares one K/V head (Ainslie et al., 2023).

$$G=h:\ \text{MHA}\qquad 1<G<h:\ \text{GQA}\qquad G=1:\ \text{MQA}$$

:::analogy
Eight students (query heads) each need a reference book. MHA gives each their own copy; MQA makes all eight share one book; GQA makes pairs share a book.
:::

The GQA paper also shows that you need not train from scratch: **up-train** an existing MHA checkpoint by *mean-pooling* the K and V heads inside each group, then continue training briefly.

## Parameter Counts (MidSem Q4(d))

> [!compiler] Worked example
> Setting: $h=6$ query heads, $d_{\text{model}}=4$, $d_k=d_v=3$. Each projection of one head is $4\times3=12$ parameters. Biases ignored. $W_O\in\mathbb{R}^{(hd_v)\times d_{\text{model}}}$ has $18\cdot4=72$.
>
> | Variant | $W_Q$ | $W_K$ | $W_V$ | $W_O$ | Total |
> |---|---|---|---|---|---|
> | (i) MHA, 6 K/V heads | 72 | 72 | 72 | 72 | **288** |
> | (ii) MQA, 1 K/V head | 72 | 12 | 12 | 72 | **168** |
> | (iii) GQA, group size 2, so 3 K/V heads | 72 | 36 | 36 | 72 | **216** |
>
> **Ambiguity to flag in the exam:** "group size 2" most naturally means 2 query heads per group (3 groups). If the examiner meant *2 groups*, the answer is $72+24+24+72=192$. Write your reading next to the number.

```python
def params(d_model, h, dk, kv_heads):
    wq = d_model * dk * h
    wk = wv = d_model * dk * kv_heads
    wo = h * dk * d_model
    return wq + wk + wv + wo

print("MHA:", params(4, 6, 3, 6))
print("MQA:", params(4, 6, 3, 1))
print("GQA (3 kv heads):", params(4, 6, 3, 3))
print("GQA (2 kv heads):", params(4, 6, 3, 2))
print("rule d*dk*(h+2G) + h*dk*d, d=8,h=8,dk=4,G=2:", 8 * 4 * (8 + 2 * 2) + 8 * 4 * 8)
```

```output
MHA: 288
MQA: 168
GQA (3 kv heads): 216
GQA (2 kv heads): 192
rule d*dk*(h+2G) + h*dk*d, d=8,h=8,dk=4,G=2: 640
```

> [!compiler] What MQA and GQA do and do not buy
> They cut the **cache size** and the **memory traffic per decoded token** by $h/G$, so decode gets faster and batches get larger. They cut few FLOPs. In training they are not a speed-up in the way FlashAttention is.

:::def
**Grouped-query attention** uses $h$ query heads but only $G$ key/value heads, each shared by $h/G$ query heads. $G=h$ is MHA and $G=1$ is MQA.
:::

:::key
- Cache shrinks by $h/G$. Only $W_K,W_V$ shrink; $W_Q$ and $W_O$ do not.
- MQA risks quality; GQA is the compromise; GQA can be up-trained by mean-pooling heads.
- Total parameters: $d\,d_k(h+2G)+h\,d_k\,d$.
:::

:::mistakes
- "MQA/GQA reduce attention FLOPs a lot." They mainly reduce the cache and its bandwidth.
:::

:::exam
- *Parameter count for MHA, MQA, GQA:* show the table, flag the "group size" ambiguity.
:::

## Quick Check

:::q Q1 · For $h=32$ and $G=8$, by what factor does the KV-cache shrink?
:::ans
$h/G=4$.
:::
:::

:::q Q2 · Which projection matrices shrink under MQA?
:::ans
Only $W_K$ and $W_V$.
:::
:::
