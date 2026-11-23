**Sliding-window attention lets each token attend only to the last $w$ tokens, giving $O(nw)$ cost and a fixed-size rolling cache, while stacked layers still reach about $L\cdot w$ tokens back.** Longformer adds a few global tokens.

## The Problem: Why Should Token 50,000 Attend to Token 3?

Even with small $G$, the cache grows linearly with $n$ and attention costs $O(n^2)$ over a whole document. For language, nearby tokens carry most of the signal (the same locality argument that made CNNs work).

**Attempt 1: cap the history at $w$ tokens.** Token $i$ attends only to $i-w+1,\dots,i$ (Longformer's local attention; Mistral 7B uses $w=4096$). The mask becomes a *band*:

$$M_{ij}=\begin{cases}0 & i-w<j\le i\\ -\infty&\text{otherwise}\end{cases}\qquad\text{cost }O(n\,w)\text{ instead of }O(n^2)$$

:::analogy
Reading a long novel with a sticky note covering everything except the last few pages. Still, each page you read was itself shaped by earlier pages you read before, so some older context leaks through.
:::

**It is not just "forgetting everything older than $w$".** Layers stack. Token $i$ in layer 1 sees $w$ tokens back; those tokens already absorbed *their* predecessors, so at layer 2 token $i$ is indirectly influenced by up to $2w$ tokens back. After $L$ layers the **receptive field** is about $L\cdot w$. The information is diluted with depth, so this is a theoretical reach, not a guarantee of recall.

**The cache win.** Only the last $w$ keys and values can ever be used, so the cache is a **rolling buffer** of fixed size:

```text
slot = t mod w
K_cache[slot], V_cache[slot] = k_t, v_t        # overwrite the oldest entry
attend over all w slots  (positions are tracked separately, e.g. via RoPE)
```

```python
def reach(w, layers):
    return layers * (w - 1)          # each layer reaches w-1 positions back

print("w=4096, 32 layers: about", 32 * 4096, "tokens")
print("w=8, L=4: position 40 is influenced by positions", 40 - reach(8, 4), "to 40")

w, n = 4, 10
band = [[1 if 0 <= i - j < w else 0 for j in range(n)] for i in range(n)]
print("row 9 attends to:", [j for j in range(n) if band[9][j]])
print("cost n*w =", n * w, "vs n^2 =", n * n)
```

```output
w=4096, 32 layers: about 131072 tokens
w=8, L=4: position 40 is influenced by positions 12 to 40
row 9 attends to: [6, 7, 8, 9]
cost n*w = 40 vs n^2 = 100
```

> [!compiler] Longformer's refinement
> Pure local attention cannot gather a document-level summary. Longformer adds a few **global tokens** (for example `[CLS]`, or question tokens) that attend to everything and are attended by everything, plus optionally *dilated* windows. Cost stays linear in $n$.

:::def
**Sliding-window attention** restricts token $i$ to attend to positions $i-w+1,\dots,i$, giving $O(nw)$ time and a rolling KV-cache of fixed size $w$.
:::

:::key
- Band mask; cost $O(nw)$; fixed rolling cache (memory constant in $n$).
- Receptive field grows to about $L\cdot w$ with depth, with diluted information.
- Global tokens (Longformer) restore some long-range access.
:::

:::mistakes
- "Sliding window means the model cannot see past $w$." Stacked layers extend the reach.
:::

:::exam
- *"Which positions can influence position 40 in the last layer if $w=8$, $L=4$?"* Positions 12 to 40.
- *"When would you choose sliding window over GQA, or both?"* Cache size is $h_{kv}\cdot\min(n,w)$, so they combine.
:::

## Quick Check

:::q Q1 · What is the cost of sliding-window attention for sequence length $n$ and window $w$?
:::ans
$O(nw)$.
:::
:::

:::q Q2 · Why is the cache of fixed size?
:::ans
Only the last $w$ keys and values can be used, so older entries are overwritten in a rolling buffer.
:::
:::
