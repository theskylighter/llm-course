**In a decoder, the keys and values of past tokens never change, so we store them and compute only the new token's projections at each step.** The cost of generation falls from quadratic to linear in the number of weight-matrix multiplications, and the price is a large memory cache.

> [!compiler] Source note
> There is no NPTEL transcript for this chapter, so there is no professor's voice here. Every callout is a Compiler's Note, built from the cited papers and checked numerically. It has real exam relevance: the 2024-25 mid-semester paper has one MCQ and four sub-parts on this chapter.

## The Problem: Generation Recomputes the Past

Chapter 3 ended with a clean story: attention costs $O(n^2d)$, but it is parallel. That story is about **training**. At **generation** time a decoder produces one token, appends it, and runs again. Naively:

```text
tokens = prompt
repeat until EOS:
    H = transformer(tokens)            # full forward pass over ALL t tokens
    next = sample(softmax(H[-1] @ W_vocab))
    tokens.append(next)
```

At step $t$ this recomputes $\mathbf{k}_1,\dots,\mathbf{k}_{t-1}$ and $\mathbf{v}_1,\dots,\mathbf{v}_{t-1}$, which we already computed one step ago. Is anything different this time? Think about it carefully.

> [!compiler] The key observation: the causal mask makes the past immutable
> In a decoder, position $i$ attends only to positions $\le i$ (lesson 3.8). So the hidden state, key and value at position $i$ in **every layer** depend only on tokens $1..i$. Appending token $t+1$ changes nothing about positions $1..t$. Their $K$ and $V$ are *constants* from now on. (A bidirectional encoder has no such property, which is why BERT has no KV-cache.)

$$\mathbf{a}_t=\operatorname{softmax}\!\Big(\frac{\mathbf{q}_t\,K_{1:t}^{\top}}{\sqrt{d_k}}\Big)V_{1:t},\qquad K_{1:t}=[K_{1:t-1};\mathbf{k}_t]$$

:::analogy
Taking minutes in a meeting. When a new person speaks you do not rewrite all earlier minutes from scratch; you keep the notes and append one line.
:::

> [!compiler] Why we cache K and V but not Q
> $\mathbf{q}_i$ is used exactly once, to produce the output at position $i$. No later token ever needs an old query. Keys and values are needed by *every* later token.

## Counting the Saving (MidSem Q4(b))

The exam asks for the number of matrix multiplications for a single decoder layer (one head, masked attention plus MLP, no layer norm) generating $n$ tokens, with and without the cache. State your convention. Ours: **one (row vector times weight matrix) product = one multiplication**, only the last position's output is needed.

Without cache at step $t$: $\mathbf{k},\mathbf{v}$ for all $t$ tokens ($2t$), $\mathbf{q}$ for the last (1), scores (1), weighted sum (1), $W_O$ (1), MLP (2): $2t+6$. With cache: $\mathbf{q},\mathbf{k},\mathbf{v}$ (3) + scores (1) + sum (1) + $W_O$ (1) + MLP (2) $=8$.

$$\sum_{t=1}^{n}(2t+6)=n^2+7n=O(n^2),\qquad 8n=O(n)$$

```python
def without_cache(n):
    return sum(2 * t + 6 for t in range(1, n + 1))

def with_cache(n):
    return 8 * n

for n in (10, 100, 1000):
    print(n, without_cache(n), with_cache(n))
print("closed form n^2 + 7n at n=10:", 10 ** 2 + 7 * 10)
```

```output
10 170 80
100 10700 800
1000 1007000 8000
closed form n^2 + 7n at n=10: 170
```

> [!compiler] Precision on "the cache makes attention O(n)"
> The cache removes the **recomputation of projections**. The attention step itself still reads all $t$ cached keys, so it costs $O(td)$ per token and $O(n^2d)$ in total. What fell from quadratic to linear is the *number of weight-matrix multiplications*, not the whole algorithm.

## The New Problem: the Cache Is a Memory Monster

$$\text{KV bytes}=2\;(K,V)\times L_{\text{layers}}\times h_{kv}\times d_{\text{head}}\times n\times B\times\text{bytes}$$

```python
layers, heads, dhead, nbytes = 32, 32, 128, 2          # a 7B-class model in fp16
per_token = 2 * layers * heads * dhead * nbytes
print("bytes per token:", per_token, "=", per_token / 2 ** 20, "MiB")
ctx = 4096
print("one 4096-token sequence:", per_token * ctx / 2 ** 30, "GiB")
print("batch of 16:", 16 * per_token * ctx / 2 ** 30, "GiB")
```

```output
bytes per token: 524288 = 0.5 MiB
one 4096-token sequence: 2.0 GiB
batch of 16: 32.0 GiB
```

> [!compiler] Why this is the real bottleneck
> **Prefill** (processing the prompt) is a large, parallel matrix multiply: compute-bound. **Decode** produces one token per pass, so for each token the GPU must stream all the weights *and* the whole KV-cache from memory to do a tiny amount of arithmetic. Decode is **memory-bandwidth bound**. Contrast with an RNN: its state is $O(1)$ in sequence length; the Transformer's "state" is the cache, $O(n)$.

:::def
A **KV-cache** stores the key and value vectors of all previous tokens in every layer during autoregressive decoding, so that each step computes projections only for the newest token.
:::

:::key
- Decoder K and V of past tokens never change (causal mask), so cache them. Queries are never cached.
- Multiplication count per layer: $O(n^2)\to O(n)$; the attention reads are still $O(n^2d)$ overall.
- Cache size: $2\,L\,h_{kv}\,d_{\text{head}}$ values per token; decode is memory-bandwidth bound.
:::

:::mistakes
- "KV-cache speeds up BERT." No: an encoder has no autoregressive loop.
- "KV-cache is used in training." No: training processes all positions in parallel.
- "Cache Q too." A query is used once.
:::

:::exam
- *Count the multiplications* (MidSem Q4(b)): state your convention; answer $n^2+7n$ vs $8n$.
- *MCQ trap:* a statement that KV-cache "reduces the complexity of attention computation": the cache avoids recomputation but attention still reads all keys.
:::

## Quick Check

:::q Q1 · Why can we cache K and V in a decoder but not in a BERT encoder?
:::ans
Decoder states depend only on earlier tokens, so appending a token leaves old K and V unchanged. In an encoder every token attends to all others, so there is no autoregressive loop to reuse anything.
:::
:::

:::q Q2 · Cache size per token for $L=80$, $h_{kv}=8$, $d_{\text{head}}=128$, fp16?
:::ans
$2\cdot80\cdot8\cdot128\cdot2=327{,}680$ bytes per token, about 1.25 GiB for 4096 tokens.
:::
:::
