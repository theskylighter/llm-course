**The residual stream is a shared sum to which attention and FFN blocks write; QK circuits decide what a head reads, OV circuits what it writes; patching gives causal evidence and probes need control tasks.** Patchscopes generalize the logit lens.

## The Problem: "Why Did It Say That?"

First attempt: look at the attention map. It is only part of the story. Fix: treat the network as a computation to reverse engineer.

> [!slides] Opening (Lec_31)
> **Why do we need such a formulation?** By decomposing the Transformer into simpler components, like the **query-key circuit $W_{QK}$** and the **output-value circuit $W_{OV}$**, we can better understand the information flow within Transformer-based LLMs.
> — *Lec_31 slides, Tanmoy Chakraborty*

> [!slides] The residual stream formulation (Lec_31)
> Each input embedding is **updated through vector additions** from the attention and feed-forward blocks, producing **residual stream states**. The final-layer state is projected into vocabulary space by the **unembedding matrix $W_U\in\mathbb{R}^{d\times|\mathcal V|}$** and normalized by softmax. **QK circuits read from the residual stream, OV circuits write to it.**

:::analogy
The residual stream is a shared whiteboard. Each layer's heads and FFNs read a few lines, then write their contribution on the board; the final answer is read off the board.
:::

> [!compiler] EndSem Q5(b): QK and OV circuits (2 marks)
> $W_{QK}^{h}=W_Q^{h}W_K^{h\top}$ is the bilinear form that scores a pair (query token, key token): it decides **which positions the head attends to**, i.e. **what it reads**. $W_{OV}^{h}=W_V^{h}W_O^{h}$ maps an attended token's residual to the vector the head **writes** into the destination's residual stream.

## Direct Logit Attribution

Ignoring the final layer norm, $\text{logits}=W_U\,x^{L}=W_U\big(x^{0}+\sum_{l}\text{Attn}^{l}+\sum_{l}\text{FFN}^{l}\big)$. Because it is linear, the logit of token $t$ is a **sum of per-component terms** $c_k=(W_U)_{:,t}\cdot o_k$.

```python
import numpy as np
rng = np.random.default_rng(0)
d, V = 8, 5
WU = rng.normal(size=(d, V))
parts = {"embedding": rng.normal(size=d), "attn L1": rng.normal(size=d),
         "ffn L1": rng.normal(size=d), "attn L2": rng.normal(size=d)}
resid = sum(parts.values())
tok = 3
total = resid @ WU[:, tok]
contrib = {k: float(v @ WU[:, tok]) for k, v in parts.items()}
print({k: round(c, 2) for k, c in contrib.items()})
print("sum of contributions equals the logit:", np.isclose(sum(contrib.values()), total))
```

```output
{'embedding': 1.39, 'attn L1': -2.0, 'ffn L1': 0.48, 'attn L2': 2.21}
sum of contributions equals the logit: True
```

## Activation Patching

> [!slides] Circuits, workflow, patching (Lec_32)
> A **circuit** is a computational subgraph of the network. Example: the **indirect object identification** circuit in GPT-2 small. **Activation patching:** run a clean and a corrupted prompt; copy one component's activation from the clean run into the corrupted run and see how much the behavior is restored.

```python
p_clean, p_corrupt, p_patched = 0.80, 0.10, 0.55
effect = (p_patched - p_corrupt) / (p_clean - p_corrupt)
print("normalized restoration:", round(effect, 2))
print("EndSem-style exercise (0.6, 0.9, 0.2):", round((0.6 - 0.2) / (0.9 - 0.2), 2))
```

```output
normalized restoration: 0.64
EndSem-style exercise (0.6, 0.9, 0.2): 0.57
```

> [!slides] Induction heads (Lec_32, Olsson et al.)
> An **induction head** looks back for an earlier occurrence of the current token $A$, finds the token $B$ that followed it, and predicts $B$: the pattern $[A][B]\dots[A]\to[B]$. It is implemented by **two heads**: a **previous-token head** and the **induction head** itself.

## Probing, Control Tasks, Logit Lens, Patchscopes

> [!slides] Probing and control tasks (Lec_32; EndSem Q5(a))
> A **probing classifier** $g:f^{l}(x)\to z$ maps intermediate representations to a property $z$ (for example part of speech). If it can predict the property, the property is encoded somewhere.

> [!compiler] EndSem Q5(a), answered (1.5 marks)
> **What:** a control task assigns arbitrary labels to word types, so it contains no linguistic structure and a probe can only *memorize*. **Why needed:** a high probing accuracy is ambiguous: the representation may encode the property, or the probe may be **expressive enough to learn the task itself**. Selectivity = accuracy on the linguistic task minus accuracy on the control task.

> [!slides] Logit lens, Patchscopes (Lec_32)
> **Logit lens:** project an intermediate residual state $x^{l}$ through $W_U$, predicting as if all later layers were skipped; it can fail on some models. **Patchscopes:** modify activations at inference time to ask where information is encoded. **Linear representation hypothesis** and sparse autoencoders are listed too.

> [!compiler] EndSem Q5(c): Patchscopes (4.5 marks, from the paper)
> **Setup:** take $h_i^{l}$ from the *source* run of $M$ on $S$, apply the mapping $f$, and **patch** $f(h_i^{l})$ in place of $h_j^{l^*}$ in a separate inference pass of $M^*$ on the *target prompt* $T$; whatever $M^*$ generates is the "explanation" of $h_i^{l}$.
> **(i) Equivalence to the logit lens:** $M^*=M$, $f$ is the identity, and the patch is inserted at the **last layer** $l^*=L$ so no further layers process it and the next-token distribution is read directly through the unembedding.
> **(ii) Checking whether an attribute is encoded:** use a target prompt that elicits the attribute by few-shot description (for example "Syria: country in the Middle East; Leonardo DiCaprio: American actor; $x$"), patch into the placeholder $x$ and read the generated text.

:::def
The **residual stream** is the running sum $x^{l+1}=x^{l}+\text{Attn}^{l}(x^{l})+\text{FFN}^{l}(x^{l})$ that carries information between layers. **Activation patching** replaces one component's activation in a corrupted run with the clean one to test its causal role.
:::

:::key
- QK reads, OV writes; logits decompose into component contributions.
- Activation patching gives causal evidence; normalized restoration $=\dfrac{p_{patched}-p_{corrupt}}{p_{clean}-p_{corrupt}}$.
- Probes need control tasks (selectivity); the logit lens is a special case of Patchscopes.
:::

:::mistakes
- "Attention weights explain the model." They show where a head reads, not what it writes or whether it matters.
- "Probe accuracy shows the model uses the feature." It shows the information is decodable.
- "Logit lens works for every model." It can fail.
:::

:::exam
- *(EndSem Q5, 8 marks)* (a) control tasks (1.5), (b) QK and OV circuits (2), (c) Patchscopes versus logit lens and for checking attributes (4.5).
:::

## Quick Check

:::q Q1 · Compute the normalized restoration for $p_{clean}=0.9$, $p_{corrupt}=0.2$, $p_{patched}=0.6$.
:::ans
$(0.6-0.2)/(0.9-0.2)\approx0.57$.
:::
:::

:::q Q2 · Why is a control task used with probing?
:::ans
It shows whether a probe can reach high accuracy on random labels by memorising; a representation encodes the property only if the real-task accuracy greatly exceeds the control accuracy.
:::
:::
