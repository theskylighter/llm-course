**QLoRA fine-tunes LoRA adapters on top of a frozen 4-bit quantized base model, bringing a 65B model from over 780 GB to under 48 GB; pruning removes weights; distillation trains a small student to imitate a large teacher.** Together they make big models cheap.

## The Problem: The Frozen Model Is Still Too Big to Train Against

Even with LoRA, a 16-bit 65B model needs 130 GB just for weights. So we compress the base itself.

## QLoRA

> [!slides] Lec_30: Dettmers et al. 2023
> Average memory to fine-tune a 65B model is $>780$ GB; QLoRA brings it to **$<48$ GB without degrading predictive performance**. Four pieces: **(1) 4-bit NormalFloat (NF4) quantization**, **(2) double quantization**, **(3) paged optimizers**, **(4) LoRA**.

> [!compiler] EndSem Q1(a) answer and the arithmetic
> - *NF4:* the 4-bit levels are the quantiles of a standard normal (weights are roughly normal), rescaled to $[-1,1]$ with **block-wise** absmax (blocks of 64). A uniform grid wastes levels in the tails.
> - *Double quantization:* per-block FP32 constants cost $32/64=0.5$ bit per parameter. Quantizing them to 8 bit with a second-level block of 256 costs $8/64+32/(64\cdot256)\approx0.127$, saving about **0.373 bit per parameter**.
> - *Paged optimizers:* optimizer state is **evicted to CPU RAM and paged back** during memory spikes.
> - 65B at 4 bits is **32.5 GB** of weights (versus 130 GB at 16 bits), so $<48$ GB leaves room for LoRA, activations and optimizer state. This is not QAT: the base is quantized once and never retrained.

```python
params = 65e9
print("fp16 weights GB:", params * 2 / 1e9)
print("4-bit weights GB:", params * 0.5 / 1e9)
print("full fine-tune (12 bytes/param) GB:", params * 12 / 1e9)
first = 32 / 64
second = 8 / 64 + 32 / (64 * 256)
print("double quantization saves bits/param:", round(first - second, 3), "( constants", first, "->", round(second, 3), ")")
```

```output
fp16 weights GB: 130.0
4-bit weights GB: 32.5
full fine-tune (12 bytes/param) GB: 780.0
double quantization saves bits/param: 0.373 ( constants 0.5 -> 0.127 )
```

## Pruning

> [!slides] Pruning (Lec_30)
> **Magnitude pruning** (Han et al. 2015): remove weights with the smallest absolute value; about **40%** can go with negligible loss, and retraining after pruning allows about **80%**. **Wanda** (Sun et al. 2023) is the LLM-era method: score a weight by $|W_{ij}|\cdot\|X_j\|_2$ and prune without retraining. **Unstructured** pruning works only if the hardware supports it; **structured** pruning removes whole components (for example the 2:4 pattern).

## Distillation

> [!slides] Distillation (Hinton et al. 2015)
> Train a small *student* to imitate a *teacher*. Gold label $[0,0,1,0,0]$; teacher prediction $[0.10,0.20,0.50,0.15,0.05]$. The **soft target** carries more information than the one-hot **hard target** (class 2 is more plausible than class 5). Pros: no restriction on the student structure, biggest potential speed gain. Cons: needs training data; expensive to train the student and to get soft labels.

> [!compiler] EndSem Q1(b): the two losses
> **(i) Word-level KD**, student $p_S$ matched to teacher $p_T$ at each position of $s=(s_1,\dots,s_L)$:
> $$\mathcal{L}_{\text{word}}=-\sum_{t=1}^{L}\sum_{v\in V}p_T(v\mid s_{<t};\theta_T)\,\log p_S(v\mid s_{<t};\theta_S)$$
> **(ii) Sequence-level KD** (Kim and Rush 2016): the teacher's distribution over whole *sequences* is intractable, so approximate it with its mode: run **beam search** with the teacher to get $\hat y=\arg\max p_T(y|x)$ and train the student by ordinary maximum likelihood, $\mathcal{L}_{\text{seq}}=-\log p_S(\hat y\mid x;\theta_S)$.

:::analogy
Distillation is a master chef writing a detailed recipe card including "this is nearly as good as that" notes; the apprentice learns from the full card, not only from "dish A won".
:::

```python
import math
teacher = [0.10, 0.20, 0.50, 0.15, 0.05]
student = [0.20, 0.20, 0.30, 0.20, 0.10]
gold = [0, 0, 1, 0, 0]
ce = lambda target, p: -sum(t * math.log(q) for t, q in zip(target, p))
print("hard-target loss:", round(ce(gold, student), 3))
print("soft-target loss:", round(ce(teacher, student), 3))
# magnitude pruning
w = [0.9, -0.05, 0.4, 0.01, -0.7, 0.2]
thr = sorted(abs(x) for x in w)[int(0.4 * len(w))]
print("pruned weights (40%):", [x if abs(x) >= thr else 0 for x in w])
```

```output
hard-target loss: 1.204
soft-target loss: 1.441
pruned weights (40%): [0.9, 0, 0.4, 0, -0.7, 0.2]
```

:::def
**Knowledge distillation** trains a student model to match the teacher's output distribution (soft targets) at each token (word-level) or to maximise the likelihood of the teacher's beam-search output (sequence-level).
:::

:::key
- QLoRA = NF4 + double quantization + paged optimizers + LoRA; 65B from over 780 GB to under 48 GB.
- Pruning: magnitude, Wanda; unstructured needs hardware support (2:4).
- Distillation: soft targets (word-level) or beam-search outputs (sequence-level). Self-Instruct is distillation from a bigger model.
:::

:::mistakes
- "QLoRA is quantization-aware training." The 4-bit base is frozen and quantized once; only LoRA weights train.
- "Pruning always speeds things up." Unstructured sparsity needs hardware support.
:::

:::exam
- *(EndSem Q1, 10 marks)* (a) explain NF4, double quantization, paged optimizers (4 marks); (b) word-level and sequence-level KD losses; (c) prompt versus prefix tuning.
:::

## Quick Check

:::q Q1 · What are the three strategies QLoRA adds on top of LoRA?
:::ans
NF4 4-bit quantization, double quantization, and paged optimizers.
:::
:::

:::q Q2 · How does sequence-level KD change the word-level loss?
:::ans
Instead of summing over the whole vocabulary at each position, it uses the teacher's beam-search output as a single target sequence and trains the student by maximum likelihood.
:::
:::
