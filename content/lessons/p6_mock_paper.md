**Time: 2 hours. Maximum marks: 100.** Attempt every question before opening any answer. The mock follows the shape of the 2024-25 papers described in the source notes: True/False with justification, multi-correct MCQ, short derivations, numericals and longer written answers. It mixes every unit.

:::tip
Suggested timing: Section A 15 min, B 10 min, C 35 min, D 25 min, E 35 min. Write the method before the number; examiners award marks for the method.
:::

## Section A: True or False with Justification (8 x 2 = 16 marks)

:::q A1 · Self-attention eliminates the need for positional encoding.
:::ans
**False.** It is permutation-equivariant; order is invisible without positional information. (Lesson 3.1)
:::
:::

:::q A2 · RoPE represents positions as complex numbers and tokens as pure rotations.
:::ans
**False.** Tokens ($q,k$) are the complex numbers; positions are the rotations. (Lesson 3.10)
:::
:::

:::q A3 · ALiBi adds positional embeddings to the token embeddings.
:::ans
**False.** It adds a distance penalty to attention scores. (Lesson 11.2)
:::
:::

:::q A4 · The KV-cache speeds up BERT inference.
:::ans
**False.** An encoder has no autoregressive loop. (Lesson 4.1)
:::
:::

:::q A5 · QLoRA is quantization-aware training.
:::ans
**False.** The base is quantized once and frozen. (Lesson 7.4)
:::
:::

:::q A6 · Kneser-Ney smoothing uses unigram frequency as the lower-order distribution.
:::ans
**False.** It uses continuation probability (number of distinct left contexts). (Lesson 1.5)
:::
:::

:::q A7 · TALMs need extensive pre-training on tool datasets.
:::ans
**False.** A few demonstrations plus self-play. (Lesson 8.2)
:::
:::

:::q A8 · Emergence is a sudden spike regardless of the metric used.
:::ans
**False.** Non-linear or discontinuous metrics create the jump; smooth metrics show gradual progress. (Lesson 5.6)
:::
:::

## Section B: Multiple-Correct MCQ (4 x 3 = 12 marks; no partial marking)

:::q B1 · Which are true? (A) KV-cache reuses past keys and values. (B) FlashAttention reduces HBM-SRAM I/O. (C) FlashAttention reduces FLOPs from $O(n^2d)$. (D) MQA reduces the KV-cache by $h$.
:::ans
**A, B, D.** C is false: FLOPs are unchanged. (Lessons 4.1, 4.2, 4.4)
:::
:::

:::q B2 · BERT's MLM objective: which are true? (A) 15% of tokens are chosen. (B) Of those, 80% become `[MASK]`. (C) Loss is computed at all positions. (D) 10% of the chosen tokens are left unchanged but still predicted.
:::ans
**A, B, D.** C is false: only chosen positions. (Lesson 5.2)
:::
:::

:::q B3 · Which statements about the reward model and DPO are true? (A) BT gives $P(y^+\succ y^-)=\sigma(r^+-r^-)$. (B) PPO-RLHF keeps four models. (C) DPO needs a separately trained reward model. (D) DPO's $\log Z(x)$ cancels.
:::ans
**A, B, D.** C is false: the reward is implicit. (Lessons 6.4 to 6.6)
:::
:::

:::q B4 · Which are true of LoRA? (A) $B$ is initialised to zero. (B) It adds inference latency. (C) $\Delta W=BA$. (D) Trainable parameters per matrix are $r(d+k)$.
:::ans
**A, C, D.** B is false: $BA$ can be merged into $W$. (Lesson 7.2)
:::
:::

## Section C: Short Answers and Derivations (6 x 5 = 30 marks)

:::q C1 · Why is the attention score divided by $\sqrt{d_k}$?
:::ans
Unit-variance components give $\operatorname{Var}(\mathbf{q}^\top\mathbf{k})=d_k$; large scores saturate the softmax and kill gradients; dividing by $\sqrt{d_k}$ restores unit variance. (Lesson 3.3)
:::
:::

:::q C2 · Derive the bigram MLE and show why add-1 uses $\lvert V\rvert$ in the denominator.
:::ans
$P(w_i\mid w_{i-1})=c(w_{i-1},w_i)/c(w_{i-1})$. Add one pseudo-count to each of the $\lvert V\rvert$ possible next words, so the denominator becomes $c(w_{i-1})+\lvert V\rvert$ and the probabilities still sum to 1. (Lessons 1.3 and 1.4)
:::
:::

:::q C3 · Compare BatchNorm and LayerNorm and say why Transformers use LayerNorm.
:::ans
BN normalises each feature across the batch (batch-size dependent, different at train and test, awkward for variable-length sequences); LN normalises each token across its features (no batch dependence, same at train and test). (Lesson 3.7)
:::
:::

:::q C4 · Explain T5 span corruption and how it differs from BART.
:::ans
Spans are replaced by unique sentinels (`<X>`, `<Y>`); the decoder outputs only the spans, each preceded by its sentinel. BART's decoder reconstructs the whole clean text. (Lesson 5.4)
:::
:::

:::q C5 · Why does a KL term appear in the RLHF objective, and what is the REINFORCE gradient?
:::ans
It keeps the policy near the reference and prevents reward hacking. $\nabla_\theta J=\mathbb{E}_{y\sim\pi_\theta}[r_s(x,y)\nabla_\theta\log\pi_\theta(y|x)]$ by the log-derivative trick. (Lessons 6.4, 6.5)
:::
:::

:::q C6 · Explain prompt tuning versus prefix tuning.
:::ans
Prompt tuning trains vectors at the input embedding layer only; prefix tuning trains vectors at every Transformer layer (keys and values), generated through an MLP. (Lesson 7.1)
:::
:::

## Section D: Numericals (3 x 6 = 18 marks)

:::q D1 · MHA, MQA and GQA parameter counts for $h=6$, $d_{\text{model}}=4$, $d_k=d_v=3$, group size 2.
:::ans
MHA 288; MQA 168; GQA with 3 K/V heads 216 (or 192 if 2 groups). State your reading of "group size". (Lesson 4.2)
:::
:::

:::q D2 · Predict the output.
```python
import math
def pp(probs):
    return 2 ** (-sum(math.log2(p) for p in probs) / len(probs))
print(round(pp([0.5, 0.25, 0.125]), 3))
```
:::ans
```output
4.0
```

Perplexity is the geometric-mean inverse probability: $(0.5\cdot0.25\cdot0.125)^{-1/3}=(1/64)^{-1/3}=4$.
:::
:::

:::q D3 · Compute-optimal $N$ and $D$ for $C=10^{23}$ FLOPs, and the memory of 4-bit weights for a 65B model.
:::ans
$C=120N^2$, so $N\approx2.9\times10^{10}$ and $D=20N\approx5.8\times10^{11}$. 65B at 4 bits is $32.5$ GB. (Lessons 5.6, 7.4)
:::
:::

## Section E: Long Answers (3 x 8 = 24 marks)

:::q E1 · Explain the Transformer encoder and decoder, including positional encoding, multi-head attention, masking and cross-attention.
:::ans
Cover: why recurrence is removed; Q, K, V and scaled dot-product attention; sinusoidal PE ladder; multi-head with $d_k=d_{\text{model}}/h$ and $W_O$; FFN with expansion 4 and the $12d^2$ per layer; residual and layer norm; causal mask added as $-\infty$; cross-attention with Q from the decoder and K, V from the final encoder output; teacher forcing for parallel training. (Lessons 3.1 to 3.9)
:::
:::

:::q E2 · Trace the path from a pre-trained base model to an aligned assistant, and list how to adapt it cheaply.
:::ans
Instruction tuning (FLAN, Self-Instruct); reward model by Bradley-Terry; PPO with KL or DPO; PEFT (prompt, prefix, adapters, LoRA, QLoRA); quantization and distillation. Mention alignment tax and prompt sensitivity. (Lessons 6.1 to 7.4)
:::
:::

:::q E3 · What makes an LLM responsible? Discuss bias, hallucination and interpretability with measurement and mitigation.
:::ans
Four dimensions: explainability, fairness, robustness, safety. Bias: StereoSet icat, Regard; sources in data, model, deployment; mitigation by CDA, triggers, safety demonstrations, constitutional AI. Hallucination: factuality versus faithfulness, FActScore, SelfCheck, RAG or KG grounding. Interpretability: residual stream, QK/OV, patching, probing with control tasks, Patchscopes. (Lessons 10.1, 10.2, 11.5)
:::
:::
