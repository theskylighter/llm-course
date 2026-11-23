**Attempt each question before opening its answer.** This set covers Chapters 4 and 5 and mixes in Chapter 3.

## A. MCQ (more than one option may be correct, no partial marking)

:::q A1 · Which statements are correct?
(A) KV-cache reuses past keys and values in autoregressive models. (B) FlashAttention reduces HBM-SRAM I/O. (C) KV-cache is for generation; FlashAttention reduces the *computational complexity* of attention. (D) Both reduce memory by sharing attention weights across layers.
:::ans
**A and B are true. D is false.** C is the trap: FlashAttention helps training and inference, but its FLOPs stay $O(n^2d)$; it reduces memory traffic and memory, not computational complexity. A strict reader marks C false. Lessons 4.1 and 4.4.
:::
:::

:::q A2 · Which statements about ELMo and BERT are correct?
(A) ELMo = bi-LSTM, BERT = Transformer encoder with MLM. (B) ELMo is context-independent. (C) BERT masks and predicts; ELMo predicts the next word. (D) Both produce embeddings that can be fine-tuned.
:::ans
**A and C.** B is false (ELMo is contextual). D is the trap: ELMo's LM is **frozen** and used as features, so a strict reading makes it false. Lesson 5.1.
:::
:::

:::q A3 · In BERT pre-training, the loss is computed at
(a) every token (b) only the chosen 15% positions (c) only `[MASK]` tokens (d) only `[CLS]`
:::ans
**(b).** Chosen positions include the 10% replaced by a random token and the 10% left unchanged, so (c) is wrong.
:::
:::

## B. Short Answers

:::q B1 · Why do we cache K and V but not Q? (3 marks)
:::ans
$\mathbf{q}_i$ is used once, for the output at position $i$. Keys and values at earlier positions are needed by every later token and, under the causal mask, never change. Lesson 4.1.
:::
:::

:::q B2 · Explain MQA and GQA and their effect on the KV-cache. (4 marks)
:::ans
MQA shares one K/V head across all $h$ query heads; GQA shares one per group of $h/G$ query heads. The cache shrinks by $h/G$; only $W_K$ and $W_V$ shrink. GQA recovers most of MHA's quality. Lesson 4.2.
:::
:::

:::q B3 · (MidSem Q4(c)) Can you use a BERT-style `[CLS]` at the start of the sequence with GPT-2? (3 marks)
:::ans
No. Under the causal mask position 0 sees only itself, so its state cannot summarise the sentence. In a decoder, read the **last** token's state. Lesson 5.3.
:::
:::

:::q B4 · (MidSem Q6) State Kaplan versus Chinchilla without numbers, and judge: "emergence is a sudden spike regardless of metric." (4 marks)
:::ans
Kaplan: put most extra compute into a bigger model. Chinchilla: scale parameters and tokens equally (about 20 tokens per parameter), so big models were under-trained. The spike claim is **false**: it appears for non-linear or discontinuous metrics; smooth metrics show gradual improvement. Lesson 5.6.
:::
:::

## C. Numericals

:::q C1 · Parameters of MHA, MQA and GQA with $h=6$, $d_{\text{model}}=4$, $d_k=d_v=3$ (group size 2). (5 marks)
:::ans
MHA 288, MQA 168, GQA 216 (3 groups). If "group size 2" meant 2 groups: 192. State your reading. Lesson 4.2.
:::
:::

:::q C2 · Compute-optimal $N$ and $D$ for $C=10^{24}$ FLOPs. (3 marks)
```python
import math
C = 1e24
N = math.sqrt(C / 120)
print(f"N = {N:.2e}, D = {20 * N:.2e}")
```
:::ans
```output
N = 9.13e+10, D = 1.83e+12
```

Using $C=6ND$ with $D=20N$ gives $C=120N^2$.
:::
:::

## D. Predict the Output

:::q D1 · What does this print?
```python
def span_corrupt(tokens, start, length):
    return tokens[:start] + ["<X>"] + tokens[start + length:], ["<X>"] + tokens[start:start + length]

enc, dec = span_corrupt("a b c d e f".split(), 2, 2)
print(enc)
print(dec)
```
:::ans
```output
['a', 'b', '<X>', 'e', 'f']
['<X>', 'c', 'd']
```

The encoder sees one sentinel in place of the span; the decoder target contains only the span.
:::
:::

## E. Long Answer

:::q E1 · Compare encoder-only, decoder-only and encoder-decoder pre-training, with objective and example for each. (8 marks)
:::ans
Encoder-only (BERT): bidirectional attention, MLM with 15% chosen and the 80/10/10 rule, plus NSP; good for classification and QA, not generation. Decoder-only (GPT, LLaMA): causal mask, next-token prediction at every position, about half the parameters of an encoder-decoder, natural generation. Encoder-decoder (BART, T5): corrupt and reconstruct; BART regenerates the whole text, T5 predicts only sentinel-marked spans and casts all tasks as text-to-text. Cite lessons 5.2 to 5.5.
:::
:::

:::q E2 · Explain how FlashAttention computes exact attention without storing the $n\times n$ matrix. (6 marks)
:::ans
Tile Q, K, V into SRAM-sized blocks; for each query block loop over key/value blocks keeping running $m$, $\ell$, $\mathbf{o}$ and rescaling by $e^{m-m'}$ (online softmax); write only $O$ and the log-sum-exp; in the backward pass recompute tiles. FLOPs are unchanged; HBM traffic falls and memory is $O(n)$. Lesson 4.4.
:::
:::
