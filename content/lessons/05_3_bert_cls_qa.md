**BERT adds next-sentence prediction, a [CLS] summary token and segment embeddings, and is fine-tuned for question answering with two heads that mark the start and end of the answer span.** The same [CLS] trick fails in GPT-style models.

## The Problem: One Objective Does Not Teach Sentence Relations

MLM teaches words in context. Many tasks (entailment, QA) need the relation *between two sentences*. So BERT trains two losses at once.

- **MLM**, from the previous lesson.
- **Next-sentence prediction (NSP):** given $(W_1,W_2)$ predict whether $W_2$ follows $W_1$. Positives are consecutive sentences, negatives are random pairs ("I enjoy reading the book." / "I finished the novel." gives yes).

Input format: `[CLS] sentence A [SEP] sentence B [SEP]`. The embedding of each token is the **sum of three** vectors: **token + position + segment** (A or B). The `[CLS]` output goes through a small linear layer for NSP; the two losses are added and back-propagated through the whole encoder.

> [!lecture] Why a [CLS] token?
> Why not concatenate all token outputs and classify? That makes the classifier bigger. And why not use some real token? `[CLS]` is not part of the sentence, so it is **unbiased** toward any word, whereas the vector at a real word like "I" is biased toward itself. Also `[CLS]` attends to all tokens through self-attention, so it holds information about all of them. The new parameters are just a $512\times2$ matrix.

> [!compiler] How strong is that argument?
> The "unbiased" argument is the lecture's intuition. It works because training makes `[CLS]` an aggregator. It also requires bidirectional attention. **That is exactly why the same trick fails in GPT-2** (MidSem Q4(c)): under the causal mask the first token sees only itself. In a decoder you read the *last* token.

:::analogy
`[CLS]` is a class monitor who listens to every student and then reports to the teacher on behalf of the whole class, instead of the teacher polling each student.
:::

## Setup and Sizes

> [!lecture] Setup
> 2.5 billion words of Wikipedia plus a book corpus of 800 million words. **BERT-base:** 12 encoder layers. **BERT-large:** 24 layers, with a larger hidden size and 12 / 16 attention heads.

> [!compiler] Exact sizes
> From the paper: base 110M parameters (768 hidden, 12 heads), large 340M (1024 hidden, 16 heads).

## Fine-Tuning for Question Answering

How do you use BERT on a task unlike MLM, such as extractive QA? Fine-tune with a **span head**. Input: `[CLS] question [SEP] passage [SEP]`. Add two linear heads on every passage token: $p_b$ (this token *begins* the answer) and $p_e$ (it *ends* it). At test time choose

$$(i^*,j^*)=\arg\max_{i\le j}\;p_b(i)\,p_e(j)$$

with $j\ge i$ ($j=i$ is allowed: a one-token answer). If the answer occurs several times, take the first occurrence.

```python
import numpy as np
pb = np.array([0.05, 0.10, 0.60, 0.05, 0.10, 0.10])   # begin probabilities
pe = np.array([0.05, 0.05, 0.10, 0.65, 0.05, 0.10])   # end probabilities

# O(n): keep the best begin seen so far
best, span, best_b = -1.0, None, 0
for j in range(len(pe)):
    if pb[j] > pb[best_b]:
        best_b = j
    score = pb[best_b] * pe[j]
    if score > best:
        best, span = score, (best_b, j)
print("answer span (begin, end):", span, "score", round(float(best), 3))
```

```output
answer span (begin, end): (2, 3) score 0.39
```

> [!lecture] After BERT
> BERT was state of the art until about 2020. Follow-ups: DistilBERT and TinyBERT (squeeze the model), **RoBERTa** (BERT was *under-trained*, train it more), and multi-task variants.

> [!compiler] What later work changed
> RoBERTa also **dropped NSP** (it found it unhelpful) and used *dynamic masking*.

:::def
**BERT** is a stack of bidirectional Transformer encoder layers pre-trained with masked language modeling and next-sentence prediction, whose input embedding is the sum of token, position and segment embeddings, and whose `[CLS]` output summarises the input for classification.
:::

:::key
- Input: `[CLS] A [SEP] B [SEP]`; embeddings = token + position + segment.
- `[CLS]` is a bidirectional aggregator; in decoders read the last token instead.
- QA: begin and end heads; $\arg\max_{i\le j}p_b(i)p_e(j)$.
- RoBERTa: train longer, drop NSP.
:::

:::mistakes
- Putting `[CLS]` at the start for a GPT-style (causal) classifier. It cannot see anything.
:::

:::exam
- *(MidSem Q4(c))* "Can you use a BERT-style `[CLS]` at the start with GPT-2?" **No**: the causal mask means position 0 sees only itself.
- *"Write pseudo-code for the BERT QA span decoder in $O(n)$."*
:::

## Quick Check

:::q Q1 · Which three embeddings are summed in BERT's input?
:::ans
Token, position and segment (A or B).
:::
:::

:::q Q2 · Why is $j\ge i$ imposed in span selection?
:::ans
The end of an answer cannot precede its start; $j=i$ gives a one-token answer.
:::
:::
