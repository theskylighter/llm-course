<<SVG:elmo_contextual_word_vectors>>

*Figure: ELMo’s frozen forward/backward LSTM produces context-dependent token states, which a task-specific mix combines.*

**Static word vectors give "bat" the same vector in every sentence; ELMo fixes this by running a language model over the sentence so each token's vector depends on its context.** It was the first step from pre-trained vectors to pre-trained models.

## The Problem: One Vector per Word

Take "I **record** the record" or "A **bat** flew out of the cave. He hit the ball with the **bat**." The same word, two meanings. Follow the ladder.

> [!lecture] Opening
> Today we talk about pre-training strategies. We will see how the paradigm shifted from pre-trained word embeddings to pre-trained language models. Remember the quote: *you shall know a word by the company it keeps.* The same person later modified it: the complete meaning of a word is always contextual, and no study of meaning apart from a complete context can be taken seriously.
> — *NPTEL Lecture 18*

**Attempt 1: one vector per word (word2vec, GloVe).** Both occurrences of *bat* get the *same* vector. **We need embeddings that change with context.**
**Attempt 2: pre-train only the input vectors.** Only the embedding layer is pre-trained and frozen; every other weight is learned from scratch on a small dataset.
**Attempt 3: pre-train the whole model.** All parameters are learned on raw text, then adapted to a task.

| | Pre-trained word vectors | Pre-trained models |
|---|---|---|
| What is pre-trained | the input embedding table | **all** parameters |
| Embedding of *bat* | static | depends on the sentence |
| Task-specific part | everything above the embeddings | a small head, or nothing |

The remaining question is: **what task do we pre-train on when we have no labels?** Every model in this chapter is one answer, and all are **self-supervised**: the labels come from the text itself.

> [!lecture] Three kinds of pre-training
> Depending on which part of the Transformer you use. **Decoder only** (GPT series). **Encoder only** (BERT, ALBERT, RoBERTa). **Encoder-decoder**, the original structure (T5, BART).

| Family | Attention mask | Pre-training signal | Examples |
|---|---|---|---|
| Encoder-only | full (bidirectional) | predict masked tokens | BERT, RoBERTa |
| Decoder-only | causal (left only) | predict next token | GPT, LLaMA |
| Encoder-decoder | full in encoder, causal plus cross in decoder | reconstruct corrupted input | T5, BART |

## ELMo: Contextual Vectors from a Language Model

> [!lecture] Intuition
> ELMo (2018) is *not* a Transformer model. ELMo stands for *Embeddings from Language Models*. It relies on RNNs (LSTM or GRU). Train an RNN with the **language-model objective**, next-word prediction, over many sequences. The hidden state at each token is then used as that token's embedding.

1. **Bottom layer: a language model.** Pre-train a stacked RNN on next-word prediction. When done, **freeze** it.
2. **Top layer: your task.** Put a new task-specific RNN on top of the frozen vectors and train that on the labelled data.

> [!lecture] At inference time
> You feed a *new* sentence, say "Akshay is a great cricket player", through the bottom layer. The hidden state for *great* is now different from anything stored earlier, because it is a function of the previous words: it is **contextual**.

One direction is not enough. ELMo runs a **forward LM** (left to right) and a **backward LM** (right to left). The objective is the sum of the two log-likelihoods:

$$\sum_{k=1}^{N}\Big[\log p(t_k\mid t_1..t_{k-1};\Theta_x,\overrightarrow{\Theta}_{LSTM},\Theta_s)+\log p(t_k\mid t_{k+1}..t_N;\Theta_x,\overleftarrow{\Theta}_{LSTM},\Theta_s)\Big]$$

For each token we have several vectors (per layer and direction). The last refinement: learn **task-specific weights** over layers, so the LM stays frozen but each task chooses its own mix:

$$\mathrm{ELMo}_k^{\text{task}}=\gamma^{\text{task}}\sum_{j=0}^{n}s_j^{\text{task}}\,\mathbf{h}_{k,j},\qquad \mathbf{h}_{k,j}=[\overrightarrow{\mathbf{h}}_{k,j};\overleftarrow{\mathbf{h}}_{k,j}]$$

$s_j$ are softmax-normalised layer weights and $\gamma$ scales the whole vector. Inputs are **character-level** (a character CNN), which avoids the UNK problem.

:::analogy
A word is an actor; static vectors are the actor's headshot, the same in every film. ELMo is the actor on set: how they look depends on the scene around them.
:::

```python
import numpy as np
# Toy layer-mixing: three layers of hidden vectors for one token, task-specific softmax weights.
h = np.array([[0.1, 0.2], [0.5, -0.1], [0.9, 0.4]])   # layer 0 (chars), layer 1, layer 2
raw = np.array([0.0, 1.0, 2.0])
s = np.exp(raw) / np.exp(raw).sum()
gamma = 1.5
elmo = gamma * (s[:, None] * h).sum(0)
print("layer weights:", np.round(s, 3), "sum", round(float(s.sum()), 3))
print("ELMo vector  :", np.round(elmo, 3))
```

```output
layer weights: [0.09  0.245 0.665] sum 1.0
ELMo vector  : [1.095 0.389]
```

> [!compiler] Precision on the lecture's "dictionary" story
> The lecture first describes a lookup table whose rows get *replaced* by hidden states. That is a teaching simplification: what ELMo stores is the **trained biLM**, and a token's vector is computed by running the biLM over its sentence. Treat ELMo as **feature-based**: the LM is not fine-tuned (this matters for MidSem Q4).

:::def
**ELMo** produces contextual word representations as a task-specific weighted sum of the hidden states of a frozen, pre-trained bidirectional LSTM language model.
:::

:::key
- Static vectors cannot separate word senses; pre-train the whole network so representations depend on context.
- ELMo: bi-LSTM LM, frozen, feature-based, task-specific layer weights; character inputs.
- Self-supervised: labels come from the text itself.
:::

:::mistakes
- "ELMo is a Transformer." It is LSTM-based, from the same era.
- "ELMo is context-independent." It is the contextual one; word2vec is static.
:::

:::exam
- *(MidSem Q4)* A: ELMo = bi-LSTM, BERT = Transformer encoder with MLM: **true**. B: "ELMo is context-independent": **false**. C: BERT masks and predicts, ELMo predicts next word: **true**. D: "both produce embeddings that can be fine-tuned": the trap; in the lecture ELMo's LM is **frozen**.
:::

## Quick Check

:::q Q1 · What are the two parts of the ELMo recipe?
:::ans
A frozen pre-trained bidirectional LM, and a task-specific model on top that learns a softmax-weighted mix of the LM's layers.
:::
:::

:::q Q2 · Why is "bat" a motivating example?
:::ans
A static embedding gives both senses the same vector; contextual embeddings differ by sentence.
:::
:::
