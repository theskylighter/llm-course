**Decoder-only models are trained to predict the next token with a causal mask, so every position gives a training signal in one pass, and the same model generates text by sampling.** They need roughly half the parameters of an encoder-decoder.

## The Problem: Encoders Cannot Generate, Encoder-Decoders Are Heavy

> [!lecture] Why decoders won
> The encoder-only model is not the right solution for a **generative** task, which needs an autoregressive setup. The encoder-decoder needs a lot of memory and parameters. The decoder-only model fits next-token generation and needs about **half the parameters** of an encoder-decoder.

> [!slides] Encoders versus decoders (Lec_19)
> BERT is a Transformer encoder: bidirectional attention, trained with MLM, $P(x_i\mid x_1..x_{i-1},x_{i+1}..x_n)$. GPT and LLaMA are decoders: unidirectional attention, trained to predict the next token, $P(x_i\mid x_{<i})$. For an encoder you ignore this distribution and fine-tune for $P(y\mid x)$. A decoder you *use as a generator*.

## The Objective

Plain maximum-likelihood on sentences. By the chain rule

$$\log p_\theta(s)=\log p_1(t_1)+\sum_{j=1}^{m-1}\log p_{j+1}(t_{j+1}\mid t_1,\dots,t_j)$$

so training is **classification over the vocabulary $|V|$ at every position**, and with the causal mask all positions are trained in parallel (teacher forcing, lesson 3.9). Sampling is the reverse: draw $t_1$, then repeatedly draw $t_{j+1}\sim p_{j+1}(\cdot\mid t_{\le j})$.

:::analogy
A student who completes sentences. Every word in a textbook is a free exercise: cover the next word and guess it from everything before.
:::

```python
import numpy as np
text = [2, 0, 3, 1]                       # token ids
vocab = 4
rng = np.random.default_rng(1)
logits = rng.normal(size=(len(text) - 1, vocab))   # one row per position that predicts the next token
p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
nll = -np.log(p[np.arange(len(text) - 1), text[1:]])
print("per-position loss:", np.round(nll, 3))
print("mean loss:", round(float(nll.mean()), 3))
```

```output
per-position loss: [1.332 1.276 1.418]
mean loss: 1.342
```

## The GPT Line

> [!slides] The GPT line (Lec_19)
> **GPT-1 (2018):** 12-layer Transformer decoder, 117M parameters, 768-d hidden, 3072-d FFN, BPE with 40,000 merges, trained on **BooksCorpus** (7000+ books). **GPT-2:** 1.5B parameters, layer norm placed in each sub-block, vocabulary 50,257, context 512 to 1024, 8M web documents. **GPT-3:** 175B parameters, 96 layers, 96 heads, 12k-dim vectors. **GPT-4:** Transformer-based, "the rest is a mystery". **LLaMA:** BPE, **rotary** positional encodings, an optimised mix of Common Crawl, GitHub, Wikipedia, books.
> Cost anchors: BERT base USD 500 / large USD 7,000; GPT-2 USD 25,000 for one run.

> [!compiler] Notes on the GPT slides
> GPT-2's "layer norm in between each sub-block" is the move to **pre-LN**. From GPT-3 on, models are used **without fine-tuning**: you put examples in the prompt (**in-context learning**, Chapter 6). The slides' cost figures are third-party estimates, not paper values.

:::def
A **decoder-only language model** is a Transformer with causal self-attention trained by maximum likelihood to predict each token from the preceding tokens; it generates text autoregressively.
:::

:::key
- Decoder-only: causal mask, next-token loss at every position in parallel, about half the parameters of an encoder-decoder.
- GPT-1 117M, GPT-2 1.5B, GPT-3 175B (96 layers); LLaMA uses RoPE.
- GPT-3 onward: prompt instead of fine-tune (in-context learning).
:::

:::mistakes
- "Autoregressive means training is sequential." The mask makes training parallel; only sampling is sequential.
:::

:::exam
- *"Why does a decoder-only model give a parallel training loss despite being autoregressive?"* (Tier 2) Causal mask and teacher forcing.
:::

## Watch One Forward Pass

Everything in this lesson, and in Chapter 3, runs inside a GPT-2 small model that you can watch live in your browser: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Polo Club of Data Science, Georgia Tech). It opens in a new tab, and a laptop screen works better than a phone.

What to try:
1. Type a short sentence and follow one token through embedding, positional encoding, Q/K/V and the attention heads.
2. Open a head and look at the attention matrix. The upper triangle is empty because of the causal mask.
3. Raise the temperature and watch the next-token distribution flatten. Lower it and the top token takes over.

## Quick Check

:::q Q1 · What does a decoder-only model predict at each position?
:::ans
A distribution over the vocabulary for the next token, given all earlier tokens.
:::
:::

:::q Q2 · Give one architectural change GPT-2 made over GPT-1.
:::ans
Layer norm moved into each sub-block (pre-LN), vocabulary 50,257, context raised from 512 to 1024.
:::
:::
