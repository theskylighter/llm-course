<<SVG:language_model_score_generate>>

*Figure: A language model scores whole sequences and generates by sampling the next token from a context-conditioned distribution.*

**A language model assigns a probability to a sequence of tokens, and generating text is just sampling the next token from that distribution.** That is the whole job. Everything else in this course is a better way of doing it.

## The Problem: How Does a Machine Choose Between Two Sentences?

Take the vocabulary $\mathcal{V}=\{\text{arrived, delhi, have, is, monsoon, rains, the}\}$. Two orderings of the same words:

- "the monsoon rains have arrived"
- "monsoon the have rains arrived"

A human knows at once that the first is English and the second is word salad. A computer has no such sense. What we want is a number that is **high for the first and tiny for the second**.

> [!lecture] Opening
> A language model gives a probability distribution over a sequence of tokens. The slide uses 0.2 for "the monsoon rains have arrived" and 0.001 for the shuffled version. Generation is just this: compute $P(x_i \mid \text{the monsoon rains have})$ for every $x_i\in\mathcal{V}$ and sample the next token. Think of texting autocomplete.
> — *NPTEL Lecture 1*

## The Idea in Plain Words

A language model (LM) answers one of two equivalent questions:

1. How likely is the whole sentence, $P(w_1,\dots,w_n)$?
2. Given the words so far, how likely is each possible next word, $P(w_n\mid w_{<n})$?

The second form is the useful one, because it lets us **generate**: look at the distribution, pick a word, append it, repeat.

:::analogy
Your phone's keyboard suggests the next word as you type. It is a tiny language model: it ranks every word in its vocabulary by how likely it is to come next.
:::

## How Generation Works

1. Start with a prompt, for example `the monsoon rains have`.
2. The model computes a probability for every word in $\mathcal{V}$.
3. Sample (or pick the largest) to choose the next word.
4. Append it to the prompt and go back to step 2.

```python
# A toy next-word distribution, written by hand.
probs = {"arrived": 0.60, "begun": 0.25, "is": 0.10, "delhi": 0.05}
print("sum of probabilities:", round(sum(probs.values()), 2))
best = max(probs, key=probs.get)
print("most likely next word:", best)
```

```output
sum of probabilities: 1.0
most likely next word: arrived
```

> [!slides] Three things to remember from lecture 1
> (1) "Large" means parameters **and** data. (2) The roadmap is Basics, Architecture, Learnability, Knowledge/Retrieval, Ethics. (3) The course does **not** cover deep NLP or non-text generation.

:::def
A **language model** is a probability distribution over sequences of tokens, $P(w_1,\dots,w_n)$, usually factorised as a product of next-token distributions $P(w_t\mid w_{<t})$.
:::

:::key
- An LM scores sentences and generates by sampling the next token.
- The probabilities for the next token sum to 1 over the vocabulary.
- "Large" means large parameter count and large training data.
:::

:::mistakes
- "An LM only generates text." It also **scores** text, which is how it is evaluated (perplexity, lesson 1.6).
- "A high probability means the sentence is true." It means the sentence is **fluent** given the training data.
:::

:::exam
- *"Define a language model."* Give the distribution-over-sequences definition and the chain-rule form.
- *"How does an LM generate text?"* Compute $P(x_i\mid \text{context})$ for all $x_i\in\mathcal{V}$, then sample.
:::

## Quick Check

:::q Q1 · Why is generation described as sampling from $P(x_i\mid\text{context})$?
:::ans
At each step the model gives a distribution over the whole vocabulary. Choosing the next token from that distribution (by sampling or taking the maximum) and appending it produces text one token at a time.
:::
:::

:::q Q2 · What does the word "large" refer to in "large language model"?
:::ans
Both the number of parameters and the amount of training data.
:::
:::
