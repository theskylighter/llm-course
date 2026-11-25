<<SVG:ngram_markov_context>>

*Figure: N-grams shorten the history to the previous N−1 words, making contexts countable at the cost of information.*

**The chain rule gives the exact probability of a sentence, but it needs histories that never repeat; the Markov assumption shortens the history so that we can count.** An N-gram model conditions on only the previous $N-1$ words.

## The Problem: We Cannot Count Whole Sentences

We want $P(w_1,\dots,w_n)$. The chain rule says this equals a product of next-word probabilities, and that is exact.

> [!slides] Chain rule and Markov assumption
> $$P(w_1\dots w_n)=\prod_{i} P(w_i\mid w_1\dots w_{i-1})\;\approx\;\prod_i P(w_i\mid w_{i-k}\dots w_{i-1})$$
> An **N-gram** model conditions on the previous $N-1$ words, so an N-gram LM is an $(N-1)$-order Markov model. Unigram $P(\text{begun})$, bigram $P(\text{begun}\mid\text{has})$, trigram $P(\text{begun}\mid\text{season has})$.

Why not simply count how often "The monsoon season has" is followed by "begun"? Because that full history almost never appears in any corpus, so the count is zero. The Markov assumption **shortens the history until counts exist**.

## Estimating a Bigram Model

For a bigram, the maximum-likelihood estimate (MLE) is a ratio of counts:

$$P(w_i\mid w_{i-1})=\frac{c(w_{i-1},w_i)}{c(w_{i-1})}$$

The markers `<s>` and `</s>` are tokens too, so the table has a row for sentence-start and a column for sentence-end.

:::analogy
Predicting the next word from only the last word is like guessing a person's next sentence by hearing only their last word. You will often be right about common phrases and wrong about everything that needs memory.
:::

```python
from collections import Counter

corpus = [
    "the monsoon has begun",
    "the rains have arrived",
    "the monsoon has arrived",
]
uni, bi = Counter(), Counter()
for sent in corpus:
    toks = ["<s>"] + sent.split() + ["</s>"]
    for a, b in zip(toks, toks[1:]):
        bi[(a, b)] += 1
        uni[a] += 1

def p(b, a):
    return bi[(a, b)] / uni[a]

print("P(monsoon | the) =", round(p("monsoon", "the"), 3))
print("P(begun | has)   =", round(p("begun", "has"), 3))
print("P(arrived | has) =", round(p("arrived", "has"), 3))
sent = ["<s>", "the", "monsoon", "has", "begun", "</s>"]
prob = 1.0
for a, b in zip(sent, sent[1:]):
    prob *= p(b, a)
print("P(the monsoon has begun) =", round(prob, 4))
```

```output
P(monsoon | the) = 0.667
P(begun | has)   = 0.5
P(arrived | has) = 0.5
P(the monsoon has begun) = 0.3333
```

> [!compiler] Precision on the name
> An N-gram LM is an $(N-1)$-order Markov model. A bigram looks one word back, so it is first-order.

:::def
An **N-gram language model** approximates $P(w_i\mid w_{1:i-1})$ by $P(w_i\mid w_{i-N+1:i-1})$, estimated by relative frequency of counts in a training corpus.
:::

:::key
- Chain rule: exact. Markov assumption: the approximation that makes counting possible.
- Bigram MLE is $c(w_{i-1},w_i)/c(w_{i-1})$.
- Use `<s>` and `</s>` so the first and last words are modelled.
:::

:::mistakes
- Saying an N-gram model has "order $N$". It is order $N-1$.
- Forgetting that every sentence is wrapped in `<s>` ... `</s>`.
:::

:::exam
- *"Derive the bigram MLE and show why a test sentence can get probability 0."* (2 marks)
- *"Given a small corpus, compute $P(w_i\mid w_{i-1})$."* Build the count table, then divide.
:::

## Quick Check

:::q Q1 · Why is the chain rule alone not enough to build an LM from counts?
:::ans
The conditioning histories $w_1\dots w_{i-1}$ get longer and almost never repeat in a corpus, so their counts are zero or tiny.
:::
:::

:::q Q2 · In the corpus above, what is $c(\text{has})$ and why is $P(\text{begun}\mid\text{has})=0.5$?
:::ans
"has" occurs twice (once followed by "begun", once by "arrived"), so $0.5=1/2$.
:::
:::
