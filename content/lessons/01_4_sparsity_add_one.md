<<SVG:add_one_smoothing_mass>>

*Figure: Add-one changes the unseen “rains” continuation from zero to a positive probability by adding one pseudo-count per vocabulary item.*

**Most possible bigrams never appear in training, so an N-gram model gives them probability zero; smoothing moves a little probability mass to unseen events.** The simplest version is add-one (Laplace) smoothing.

## The Problem: 99.95% of the Table Is Zero

Think of a million-word vocabulary: a million-by-million matrix of bigrams.

> [!lecture] Look at this table very carefully
> In the Berkeley Restaurant corpus (9332 sentences, $V=1446$), **99.95% of bigram entries are zero**. A zero does not mean the pair is invalid. It means we did not see it. "enjoyed the movie / food / game" in training, "enjoyed the concert" in test: $P(\text{concert}\mid\text{enjoyed the})=0$, so the whole test sentence gets probability 0 and perplexity cannot be computed.
> — *NPTEL Lecture 3*

Limits of N-grams in one list: **data sparsity**, a **fixed window** (cannot link "project" to "approved" across a long clause), memory blow-up with large $V$, and **no generalisation** to similar words.

## The Cheapest Fix: Add One

Pretend every possible bigram was seen once more than it really was.

> [!slides] Laplace (add-1)
> $$P_{\text{add-1}}(w_i\mid w_{i-1})=\frac{c(w_{i-1},w_i)+1}{c(w_{i-1})+\lvert V\rvert}$$

The denominator grows by $\lvert V\rvert$, not by 1, because we added one pseudo-count for **each** of the $\lvert V\rvert$ possible next words. Now every probability is positive, and they still sum to 1.

:::analogy
A teacher with 100 marks to give away and 5 unseen students: to give each unseen student a mark she must take a little from the students who already have many.
:::

```python
from collections import Counter

corpus = ["the monsoon has begun", "the rains have arrived", "the monsoon has arrived"]
uni, bi, vocab = Counter(), Counter(), set()
for sent in corpus:
    toks = ["<s>"] + sent.split() + ["</s>"]
    vocab.update(toks)
    for a, b in zip(toks, toks[1:]):
        bi[(a, b)] += 1
        uni[a] += 1
V = len(vocab)
print("V =", V)

def p_mle(b, a):
    return bi[(a, b)] / uni[a]

def p_add1(b, a):
    return (bi[(a, b)] + 1) / (uni[a] + V)

for b in ["begun", "arrived", "rains"]:
    print(b, "| MLE", round(p_mle(b, "has"), 3), "| add-1", round(p_add1(b, "has"), 3))
print("add-1 sums to", round(sum(p_add1(b, "has") for b in vocab), 6))
```

```output
V = 9
begun | MLE 0.5 | add-1 0.182
arrived | MLE 0.5 | add-1 0.182
rains | MLE 0.0 | add-1 0.091
add-1 sums to 1.0
```

> [!compiler] Reconstituted counts
> The slide lists $c^*(w_{n-1}w_n)=\dfrac{(c+1)\,c(w_{n-1})}{c(w_{n-1})+\lvert V\rvert}$, the effective count after add-1. It shows how much add-1 distorts seen counts: with large $V$ it steals too much mass from seen events.

:::def
**Smoothing** redistributes probability mass from observed N-grams to unseen ones so that no N-gram has zero probability. **Add-one (Laplace) smoothing** adds 1 to every count and $\lvert V\rvert$ to the denominator.
:::

:::key
- Zero count means unseen, not impossible.
- Add-1: $\dfrac{c+1}{c(w_{i-1})+\lvert V\rvert}$.
- Add-$k$ uses $k<1$ and is tuned on held-out data, but is still crude.
:::

:::mistakes
- Using $c(w_{i-1})+1$ in the denominator. It must be $+\lvert V\rvert$.
- Reading a zero as an invalid sentence.
:::

:::exam
- *"Compute $P(w_i\mid w_{i-1})$ with and without add-1."* Show both rows of a small table.
- *"Why does add-1 fail for large vocabularies?"* It moves too much mass to unseen events.
:::

## Quick Check

:::q Q1 · A bigram has count 0 and $c(w_{i-1})=10$, $\lvert V\rvert=1000$. What is its add-1 probability?
:::ans
$\dfrac{0+1}{10+1000}=\dfrac{1}{1010}\approx 0.00099$.
:::
:::

:::q Q2 · Why does the denominator add $\lvert V\rvert$?
:::ans
One pseudo-count is added for each of the $\lvert V\rvert$ possible next words, so the probabilities still sum to 1.
:::
:::
