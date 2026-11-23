**Attempt each question before opening its answer.** This set covers Chapters 1 and 2: language models, N-grams, smoothing, perplexity, perceptrons, MLPs, backpropagation and the road to attention.

## A. Definitions and Short Answers

:::q A1 · Define a language model and state the chain-rule factorisation. (2 marks)
:::ans
A language model is a probability distribution over token sequences. By the chain rule $P(w_1\dots w_n)=\prod_i P(w_i\mid w_{<i})$. See lesson 1.1.
:::
:::

:::q A2 · Differentiate back-off and interpolation. (3 marks)
:::ans
| | Back-off | Interpolation |
|---|---|---|
| Idea | use the highest order with evidence, else fall back | mix all orders with weights |
| Formula | trigram if seen, else bigram, else unigram | $\lambda_1P_3+\lambda_2P_2+\lambda_3P_1$ |
| Switch | hard | soft |

Lesson 1.5.
:::
:::

:::q A3 · Why does Kneser-Ney use continuation probability? Use "Los Angeles". (3 marks)
:::ans
"Angeles" is frequent but almost always follows "Los", so a plain unigram probability overestimates it in new contexts. Continuation probability counts how many distinct left contexts a word completes, which is what predicts appearing in a *new* context. Lesson 1.5.
:::
:::

:::q A4 · List three limitations of N-gram LMs. (3 marks)
:::ans
Data sparsity (99.95% zero bigrams in the Berkeley corpus), a fixed short window, and no generalisation between similar words. Memory also grows with vocabulary size.
:::
:::

:::q A5 · Why can a single perceptron not compute XOR? (2 marks)
:::ans
It draws one linear boundary, and XOR is not linearly separable. A hidden layer computing OR and AND makes XOR (OR minus AND) separable. Lesson 2.1.
:::
:::

## B. MCQ (more than one option may be correct)

:::q B1 · Which statements about perplexity are correct?
(a) Lower is better. (b) It should be computed on training data. (c) It equals $P(W)^{-1/n}$. (d) It can be compared across any two vocabularies.
:::ans
**(a) and (c).** (b) is wrong: use held-out data. (d) is wrong: the vocabulary and test set must match.
:::
:::

:::q B2 · The add-1 smoothed bigram probability has denominator
(a) $c(w_{i-1})+1$ (b) $c(w_{i-1})+\lvert V\rvert$ (c) $c(w_{i-1})\cdot\lvert V\rvert$ (d) $\lvert V\rvert$
:::ans
**(b).** One pseudo-count for each of the $\lvert V\rvert$ possible next words.
:::
:::

:::q B3 · Which activation can suffer from "dead neurons"?
(a) sigmoid (b) tanh (c) ReLU (d) softmax
:::ans
**(c) ReLU.** Negative pre-activation gives zero output and zero gradient.
:::
:::

:::q B4 · The maximum value of the sigmoid derivative is
(a) 1 (b) 0.5 (c) 0.25 (d) 0.1
:::ans
**(c).** $\sigma'(0)=0.5\times0.5=0.25$. The lecture's "0.5" refers to $\sigma(0)$.
:::
:::

## C. Predict the Output

:::q C1 · What does this print?
```python
from collections import Counter
toks = "<s> a b a b a </s>".split()
bi = Counter(zip(toks, toks[1:]))
uni = Counter(toks[:-1])
print(bi[("a", "b")], uni["a"], bi[("a", "b")] / uni["a"])
```
:::ans
```output
2 3 0.6666666666666666
```

Explanation: "a" is followed by "b" twice and by "</s>" once, so $c(a)=3$ and $P(b\mid a)=2/3$.
:::
:::

:::q C2 · What does this print?
```python
import math
p = [0.25, 0.25, 0.25, 0.25]
pp = 2 ** (-sum(math.log2(x) for x in p) / len(p))
print(pp)
```
:::ans
```output
4.0
```

Uniform over 4 choices at every step gives perplexity 4.
:::
:::

## D. Find the Error

:::q D1 · The code below should compute add-1 probabilities. What is wrong?
```python
#@expect: error
bi = {("a", "b"): 2}
uni = {"a": 3}
V = 5
p = (bi[("a", "c")] + 1) / (uni["a"] + V)
print(p)
```
:::ans
The pair `("a", "c")` is unseen, so the dictionary lookup raises `KeyError`. A smoothing implementation must use `bi.get(("a", "c"), 0)` (or a `Counter`, which returns 0). This is the code-level version of "unseen means count 0, not an error".
:::
:::

## E. Write the Program

:::q E1 · Write a function `add1(bi, uni, V, a, b)` returning the add-1 bigram probability, and test it. (4 marks)
:::ans
```python
from collections import Counter
def add1(bi, uni, V, a, b):
    return (bi[(a, b)] + 1) / (uni[a] + V)

bi = Counter({("the", "cat"): 2})
uni = Counter({"the": 4})
print(add1(bi, uni, 10, "the", "cat"))
print(add1(bi, uni, 10, "the", "dog"))
```

```output
0.21428571428571427
0.07142857142857142
```
:::
:::

## F. Long Answer

:::q F1 · Explain the smoothing ladder from add-1 to Kneser-Ney, and say why each step is needed. (8 marks)
:::ans
Outline: (1) zero counts break perplexity; (2) add-1 fixes zeros but steals mass, so add-$k$; (3) one global weight is crude, so back-off and interpolation; (4) Good-Turing estimates unseen mass by $N_1/N$ with $c^*=(c+1)N_{c+1}/N_c$ (bird example, $3/18$); (5) discounting by about 0.75 gives absolute discounting; (6) Kneser-Ney replaces the unigram by continuation probability. Cite lessons 1.4 and 1.5.
:::
:::

:::q F2 · Derive $\partial E/\partial w^{(1)}_{ji}$ for a one-hidden-layer tanh MLP with squared error. (6 marks)
:::ans
$E=\tfrac12\sum_k(y_k-t_k)^2$ and the chain $\dfrac{\partial E}{\partial w_{ji}}=\dfrac{\partial E}{\partial z_j}\dfrac{\partial z_j}{\partial a_j}\dfrac{\partial a_j}{\partial w_{ji}}$ with $\partial E/\partial z_j=\sum_k(y_k-t_k)w^{(2)}_{kj}$, $\partial z_j/\partial a_j=1-z_j^2$ and $\partial a_j/\partial w_{ji}=x_i$. So $\partial E/\partial w^{(1)}_{ji}=\big[\sum_k(y_k-t_k)w^{(2)}_{kj}\big](1-z_j^2)x_i$. Lesson 2.3.
:::
:::
