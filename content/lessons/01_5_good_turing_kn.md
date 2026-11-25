<<SVG:good_turing_kneser_ney>>

*Figure: Good–Turing estimates unseen mass from singleton counts; Kneser–Ney rewards words with diverse left contexts.*

**Add-one is too blunt, so the smoothing ladder climbs: Good-Turing estimates unseen mass from things seen once, interpolation mixes N-gram orders, and Kneser-Ney asks how likely a word is to appear as a *new continuation*.** Examiners like this ladder.

## The Problem: Each Fix Fails in a New Way

Add-1 steals too much mass. Add-$k$ needs a tuned $k$. A unigram prior uses one global weight. Each failure motivates the next step.

> [!slides] The smoothing ladder
> | Method | Formula | Why it is not enough |
> |---|---|---|
> | Laplace (add-1) | $\dfrac{c+1}{c(w_{i-1})+\lvert V\rvert}$ | steals too much mass from seen events when $V$ is large |
> | Add-$k$ | $\dfrac{c+k}{c(w_{i-1})+k\lvert V\rvert}$ | $k$ tuned on held-out data, still crude |
> | Back-off | use trigram if evidence, else bigram, else unigram | hard switch |
> | Interpolation | $\lambda_1 P_3+\lambda_2 P_2+\lambda_3 P_1$ | usually beats back-off; $\lambda$ may depend on context |
> | Good-Turing | $c^*=\dfrac{(c+1)N_{c+1}}{N_c}$, $P_{GT}(\text{unseen})=\dfrac{N_1}{N}$ | needs $N_c$ smooth for large $c$ |
> | Absolute discounting | $\dfrac{\max(c-d,0)}{c(w_{i-1})}+\lambda(w_{i-1})P(w_i)$ | uses plain unigram $P(w)$ |
> | **Kneser-Ney** | same, but $P(w)\to P_{\text{continuation}}(w)$ | the state of the art here |

## Good-Turing: Use Things Seen Once to Estimate the Unseen

> [!scribe] Worked Example: Good-Turing birdwatching (lecture 4)
> Seen: 10 flamingo, 3 kingfisher, 2 roller, 1 woodpecker, 1 peacock, 1 crane. $N=18$, $N_1=3$, $N_2=1$.
> - Next bird is a **new species**: $P^*_{GT}=N_1/N=3/18=1/6$.
> - Woodpecker (seen once): $c^*=2\cdot N_2/N_1=2/3$, so $P^*_{GT}=\tfrac{2/3}{18}=\tfrac1{27}<\tfrac1{18}$ (MLE). Mass moved from seen to unseen.
> - Church-Gale: empirically $c^*\approx c-0.75$ for $c\ge 2$. That is the idea behind absolute discounting.

```python
from fractions import Fraction
from collections import Counter

counts = {"flamingo": 10, "kingfisher": 3, "roller": 2, "woodpecker": 1, "peacock": 1, "crane": 1}
N = sum(counts.values())
Nc = Counter(counts.values())          # frequency of frequencies
print("N =", N, " N1 =", Nc[1], " N2 =", Nc[2])
print("P(new species) =", Fraction(Nc[1], N))
c_star = Fraction(2 * Nc[2], Nc[1])
print("c* for count 1 =", c_star)
print("P_GT(woodpecker) =", c_star / N, "versus MLE", Fraction(1, N))
```

```output
N = 18  N1 = 3  N2 = 1
P(new species) = 1/6
c* for count 1 = 2/3
P_GT(woodpecker) = 1/27 versus MLE 1/18
```

## Kneser-Ney: Continuation, Not Frequency

> [!lecture] Kneser-Ney intuition: the Shannon game
> "My breakfast is incomplete without a cup of ___": *coffee* or *Angeles*? If "Angeles" is frequent in the corpus, a plain unigram says Angeles. But "Angeles" nearly always follows "Los". So ask a different question, not "how likely is $w$?" but "**how likely is $w$ to appear as a novel continuation?**" Count how many *different bigram types* the word completes.
> — *NPTEL Lecture 4*

> [!slides] Kneser-Ney
> $$P_{\text{cont}}(w)=\frac{\lvert\{w' : c(w',w)>0\}\rvert}{\lvert\{(w'_{j-1},w'_j): c(w'_{j-1},w'_j)>0\}\rvert}$$
> $$P_{KN}(w_i\mid w_{i-1})=\frac{\max(c(w_{i-1},w_i)-d,0)}{c(w_{i-1})}+\lambda(w_{i-1})P_{\text{cont}}(w_i),\qquad \lambda(w_{i-1})=\frac{d}{c(w_{i-1})}\lvert\{w:c(w_{i-1},w)>0\}\rvert$$

```python
bigrams = {("los", "angeles"): 50, ("a", "coffee"): 2, ("the", "coffee"): 3, ("hot", "coffee"): 1,
           ("the", "city"): 4, ("big", "city"): 2}
freq = {}
for (a, b), c in bigrams.items():
    freq[b] = freq.get(b, 0) + c
types = {}
for (a, b) in bigrams:
    types[b] = types.get(b, 0) + 1
total_types = len(bigrams)
print("raw frequency:", {w: freq[w] for w in ("angeles", "coffee")})
for w in ("angeles", "coffee"):
    print(w, "P_cont =", round(types[w] / total_types, 3))
```

```output
raw frequency: {'angeles': 50, 'coffee': 6}
angeles P_cont = 0.167
coffee P_cont = 0.5
```

:::def
**Kneser-Ney smoothing** is absolute discounting with the lower-order distribution replaced by the **continuation probability**: the number of distinct left contexts a word completes, divided by the total number of distinct bigram types.
:::

:::key
- Good-Turing: $c^*=(c+1)N_{c+1}/N_c$ and $P(\text{unseen})=N_1/N$.
- Back-off picks one order. Interpolation **mixes** all orders.
- Kneser-Ney uses continuation counts, not unigram frequency ("Los Angeles").
:::

:::mistakes
- "Kneser-Ney uses unigram frequency." It uses the count of distinct left contexts.
- "Back-off equals interpolation." Back-off switches; interpolation blends.
:::

:::exam
- *"Why does Kneser-Ney use continuation probability? Use Los Angeles."* (short)
- *Good-Turing numerical:* given frequency-of-frequency counts, find $c^*$ and the unseen mass.
:::

## Quick Check

:::q Q1 · In the bird example, why is $P(\text{new species})=3/18$?
:::ans
$N_1=3$ species were seen exactly once and $N=18$ birds were seen in total; Good-Turing uses $N_1/N$ as the probability mass of unseen events.
:::
:::

:::q Q2 · Why is "Angeles" a poor choice after "a cup of" in Kneser-Ney?
:::ans
It is frequent but almost always follows "Los", so it completes very few distinct bigram types and has a low continuation probability.
:::
:::
