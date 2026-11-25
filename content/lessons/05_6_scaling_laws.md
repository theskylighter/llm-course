<<SVG:scaling_laws_chinchilla_balance>>

*Figure: Under a fixed compute budget, Chinchilla’s rule scales data and parameters together at roughly 20 tokens per parameter.*

**Test loss falls as a power law in parameters, data and compute, and Chinchilla showed that for a fixed compute budget model size and tokens should grow together, about 20 tokens per parameter.** Many earlier large models were under-trained.

## The Problem: How Do We Split a Compute Budget?

Pre-training is expensive, so a pointed question follows: **given a fixed compute budget, how do we split it between model size and data?**

**Setting the stage: emergence.** Larger models appear to *suddenly* do new things. If this were true we could never predict *which* abilities appear, or when. So the lecture looks for a curve that *does* predict loss.

## Kaplan Laws (OpenAI, 2020)

Test loss falls as a **power law** in each resource when the others are not a bottleneck: $L(N)\propto N^{-\alpha_N}$, $L(D)\propto D^{-\alpha_D}$, $L(C)\propto C^{-\alpha_C}$ ($N$ non-embedding parameters, $D$ tokens, $C$ compute). On log-log axes these are straight lines: "scale is predictable". Takeaways: (1) scale parameters **and** data together, or you overfit; (2) training time follows a similar smooth law; (3) model shape (depth versus width) matters little; (4) the embedding matrix does not matter; (5) dataset composition matters little for the trend.

## Chinchilla (Hoffmann et al., 2022)

> [!compiler] The result, from the paper (slide equations are images)
> Fit $L(N,D)=E+\dfrac{A}{N^{\alpha}}+\dfrac{B}{D^{\beta}}$ with $E$ the irreducible loss. Under a compute budget $C\approx6ND$, the optimum grows **$N$ and $D$ in equal proportion** ($N_{\rm opt},D_{\rm opt}\propto C^{0.5}$), giving roughly **20 tokens per parameter**. Kaplan's fit put most extra compute into model size. So earlier large models were **under-trained**: Gopher (280B on about 300B tokens) used about the same compute as Chinchilla (70B on 1.4T tokens) and lost to it.

:::analogy
Baking: a bigger oven (parameters) is useless without enough dough (data); a mountain of dough with a tiny oven is also wasted. Chinchilla says to grow both together.
:::

```python
import math

def chinchilla(C):
    N = math.sqrt(C / 120)          # C = 6*N*D with D = 20*N
    return N, 20 * N

for C in (1e23, 1e24):
    N, D = chinchilla(C)
    print(f"C={C:.0e}: N={N:.2e} params, D={D:.2e} tokens")

print("Gopher  6*N*D:", f"{6 * 280e9 * 300e9:.2e}")
print("Chinchilla 6*N*D:", f"{6 * 70e9 * 1.4e12:.2e}")
print("tokens per param: LLaMA-1 7B ~", round(1e12 / 7e9), "; GPT-3 ~", round(300e9 / 175e9, 1))
```

```output
C=1e+23: N=2.89e+10 params, D=5.77e+11 tokens
C=1e+24: N=9.13e+10 params, D=1.83e+12 tokens
Gopher  6*N*D: 5.04e+23
Chinchilla 6*N*D: 5.88e+23
tokens per param: LLaMA-1 7B ~ 143 ; GPT-3 ~ 1.7
```

> [!compiler] Chinchilla-optimal is not "best to deploy"
> The law minimises loss for a fixed *training* budget. If a model will serve billions of queries, a **smaller model trained far past 20 tokens/parameter** is cheaper at inference. LLaMA-1 7B was trained on about 1T tokens, roughly 140 tokens per parameter. GPT-3 (about 1.7 tokens per parameter) is under-trained by this rule.

## Is Emergence Real? (Schaeffer et al., 2023)

The "jump" can be an artifact of the **metric**: more than 92% of the BIG-bench emergence claims use metrics that are **non-linear** (exact string match) or **discontinuous** (multiple-choice grade). Switching to **edit distance** or the **Brier score** gives smooth, predictable curves.

:::def
**Chinchilla scaling:** for a compute budget $C\approx6ND$, loss is minimised when parameters $N$ and tokens $D$ are scaled in equal proportion, about $D\approx20N$.
:::

:::key
- Loss follows power laws in $N$, $D$ and $C$.
- Kaplan: spend extra compute mostly on model size. Chinchilla: scale $N$ and $D$ equally (about 20 tokens per parameter).
- Emergence claims depend on the metric.
:::

:::mistakes
- "Chinchilla says smaller models are better." It says *balanced* scaling for a fixed training budget.
:::

:::exam
- *(MidSem Q6, 4 marks)* (a) Kaplan versus Chinchilla, no numbers. (b) "Emergence is a sudden spike regardless of metric": **false**.
- *Numerical:* compute-optimal $N$ and $D$ for $C=10^{23}$: $N\approx2.9\times10^{10}$, $D\approx5.8\times10^{11}$.
:::

## Quick Check

:::q Q1 · What does $C\approx6ND$ mean?
:::ans
Training compute in FLOPs is about six times the product of parameters and training tokens.
:::
:::

:::q Q2 · Why can a metric create an illusion of emergence?
:::ans
Non-linear or discontinuous metrics (exact match, multiple-choice grade) turn smooth improvement into an apparent jump; smooth metrics such as edit distance show gradual progress.
:::
:::
