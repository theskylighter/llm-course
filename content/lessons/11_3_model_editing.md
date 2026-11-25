<<SVG:model_editing_three_tests>>

*Figure: Test an edit on the target, paraphrases, and unrelated prompts to measure efficacy, generalization, and locality.*

**Model editing changes one fact in a trained model without retraining; a good edit is efficacious, generalises to paraphrases and leaves unrelated behaviour alone.** KnowledgeEditor predicts the weight update with a hypernetwork under a KL constraint; ROME writes a rank-one update into a mid-layer MLP.

## The Problem: "A Fact Changed. Can I Fix It Without Retraining?"

"The CEO of X is now Y." Retrain? No. **Fine-tune on the one fact?** The model may forget unrelated things or only fix the exact wording. Good editing has three properties:

- **Efficacy:** the edited prompt gives the new answer.
- **Generalization:** paraphrases do too.
- **Specificity (locality):** unrelated prompts are unchanged.

## KnowledgeEditor (EndSem Q3, 6 marks)

> [!compiler] Worked answer (De Cao et al. 2021; no class notes in the repo)
> **(a) Representation and computation of $\Delta\theta$.** Rather than fine-tune, KnowledgeEditor trains a **hypernetwork $g_\phi$** that *predicts the parameter update*: $\theta'=\theta+\Delta\theta$ with $\Delta\theta=g_\phi(x,\ y,\ a,\ \nabla_\theta\ell)$, using the edit input $x$, the model's old output $y$, the desired new output $a$ and the gradient of the loss.
> **(b) Constrained optimization.** (i) Train $\phi$ to minimize
> $$\min_{\phi}\ \sum_{x\in\mathcal{E}}\ell\big(f(x;\theta+\Delta\theta),\,a\big)\quad\text{s.t.}\quad\sum_{x'\in\mathcal{N}}\mathrm{KL}\big(f(x';\theta)\,\Vert\,f(x';\theta+\Delta\theta)\big)\le m,$$
> where $\mathcal{E}$ is the edit prompt and its paraphrases (success and generalization) and $\mathcal{N}$ is a set of **other inputs** (locality), $m$ a margin. In practice it is trained with a Lagrangian-style penalty. (ii) **Purpose of the constraint:** the edit must not change the model's behavior on unrelated inputs. KL measures **behavior** change; $L_2$ measures weight change.

:::analogy
Correcting a wrong page in a printed encyclopedia: you want one erratum slip that fixes that entry and every reference to it, without the printer reshuffling other pages.
:::

## ROME: Rank-One Model Editing

> [!compiler] ROME, MEMIT, IKE, GRACE (from the papers)
> **ROME** (*Locating and Editing Factual Associations*): *causal tracing* finds that **mid-layer MLPs at the last subject token** carry the fact. Treat the MLP output matrix $W$ as a **linear associative memory** $Wk\approx v$ with key $k$ (the subject) and value $v$ (the fact). To store a new pair $(k_*,v_*)$:
> $$W'=W+\frac{(v_*-Wk_*)\,(C^{-1}k_*)^{\top}}{(C^{-1}k_*)^{\top}k_*},\qquad C=\mathbb{E}[kk^{\top}]$$
> $C$ (key covariance) ensures the update touches directions *unused by other keys*. **MEMIT** spreads many edits over several layers. **IKE** (in-context knowledge editing) changes no weights: the new fact goes in the **prompt**. **GRACE** keeps a discrete codebook of edits.

```python
import numpy as np
rng = np.random.default_rng(0)
d = 6
W = rng.normal(size=(d, d))
K = rng.normal(size=(d, 500))                 # many existing keys
C = K @ K.T / 500
k_star = rng.normal(size=d)
v_star = rng.normal(size=d)
u = np.linalg.solve(C, k_star)                # C^{-1} k*
W2 = W + np.outer(v_star - W @ k_star, u) / (u @ k_star)
print("W' k* == v*:", np.allclose(W2 @ k_star, v_star))
other = rng.normal(size=d)
print("a random other key changes by", round(float(np.linalg.norm(W2 @ other - W @ other)), 3),
      "vs a typical output norm", round(float(np.linalg.norm(W @ other)), 3))
print("rank of the update:", np.linalg.matrix_rank(W2 - W))
```

```output
W' k* == v*: True
a random other key changes by 0.586 vs a typical output norm 3.731
rank of the update: 1
```

:::def
**Model editing** modifies a pre-trained model so that a specific fact changes, judged by efficacy, generalization and locality. **ROME** applies a rank-one update $W'=W+\dfrac{(v_*-Wk_*)(C^{-1}k_*)^\top}{(C^{-1}k_*)^\top k_*}$ to a mid-layer MLP.
:::

:::key
- Three properties: efficacy, generalization, locality.
- KnowledgeEditor = hypernetwork predicting $\Delta\theta$ + KL locality constraint.
- ROME = rank-one update with $W'k_*=v_*$; MEMIT batches; IKE prompts; GRACE codebook.
:::

:::mistakes
- "Fine-tuning on a fact is editing." It damages locality.
- "KL and $L_2$ are interchangeable." KL measures behavior, $L_2$ measures weights.
:::

:::exam
- *(EndSem Q3, 6 marks)* KnowledgeEditor: representation of $\Delta\theta$, constrained optimization with KL, purpose of the constraint.
- *"Show ROME's update stores $(k_*,v_*)$."* $W'k_*=Wk_*+(v_*-Wk_*)\cdot1=v_*$.
:::

## Quick Check

:::q Q1 · What is the purpose of the KL constraint in KnowledgeEditor?
:::ans
To keep the model's predictions on unrelated inputs unchanged (locality) while the edit is applied.
:::
:::

:::q Q2 · Which part of the transformer does ROME edit?
:::ans
A mid-layer MLP output matrix at the last subject token, viewed as a key-value associative memory.
:::
:::
