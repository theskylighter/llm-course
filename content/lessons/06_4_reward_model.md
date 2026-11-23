**A reward model scores a response with a single number, and it is trained from human comparisons using the Bradley-Terry model, whose loss is the negative log-sigmoid of the reward gap.** Asking humans which of two answers is better is easier than asking for a score.

## The Problem: Humans Cannot Be in the Loop at Every Step

After instruction tuning the model knows how to respond but not what the *appropriate* response is. Humans must say which is better. An ideal reward is a human, but that is costly, slow and needed at every training iteration. So train another LLM as the proxy.

Asking humans for a *number* is hard and subjective (a poem scored 0.5 versus 0.7), so ask for a *comparison*: given $x$ and two outputs, which is better? Data: $(x,y^+,y^-)$. If a human labels, it is **RLHF**; if an AI labels, **RLAIF**.

**Architecture:** an encoder or decoder LM, average the final token embeddings, a freshly initialised **linear layer** gives a scalar $r_\theta(x,y)$.

## Bradley-Terry

> [!lecture] Bradley-Terry
> Entities with scores $p_i>0$: $P(y_i\succ y_j)=\frac{p_i}{p_i+p_j}$. Write $p_i=e^{r_i}$ (that is, $r_i=\log p_i$) and parameterize the unknown oracle reward by the reward model:
> $$P(y^+\succ y^-\mid x)=\frac{e^{r_\theta(x,y^+)}}{e^{r_\theta(x,y^+)}+e^{r_\theta(x,y^-)}}=\sigma\!\big(r_\theta(x,y^+)-r_\theta(x,y^-)\big)$$
> (divide numerator and denominator by $e^{r_\theta(x,y^-)}$). Maximum likelihood gives the reward-model loss.

$$\mathcal{L}_{RM}(\theta)=-\mathbb{E}_{(x,y^+,y^-)}\log\sigma\!\big(r_\theta(x,y^+)-r_\theta(x,y^-)\big)$$

Pairs are needed only for training; at inference the model scores a single $(x,y)$.

:::analogy
Chess ratings: you never see a "true skill" number, only who beat whom. The rating gap is turned into a win probability, and ratings are fitted so observed wins are likely.
:::

```python
import math
sigma = lambda z: 1 / (1 + math.exp(-z))

def loss(r_plus, r_minus):
    return -math.log(sigma(r_plus - r_minus))

for rp, rm in [(2.0, 0.0), (0.0, 0.0), (0.0, 2.0)]:
    print(f"r+={rp}, r-={rm}: P(y+ preferred)={sigma(rp - rm):.3f}, loss={loss(rp, rm):.3f}")
# adding a constant to both rewards changes nothing
print("shift-invariant:", abs(loss(5.0, 3.0) - loss(2.0, 0.0)) < 1e-12)
```

```output
r+=2.0, r-=0.0: P(y+ preferred)=0.881, loss=0.127
r+=0.0, r-=0.0: P(y+ preferred)=0.500, loss=0.693
r+=0.0, r-=2.0: P(y+ preferred)=0.119, loss=2.127
shift-invariant: True
```

**Where the data comes from:** prompts $x$ from a held-out slice of the instruction-tuning set (or user prompts); outputs from the model being aligned or from other LLMs for diversity; preferences from humans or an AI judge. Public sets: OpenAI *Summarize from Feedback* and **UltraFeedback** (GPT-4 as judge). **Constitutional AI** (Anthropic) asks an LLM, with 16 written principles, which of two responses is less harmful.

## Reward Hacking and the KL Leash

If the policy only maximizes $r$, it learns the reward model's loopholes: verbose answers, emoticons, special symbols. The lecture's analogy: the student who fills the whole page because the examiner may reward volume. So keep the policy near the **reference policy** $\pi_{ref}$ (the instruction-tuned model) using KL divergence:

$$\max_\theta\ \mathbb{E}_{x}\ \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\Big[r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}\Big]$$

:::def
A **reward model** $r_\theta(x,y)$ maps a prompt and response to a scalar, trained by minimising $-\log\sigma(r_\theta(x,y^+)-r_\theta(x,y^-))$ on human preference pairs.
:::

:::key
- Compare, do not score: data are $(x,y^+,y^-)$.
- Bradley-Terry with $p_i=e^{r_i}$ gives $\sigma$ of the reward difference.
- Reward hacking is controlled by a KL penalty with weight $\beta$ to the reference policy.
:::

:::mistakes
- "Reward model equals policy." Two LLMs: the policy generates; the reward model scores a full $(x,y)$.
- "The reward is given per token." Only at the end of the sentence.
:::

:::exam
- *"Derive $\mathcal{L}_{RM}$ from Bradley-Terry with $p_i=e^{r_i}$."* (Tier 2)
:::

## Quick Check

:::q Q1 · Why train on comparisons rather than scores?
:::ans
People disagree and are inconsistent about absolute numbers, but comparing two outputs is easier and more reliable.
:::
:::

:::q Q2 · Why does a KL term protect against reward hacking?
:::ans
It penalises the policy for drifting from the reference model, which limits how far it can move to exploit reward-model loopholes.
:::
:::
