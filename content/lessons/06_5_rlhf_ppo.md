**RLHF treats the language model as a policy that picks the next token, scores the finished answer with a reward model, and updates the policy with PPO while a KL term keeps it near the reference model.** REINFORCE gives the gradient; Q-values, advantages and clipping tame its variance.

## The Problem: The Reward Arrives Only at the End

> [!lecture] The cast
> Reinforcement learning has a **policy** that chooses an **action** in a **state**, and a **reward**. A blind person exploring a floor takes steps; walking into a closed door is a negative reward and he learns not to go there. Here the **policy $\pi_\theta$ is the LLM itself**. The **state** $s_t$ is the prompt plus the tokens generated so far; the **action** $a_t$ is the next token (a $|V|\approx$ 50K-way choice); the new state is $s_{t+1}=(s_t,a_t)$. There is no explicit environment, and **the reward arrives only at the end of the sentence**: "Taj Mahal is in Agra <eos>" gives 0, 0, 0, ..., $+1$.

The objective from the previous lesson is the **regularized reward** $r_s(x,y)=r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$.

## Why Ordinary Gradient Descent Fails: REINFORCE

The expectation is over *samples* from the policy, which is not differentiable. Use the **log-derivative trick** $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$:

$$\nabla_\theta J=\sum_y\nabla_\theta\pi_\theta(y\mid x)\,r_s(x,y)=\mathbb{E}_{y\sim\pi_\theta}\big[r_s(x,y)\,\nabla_\theta\log\pi_\theta(y\mid x)\big]\approx\frac1n\sum_{k=1}^{n}r_s(x,y_k)\,\nabla_\theta\log\pi_\theta(y_k\mid x)$$

The sum over all $y$ is intractable; the sampled average is the **Monte Carlo** estimate. The sequence log-probability splits into tokens, and plain REINFORCE multiplies **every** token's gradient by the one final reward.

> [!compiler] Two precision points
> (a) The lecture treats $r_s$ as constant in $\theta$, but $r_s$ contains $\log\pi_\theta$. The missing term is $-\beta\,\mathbb{E}_{y\sim\pi_\theta}[\nabla_\theta\log\pi_\theta(y|x)]$, which is **zero** (the expected score is zero), so the formula is exact. (b) Lecture 25 says "the derivative of the log of the reward"; it means $\nabla_\theta\log\pi_\theta$ (a transcript slip).

## The Credit-Assignment Problem

If the model writes "Taj Mahal is in **Paris**", the negative reward also pushes down "Taj", which was correct. The reward at "Taj" depends on the *future*, hence high variance. Fix 1: replace $r$ by the **Q-function**:

$$Q(s_t,a_t)=\mathbb{E}\Big[\sum_{k\ge t}\gamma^{\,k-t}\,r_k\Big],\qquad V(s_t)=\mathbb{E}_{a\sim\pi}\,Q(s_t,a)$$

with $\gamma\in(0,1)$ so nearer tokens matter more. Since the next state is determined, $Q(s_t,a_t)=r_t+\gamma V(s_{t+1})$. Fix 2: the **advantage** $A(s_t,a_t)=Q-V=r_t+\gamma V(s_{t+1})-V(s_t)$, "the contribution of the action over an average action at the same state". The value $V$ is a **linear head on every token embedding**, trained by MSE against the reward-to-go.

:::analogy
A team wins a match. Giving every player the same reward ignores who played well. The advantage says how much better this player's move was than the average move in the same position.
:::

```python
gamma = 0.9
rewards = [0, 0, 0, 1]                       # reward only at <eos>
V = [0.3, 0.4, 0.6, 0.8, 0.0]                # value head outputs for s_0..s_3, then terminal
rtg = []
for t in range(len(rewards)):
    rtg.append(sum(gamma ** (k - t) * rewards[k] for k in range(t, len(rewards))))
adv = [rewards[t] + gamma * V[t + 1] - V[t] for t in range(len(rewards))]
print("reward-to-go:", [round(x, 3) for x in rtg])
print("advantage   :", [round(x, 3) for x in adv])
```

```output
reward-to-go: [0.729, 0.81, 0.9, 1.0]
advantage   : [0.06, 0.14, 0.12, 0.2]
```

## Reuse Samples: Importance Weights and PPO-Clip

Regenerating from the new policy after every update is slow. Sample from an older fixed $\pi_{\theta_k}$ and reweight by $\rho_t=\pi_\theta(a_t|s_t)/\pi_{\theta_k}(a_t|s_t)$. If we update too far, $\rho_t$ explodes. **PPO-Clip** keeps it in $[1-\epsilon,1+\epsilon]$:

$$\mathcal{J}_{\text{clip}}=\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\ \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\Big]$$

```python
def ppo_term(rho, A, eps=0.2):
    clipped = max(1 - eps, min(1 + eps, rho))
    return min(rho * A, clipped * A)

for A in (1.0, -1.0):
    print("A =", A, [round(ppo_term(r, A), 2) for r in (0.5, 1.0, 1.5)])
```

```output
A = 1.0 [0.5, 1.0, 1.2]
A = -1.0 [-0.8, -1.0, -1.5]
```

> [!compiler] Reading the clip, and the cost
> For $A>0$ the objective stops rewarding once $\rho>1+\epsilon$; for $A<0$ it stops rewarding once $\rho<1-\epsilon$. Standard PPO-RLHF keeps **four models** in memory: policy, frozen reference, reward model, value head. Hold this thought for DPO.

```text
# PPO-RLHF loop
repeat until convergence:
    sample prompts; generate y with pi_old (once, reuse)
    r = reward_model(x, y)                   # only at <eos>; KL term adds penalty
    R[t] = sum_{k>=t} gamma^(k-t) r[k];  A[t] = r[t] + gamma V(s[t+1]) - V(s[t])
    for a few epochs:
        loss_pi = -mean(min(rho*A, clip(rho,1-eps,1+eps)*A));  loss_v = mean((V - R)^2)
```

:::def
**PPO** (Proximal Policy Optimization) maximises the clipped surrogate $\mathbb{E}_t[\min(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]$, reusing samples from an old policy while preventing large policy updates.
:::

:::key
- State = prompt + tokens so far; action = next token; reward only at the end.
- Objective: reward minus $\beta\cdot$KL to the reference. REINFORCE gradient via the log-derivative trick.
- $Q$, advantage and PPO-Clip reduce variance and allow sample reuse.
- PPO-RLHF keeps four models: policy, reference, reward, value.
:::

:::mistakes
- "RLHF equals PPO." RLHF is the human-preference setup; PPO is one optimizer, DPO another.
- "Maximize reward is the goal." Maximize reward **minus** $\beta\cdot$KL, or the policy hacks the reward.
:::

:::exam
- *(MidSem Q7(b))* PPO loss with $\pi_{ref}$ the instruction-tuned model: $\mathcal{L}_{PPO}=-\mathbb{E}_{x,\,y\sim\pi_\theta}\big[r(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\big]$.
- *"Pseudo-code GAE-free advantages for $r=(0,0,0,1)$, $\gamma=0.9$ and given $V$."* (Tier 3)
:::

## Quick Check

:::q Q1 · What is the advantage and why use it instead of the raw reward?
:::ans
$A=Q-V$: how much better the chosen token is than an average token in the same state. It credits individual tokens and reduces variance compared with giving every token the final reward.
:::
:::

:::q Q2 · What does the clip do when $A>0$ and $\rho$ is far above $1+\epsilon$?
:::ans
It stops increasing the objective, removing the incentive to push the policy further from the old policy.
:::
:::
