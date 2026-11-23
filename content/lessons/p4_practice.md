**Attempt each question before opening its answer.** This set covers Chapters 6 and 7 and revisits attention and pretraining.

## A. Short Answers

:::q A1 · Distinguish instruction tuning from in-context learning. (3 marks)
:::ans
Instruction tuning fine-tunes the weights on (instruction, optional input, output) examples with cross-entropy on the output. In-context learning supplies the instruction and examples in the prompt and updates **no** parameters, so the prompt must be sent on every call. Lessons 6.1 and 6.3.
:::
:::

:::q A2 · Why does Self-Instruct use output-first generation for classification tasks? (2 marks)
:::ans
When asked to generate inputs first, the model favours popular labels and the data become unbalanced. Giving the label first and asking for a matching input balances the classes. Lesson 6.2.
:::
:::

:::q A3 · Why is a reward model trained from comparisons, and what loss is used? (4 marks)
:::ans
Absolute scores are subjective; comparisons are easier. By Bradley-Terry with $p_i=e^{r_i}$, $P(y^+\succ y^-)=\sigma(r_\theta(x,y^+)-r_\theta(x,y^-))$, and the loss is $-\log\sigma(\cdot)$ of the gap. Lesson 6.4.
:::
:::

:::q A4 · Why is the KL term needed in the RLHF objective? (2 marks)
:::ans
Without it the policy exploits reward-model loopholes (verbosity, symbols). The KL penalty $\beta\log\pi_\theta/\pi_{ref}$ keeps the policy near the instruction-tuned reference. Lesson 6.4.
:::
:::

:::q A5 · Compare PPO and DPO. (4 marks)
:::ans
| | PPO-RLHF | DPO |
|---|---|---|
| Online / offline | online | offline |
| Reward model | explicit | implicit, $\beta\log\pi_\theta/\pi_{ref}$ |
| Models in memory | policy, reference, reward, value | policy, reference |

Lesson 6.6.
:::
:::

## B. True or False with Justification

:::q B1 · "QLoRA is quantization-aware training." (2 marks)
:::ans
**False.** The 4-bit base is quantized once after pre-training and kept frozen; only the LoRA matrices train. Lesson 7.4.
:::
:::

:::q B2 · "TALMs require extensive pre-training on tool-specific datasets." (2 marks)
:::ans
**False.** They bootstrap from a few demonstrations by self-play and fine-tuning. See lesson 8.2.
:::
:::

:::q B3 · "LoRA adds inference latency because it adds parameters." (2 marks)
:::ans
**False.** The update $BA$ can be merged into $W$ after training. Adapters, which sit in series, do add latency. Lesson 7.2.
:::
:::

## C. Predict the Output

:::q C1 · What does this print?
```python
import math
sigma = lambda z: 1 / (1 + math.exp(-z))
print(round(-math.log(sigma(3.0 - 1.0)), 3))
```
:::ans
```output
0.127
```

The reward gap is 2, so the probability that $y^+$ wins is $\sigma(2)\approx0.881$ and the loss is $-\ln0.881$.
:::
:::

:::q C2 · What does this print?
```python
def ppo_term(rho, A, eps=0.2):
    return min(rho * A, max(1 - eps, min(1 + eps, rho)) * A)
print(ppo_term(1.5, 2.0), ppo_term(0.5, -2.0))
```
:::ans
```output
2.4 -1.6
```

For $A>0$ the ratio is clipped at $1.2$ (so $2.4$); for $A<0$ the unclipped $\rho A=-1.0$ is larger than the clipped $0.8\cdot(-2)$, so `min` picks $-1.6$.
:::
:::

:::q C3 · What does this print?
```python
import numpy as np
x = np.array([0.0, 0.5, -1.27, 2.54])
s = 127 / abs(x).max()
print(s, np.round(s * x).astype(int))
```
:::ans
```output
50.0 [  0  25 -64 127]
```

The scale is 127 divided by the largest absolute value.
:::
:::

## D. Numericals

:::q D1 · LoRA on a $4096\times4096$ matrix with $r=8$: parameters and fraction. (3 marks)
:::ans
$r(d+k)=8\times8192=65{,}536$ against $16{,}777{,}216$, about 0.39%. Lesson 7.2.
:::
:::

:::q D2 · Memory to fine-tune a 65B model fully, and weights alone at 4 bits. (3 marks)
:::ans
$65\text{B}\times12$ bytes $=780$ GB; at 4 bits the weights alone are $65\text{B}\times0.5=32.5$ GB. Lesson 7.4.
:::
:::

## E. Long Answer

:::q E1 · Trace the pipeline from a base model to an aligned assistant, naming the data and loss at each stage. (8 marks)
:::ans
Pre-training: next-token prediction on raw text. Instruction tuning: (instruction, input, output) triples, cross-entropy on output tokens (FLAN, Self-Instruct). Reward model: preference pairs, $-\log\sigma(r^+-r^-)$. Policy optimisation: PPO with KL penalty to the reference, or DPO with the closed-form loss. Mention the alignment tax and sensitivity. Lessons 6.1 to 6.6.
:::
:::

:::q E2 · Compare soft prompt tuning, prefix tuning, adapters and LoRA. (8 marks)
:::ans
Prompt tuning: vectors at the input only, model frozen, weak on small models. Prefix tuning: vectors at every layer, MLP reparameterization, about 0.1% of parameters. Adapters: bottleneck modules in series, about 3.6% parameters, latency cost. LoRA: low-rank $\Delta W=BA$, $B=0$ init, mergeable, no latency. Lessons 7.1 and 7.2.
:::
:::
