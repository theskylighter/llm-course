**Attempt each question before opening its answer.** This set covers Chapter 3 and also revisits Chapters 1 and 2.

## A. True or False, with Justification

:::q A1 · "Self-attention eliminates the need for positional encoding." (2 marks)
:::ans
**False.** Self-attention is permutation-equivariant: shuffling the tokens shuffles the outputs and nothing else, so the model cannot see order without positional information. Lesson 3.1.
:::
:::

:::q A2 · "RoPE represents positions as complex numbers and tokens as pure rotations." (2 marks)
:::ans
**False**, the wrong way round. The query and key vectors are the (pairs of) complex numbers; positions are the rotations $e^{im\theta}$ applied to them. Lesson 3.10.
:::
:::

:::q A3 · "In cross-attention, the decoder state of layer $k$ attends to the encoder state of layer $k$." (2 marks)
:::ans
**False.** Every decoder layer attends to the **final** encoder output. The queries come from the decoder; keys and values from the encoder.
:::
:::

:::q A4 · "The causal mask is multiplied with the score matrix." (2 marks)
:::ans
**False.** It is added ($-\infty$ above the diagonal). Multiplying would give $0\times(-\infty)=\text{NaN}$ on allowed entries.
:::
:::

## B. Short Answers and Derivations

:::q B1 · Why is the attention score divided by $\sqrt{d_k}$? (3 marks)
:::ans
If $\mathbf{q}$ and $\mathbf{k}$ have independent unit-variance entries, $\mathbf{q}^\top\mathbf{k}$ has variance $d_k$, so for $d_k=64$ the standard deviation is 8. The softmax saturates to one-hot and its gradient vanishes. Dividing by $\sqrt{d_k}$ restores unit variance. Lesson 3.3.
:::
:::

:::q B2 · List the three attempts at positional encoding before the sinusoid and why each fails. (4 marks)
:::ans
(1) Raw index: unbounded and ambiguous about the sequence end. (2) Index divided by length: the same position gets different values in different sequence lengths. (3) Binary: unique but discrete, and adding 0/1 vectors to real embeddings is awkward; but each bit is periodic, which motivates sinusoids. Lesson 3.4.
:::
:::

:::q B3 · Compare BatchNorm and LayerNorm. (3 marks)
:::ans
| | BatchNorm | LayerNorm |
|---|---|---|
| Normalises | one feature across the batch | one token across its features |
| Batch dependence | yes | no |
| Train vs inference | differ | same |
| Variable length | awkward | natural |
:::
:::

:::q B4 · Parameters of one encoder layer for $d_{\text{model}}=512$, $d_{ff}=2048$ (no biases). (3 marks)
:::ans
Attention $4d^2=1{,}048{,}576$; FFN $2\cdot512\cdot2048=2{,}097{,}152$; total $3{,}145{,}728=12d^2$.
:::
:::

## C. Predict the Output

:::q C1 · What does this print?
```python
import numpy as np
S = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
S = np.where(np.triu(np.ones((3, 3), bool), 1), -np.inf, S)
A = np.exp(S - S.max(1, keepdims=True))
A /= A.sum(1, keepdims=True)
print(np.round(A, 3))
```
:::ans
```output
[[1.    0.    0.   ]
 [0.269 0.731 0.   ]
 [0.09  0.245 0.665]]
```

Row 0 attends only to itself (weight 1); row 1 to positions 0 and 1; row 2 to all three.
:::
:::

:::q C2 · What does this print?
```python
import numpy as np
x = np.array([2.0, 4.0, 6.0, 8.0])
y = (x - x.mean()) / x.std()
print(round(float(x.mean()), 2), round(float(x.std()), 3), np.round(y, 3))
```
:::ans
```output
5.0 2.236 [-1.342 -0.447  0.447  1.342]
```

The mean is 5 and the population standard deviation is $\sqrt5\approx2.236$; the normalised vector has mean 0 and standard deviation 1.
:::
:::

## D. Find the Error

:::q D1 · The programmer wants multi-head attention with $d=10$ and $h=4$. What goes wrong?
```python
#@expect: error
d, h = 10, 4
assert d % h == 0, "embed_size must be divisible by heads"
dk = d // h
```
:::ans
$10$ is not divisible by $4$, so $d_k=d/h$ is not an integer and the concatenated heads would not be $d$ wide. The assertion fails, as in the course code. Lesson 3.5.
:::
:::

## E. Long Answer

:::q E1 · Describe the encoder and decoder blocks of the Transformer and explain what each part is for. (8 marks)
:::ans
Encoder block: multi-head self-attention, Add & Norm, FFN, Add & Norm. Decoder block: masked multi-head self-attention, Add & Norm, cross-attention (Q from decoder, K/V from final encoder output), Add & Norm, FFN, Add & Norm. Purposes: positional encoding gives order; multi-head attention queries several positions; FFN adds non-linearity and memory; the mask prevents lookups into the future; residuals improve gradient flow; layer norm stabilises. Final linear and softmax predict the next token. Lessons 3.5 to 3.9.
:::
:::

:::q E2 · Derive why RoPE attention scores depend only on relative position. (5 marks)
:::ans
Rotate $\mathbf{q}$ by $m\theta$ and $\mathbf{k}$ by $n\theta$. Then $(R_{m\theta}\mathbf{q})^\top(R_{n\theta}\mathbf{k})=\mathbf{q}^\top R_{m\theta}^\top R_{n\theta}\mathbf{k}=\mathbf{q}^\top R_{(n-m)\theta}\mathbf{k}$ because $R_a^\top=R_{-a}$ and rotations add angles. Only $n-m$ remains. Lesson 3.10.
:::
:::
