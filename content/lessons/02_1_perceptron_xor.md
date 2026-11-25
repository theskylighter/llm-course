<<SVG:perceptron_xor_hidden_layer>>

*Figure: A single threshold can separate AND and OR, but XOR requires a hidden layer to build useful features.*

**A perceptron computes a weighted sum, adds a bias and applies a step; it can draw one straight line, which is why it can learn AND and OR but never XOR.** The fix is a hidden layer.

## The Problem: Every Input Counts the Same

> [!lecture] Opening
> Today's lecture is a very brief overview, because at the end of the day LLMs are also neural networks. History: 1943 McCulloch and Pitts proposed the MP neuron. 1957-58 Rosenblatt's perceptron. 1969 Minsky and Papert showed XOR cannot be modelled, which began the AI winter. 1986 backpropagation. 1989-90 the **universal approximation theorem**: an MLP with a single hidden layer can approximate any continuous function to any desired precision.
> — *NPTEL Lecture 5*

> [!lecture] Now remember...
> If you call today's LLMs "neural networks" mimicking the brain, people will essentially kill you. They are just non-linear functions. Nobody cares any more whether they mimic the human brain.

The first model, the **McCulloch-Pitts (MP) neuron**, takes binary inputs and outputs 1 if $\sum_i x_i\ge\theta$. In "should I watch this movie?", the hero may matter far more than the duration, but the MP neuron weighs them equally. So add **weights**.

## The Perceptron

> [!slides] Perceptron
> $$y=g\Big(\sum_i w_i x_i+w_0\Big),\qquad g(a)=\begin{cases}1&a\ge0\\0&a<0\end{cases}$$
> $w_0$ is the bias. AND: $w_1=w_2=1,\ w_0=-1.5$. OR: $w_1=w_2=1,\ w_0=-0.5$. Everything an MP neuron computes, a perceptron computes, with weighted emphasis.

:::analogy
A perceptron is a bouncer with a scoreboard: each fact about you adds or removes points, and if the total clears the bar (the bias), you get in.
:::

```python
def perceptron(x1, x2, w1, w2, w0):
    return 1 if w1 * x1 + w2 * x2 + w0 >= 0 else 0

points = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("AND:", [perceptron(a, b, 1, 1, -1.5) for a, b in points])
print("OR :", [perceptron(a, b, 1, 1, -0.5) for a, b in points])
# Try every weight setting on a coarse grid for XOR (target 0,1,1,0)
target = [0, 1, 1, 0]
grid = [x / 2 for x in range(-8, 9)]
found = any(
    [perceptron(a, b, w1, w2, w0) for a, b in points] == target
    for w1 in grid for w2 in grid for w0 in grid
)
print("a single perceptron solves XOR:", found)
```

```output
AND: [0, 0, 0, 1]
OR : [0, 1, 1, 1]
a single perceptron solves XOR: False
```

## XOR Still Fails, and a Hidden Layer Fixes It

There is no non-linearity in a perceptron: it separates the plane with **one line**. The four XOR points cannot be split by one line. Think about it carefully: this is the question examiners reuse.

> [!slides] Two-layer XOR (the lecture's weights)
> | $x_1,x_2$ | $h_1=g(x_1+x_2-0.5)$ (OR) | $h_2=g(x_1+x_2-1.5)$ (AND) | $y=g(h_1-h_2-0.5)$ |
> |---|---|---|---|
> | 0,0 | 0 | 0 | 0 |
> | 0,1 | 1 | 0 | 1 |
> | 1,0 | 1 | 0 | 1 |
> | 1,1 | 1 | 1 | 0 |
>
> XOR = OR minus AND. A hidden layer re-represents the points so that they become linearly separable.

```python
def g(a):
    return 1 if a >= 0 else 0

def xor(x1, x2):
    h1 = g(x1 + x2 - 0.5)      # OR
    h2 = g(x1 + x2 - 1.5)      # AND
    return g(h1 - h2 - 0.5)

for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(a, b, "->", xor(a, b))
```

```output
0 0 -> 0
0 1 -> 1
1 0 -> 1
1 1 -> 0
```

:::def
A **perceptron** is a linear threshold unit $y=g(\mathbf{w}^\top\mathbf{x}+w_0)$. It can represent only **linearly separable** functions. XOR is not linearly separable, so it needs at least one hidden layer.
:::

:::key
- Perceptron = weighted sum + bias + step; one linear boundary.
- AND: weights 1, 1, bias $-1.5$. OR: weights 1, 1, bias $-0.5$.
- XOR = OR minus AND, which needs a hidden layer.
- Universal approximation: one hidden layer can approximate any continuous function (existence, not ease of training).
:::

:::mistakes
- "A perceptron can do XOR with better weights." No: one line cannot separate XOR.
- Saying LLMs "mimic the brain" in an exam. The professor warns against it.
:::

:::exam
- *"Design an MP neuron or perceptron for AND/OR (give weights and bias). Why does XOR fail?"* (the classic)
- *"Build XOR with a 2-layer MLP and verify with a truth table."*
:::

## Quick Check

:::q Q1 · Give weights and bias for a perceptron computing OR of two binary inputs.
:::ans
$w_1=w_2=1$ and $w_0=-0.5$: the sum clears the threshold when at least one input is 1.
:::
:::

:::q Q2 · Why does the hidden layer make XOR solvable?
:::ans
It maps the inputs to $(h_1,h_2)=(\text{OR},\text{AND})$, where the XOR classes become linearly separable.
:::
:::
