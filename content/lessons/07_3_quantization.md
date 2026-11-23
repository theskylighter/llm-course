**Quantization stores weights and activations in fewer bits, such as INT8 instead of FP32, trading a small rounding error for large savings in memory and cost.** Outliers in large models break the naive version.

## The Problem: The Model Is Too Big to Run

> [!slides] Lec_30: the problem
> Larger model means larger GPU memory, latency, inference cost and environmental cost. *Why is gpt-4o-mini so much cheaper than gpt-4o?* Two families: **model compression (lossy)** and **efficient engineering (lossless)**. Compression has three tools: **quantization** (same model, fewer bits), **pruning** (remove parts), **distillation** (train a smaller model to imitate the bigger one). Efficient engineering (KV cache, FlashAttention) is Chapter 4.

Billions of weights are expensive to store, and **activations** (input times weights) are produced at inference and are also expensive. Represent them with fewer bits, for example FP32 to INT8, then **dequantize** back for use.

:::analogy
Rounding prices to the nearest 5 rupees: bills are shorter to write and add up almost the same, but a single huge item can force a coarser rounding for everything else.
:::

```python
import numpy as np
x = np.array([0.0, 0.5, -1.27, 2.54, 0.013])
s = 127 / np.abs(x).max()                  # absmax scale
xq = np.round(s * x).astype(int)           # int8 values in [-127, 127]
xd = xq / s                                # dequantize
print("scale s =", s)
print("quantized  :", xq)
print("dequantized:", np.round(xd, 3))
print("worst error:", round(float(np.abs(x - xd).max()), 3), "(bound 0.5/s =", round(0.5 / s, 3), ")")
```

```output
scale s = 50.0
quantized  : [  0  25 -64 127   1]
dequantized: [ 0.    0.5  -1.28  2.54  0.02]
worst error: 0.01 (bound 0.5/s = 0.01 )
```

**Post-training quantization (PTQ).** No retraining and no architecture change. **Weights and biases are constants**, so their scale factors are easy to compute. **Inputs and activations vary**, so use a **calibration dataset** to estimate their scale factors. **Quantization-aware training (QAT)** is the other branch.

> [!slides] LLM.int8() (Dettmers et al. 2022): why plain INT8 breaks
> Regular quantization keeps performance up to about **2.7B** parameters. Once **systematic outliers** appear at about **6.7B**, regular methods fail. LLM.int8() keeps 16-bit accuracy at every scale. (One huge value in a vector makes the scale tiny, so everything else collapses to a few levels.)

```python
import numpy as np
normal = np.array([0.1, -0.2, 0.3, 0.05])
with_outlier = np.array([0.1, -0.2, 0.3, 60.0])
for name, v in (("no outlier", normal), ("one outlier", with_outlier)):
    s = 127 / np.abs(v).max()
    print(f"{name:12s} levels used by the small values:", np.round(s * v[:3]).astype(int))
```

```output
no outlier   levels used by the small values: [ 42 -85 127]
one outlier  levels used by the small values: [0 0 1]
```

> [!compiler] What the compiler adds (none of this is in the repo)
> *LLM.int8() recipe (from the paper):* **vector-wise** scales plus a **mixed-precision decomposition**: the few outlier feature dimensions are multiplied in FP16 and the rest in INT8. *QAT:* simulate quantization in the forward pass ("fake quant") and pass gradients through the rounding with a **straight-through estimator**. It costs a training run; PTQ does not. "Mixed precision" has two meanings: this decomposition, and FP16/BF16 training.

:::def
**Quantization** maps high-precision values to a small set of discrete levels, for example absmax INT8: $x_q=\operatorname{round}(sx)$ with $s=127/\max|x|$, and $x\approx x_q/s$.
:::

:::key
- Absmax INT8: $s=127/\max|x|$; error at most $0.5/s$.
- PTQ needs no retraining; activations need a calibration set.
- Outliers (from about 6.7B parameters) break naive INT8; LLM.int8() handles them in FP16.
:::

:::mistakes
- "Quantization is only for weights." Activations need calibration and cause the outlier problem.
- "Distillation equals quantization." Distillation trains a different, smaller model.
:::

:::exam
- *"Quantize $x=[0,0.5,-1.27,2.54,0.013]$ to INT8 with absmax."* $s=50$, $x_q=[0,25,-64,127,1]$.
:::

## Quick Check

:::q Q1 · Why do outliers hurt absmax quantization?
:::ans
One large value makes the scale tiny, so all other values map to very few integer levels and lose precision.
:::
:::

:::q Q2 · How does PTQ get scale factors for activations?
:::ans
By running a calibration dataset through the model and estimating their ranges.
:::
:::
