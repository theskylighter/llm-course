**A Transformer is trained in parallel by feeding the whole target shifted right and computing a cross-entropy loss at every position at once; generation is still one token at a time.** This is teacher forcing combined with the causal mask.

## The Problem: How Do We Train an Encoder-Decoder in One Pass?

An RNN decoder must wait for step $t-1$ before computing step $t$. We removed recurrence, so we should not have to wait.

> [!lecture] Lecture
> The task in the paper is machine translation: given the source sentence, generate the target. During **training** we already know the translated sentence, so the whole target is fed in (shifted right, starting with `<START>`) and *all positions are processed in parallel* (this is why we needed the mask). Each top decoder state goes through a linear layer and a softmax to give a distribution over the vocabulary. We know the ground-truth token at that position, so we compute a cross-entropy error at **every position at the same time**, sum them, and backpropagate end to end. Unlike an RNN, we don't wait. At **test time** we generate token by token: sample or take the argmax, feed it back in, and generate the next.

$$\mathcal{L}=-\sum_{t=1}^{T}\log P_\theta\!\left(y_t\mid y_{<t},\,x_{1:S}\right)\qquad\text{(teacher forcing; all $t$ computed in parallel)}$$

> [!compiler] The thing that is easy to miss
> In code this is `model(src, trg[:, :-1])` giving logits compared with `trg[:, 1:]`: the decoder input is the target *shifted right*, the labels are the target *shifted left*. The model returns **logits**; the softmax is inside the loss (`nn.CrossEntropyLoss`) or the sampler.

:::analogy
A student learning to write sentences with a teacher who always supplies the correct previous words: at every position they only have to predict the next word, so all positions can be graded at once.
:::

```python
import numpy as np
target = ["<s>", "le", "chat", "dort", "</s>"]
decoder_input = target[:-1]
labels = target[1:]
for i, (inp, lab) in enumerate(zip(decoder_input, labels)):
    print(f"position {i}: sees {decoder_input[:i + 1]} -> predicts {lab!r}")

# cross-entropy at every position at the same time
rng = np.random.default_rng(0)
vocab = ["<s>", "le", "chat", "dort", "</s>"]
logits = rng.normal(size=(4, len(vocab)))
probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
idx = [vocab.index(t) for t in labels]
loss = -np.log(probs[np.arange(4), idx]).sum()
print("summed loss over 4 positions:", round(float(loss), 3))
```

```output
position 0: sees ['<s>'] -> predicts 'le'
position 1: sees ['<s>', 'le'] -> predicts 'chat'
position 2: sees ['<s>', 'le', 'chat'] -> predicts 'dort'
position 3: sees ['<s>', 'le', 'chat', 'dort'] -> predicts '</s>'
summed loss over 4 positions: 5.054
```

> [!compiler] What each piece is for
> Positional encoding adds positional information. Multi-head attention queries multiple positions per layer. FFN adds non-linearity. Masked attention prevents lookups into the future. Residual connections improve gradient flow. Layer norm makes the system stable.

:::def
**Teacher forcing** trains a sequence model by feeding the ground-truth previous tokens as input and maximising the log-likelihood of the next ground-truth token at every position.
:::

:::key
- Training is parallel: shifted-right target in, shifted-left labels out, loss at every position.
- Generation is sequential: argmax or sample, feed back.
- Gradients flow through all parameters including embeddings and (if trainable) position embeddings.
- Autoregressive training in parallel needs the causal mask.
:::

:::mistakes
- "Generation is parallel too." Only training is.
- "The model outputs probabilities." The model outputs logits; softmax lives in the loss or sampler.
:::

:::exam
- *"Explain how a Transformer is trained and how it generates."* Teacher forcing, parallel loss, then token-by-token at test time.
- *Self-attention vs FFN:* message passing versus processing.
:::

## Quick Check

:::q Q1 · What are the decoder input and the labels for target `<s> a b </s>`?
:::ans
Input `<s> a b`; labels `a b </s>`.
:::
:::

:::q Q2 · Why does training need the causal mask even though the target is known?
:::ans
Without it position $i$ could attend to position $i+1$, whose token is the label being predicted, so the model would cheat.
:::
:::
