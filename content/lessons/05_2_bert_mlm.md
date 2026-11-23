**BERT is an encoder-only Transformer trained to predict hidden tokens from both sides: 15% of tokens are chosen, 80% become [MASK], 10% a random token, 10% stay unchanged.** The loss is computed only at the chosen positions.

## The Problem: A Bidirectional Model Cannot Be a Left-to-Right LM

> [!lecture] Lecture
> BERT stands for *Bidirectional Encoder Representations from Transformers*, proposed by Google. We keep only the encoder part. The beauty of the encoder is that there is **no masked self-attention**: it is unmasked, every token attends to all other tokens, so it is bidirectional in nature.

We have raw text only, so what is the objective? Think about it carefully. A left-to-right LM cannot be bidirectional, because it would be cheating (the answer is in the right context). The trick: **hide** some tokens and predict them from *both* sides. That is the **masked language model (MLM)**: self-supervised, since the "labels" are the hidden tokens.

:::analogy
A fill-in-the-blanks worksheet made automatically from any book: cover some words and ask the student to recover them using the words on both sides.
:::

## Which Tokens, and How Many

> [!lecture] The 15% and the 80/10/10 rule
> You mask **15%** of the tokens. (One sentence of the transcript reads "50% masking is enough"; this is a transcription slip, since the very next sentences say 15% and argue that masking *more* destroys the context.) Of the chosen 15%: **80%** are replaced by a new `[MASK]` token; **10%** are replaced by a *random other* token (park becomes store); **10%** are left unchanged. In all three cases the model must predict the original word, and the **loss is computed only at those positions**.

Why not always use `[MASK]`? If the model only learns to predict `[MASK]`, it forgets about predicting actual tokens, so we sometimes show it a real or a wrong token and make it still decide what is right.

> [!compiler] Why 15%?
> The lecture's argument: too much masking removes the context, and 15% on a corpus of billions of tokens is empirically enough. The counter-argument (too *little* masking makes each pass uninformative and expensive) comes from the paper. Also, `[MASK]` never appears at fine-tuning time; the paper cites this mismatch as the reason for the 10% plus 10%.

```python
import random

def corrupt(tokens, rate=0.15, seed=0):
    rng = random.Random(seed)
    vocab = ["store", "river", "ball", "green"]
    inp, targets = list(tokens), {}
    for i, t in enumerate(tokens):
        if rng.random() < rate:
            targets[i] = t                       # loss only here
            r = rng.random()
            if r < 0.8:
                inp[i] = "[MASK]"
            elif r < 0.9:
                inp[i] = rng.choice(vocab)
            # else: unchanged but still a target
    return inp, targets

tokens = "he went to the park and then to the market near the old bridge".split()
inp, targets = corrupt(tokens, rate=0.3, seed=2)
print(" ".join(inp))
print("loss positions:", sorted(targets), "->", [targets[i] for i in sorted(targets)])
print("fraction of positions with loss:", round(len(targets) / len(tokens), 2))
```

```output
he went [MASK] the park and then to the market [MASK] the old bridge
loss positions: [2, 10] -> ['to', 'near']
fraction of positions with loss: 0.14
```

```text
# BERT MLM loss
loss = sum over i in chosen of  -log P(original[i] | corrupted input)
```

:::def
**Masked language modeling (MLM)** corrupts a fraction (15%) of the input tokens and trains a bidirectional encoder to recover the originals, with the loss computed only at the corrupted positions.
:::

:::key
- Encoder-only, unmasked attention, bidirectional.
- 15% chosen; 80% [MASK], 10% random token, 10% unchanged; loss only on chosen positions.
- Self-supervised: labels are the hidden tokens.
- BERT is not a generative model in the left-to-right sense.
:::

:::mistakes
- "Loss in BERT is over all tokens." Only the roughly 15% chosen positions.
- "BERT can generate text." Not naturally: MLM gives a bidirectional distribution, not a left-to-right sampler.
:::

:::exam
- *"List the three corruptions BERT applies to a chosen token and their probabilities. Where is the loss computed?"* (Tier 1)
:::

## Quick Check

:::q Q1 · Why keep 10% of chosen tokens unchanged?
:::ans
So the model cannot learn that "unmasked means correct": it must still decide for real tokens, and it reduces the mismatch with fine-tuning where no [MASK] appears.
:::
:::

:::q Q2 · Why can BERT not be trained with ordinary next-token prediction while being bidirectional?
:::ans
Each position would see the answer in its right context, which is cheating.
:::
:::
