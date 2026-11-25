<<SVG:vit_clip_patches_alignment>>

*Figure: ViT feeds patch tokens to a standard encoder; CLIP pulls matched image/text pairs together and separates mismatches.*

**A Vision Transformer cuts an image into patches and treats them as tokens for a standard Transformer encoder; CLIP aligns image and text encoders with a contrastive loss; vision-language models connect a vision encoder to an LLM through a small trainable connector.** The language model gets eyes without being retrained from scratch.

## The Problem: A Transformer Expects a 1-D Sequence of Tokens

Radiology scans, charts, screenshots, video: a language model has no eyes. An image is an $H\times W\times C$ array. Treating every pixel as a token fails: a $224\times224$ image has $50{,}176$ pixels, and self-attention is $O(n^2)$ in sequence length. We need a *tokenizer for images*.

## Vision Transformer (ViT)

> [!compiler] EndSem Q4(a), worked (Dosovitskiy et al.)
> **Input format.** Cut the image into non-overlapping patches of size $P\times P$. The number of patches is $N=\dfrac{HW}{P^2}$. Flatten each patch into a vector of length $P^2C$. A **trainable linear projection** $E\in\mathbb{R}^{(P^2C)\times D}$ maps each flattened patch to a $D$-dimensional **patch embedding**. Prepend a learnable `[CLS]` token and add position embeddings.
> **Architectural change?** **None.** ViT is a *standard Transformer encoder*; the only new parts are the patch embedding in front and a classification head on the `[CLS]` output. Title of the paper: "An image is worth $16\times16$ words."
> **Caveat:** ViT has no convolutional inductive bias, so it needs **large-scale pre-training** to beat CNNs.

:::analogy
Cutting a photograph into postage-stamp squares and laying them in a row like words in a sentence.
:::

```python
H = W = 224; C = 3; P = 16; D = 768
N = H * W // P ** 2
print("patches:", N, "=", H // P, "x", W // P)
print("flattened patch length:", P * P * C)
print("tokens seen by the Transformer (with [CLS]):", N + 1, "x", D)
print("pixels as tokens would give:", H * W)
```

```output
patches: 196 = 14 x 14
flattened patch length: 768
tokens seen by the Transformer (with [CLS]): 197 x 768
pixels as tokens would give: 50176
```

## CLIP: A Shared Space with a Contrastive Loss

> [!compiler] The idea (Radford et al. 2021)
> Train an **image encoder** and a **text encoder** so that a matching image-caption pair lands close together and mismatched pairs land far apart, using about 400M web image-text pairs. For a batch of $N$ pairs let $s_{ij}=\cos(\mathbf{z}^{img}_i,\mathbf{z}^{txt}_j)/\tau$. The $N\times N$ similarity matrix should have its **diagonal** large. The loss is a symmetric cross-entropy (InfoNCE):
> $$\mathcal{L}=-\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{s_{ii}}}{\sum_{j}e^{s_{ij}}}+\log\frac{e^{s_{ii}}}{\sum_{j}e^{s_{ji}}}\right]$$
> **Zero-shot classification:** embed prompts "a photo of a {class}" for every class, embed the image, pick the closest.

> [!compiler] EndSem Q4(b): the VideoCLIP loss (Xu et al. 2021, reconstruction)
> With $z_v$ and $z_t$ the pooled states of a video clip and its text clip, and $\tau$ a temperature, the loss over a batch $\mathcal{B}$ of positive pairs is video-to-text plus text-to-video InfoNCE, with the other pairs in the batch as negatives. The paper's twists (from memory): positives are **temporally overlapping** clips and **hard negatives are retrieved** from similar videos. The exam's wording is not in the repo, so your class notes' conventions take priority.

```python
import numpy as np
img = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
txt = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.8]])
norm = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
S = norm(img) @ norm(txt).T / 0.1
def xent(S):
    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return -np.log(np.diag(P)).mean()
print("image->text loss:", round(float(xent(S)), 4))
print("text->image loss:", round(float(xent(S.T)), 4))
print("symmetric CLIP loss:", round(float((xent(S) + xent(S.T)) / 2), 4))
print("predicted caption for each image:", S.argmax(1))
```

```output
image->text loss: 0.125
text->image loss: 0.1272
symmetric CLIP loss: 0.1261
predicted caption for each image: [0 1 2]
```

## From a Shared Space to a Model That Talks About Images

CLIP *scores* pairs; it cannot write a caption or answer a question. For that, connect a vision encoder to a **generative LM**.

| Model | Recipe (from the papers) |
|---|---|
| **Frozen** | freeze the LM; train a vision encoder whose output is a **prefix** of embeddings (the soft-prompt idea of lesson 7.1) |
| **Flamingo** | freeze vision encoder and LM; a **Perceiver Resampler** compresses image features; **gated cross-attention** layers between frozen LM layers; gate $\tanh(\alpha)$ with $\alpha=0$ at start |
| **BLIP-2** | frozen image encoder and frozen LLM bridged by a small trainable **Q-Former** (32 learned queries) |
| **LLaVA / MiniGPT-4** | project CLIP features by a linear or MLP layer into the LLM's embedding space; **visual instruction tuning** |

The common pattern is a **connector**: image, frozen encoder, small trainable module, tokens the LLM can read. A common pipeline: stage 1 align the connector on captions; stage 2 instruction tune on visual dialogues.

:::def
**CLIP** jointly trains an image encoder and a text encoder with a symmetric InfoNCE loss on a batch of matched image-text pairs so that matching pairs have high cosine similarity and the other pairs in the batch act as negatives.
:::

:::key
- ViT: $N=HW/P^2$ patches, flatten to $P^2C$, linear projection to $D$, `[CLS]` and positions; no change to the encoder.
- CLIP: symmetric contrastive loss, zero-shot classification by prompt matching.
- VLMs: frozen encoder, small trainable connector, frozen or LoRA LLM.
:::

:::mistakes
- "ViT changes the Transformer architecture." Only the front end (patch embedding) is new.
- "CLIP can write captions." It only scores pairs.
:::

:::exam
- *(EndSem Q4(a))* "How is an image fed to ViT? Does the architecture change?"
- *(EndSem Q4(b))* "Write the VideoCLIP contrastive loss."
:::

## Quick Check

:::q Q1 · For a $224\times224\times3$ image and $P=16$, how many patches and what is each flattened length?
:::ans
$196$ patches ($14\times14$), each of length $16\cdot16\cdot3=768$.
:::
:::

:::q Q2 · Why does a bigger batch make CLIP training harder?
:::ans
More negatives per positive pair: each image must pick its caption among more candidates.
:::
:::
