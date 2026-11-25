<<SVG:bart_t5_corruption_targets>>

*Figure: BART reconstructs the complete clean sequence; T5 predicts only sentinel-marked spans.*

**BART corrupts the input and makes the decoder regenerate the whole clean text; T5 casts every task as text-to-text and pre-trains by replacing spans with sentinels and predicting only the spans.** Both use the full encoder-decoder.

## The Problem: Encoder-Decoder Needs Labelled Pairs

> [!lecture] The problem
> In an encoder-decoder the usual training needs labelled pairs (source to target). Labelled data is hard to collect. We have seen BERT use raw text. So: **how do we use unlabelled data in an encoder-decoder, and force the decoder to attend to the whole input?**

## BART: Corrupt, Then Reconstruct

BART = *Bidirectional and Auto-Regressive Transformer* (Facebook). It joins BERT's bidirectional encoder with GPT's autoregressive decoder. The lecture's corruptions, input $A\,B\,C\,D\,E$:

| Corruption | Encoder sees | Decoder must output |
|---|---|---|
| Token masking | `A [M] C [M] E` | `A B C D E` |
| Permutation | `D E A B C` | `A B C D E` |
| Rotation | `C D E A B` | `A B C D E` |
| Token deletion | `A C E` | `A B C D E` |
| Span masking | `A [M] D E` | `A B C D E` |

> [!lecture] BERT versus BART
> BERT predicts only the masked tokens. BART predicts **all tokens**. BART uses an encoder and a decoder, so roughly double the parameters.

## T5: Everything Is Text-to-Text

T5 = *Text-to-Text Transfer Transformer* (Google). Its ambition: **any task is a text-to-text problem**, with the task named in a prefix:

```text
INPUT:  "translate English to German: That is good."    OUTPUT: "Das ist gut."
INPUT:  "cola sentence: The course is jumping well."    OUTPUT: "not acceptable"
INPUT:  "stsb sentence1: ...rhino grazed... sentence2: ...rhino grazing..."   OUTPUT: "3.8"
INPUT:  "summarize: state authorities dispatched emergency crews..."          OUTPUT: <summary>
```

The lecture stresses the odd case: for similarity the output **3.8 is a *string*** that the model prints, not a number it regresses. "Believe it, this framework actually works." The model is the *vanilla* encoder-decoder.

> [!lecture] T5's pre-training: span corruption with sentinels
> Take "Thank you for inviting me to your party last week". Replace different-length spans with **unique sentinel tokens** `<X>`, `<Y>`: input `Thank you <X> me to your party <Y> week`. In BERT there is a single `[MASK]`; here every masked span gets *its own* sentinel. And the decoder only produces the **masked spans**: `<X> for inviting <Y> last <Z>`, **not** the whole sentence. That is the difference from BART.

```python
def span_corrupt(tokens, spans):
    """spans: list of (start, length). Returns encoder input and decoder target."""
    names = ["<X>", "<Y>", "<Z>"]
    enc, dec, i, k = [], [], 0, 0
    spans = sorted(spans)
    for start, length in spans:
        enc += tokens[i:start] + [names[k]]
        dec += [names[k]] + tokens[start:start + length]
        i, k = start + length, k + 1
    enc += tokens[i:]
    dec += [names[k]]
    return enc, dec

toks = "Thank you for inviting me to your party last week".split()
enc, dec = span_corrupt(toks, [(2, 2), (8, 1)])
print("encoder input :", " ".join(enc))
print("decoder target:", " ".join(dec))
```

```output
encoder input : Thank you <X> me to your party <Y> week
decoder target: <X> for inviting <Y> last <Z>
```

Data: Common Crawl is noisy, so T5 filters it: keep lines ending in terminal punctuation, keep only English, drop placeholder text and code, drop duplicates. The result is **C4** (*Colossal Clean Crawled Corpus*), about 745 GB. Cleaned data beat un-cleaned by roughly 2-3 points.

> [!slides] What T5's ablations found (Lec_19)
> *Architectures:* standard encoder-decoder, decoder-only LM, and **prefix LM** (bidirectional over the prefix, causal over the target). *Objectives:* **all variants perform similarly**; "replace corrupted spans" and "drop corrupted tokens" are appealing because the **targets are shorter, which speeds up training.**

> [!compiler] Where the lecture is imprecise about T5
> The lecture says T5 "doesn't need fine-tuning". In the paper the main pre-training is **unsupervised span corruption on C4**, followed by fine-tuning on each task. What is distinctive is the *unified text interface*.

:::def
**T5** is an encoder-decoder Transformer that treats every NLP task as mapping an input string (with a task prefix) to an output string, pre-trained by span corruption with sentinel tokens on the C4 corpus.
:::

:::key
- BART: corrupt and reconstruct the **whole** sequence. T5: sentinels, decoder outputs **only** the masked spans.
- T5 unifies tasks via text prefixes; numeric outputs are strings.
- Corruption variants perform similarly; shorter targets train faster.
:::

:::mistakes
- "BART and BERT both predict masked tokens." BART reconstructs the whole clean sequence.
- "`[MASK]` in T5 and BERT are the same." T5 uses a distinct sentinel per span.
:::

:::exam
- *"Write pseudo-code for T5 span corruption: given tokens and span lengths, produce the encoder input and decoder target."* (Tier 3)
:::

## Quick Check

:::q Q1 · What does the T5 decoder output for input `Thank you <X> me to your party <Y> week`?
:::ans
Only the masked spans, each preceded by its sentinel: `<X> for inviting <Y> last <Z>`.
:::
:::

:::q Q2 · Name three BART corruptions.
:::ans
Token masking, permutation, rotation, token deletion, span masking (any three).
:::
:::
