**Multilingual models share one vocabulary and one set of Transformer weights across about 100 languages, so what the model learns in a high-resource language transfers to others; the price is the curse of multilinguality and uneven transfer.** The objective stays monolingual; cross-lingual ability is a side effect.

> [!compiler] Source note
> No NPTEL lecture covers multilingual or vision-language models. Real material in the repo: fragments of lecture 22 (GPT-3 translation results, 7% non-English data), lecture 1 and the one cultural-bias slide (Lec_36). The model line and recipes are Compiler's Notes from the papers.

## The Problem: One Model per Language Does Not Scale

> [!lecture] Opening (lecture 22)
> If you look at the training data, **7% of it is in languages other than English; the remaining 93% is English.** So what happens when you ask for a non-English task, say translation from English to French, French to English, English to German? Although only 7% of the data is not English, and there are so many languages around the world, Indian languages among them, **the GPT model without fine-tuning performs quite competitively with a state-of-the-art supervised model.**
> — *Prof. Tanmoy Chakraborty, lecture 22 (discussing GPT-3)*

**Attempt 1: one model per language.** Needs a large corpus per language, and most languages lack one. A Tamil sentiment classifier with 500 labelled examples will be poor.
**Attempt 2: translate everything to English at the door.** The pipeline doubles the error and loses culture-specific words.
**Attempt 3: one model, many languages, shared parameters.** Pre-train a single Transformer on text from 100 languages. Fine-tune on English labels, test on Hindi: **zero-shot cross-lingual transfer**.

> [!compiler] Two corrections to keep the numbers honest
> The lecture says "37% accuracy" and "BLEU score" in the same breath: these are **BLEU scores (0-100)**, not accuracies. And GPT-3's "93% English" is by the paper's count of **words**, not documents.

## The Model Line

| Model | Year | Data | Objective | Key idea |
|---|---|---|---|---|
| **mBERT** | 2018 | Wikipedia, 104 languages | masked LM only | one shared WordPiece vocabulary; **no parallel data, no language ID**, yet it transfers |
| **XLM** | 2019 | monolingual plus parallel text | MLM, causal LM, **TLM** | **Translation LM**: concatenate a sentence and its translation and mask tokens in both |
| **XLM-R** | 2019 | CommonCrawl, 100 languages (about 2.5 TB) | MLM only | far more data, SentencePiece vocabulary of about 250k, **no TLM needed** |
| **mT5** | 2020 | mC4, 101 languages | span corruption | T5 recipe, multilingual; sampling by $p_i\propto n_i^{\alpha}$ to up-weight small languages |
| **BLOOM, mGPT** | 2022+ | multilingual decoder models | causal LM | multilingual generation |

The **objective stays monolingual** (mask a token, predict it). Cross-lingual ability is a *side effect* of sharing parameters and vocabulary, plus (for XLM) an explicit translation signal.

:::analogy
A polyglot who learns grammar patterns in French and finds Spanish easier afterwards, because both share Latin roots and the same ideas of subject and verb.
:::

## Why Transfer Works, and Why It Is Uneven

> [!compiler] Three ingredients
> **Shared subwords:** numerals, names, URLs, cognates and loanwords appear in several languages and act as anchors. **Shared structure:** the same weights process every language. **Shared geometry:** the hidden states for a sentence and its translation lie close together.

```text
model = pretrained_multilingual_encoder           # mBERT / XLM-R
train(model, labelled_data["en"])                 # fine-tune on English only
for lang in ["hi", "sw", "ta"]:
    score[lang] = evaluate(model, test_data[lang])    # no target-language labels
# gap = score["en"] - score[lang]  : the cross-lingual transfer gap
```

**XNLI** (natural language inference translated into 15 languages) reports exactly this gap. Closing it: **translate-train**, **translate-test**, or add a few target-language labels.

> [!compiler] Curse of multilinguality: capacity, data and tokenizer
> **Capacity:** with a fixed number of parameters, adding languages helps low-resource ones up to a point, then every language gets worse. The fix is a bigger model, not fewer languages. **Data imbalance:** high-resource languages dominate the gradient; sampling in proportion to $n_i^{\alpha}$ with $\alpha<1$ flattens it.

```python
sizes = {"en": 1000.0, "hi": 100.0, "sw": 10.0}      # corpus sizes (millions of sentences)
for alpha in (1.0, 0.7, 0.3):
    w = {k: v ** alpha for k, v in sizes.items()}
    tot = sum(w.values())
    print(f"alpha={alpha}:", {k: round(v / tot, 3) for k, v in w.items()})
```

```output
alpha=1.0: {'en': 0.901, 'hi': 0.09, 'sw': 0.009}
alpha=0.7: {'en': 0.807, 'hi': 0.161, 'sw': 0.032}
alpha=0.3: {'en': 0.571, 'hi': 0.286, 'sw': 0.143}
```

> [!slides] The cultural-bias slide (Lec_36)
> **Majority of research and development focuses on high-resource languages such as English**: data collection, linguists and annotators are easier to find. This is an **adverse cycle that widens the imbalance of information and reinforces the dominance of the high-resource language.** Each language is tied to a distinct culture, so a model learns **underestimation of cultural importance for low-resource languages and over-generalization for high-resource ones.**

> [!lecture] The same gap in knowledge graphs (lecture 26)
> As soon as you drop off the popular or head entities into the torso or less-known entities, or entities in lower-resource languages and cultures, the coverage of knowledge graphs falls quite drastically.

:::def
**Zero-shot cross-lingual transfer** is fine-tuning a multilingual pre-trained model on labelled data in one language (usually English) and evaluating it on other languages with no target-language labels.
:::

:::key
- mBERT, XLM (TLM), XLM-R (no TLM, 100 languages), mT5 (span corruption).
- Objective is monolingual; transfer comes from shared subwords, structure and geometry.
- Curse of multilinguality; sample languages by $n_i^{\alpha}$ with $\alpha<1$.
:::

:::mistakes
- "Multilingual models need parallel data." mBERT and XLM-R use none.
- Quoting BLEU as accuracy.
:::

:::exam
- *"Why does cross-lingual transfer work in mBERT?"* Shared vocabulary, shared parameters, shared geometry.
:::

## Quick Check

:::q Q1 · What does TLM add to XLM?
:::ans
It concatenates a sentence with its translation and masks tokens in both so the model can use the other language to fill a gap.
:::
:::

:::q Q2 · Why sample low-resource languages with $n_i^{\alpha}$, $\alpha<1$?
:::ans
It up-weights small languages relative to their raw size so high-resource languages do not dominate training.
:::
:::
