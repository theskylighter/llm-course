<<SVG:toxicity_hallucination_checks>>

*Figure: Toxicity concerns harmful content; hallucination concerns unsupported facts or input conflicts. Classify, verify claims, ground, or abstain.*

**Toxicity is harmful generated text and hallucination is fluent text that conflicts with facts or with the given input; both are properties of fitting $P(x_t\mid x_{<t})$, so we measure them with classifiers and claim-level scores and reduce them by grounding, abstention and alignment.** Hallucination is not lying: there is no intent.

## The Problem: The Loss Never Says "I Don't Know"

A model trained on next-token prediction is rewarded for a fluent guess. Nothing in the loss rewards saying "I don't know". So "the Eiffel Tower is in Rome" is as easy to produce as the truth.

> [!lecture] The KG-as-guard passage (lecture 26)
> In other parts of the course we discuss the tendency of LLMs **to make up fictitious things through their token sampling process.** Consulting a knowledge base, by itself or after probabilistic completion, is a great way to prevent the LLM from hallucinating. It also makes LLMs **more interpretable**: if you can look at which regions of the KG the model consulted, it is easier to decide how a query went wrong.

## Toxicity

> [!compiler] What it is and how to measure it (no source in the repo)
> **Toxicity** is generated text that is rude, hateful, threatening or otherwise harmful, even when the prompt looks innocent. Standard practice: score continuations with a **toxicity classifier** (Perspective API is the usual one) over many prompts, and report the **expected maximum toxicity** over $n$ samples per prompt and the **fraction of prompts** with at least one toxic continuation. **Mitigation along the pipeline:** filter pretraining data; **safety fine-tuning** on refusals; **alignment**; **decoding filters** (block lists, classifier-guided decoding); **guardrail models**. Each has a false-positive cost: over-refusal makes the model useless for legitimate medical or legal questions.

## Hallucination: Types, Causes, Metrics

> [!compiler] Taxonomy (from Huang et al. 2023)
> | Type | Meaning | Example |
> |---|---|---|
> | **Factuality hallucination** | output conflicts with real-world facts | "The Eiffel Tower is in Rome" |
> | **Faithfulness hallucination** | output conflicts with the *given* input or instruction, or with itself | a summary claims something the document never says |
> | **Intrinsic** versus **extrinsic** (older terms) | contradicts the source versus adds unverifiable content | |

**Causes:** training data errors and gaps; **next-token training rewards fluent guessing**; exposure bias when decoding; high-temperature sampling; knowledge cut-offs. Hallucination is **not a bug in a component**, it is a property of fitting $P(x_t\mid x_{<t})$.

> [!compiler] Metrics (from the papers)
> **TruthfulQA:** questions where common misconceptions are tempting. **FActScore** splits a generation into **atomic facts**: $\text{FActScore}=\dfrac{\#\text{facts supported by a knowledge source}}{\#\text{atomic facts}}$. Example: 20 facts, 15 supported gives $0.75$. This is the same idea as **faithfulness** in RAG evaluation.

:::analogy
A student who is never penalised for a wrong answer but is marked down for leaving a blank will always write something confident-sounding.
:::

## Detecting and Reducing Hallucination

> [!compiler] Detection without a reference
> **Consistency (SelfCheckGPT):** sample several answers; a fact the model *knows* shows up in most of them, an invented one varies. **Uncertainty:** low token probability or high entropy over the answer span signals risk, with the caveat that models are often **confidently wrong**. **Verification:** ask a second model or a retriever to check each claim.

```python
def selfcheck(greedy_claims, sampled_answers, threshold=0.5):
    flagged = []
    for claim in greedy_claims:
        support = sum(claim in a for a in sampled_answers) / len(sampled_answers)
        if support < threshold:
            flagged.append((claim, round(support, 2)))
    return flagged

samples = [
    {"paris is the capital", "paris is in france"},
    {"paris is the capital", "paris is in france"},
    {"paris is the capital", "paris was founded in 1500"},
    {"paris is the capital", "paris is in france"},
]
greedy = ["paris is the capital", "paris is in france", "paris was founded in 1500"]
print("flagged claims:", selfcheck(greedy, samples))

facts, supported = 20, 15
print("FActScore:", supported / facts)
```

```output
flagged claims: [('paris was founded in 1500', 0.25)]
FActScore: 0.75
```

> [!compiler] Mitigation: ground, abstain, align
> **Grounding via RAG** (lesson 8.1): put the evidence in the prompt and require citations. **Knowledge graphs** (lecture 26): structured, editable, checkable. **Training for abstention:** reward "I don't know" when unsure. **Lower temperature** reduces sampling noise but not wrong beliefs.

## Ethics as a Practice

> [!compiler] A working framework (the compiler's synthesis)
> **Measure before you mitigate**, on the *deployment* distribution: pick a metric per dimension (icat or Regard for fairness; POSIX for robustness; faithfulness for hallucination; toxicity rate for safety). **Group fairness checks:** *demographic parity* and *equalized odds*; the two can be impossible to satisfy together. A concrete reading of the slides' message: *there is no "unbiased model", only a measured, documented and monitored one.*

:::def
**Hallucination** is generated content that is fluent but unsupported by, or contradicts, real-world facts (**factuality**) or the provided input (**faithfulness**).
:::

:::key
- Factuality (against the world) versus faithfulness (against the input).
- Causes: next-token training rewards fluent guessing; sampling; knowledge gaps.
- Detect by consistency across samples; reduce by RAG or KG grounding, abstention, alignment.
- FActScore = supported atomic facts over total.
:::

:::mistakes
- "Hallucination equals lying." No intent; it is a fluent guess.
- "RAG fixes hallucination." It reduces factual errors; faithfulness errors remain.
- "Temperature 0 means no hallucination." It removes randomness, not wrong beliefs.
:::

:::exam
- *"Define faithfulness versus factuality hallucination."* (compiler's guess of the style)
:::

## Quick Check

:::q Q1 · A summary says the document mentions a merger that it never mentions. Which type of hallucination is it?
:::ans
A **faithfulness** hallucination: it conflicts with the given input.
:::
:::

:::q Q2 · Why does sampling consistency help detect hallucination?
:::ans
Facts the model genuinely knows appear in most samples, while invented details vary from sample to sample.
:::
:::
