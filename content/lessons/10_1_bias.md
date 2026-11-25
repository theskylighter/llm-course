<<SVG:bias_lifecycle_measure_mitigate>>

*Figure: Bias can arise in data, model behavior, or deployment; mitigate at each stage and measure outcomes with appropriate metrics.*

**Bias is a distortion in a model's responses that reflects stereotypical beliefs in society; we see it in names, completions and preferences, measure it with StereoSet and Regard, and mitigate it at several points in the pipeline.** There is no unbiased model, only one that is measured, documented and monitored.

> [!compiler] Source note
> There is no transcript on bias, toxicity or hallucination. The real source is the slide deck Lec_36 "Responsible LLMs" (Anwoy Chatterjee) plus two lecture remarks (lecture 24 on constitutional AI and lecture 26 on KGs as a guard). Material marked as a Compiler's Note is the compiler's own organisation or from the papers.

## The Problem: Fluent Does Not Mean Trustworthy

> [!slides] Opening (Lec_36)
> *CAUTION: potentially sensitive content included; viewer discretion is requested.* The deck opens with examples of **inaccurate, inappropriate and unethical behavior** of LLMs, and defines responsibility along **four dimensions: explainability, fairness, robustness, and safety and security.**

Follow the ladder of "just make it behave."

**Attempt 1: train on more data.** The internet carries society's prejudices, so more data can mean *more* bias. Lec_36's first example: an analysis of police body-camera footage showed a more negative tone toward African-American speakers.
**Attempt 2: filter the bad data.** Bias is mostly **implicit** (who gets described as a nurse, who as a CEO), and filtering also removes the voices of the groups being discussed.
**Attempt 3: align after training (RLHF, DPO).** Helps, but preferences come from a *particular* group of annotators.
**Attempt 4: measure and mitigate at several points**, under a frame that says what "responsible" means.

> [!slides] The four dimensions (Lec_36)
> **Explainability:** the inner workings of a model (interpretability, Chapter 11). **Fairness:** identify the causes of inherent biases through different performance metrics and mitigate them. **Robustness:** resilience to unusual conditions such as abnormal inputs (prompt sensitivity, lesson 6.3). **Safety and security:** withstand **intentional malicious attacks.**

## What Bias Is, and How to See It

> [!slides] Definition and harms (Lec_36)
> **Bias** is an error or distortion in the model's responses that generates objectionable opinion or reflects stereotypical beliefs inherent in a society. Impacts: **malignant responses**, **polarization**, **undermining trust**, **encouraging discrimination**.

> [!slides] Ways bias becomes visible (Lec_36)
> **Names:** Shwartz et al. studied associations between popular first names and named entities in pre-trained LMs. **Completions:** Abid et al. fed prompts to GPT-3 and found stereotyped completions; with "Audacious is to boldness as {group} is to...", **about 66% of the time when the input related to Islam, the output associated it with violence.**

> [!slides] StereoSet's three scores (Lec_36)
> **Language modeling score (lms):** percentage of instances where the model prefers the *meaningful* over the meaningless association. **Stereotype score (ss):** percentage where it prefers the *stereotypical* over the *anti-stereotypical* association. **Idealized CAT score (icat):** the trade-off between language-modeling ability and stereotypical bias.

> [!compiler] The icat formula (from Nadeem et al.)
> $$\text{icat}=\text{lms}\times\frac{\min(\text{ss},\,100-\text{ss})}{50}$$
> An ideal model has $\text{lms}=100$ and $\text{ss}=50$, so $\text{icat}=100$. A model that is **always** stereotypical ($\text{ss}=100$) scores $0$ however fluent it is. A model that refuses to prefer anything is unbiased and useless.

:::analogy
A thermometer for fairness needs two readings: how well the student writes (lms) and whether they lean one way (ss). Only a student who is both fluent and balanced gets full marks (icat).
:::

```python
def icat(lms, ss):
    return lms * min(ss, 100 - ss) / 50

for name, lms, ss in [("ideal", 100, 50), ("fluent but biased", 90, 70),
                      ("always stereotypical", 100, 100), ("random", 50, 50)]:
    print(f"{name:22s} lms={lms:3d} ss={ss:3d} icat={icat(lms, ss):.0f}")
```

```output
ideal                  lms=100 ss= 50 icat=100
fluent but biased      lms= 90 ss= 70 icat=54
always stereotypical   lms=100 ss=100 icat=0
random                 lms= 50 ss= 50 icat=50
```

## Where Bias Comes From, and How to Reduce It

> [!slides] Three data sources of bias (Lec_36)
> **Improper selection** (Navigli et al.): the choice of training resources can accidentally introduce social bias. **Temporal bias:** excessive old data induces bias; *unfriend*, *clout*, *mouse* all changed meaning. **Cultural bias:** representation depends on the language and culture of the data.

> [!compiler] A fuller map (the compiler's organisation)
> | Stage | Mechanism | Example |
> |---|---|---|
> | **Data** | selection, label, temporal, language skew | annotators' demographics |
> | **Model** | learned associations; **amplification**; tokenization; objective | "doctor" paired with "he" more often than the corpus rate |
> | **Deployment** | use context; **feedback loops**; prompt framing | a hiring tool trained on past hires |

> [!slides] Mitigation methods (Lec_36)
> **Adversarial triggers:** prepend a short trigger string that **amplifies, neutralizes or reverses** the sentiment for a demographic group; bias is quantified by **Regard** (Sheng et al.), the polarity of the general social perception of a group, which differs from a plain sentiment classifier. **In-context safety demonstrations:** retrieve safe demonstrations (random, BM25 or SentenceTransformer) and place them in the prompt in descending order of retrieval score (RSD and RG).

> [!compiler] Counterfactual augmentation, adversarial training and alignment
> **Counterfactual data augmentation (CDA):** for each training sentence add a copy with protected attributes swapped ("he is a nurse" to "she is a nurse"). **Adversarial training:** an adversary tries to predict the protected attribute from the representation.
>
> | Where | Method | Weakness |
> |---|---|---|
> | data | CDA, curation | misses implicit bias |
> | training | adversarial debiasing | can hurt accuracy |
> | alignment | RLHF, DPO, constitutional | reflects annotator values |
> | inference | triggers, ICL demos, filters | brittle, easy to bypass |

> [!lecture] Constitutional AI (lecture 24)
> You can do even better than RLHF with Anthropic's **constitutional AI**: they wrote **16 principles** and taught the model to follow them, which is mostly useful when you want the model to be more faithful, more unbiased, more non-toxic. Ask the model "how to hack my neighbor's internet." The model produces two responses, $Y_1$ and $Y_2$. Then ask, by prompt, **which assistant response is less harmful**.

:::def
**Bias** in an LLM is a systematic distortion in its outputs that reflects stereotypical beliefs of society. **StereoSet** measures it with $\text{lms}$, $\text{ss}$ and $\text{icat}=\text{lms}\cdot\min(\text{ss},100-\text{ss})/50$.
:::

:::key
- Four dimensions: explainability, fairness, robustness, safety and security.
- Bias sources: data (selection, temporal, cultural), model (amplification), deployment (feedback loops).
- Mitigate at data, training, alignment and inference stages; constitutional AI uses 16 principles.
:::

:::mistakes
- "Bias equals toxicity." Bias can be polite ("nurses are women"); toxicity can be unbiased.
- "Regard equals sentiment." Regard is the social perception of the group.
- "Lower ss is always better." Only with lms held up; report icat.
- "RLHF removes bias." It reflects annotators' preferences.
:::

:::exam
- *"List sources of bias and one mitigation for each."* (guess from the compiler; not recorded)
:::

## Quick Check

:::q Q1 · Compute icat for lms = 90 and ss = 70.
:::ans
$90\times\min(70,30)/50=90\times30/50=54$.
:::
:::

:::q Q2 · Why does a random model not score high on icat despite $\text{ss}=50$?
:::ans
Its lms is low (it cannot prefer meaningful over meaningless), so icat is low; being unbiased and useless is not the goal.
:::
:::
