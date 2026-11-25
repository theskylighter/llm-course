<<SVG:knowledge_graph_triples_completion>>

*Figure: Triple paths answer multi-hop questions; link predictors estimate missing facts, and retrieved subgraphs can ground LLM responses.*

**A knowledge graph stores facts as (subject, relation, object) triples; it is useful but very incomplete, so we complete it by scoring triples with geometric models such as TransE (translation) and RotatE (rotation).** KGs also guard LLMs against hallucination.

## The Problem: LLMs Hide Facts in Opaque Weights

> [!lecture] Opening (lecture 26)
> From the Rosetta Stone and the library in Alexandria to Vannevar Bush's Memex, humans have tried to organize knowledge in one form or another. In computers that has taken the form of knowledge graphs such as Yago, WordNet, Freebase and Wikidata. Today's LLMs embed factual knowledge in an opaque, vast parameter space and are very costly to retrain. **Aesthetically, it is much better to isolate this structured, mutable knowledge from the language understanding and reasoning abilities that LLMs show.**
> — *Prof. Soumen Chakrabarti, lecture 26*

> [!lecture] What a KG is (lecture 26)
> A knowledge base or graph has **nodes that are entities** (Barack Obama, Honolulu, USA) connected by **edges labeled with a relation type** (born in, city of). A fact is a **triple: subject, relation, object**. Beyond binary relations: a table such as *president-of(person, country, start-year, end-year)* becomes a diamond-shaped relation node connected to all its arguments.

> [!lecture] Useful but incomplete
> Because facts are entered with human inspection, they are added slowly. In Freebase, **place of birth is unknown for 93.8% of people**. That makes **KG completion** an important task. The same lecture lists **KG alignment**, matching English and Chinese entities.

> [!lecture] KG question answering: two approaches
> "What city is the Mona Lisa in?" We must disambiguate *Mona Lisa* (the painting), then infer that *exhibited at* followed by *located in* puts it in Paris. **Semantic interpretation** translates the question to a structured query (SQL or SPARQL): interpretable, but tricky to build. The other approach makes the KG differentiable and trains end to end. The two parts deserve to be separated: **factual world knowledge** (can change; a KG is easy to edit) versus **inference** (LLMs are good at this).

> [!lecture] Why combine LLMs and KGs
> (i) Facts change and retraining is costly. (ii) Isolating structured, mutable knowledge **guards against hallucination**. (iii) **Interpretability**: if you can see *which regions of the KG* the model consulted, it is easier to decide how a query went wrong.

## KG Completion as a Scoring Function

> [!slides] Lec_27
> Represent each entity and relation as a continuous geometric object (a point, vector, displacement, projection or rotation) and design a **scoring function $f(s,r,o)$ for the belief in the triple**. Train on known triples, then **infer the unknown ones** (link prediction).

> [!slides] TransE and RotatE (lecture 28)
> **Motivation from word2vec/GloVe:** the vector from *man* to *king* is roughly the one from *woman* to *queen*, so a relation can be a **translation**. **TransE:** $f(s,r,o)=\lVert\mathbf{s}+\mathbf{r}-\mathbf{o}\rVert$, and a **low score means high confidence**. Hinge loss: $\frac1K\sum_k\max\big(0,\ \text{margin}-f(s'_k,r,o'_k)+f(s,r,o)\big)$ with corrupted (negative) triples.

:::analogy
Think of each relation as a signpost "walk this displacement". If you start at the subject and walk the relation's arrow, you should land on the object. TransE measures how far you land from the object.
:::

```python
import numpy as np
s = np.array([1.0, 0.0]); r = np.array([0.0, 1.0]); o = np.array([1.0, 1.0]); bad = np.array([0.0, 0.0])
f = lambda s, r, o: float(np.linalg.norm(s + r - o))
print("true triple score  :", f(s, r, o))
print("corrupted object   :", round(f(s, r, bad), 3))
margin = 1.0
print("hinge loss         :", round(max(0.0, margin - f(s, r, bad) + f(s, r, o)), 3))

# symmetric relation: needs r = -r, so TransE forces r = 0
a, b = np.array([0.2, 0.9]), np.array([0.7, 0.3])
r_ab = b - a
print("score(a,r,b) with r=b-a:", round(f(a, r_ab, b), 3), "| score(b,r,a):", round(f(b, r_ab, a), 3))
```

```output
true triple score  : 0.0
corrupted object   : 1.414
hinge loss         : 0.0
score(a,r,b) with r=b-a: 0.0 | score(b,r,a): 1.562
```

> [!compiler] Worked check of the limits
> TransE with a symmetric relation $r$ would need both $\mathbf{s}+\mathbf{r}=\mathbf{o}$ and $\mathbf{o}+\mathbf{r}=\mathbf{s}$, which forces $\mathbf{r}=\mathbf{0}$ and $\mathbf{s}=\mathbf{o}$: it cannot represent symmetric relations, nor 1-to-many or many-to-1 relations without collapsing entities. **RotatE** rotates in the complex plane, so a rotation by $\pi$ is its own inverse and symmetric relations are possible.

> [!slides] Beyond Lec_28
> **Multiplicative (factorization) models** treat the KG as a subject-relation-object tensor; lecture 26 says these reach the best performance. **Hierarchies** such as (camel, is-a, mammal) and (mammal, is-a, animal) should give (camel, is-a, animal), which none of the methods above handle.

> [!compiler] KG versus RAG
> RAG over passages needs no schema, but gives text that cannot be *joined* ("scientists who played the violin" is a set query). A KG gives exact, editable, multi-hop answers, but is incomplete and costly to build. A common pattern is to retrieve a subgraph or verbalize triples into the prompt.

:::def
A **knowledge graph** is a set of triples $(s,r,o)$ over entities and relations. **TransE** scores a triple by $\lVert\mathbf{s}+\mathbf{r}-\mathbf{o}\rVert$, where a lower value means higher confidence.
:::

:::key
- Triples (subject, relation, object); 93.8% of Freebase people lack a birth place.
- Completion = scoring $f(s,r,o)$ and link prediction; TransE translates, RotatE rotates (fixing symmetry).
- KGQA: semantic parsing (interpretable) versus differentiable KG (end to end).
- KGs guard against hallucination and aid interpretability.
:::

:::mistakes
- "TransE can model any relation." Not symmetric, 1-to-many or many-to-1 ones.
- "A KG is complete." It is not.
:::

:::exam
- *"Why combine LLMs and KGs?"* Facts change; guard hallucination; interpretability.
- *"Write the TransE score and hinge loss; state two limitations."*
:::

## Quick Check

:::q Q1 · In TransE, does a high or low score mean the triple is believed?
:::ans
A **low** score ($\lVert\mathbf{s}+\mathbf{r}-\mathbf{o}\rVert$ small) means high confidence.
:::
:::

:::q Q2 · Why can TransE not model a symmetric relation?
:::ans
It would need $\mathbf{s}+\mathbf{r}=\mathbf{o}$ and $\mathbf{o}+\mathbf{r}=\mathbf{s}$, forcing $\mathbf{r}=\mathbf{0}$.
:::
:::
