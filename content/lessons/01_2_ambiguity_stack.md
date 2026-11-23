**NLP is hard mainly because language is ambiguous at every layer, and we handle it by working through a stack: tokens, morphology, syntax, semantics, pragmatics, discourse.** Know this stack in order; it is a favourite short-answer question.

## The Problem: One Sentence, Two Meanings

"I saw a girl with a telescope." Who has the telescope? The sentence has **two parse trees**, one where "with a telescope" attaches to *saw* and one where it attaches to *girl*. Now compare "I ate rice with a spoon", "I ate rice with curd" and "I ate rice with Rahul". The surface structure is the same and the meanings are three different things.

> [!lecture] Intuition
> The real reason NLP is hard is **ambiguity**, and it is not only in language. "Virat Kohli destroyed the other team" is not violence. Add non-standard text (hashtags, "U taught us 2 lose"), segmentation issues, idioms ("kick the bucket"), neologisms ("unfriend"), world knowledge ("Mary and Juhi are sisters" versus "mothers"), and tricky entity names ("A Bug's Life").
> — *NPTEL Lecture 2*

## The Idea: Work Through Layers

We cannot solve everything at once, so NLP is organised as layers. Each layer asks one question.

> [!slides] The NLP stack (memorise this order)
> | Layer | Question it answers | Example |
> |---|---|---|
> | **Tokenization** | What are the units? | `21,53,010 COVID cases`: keep the number together |
> | **Morphology** | How are words built? A morpheme is the smallest unit with meaning | independently = in + depend + ent + ly |
> | **Syntax** | Is it grammatical? POS tags, chunking, parse tree | grammar $G=\langle V,T,P,S\rangle$, rules like $S\to NP\ VP$ |
> | **Semantics** | What does it mean? | decompositional, ontological, **distributional** |
> | **Pragmatics** | What did the speaker *intend*? | "Do you know what time it is?" means "why are you late?" |
> | **Discourse** | Meaning across sentences | resolving "It is open today" |

:::analogy
Reading a letter: first you recognise the letters and words (tokens), then how words are built (morphology), then whether the sentence is grammatical (syntax), then what it says (semantics), then what the writer wanted (pragmatics), then how the paragraphs connect (discourse).
:::

## Facts the Lecture Asks You to Remember

- A **token** is delimited by predefined characters (space, comma). Tokenization breaks a string into tokens.
- **Morphologically rich** languages: Hindi, Turkish, Hungarian. **Morphologically poor**: English, Chinese.
- **POS disambiguation:** "went to the **park** in a car" (noun) versus "went to **park** the car" (verb). The Penn Treebank has 45 tags.
- **Distributional hypothesis:** a word is characterised by the company it keeps. "cat" is understood from "pet", "meow", "feline".

> [!slides] Tasks the stack feeds
> SRL (agent, patient, instrument, time), textual entailment, coreference, NER and relation extraction, word sense disambiguation ("bank"), sentiment, MT (adequacy and fluency), summarization, QA, dialog. The lecture ends: "we will explore how LLMs solve these tasks."

:::def
**Ambiguity** is the property that one surface form of language admits more than one interpretation. It can occur at the lexical, syntactic, semantic and pragmatic levels.
:::

:::key
- NLP is hard because of ambiguity, plus noisy text, idioms, new words and missing world knowledge.
- The stack order is tokenization, morphology, syntax, semantics, pragmatics, discourse.
- The distributional hypothesis is the root of word vectors.
:::

:::mistakes
- Confusing **semantics** (literal meaning) with **pragmatics** (speaker intent).
- Thinking a word and a token are always the same thing. A token is whatever the tokenizer produces.
:::

:::exam
- *"Ambiguity: give two sentences with two readings and draw both parse trees."* Use the telescope sentence.
- *"List the layers of NLP with an example for each."* Reproduce the table above.
:::

## Quick Check

:::q Q1 · Which layer of the stack resolves "It is open today" to a specific shop?
:::ans
**Discourse** (coreference across sentences).
:::
:::

:::q Q2 · State the distributional hypothesis.
:::ans
A word is characterised by the company it keeps: its meaning is captured by the contexts in which it appears.
:::
:::

:::q Q3 · Why is "independently" a good morphology example?
:::ans
It splits into morphemes in + depend + ent + ly, each carrying part of the meaning.
:::
:::
