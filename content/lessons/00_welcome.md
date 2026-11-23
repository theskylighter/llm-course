**This course is about one question: how do we get from counting words to a model that can follow instructions, use tools and explain itself?** Every lesson answers one step of that path.

## The Problem: Notes That List Without Teaching

Lecture slides give you formulas. Slides do not tell you why the formula had to be invented. Exams, however, ask the "why": why does self-attention need positional encoding, why does Kneser-Ney use continuation counts, why is the reward model trained on comparisons and not scores. If you only memorise the final formula, a differently phrased question defeats you.

> [!lecture] How the professor teaches
> The course keeps one rhythm. We name what is broken. We show the cheapest fix. We show why the cheap fix fails. Only then do we give the real solution. The hope is that when you see the real solution, you already feel the need for it.

## The Path in One Picture

```text
counts (N-grams)  ->  vectors (neural LMs)  ->  attention (Transformer)
      -> pretraining (BERT, GPT, T5)  ->  instruction tuning + RLHF/DPO
      -> cheap adaptation (LoRA, QLoRA)  ->  retrieval, tools, agents
      -> multilingual / multimodal  ->  responsibility  ->  interpretability
```

:::analogy
Think of learning to cook for a restaurant. First you count which dishes sell (N-grams). Then you learn what ingredients are similar (embeddings). Then you build a kitchen where every cook can see every order (attention). Then the kitchen is trained on a whole library of recipes (pretraining), taught to follow a customer's request (instruction tuning), and finally taught what customers actually enjoy (alignment).
:::

## How to Use the Lessons

- Each lesson takes 4 to 10 minutes. Do one, then answer its Quick Check from memory.
- A **practice set** follows every two or three chapters. Later sets mix earlier topics on purpose.
- The final **Mock Paper** is timed and mixes every unit.
- Run the code. Every example on this site was executed, and the output shown is the real output.

:::key
- Learn the problem before the formula.
- Learn the failed attempts before the final answer.
- Learn which statement is the exam trap: the course notes flag these explicitly.
:::

## Quick Check

:::q Q1 · Why does the course teach a failed idea before the real one?
:::ans
Because each failure shows what the real solution must provide. For example, the positional-encoding lesson tries the raw index, then the normalised index, then binary numbers, and the sinusoid appears as the answer to the binary version's periodicity.
:::
:::

:::q Q2 · What does a "Compiler's Note" box mean?
:::ans
It is a bridge, correction or cross-reference added by the course compiler. It is not the professor's own words. Corrections to the lecture always go in such a box.
:::
:::
