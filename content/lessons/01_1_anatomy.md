**A good lesson teaches one idea, starts from a problem the reader recognises, and ends with a way to check understanding.** This page shows every building block the renderer supports.

## The Problem: Notes That Don't Teach

Reference-style notes list definitions and leave the reader to work out *why* they matter. Students re-read them, feel familiar with the words, and still can't answer a question that is phrased differently. A lesson should instead start from **what goes wrong without the idea**.

## The Idea in Plain Words

State the concept in two or three plain sentences first. Use **bold** for the key term, `code spans` for identifiers, and [links](https://example.com) sparingly.

:::analogy
Pick something from everyday life that works the same way. A good analogy makes the idea obvious; a forced one just adds noise. Cut it if you can't find a good one.
:::

## How It Works

Number the steps so a reader can follow them in order:

1. Show the situation.
2. Apply the idea.
3. Show the result.

| Building block | Written as | Use it for |
|---|---|---|
| Box | `:::key` ... `:::` | takeaways, mistakes, definitions, exam tips |
| Question | `:::q` with a nested `:::ans` | self-check questions, answer hidden until clicked |
| Diagram | `<<SVG:sample>>` | a figure kept in `content/diagrams/sample.svg` |
| Code | a fenced block with a language | runnable examples, with an `output` block after it |

> **A formal definition goes in a quote block**, so it stands out and can be copied into an exam answer word for word.

<<SVG:sample>>

## All the Boxes

:::def
**Definition box.** The exam-ready wording of the idea, short enough to memorise.
:::

:::example
**Example box.** A worked example or a real-world case.
:::

:::note
**Note box.** Extra context that is useful but not essential.
:::

:::tip
**Tip box.** A practical hint.
:::

:::warn
**Important box.** Something that causes serious trouble if missed.
:::

:::mistakes
- Mistakes students usually make, one bullet each.
- Say what the wrong belief is **and** what is true instead.
:::

:::exam
- *"Explain X with an example."* Define it, give the analogy, draw the diagram.
- Common MCQ trap: mixing up X and Y.
:::

:::key
- One idea per bullet.
- If a reader remembers only this box, they should still have the core of the lesson.
:::

:::summary
A recap for the end of a chapter or a revision page.
:::

## Quick Check

:::q Q1 · Why should a lesson begin with a problem rather than a definition?
:::ans
Because the reader then sees **why** the idea exists, which makes the definition easy to remember and apply to unfamiliar questions.
:::
:::

:::q Q2 · Which box holds the exam-ready wording of an idea?
:::ans
The **definition** box (`:::def`).
:::
:::
