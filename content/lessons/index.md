# Introduction to Large Language Models

**A problem-first study guide for Prof. Tanmoy Chakraborty's course on Large Language Models (IIT Delhi).** Fifty-two one-concept lessons, six practice sets and a mock paper, arranged in teaching order.

The course follows the professor's own method: name what is broken, show the cheapest fix, show why the cheap fix fails, and only then give the real solution. The lessons keep that ladder of failed attempts, because it is the pedagogy and also the easiest way to remember the answer in an exam.

## How every lesson is laid out

1. **The problem**: what goes wrong without this idea.
2. **The idea in plain words**, then an **analogy**.
3. **How it works**, step by step, with runnable code and its real output.
4. **Exam-ready definition**, **key takeaways**, **common mistakes** and **how it is asked in exams**.
5. **Quick Check** questions with hidden answers.

Coloured boxes tell you where a sentence came from:

> [!lecture] Lecture
> Paraphrased from the NPTEL lecture transcript, in the professor's order and cadence.

> [!slides] Slides
> Equations, tables and diagrams taken from the slide decks.

> [!compiler] Compiler's Note
> A bridge, a correction or a cross-reference added by the course compiler. These are not the professor's words.

## Course Roadmap

<<ROADMAP>>

## Assumptions made while building this site

:::note
The builder worked unattended, so these assumptions were made and can be changed cheaply.

- **Syllabus units.** No syllabus was supplied, so the nine "Unit" tags follow the source chapters: foundations, transformers (including efficient attention), pretraining, prompting and alignment, efficient fine-tuning, augmented LLMs, multilingual and multimodal, responsible LLMs, advanced topics.
- **Exam shape.** Modelled on the 2024-25 mid-semester and end-semester papers named in the source notes: True/False with justification, MCQ with more than one correct option, short derivations and 6 to 10 mark written answers. The mock paper follows that mix.
- **Code.** Examples are Python (standard library and NumPy), because the source notebooks use PyTorch but the site must run every example for real. NumPy versions of the same maths are used instead.
- **Math.** All formulas are written in LaTeX.
- **Teaching order.** Chapters 3 and 4 put the Transformer before efficient attention, and the pretraining chapter before alignment, because each needs the one before.
:::

## How to study

:::tip
Read a lesson, attempt its Quick Check before opening the answers, then do the practice set at the end of the chapter group. Use the "show all answers" toggle in the top bar only for revision.
:::
