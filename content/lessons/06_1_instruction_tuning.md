<<SVG:instruction_tuning_example_loss>>

*Figure: Instruction examples teach task-following through output-token loss; unlike prompting, tuning updates the model weights.*

**Instruction tuning fine-tunes a base model on (instruction, input, output) examples so it follows natural-language tasks instead of just continuing text.** Weights are updated, unlike prompting.

## The Problem: A Base Model Continues, It Does Not Answer

> [!lecture] Opening
> Today we discuss instruction fine-tuning, one of the key advancements in recent language modeling research, which lets us easily have a conversation with language models. Decoder-based models are trained on vast text from the internet with next-word prediction, so they encode a lot about the world. They are pretty amazing, but they have some major limitations. These pre-trained models are also called **base models**.
> — *NPTEL Lecture 21*

Prompt a base model with *"What is the national flower of India?"* It answers *"What is the national animal of India? What is the national bird of India?"* That is a perfectly good continuation, since such quiz pages exist on the web. Next-word prediction does not ensure the model understands or follows instructions. Follow the ladder.

**Attempt 1: classical multi-task learning.** Train on many tasks by passing input-output pairs. Nobody tells the model *which* task an example is, so a new task at test time leaves it clueless.
**Attempt 2: describe the task in words.** "Predict the sentiment of the sentence", "Translate the sentence to Hindi". Now a new task can also be described: "Is this sentence grammatical?" This is **instruction tuning**.

| Stage | What it teaches | Data |
|---|---|---|
| Pre-training | most of the reasoning power | billions of tokens of raw text |
| Instruction tuning | follow natural-language instructions | thousands of (instruction, output) pairs |
| RLHF | which outputs humans prefer | thousands of prompts plus a reward model or human |

## What an Example Looks Like

> [!lecture] Definition
> Instruction tuning is a fine-tuning technique: we fine-tune the large language model on instruction datasets. Each example has an **instruction**, an optional **input**, and the target **output**. The input is not always mandatory; the instruction and output always exist. In instruction tuning the weights are **updated**. In prompting or in-context learning **no parameter is updated**.

The training and held-out tasks are kept **disjoint**: in the FLAN example the model sees commonsense and translation tasks in training and is tested on natural language inference, which it never saw.

**Training loss.** Feed the instruction and input, predict the next token, use **cross-entropy** against the target token, append the target token (teacher forcing) and repeat until end-of-text. For a decoder-only model everything goes into the decoder. For an encoder-decoder, the instruction and input go to the **encoder** and the output is produced by the **decoder**. With instruction $i_{1:m}$, output $o_{1:n}$ and $o_{n+1}=\langle\text{eot}\rangle$:

$$\mathcal{L}(\theta)=-\sum_{j=0}^{n}\log p_\theta\!\left(o_{j+1}\mid o_{1:j},\,i_{1:m}\right)$$

:::analogy
A new employee who has read the whole library (pre-training) but has never been given a task. Instruction tuning is the onboarding binder of worked requests: "when asked X, do Y".
:::

```python
import math
# Which positions contribute to the loss? Only the output tokens, after the prompt.
prompt = ["Translate", "to", "Hindi", ":", "good"]
output = ["accha", "<eot>"]
tokens = prompt + output
mask = [0] * len(prompt) + [1] * len(output)
for t, m in zip(tokens, mask):
    print(f"{t:10s} in loss: {bool(m)}")
probs = [0.5, 0.8]                 # model probability of each output token
print("loss =", round(-sum(math.log(p) for p in probs), 3))
```

```output
Translate  in loss: False
to         in loss: False
Hindi      in loss: False
:          in loss: False
good       in loss: False
accha      in loss: True
<eot>      in loss: True
loss = 0.916
```

> [!compiler] Notation, and a note on the loss mask
> The formula sums over output positions only, conditioned on the instruction. Whether to also train on the prompt tokens is an implementation choice that the lecture does not discuss. Lecture 21 ends by noting this is **full** fine-tuning, which is compute-heavy, and promises parameter-efficient methods later (Chapter 7).

**Attempt 3: it follows instructions, but is it the answer we want?** Ask *"What's the best way to lose weight quickly?"* Both "reduce carbs, add fiber and protein, exercise" and "stop eating entirely for a few days" are fluent. After instruction tuning the model knows how to respond but not what the *appropriate* response is. That is **alignment** (RLHF, DPO), lessons 6.4 to 6.6.

:::def
**Instruction tuning** is supervised fine-tuning of a pre-trained language model on a dataset of (instruction, optional input, output) triples, with cross-entropy loss on the output tokens, so that it follows natural-language instructions on unseen tasks.
:::

:::key
- Base models continue text; instruction tuning teaches them to follow tasks.
- Example = instruction + optional input + output; weights updated; held-out tasks disjoint.
- Loss: teacher-forced cross-entropy on output tokens.
- Prompting updates no parameters; instruction tuning does.
:::

:::mistakes
- "Instruction tuning is in-context learning." Tuning updates weights; prompting never does.
- "Which part of the example is optional?" The input. Instruction and output always exist.
:::

:::exam
- *"Name the three parts of an instruction-tuning example. Which is optional?"* (Tier 1)
- *"Instruction tuning versus prompting."*
:::

## Quick Check

:::q Q1 · Why must test tasks be disjoint from training tasks?
:::ans
To measure zero-shot generalisation to unseen tasks from instructions alone.
:::
:::

:::q Q2 · Where does the input go in an encoder-decoder instruction-tuned model?
:::ans
Instruction and input go to the encoder; the output is generated by the decoder.
:::
:::
