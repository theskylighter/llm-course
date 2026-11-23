**Attempt each question before opening its answer.** This set covers Chapters 8 to 11 and mixes in earlier chapters, as the exam does.

## A. True or False with Justification

:::q A1 · "ALiBi adds positional embeddings to token embeddings." (2 marks)
:::ans
**False.** ALiBi adds no embeddings: it subtracts a fixed, non-learned penalty $m_h(i-j)$ from the attention score, with a head-specific slope. Lesson 11.2.
:::
:::

:::q A2 · "TALMs require extensive pre-training on tool-specific datasets." (2 marks)
:::ans
**False.** They bootstrap from a few demonstrations by self-play and fine-tuning. Lesson 8.2.
:::
:::

:::q A3 · "Position Interpolation extrapolates beyond the trained position range." (2 marks)
:::ans
**False.** It interpolates: new positions are squeezed into the trained range, $m\mapsto mL/L'$. Lesson 11.2.
:::
:::

:::q A4 · "RAG updates the model's weights with new facts." (2 marks)
:::ans
**False.** The index changes, not the weights. Updating knowledge means re-indexing. Lesson 8.1.
:::
:::

## B. Short Answers

:::q B1 · Explain KnowledgeEditor and the role of the KL constraint. (6 marks)
:::ans
A hypernetwork $g_\phi$ takes the edit input, old output, new target and loss gradient and predicts $\Delta\theta$. Train $\phi$ to minimise the edit loss on the prompt and its paraphrases, subject to a KL bound on a set of unrelated inputs so that their predictions stay unchanged (locality). KL measures behaviour change, not weight change. Lesson 11.3.
:::
:::

:::q B2 · How is an image given to a Vision Transformer, and does the architecture change? (4 marks)
:::ans
Cut into $N=HW/P^2$ patches, flatten each to length $P^2C$, linearly project to $D$, prepend `[CLS]`, add position embeddings, feed a standard Transformer encoder. Only the patch embedding and a classification head are new. Lesson 9.2.
:::
:::

:::q B3 · Explain QK and OV circuits. (2 marks)
:::ans
$W_{QK}=W_QW_K^\top$ decides which positions a head attends to (what it reads); $W_{OV}=W_VW_O$ decides what an attended token contributes to the destination's residual stream (what it writes). Lesson 11.5.
:::
:::

:::q B4 · What is a control task in probing and why is it needed? (3 marks)
:::ans
Arbitrary labels assigned to word types. High probe accuracy on the real task could reflect probe capacity, not the representation; selectivity is real accuracy minus control accuracy. Lesson 11.5.
:::
:::

:::q B5 · Define faithfulness versus factuality hallucination, with a mitigation for each. (4 marks)
:::ans
Factuality: output conflicts with real-world facts; mitigated by grounding on retrieval or a KG. Faithfulness: output conflicts with the given input; mitigated by requiring citations, measuring faithfulness, and instructing "use only the sources". Lesson 10.2.
:::
:::

## C. Numericals and Output

:::q C1 · What does this print?
```python
def icat(lms, ss):
    return lms * min(ss, 100 - ss) / 50
print(icat(90, 70), icat(100, 100), icat(100, 50))
```
:::ans
```output
54.0 0.0 100.0
```

Apply $\text{lms}\cdot\min(\text{ss},100-\text{ss})/50$ to each pair.
:::
:::

:::q C2 · What does this print?
```python
steps, p = 10, 0.9
print(round(p ** steps, 2))
```
:::ans
```output
0.35
```

Errors compound across agent steps: $0.9^{10}$.
:::
:::

:::q C3 · What does this print?
```python
p_clean, p_corrupt, p_patched = 0.9, 0.2, 0.6
print(round((p_patched - p_corrupt) / (p_clean - p_corrupt), 2))
```
:::ans
```output
0.57
```

Normalized restoration of the patched component.
:::
:::

:::q C4 · Chunks for a 10,000-token document with chunk size 500 and overlap 50? (3 marks)
:::ans
Stride 450; $\lceil(10000-500)/450\rceil+1=23$ chunks. Lesson 8.1.
:::
:::

## D. Long Answer

:::q D1 · Compare RAG, tool use and agents: what each fixes and what each costs. (8 marks)
:::ans
RAG fixes stale, missing and unverifiable knowledge (cost: retrieval errors, longer prompts). Tools fix computation, live data and actions (cost: latency, formatting errors, safety, prompt injection from untrusted results). Agents fix multi-step tasks (cost: compounding errors, loops, cost). The LLM only emits text; your code executes. Lessons 8.1 and 8.2.
:::
:::

:::q D2 · Describe three ways to extend the context length of a Transformer and the trade-offs. (8 marks)
:::ans
ALiBi: distance penalty on scores, no embeddings, extrapolates. RoPE with Position Interpolation: squeeze positions into the trained range, brief fine-tuning. Sliding window or sinks: constant memory but evicted text cannot be recalled. Ring attention: shards the sequence across devices to cut per-device memory; FLOPs stay $O(n^2)$. Also mention lost in the middle. Lesson 11.2.
:::
:::
