**Research Papers Journey: From Conditional Computation to Agentic MoE** — the intellectual arc that connects every paper in your vault into one coherent story.

---

## 1. The Problem: Why Read Papers at All?

> [!lecture] Opening
> You have implemented attention, you have fine-tuned LoRA adapters, you have built RAG pipelines. But when a new paper drops — *Mixture of Experts*, *Rotary Position Embeddings*, *Direct Preference Optimization* — do you understand **why it exists**, **what problem it solves**, and **how it connects to what you already know**? This lesson is the map that turns a pile of PDFs into a single, navigable landscape.

> [!compiler] How this lesson fits the course
> Every chapter so far has taught you a *technique*: attention (Ch. 3), efficient attention (Ch. 4), pretraining (Ch. 5), instruction tuning and alignment (Ch. 6), PEFT (Ch. 7), RAG (Ch. 8), multimodality (Ch. 9). This lesson stitches them together by showing the **papers that introduced each idea** and **why each paper had to exist**. Think of it as the "director's commentary" for the entire course.

---

## 2. The Seed: Conditional Computation (1991)

> [!lecture] The idea that started it all
> Before Transformers, before GPUs, before "large" meant billions of parameters — **Jacobs, Jordan, Nowlan, and Hinton (1991)** asked: *What if a neural network could decide which subset of its parameters to use for each input?*
>
> They proposed **Adaptive Mixtures of Local Experts**: a set of specialist networks ("experts") and a **gating network** that decides how much each expert contributes for a given input. Training encourages the experts to **specialise** on different cases. This is the seed of **conditional computation** — the idea that *not every parameter needs to participate in every forward pass*.

> [!slides] Why this matters for everything that follows
> | 1991 Idea | 2017 Realization | 2026 Production |
> |-----------|------------------|-----------------|
> | Experts + gating network | Sparsely-gated MoE layer: trainable gate picks a sparse top-k of experts (Shazeer et al.) | GLM-5: 744B-param MoE (40B active per token) |
> | Conditional computation | Made practical on GPU clusters, up to 137B params | Sparse experts plus DSA sparse attention |
> | "Not every weight for every input" | Only the top-k experts run per example | Async RL post-training |

> [!compiler] The thread connecting 1991 → 2017 → 2026
> The 1991 paper was **conceptual** — no GPUs, no Transformers. The 2017 paper (Shazeer et al., *Outrageously Large Neural Networks*) made it practical: a **sparsely-gated MoE layer** with up to thousands of experts, reporting **>1000× improvements in model capacity with only minor losses in computational efficiency**, and a model with up to 137B parameters. GLM-5 (Feb 2026, Zhipu AI + Tsinghua) puts the idea in a modern stack: a **744B-parameter MoE with 40B active parameters**, **DSA (DeepSeek Sparse Attention)**, **asynchronous RL**, and a focus on **agentic engineering**.

:::deep Adaptive MoE — Adaptive Mixtures of Local Experts | Jacobs, Jordan, Nowlan, Hinton, 1991 · ~4 min
**Paper:** Jacobs, Jordan, Nowlan, Hinton — *Adaptive Mixtures of Local Experts* (Neural Computation, 1991) — [PDF](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

#### The problem
A single network trained on a task that is really several sub-tasks suffers **interference**: gradients from one kind of case disturb the weights that serve another. Earlier mixtures blended the experts' outputs and trained the blend, so the experts **cooperated**: each one was pushed to fix the *combined* error, and none of them cleanly specialised.

#### The core idea
1. Several **expert networks** see the same input; each produces a complete output $o_i$.
2. A **gating network** sees the same input and outputs mixing probabilities $p_i$ (a softmax, so they sum to 1).
3. The error function treats the system as **picking one expert at random** according to $p_i$, so the experts **compete** for each case instead of jointly fitting a residual.
4. The gate learns which expert handles which kind of input, and each expert specialises on a **local region** of the input space.

#### The key equation
Cooperative error (the old way) corrects every expert to fix the blend:

$$E = \lVert d - \textstyle\sum_i p_i o_i \rVert^2$$

Competitive error (this paper):

$$E = -\log \sum_i p_i \, e^{-\frac{1}{2}\lVert d - o_i \rVert^2}$$

Its gradient for expert $i$ is $-h_i (d - o_i)$, where $h_i$ is that expert's **posterior responsibility** for the case — its prior $p_i$ re-weighted by how well it fits. The expert that already fits a case best learns most from it; the others are barely touched.

#### What it showed
On a multi-speaker vowel-recognition task, the gate learned to split the input space and different experts took different regions — specialisation emerged from the error function alone, with no hand-assigned roles.

#### Limitations → what came next
- **Every expert still runs on every input.** The gate *weights* outputs; it saves no computation.
- Small networks, no notion of scale.
- → The [Sparsely-Gated MoE](#deep-sparsely-gated-moe) (2017) keeps only the **top-k** gate values, so unchosen experts are never computed — conditional computation becomes a compute saving.

#### Exam angle
1. Why does the 1991 error function produce specialisation? *(Each expert is updated in proportion to its responsibility for the case, so experts compete instead of jointly fitting the residual.)*
2. What does the gating network output? *(A softmax distribution over experts for the current input.)*
3. Does this model save computation? *(No — all experts run; sparsity arrives in 2017.)*
:::

:::deep Sparsely-Gated MoE — Outrageously Large Neural Networks | Shazeer et al., 2017 · ~5 min
**Paper:** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean — *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017) — [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)

#### The problem
Model quality grows with **capacity** (parameters), but in a dense network compute grows with parameters too. Conditional computation promised to break that link, but in practice it ran into GPU-unfriendly branching, tiny per-expert batches, load imbalance, and network bandwidth limits.

#### The core idea
1. A **MoE layer** of $n$ feed-forward experts (up to thousands) is placed between stacked LSTM layers and applied at every position.
2. A **noisy top-k gate** scores the experts, adds tunable Gaussian noise, and **keeps only the top $k$**; the rest get weight exactly 0 and are **never computed**.
3. The layer output is the gate-weighted sum of the $k$ chosen experts.
4. **Balancing losses** stop the gate collapsing onto a few favourite experts (a self-reinforcing failure: favoured experts train faster and get chosen more).
5. Data and model parallelism are mixed so each expert still sees a large enough batch.

#### The gating equations
$$H(x)_i = (x W_g)_i + \epsilon_i \cdot \mathrm{Softplus}\big((x W_{\text{noise}})_i\big), \quad \epsilon_i \sim \mathcal{N}(0,1)$$

$$G(x) = \mathrm{Softmax}\big(\mathrm{KeepTopK}(H(x), k)\big), \qquad y = \sum_{i=1}^{n} G(x)_i \, E_i(x)$$

`KeepTopK` sets everything outside the top $k$ to $-\infty$, so those softmax weights are 0.

**Importance loss:** with $\mathrm{Importance}(X)$ = each expert's total gate value over a batch,

$$L_{\text{importance}} = w_{\text{importance}} \cdot \mathrm{CV}\big(\mathrm{Importance}(X)\big)^2$$

(CV = coefficient of variation). A second **load loss** balances the *number of examples* each expert receives.

#### Results that mattered
Models with up to **137B parameters** on language modelling (1B Word benchmark) and machine translation (WMT'14), with **>1000× more capacity for only minor losses in computational efficiency**, beating the previous state of the art at lower compute.

#### Limitations → what came next
- Built on LSTMs; heavy cross-device communication; training can be unstable.
- The auxiliary losses are hand-tuned.
- → Later Transformer MoEs inherit the same recipe — router, top-k, load balancing — up to [GLM-5](#deep-glm-5) (256 experts, 40B of 744B parameters active).

#### Exam angle
1. Where does the compute saving come from? *(Experts outside the top-k are never evaluated.)*
2. Why add the balancing loss? *(Without it the gate collapses onto a few experts; the rest are wasted capacity.)*
3. Why add noise to the gate? *(It spreads routing across experts and helps load balancing.)*
:::

---

## 3. The Architecture: Attention Is All You Need (2017)

> [!lecture] The paper that changed everything
> *Vaswani et al. (2017)* built the sequence model entirely out of **attention**, removing recurrence. Every paper in your vault either **uses** this architecture, **modifies** it, or **optimizes** it.

> [!slides] The Transformer family tree
> ```
> Transformer (2017)
>    ├── Encoder stack → BERT (2018) → ...
>    ├── Decoder stack → GPT-1 → GPT-2 → GPT-3 → ...
>    ├── Encoder–Decoder → T5 (2019), BART (2019)
>    └── Modifications → RoPE, GQA, FlashAttention, SWAT, MoE layers
> ```

> [!compiler] Not in vault but essential
> The Transformer paper is **not in your vault PDFs** — but every other paper in your vault either cites it or builds on it. We treat it as the **root node** of the progression graph. Its mechanics are Ch. 3, Lessons [3.1](03_1_why_remove_recurrence.html)–[3.10](03_10_rope.html).

:::deep Transformer — Attention Is All You Need | Vaswani et al., 2017 · ~5 min
**Paper:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin — *Attention Is All You Need* (NeurIPS 2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

#### The problem
RNNs process a sequence **one step at a time**, so training can't parallelise within a sequence, and information between distant tokens must travel through $O(n)$ steps. Attention was already used *alongside* RNNs; the question was whether it could replace them entirely.

#### The core idea
1. An **encoder–decoder** model built only from attention and feed-forward layers — no recurrence, no convolution.
2. **Scaled dot-product attention:** each token's query is compared with every key; the softmax weights mix the values.
3. **Multi-head attention:** run $h$ attentions in parallel on different learned projections, so different heads can track different relations.
4. The **decoder** uses masked self-attention (no peeking at future tokens) plus **cross-attention** to the encoder's output.
5. Each layer adds a position-wise feed-forward network, residual connections and layer normalisation; **sinusoidal positional encodings** supply word order.

#### The key equation
$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Why divide by $\sqrt{d_k}$: for large $d_k$ the dot products grow large, pushing the softmax into regions with tiny gradients.

#### Why it beats recurrence
| Layer type | Cost per layer | Sequential steps | Longest path between tokens |
|--|--|--|--|
| Self-attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n d^2)$ | $O(n)$ | $O(n)$ |

#### Model size (base)
6 encoder + 6 decoder layers, $d_{\text{model}} = 512$, 8 heads ($d_k = 64$), feed-forward inner size 2048.

#### Results that mattered
WMT 2014 English→German **28.4 BLEU**; English→French **41.8 BLEU** (single-model state of the art) after training for **3.5 days on eight P100 GPUs** — a fraction of earlier systems' training cost.

#### Limitations → what came next
- Attention is **$O(n^2)$** in sequence length → [FlashAttention](#deep-flashattention), [GQA](#deep-gqa), [SWAT](#deep-swat).
- Fixed absolute positions extrapolate poorly → [RoPE](#deep-rope).
- The encoder and decoder were soon split apart → [BERT](#deep-bert) (encoder), [GPT-1](#deep-gpt-1) (decoder), [T5](#deep-t5) (both).

#### Exam angle
1. Why scale by $\sqrt{d_k}$? *(Keep dot-product variance in check so softmax gradients don't vanish.)*
2. Why multiple heads? *(Each head attends in a different learned subspace, capturing different relations at once.)*
3. What is masked in the decoder, and why? *(Future positions, so training matches left-to-right generation.)*

**Course lessons:** [3.1 Why remove recurrence](03_1_why_remove_recurrence.html) through [3.10 RoPE](03_10_rope.html)
:::

---

## 4. Contextual Representations: ELMo → BERT → GPT-1 (2018)

> [!lecture] The problem with static embeddings
> Word2Vec and GloVe give **one vector per word** — "bank" has the same vector in "river bank" and "bank account." ELMo (Peters et al., 2018) proved **contextual embeddings** work: run a **biLSTM** language model over the sentence, take a **weighted sum of layer hidden states**, and the representation changes with context.

> [!slides] Why ELMo wasn't the final answer
> | Breakthrough | Limitation |
> |--------------|------------|
> | Contextual vectors | biLSTM = sequential (slow, not parallelizable) |
> | Layer-weighted sum | Used as extra features; the task model is still built separately |
> | Proved context matters | Architecture not scalable |

> [!compiler] The fork in the road
> ELMo's limitation — **biLSTMs don't parallelize** — forced the field to ask: *How do we get contextual representations with parallel computation?*
>
> Two answers emerged **in 2018**:
>
> 1. **BERT** (Devlin et al.): **Transformer encoder** + **Masked Language Modeling (MLM)** → bidirectional, parallelizable, great for *understanding* tasks
> 2. **GPT-1** (Radford et al.): **Transformer decoder** + **next-token prediction** → autoregressive, parallelizable at train time, great for *generation*

> [!slides] The 2018 fork: Understanding vs. Generation
> | | BERT (Encoder) | GPT-1 (Decoder) |
> |--|----------------|-----------------|
> | **Attention** | Bidirectional (full context) | Causal (left-to-right) |
> | **Objective** | MLM (predict masked tokens) | Next-token LM |
> | **Strength** | Understanding, classification | Generation, completion |
> | **Weakness** | Not a natural free-text generator | Cannot see future context |

> [!compiler] Course connection
> - **Ch. 5, [Lesson 5.1](05_1_elmo.html)**: ELMo — contextual embeddings
> - **Ch. 5, Lessons [5.2](05_2_bert_mlm.html)–[5.3](05_3_bert_cls_qa.html)**: BERT's MLM objective, [CLS] token, fine-tuning heads
> - **Ch. 5, [Lesson 5.5](05_5_decoder_only.html)**: Decoder-only models — GPT training, causal mask, next-token prediction
> - **Ch. 3, [Lesson 3.9](03_9_transformer_training.html)**: Transformer training — teacher forcing


:::deep ELMo — Deep Contextualized Word Representations | Peters et al., 2018 · ~4 min
**Paper:** Peters, Neumann, Iyyer, Gardner, Clark, Lee, Zettlemoyer — *Deep contextualized word representations* (NAACL 2018) — [arXiv:1802.05365](https://arxiv.org/abs/1802.05365)

#### The problem
Word2Vec and GloVe assign **one vector per word type**. They can't separate the senses of "bank", and can't reflect how a word's syntax and meaning change with its sentence.

#### The core idea
1. Pre-train a **bidirectional language model (biLM)**: a forward LSTM predicts the next word, a backward LSTM predicts the previous word, and both log-likelihoods are maximised jointly (sharing the token-embedding and softmax parameters).
2. Inputs go through a **character CNN**, so any word — including unseen ones — gets a representation.
3. Two **biLSTM layers** (4096 units, 512-dim projections) sit on top.
4. For a downstream task, **freeze the biLM** and feed a learned, task-specific **weighted sum of all its layers** into the task model alongside the usual word vectors.

#### The key equation
$$\mathrm{ELMo}_k^{\text{task}} = \gamma^{\text{task}} \sum_{j=0}^{L} s_j^{\text{task}} \, \mathbf{h}_{k,j}$$

$\mathbf{h}_{k,j}$ is layer $j$'s vector for token $k$ (layer 0 = the character-CNN embedding), $s^{\text{task}}$ are softmax-normalised layer weights, and $\gamma^{\text{task}}$ scales the whole vector. Each task learns *which depth* it needs.

#### What the layers learn
The paper's analysis finds **lower layers** capture syntax (they help part-of-speech tagging) and **higher layers** capture meaning in context (they help word-sense disambiguation) — which is why mixing layers beats using only the top one.

#### Results that mattered
Improved the state of the art on **six** tasks — question answering (SQuAD), textual entailment (SNLI), semantic role labelling, coreference, named entity recognition and sentiment (SST-5) — with **6–20% relative error reductions**.

#### Limitations → what came next
- LSTMs are sequential: slow, hard to scale.
- The biLM is **frozen features**; every task still needs its own architecture.
- Its two directions are trained **separately** — shallow bidirectionality.
- → [GPT-1](#deep-gpt-1) fine-tunes a whole Transformer; [BERT](#deep-bert) makes every layer bidirectional.

#### Exam angle
1. What do $s_j$ and $\gamma$ do? *(Task-specific softmax weights over layers, and an overall scale.)*
2. Why a character CNN at the input? *(No out-of-vocabulary words; morphology is captured.)*
3. How is ELMo used downstream? *(As frozen contextual features concatenated into a task-specific model.)*

**Course lesson:** [5.1 ELMo](05_1_elmo.html)
:::

:::deep BERT — Pre-training of Deep Bidirectional Transformers | Devlin et al., 2018 · ~5 min
**Paper:** Devlin, Chang, Lee, Toutanova — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (NAACL 2019) — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

#### The problem
GPT-1 pre-trains a Transformer **left-to-right**, so every token only sees its past. ELMo is bidirectional, but only by **concatenating two one-way LSTMs** trained separately. Neither lets every layer look at the **whole sentence at once**, which matters for tasks like question answering, where the answer depends on context on both sides.

#### The core idea
1. Take the **Transformer encoder** (no causal mask), so every token attends to every other token.
2. A normal language model can't be trained that way: each word would simply "see itself". So **hide some tokens and predict them** — the *Masked Language Model* (MLM).
3. Add a second task, **Next Sentence Prediction** (NSP): given sentences A and B, is B the real next sentence (50%) or a random one (50%)?
4. **Fine-tune the whole network** on each downstream task, adding only a small output layer.

#### Inputs
`[CLS] sentence A [SEP] sentence B [SEP]`, each token embedded as **token + segment (A/B) + position**, using a 30k WordPiece vocabulary.

#### The MLM objective
Choose 15% of token positions $M$. Of those, replace 80% with `[MASK]`, 10% with a random token, and leave 10% unchanged. Then minimise

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log p\left(x_i \mid \tilde{x}\right)$$

where $\tilde{x}$ is the corrupted sentence. The 80/10/10 split exists because `[MASK]` never appears at fine-tuning time; mixing in random and unchanged tokens keeps the model from relying on it.

#### Model sizes and data
| | Layers | Hidden | Heads | Parameters |
|--|--|--|--|--|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

Pre-trained on BooksCorpus (800M words) + English Wikipedia (2,500M words).

#### Results that mattered
New state of the art on eleven NLP tasks: GLUE score **80.5** (+7.7 points absolute), MultiNLI **86.7%**, SQuAD v1.1 test F1 **93.2**, SQuAD v2.0 test F1 **83.1**.

#### Limitations → what came next
- `[MASK]` is seen in pre-training but never at fine-tuning (the 80/10/10 trick only softens this).
- Only the 15% masked tokens produce a training signal, so learning is sample-inefficient.
- An encoder doesn't generate text naturally → **T5 and BART** (§5) pair an encoder with a decoder.

#### Exam angle
1. Why can't BERT be trained with an ordinary left-to-right LM objective? *(Bidirectional attention would let each token see the word it must predict.)*
2. Why the 80/10/10 masking split? *(Reduce the pre-train/fine-tune mismatch caused by `[MASK]`.)*
3. What is the `[CLS]` token for? *(Its final hidden state is the sequence representation for classification heads.)*

**Course lessons:** [5.2 BERT's MLM objective](05_2_bert_mlm.html), [5.3 [CLS] and QA heads](05_3_bert_cls_qa.html)
:::

:::deep GPT-1 — Improving Language Understanding by Generative Pre-Training | Radford et al., 2018 · ~4 min
**Paper:** Radford, Narasimhan, Salimans, Sutskever — *Improving Language Understanding by Generative Pre-Training* (OpenAI, 2018) — [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

#### The problem
Labelled data is scarce, unlabelled text is plentiful. Earlier transfer (word vectors, ELMo features) still required a **task-specific architecture** on top. Could one pre-trained model be **fine-tuned end-to-end** for many tasks with almost no new parameters?

#### The core idea
1. **Unsupervised pre-training:** train a Transformer **decoder** with the ordinary language-modelling objective on long contiguous text.
2. **Supervised fine-tuning:** add one linear layer $W_y$ on the last token's final hidden state and train on the labelled task.
3. **Keep the LM loss during fine-tuning** as an auxiliary objective — it improves generalisation and speeds up convergence.
4. **Traversal-style input transformations:** turn structured inputs into a single token sequence with start, delimiter and extract tokens, so the architecture never changes.

| Task | Input sequence(s) |
|--|--|
| Entailment | `[start] premise [delim] hypothesis [extract]` |
| Similarity | both orderings of the two sentences; the two final representations are added |
| Multiple choice | context + each answer as separate sequences; softmax over their scores |

#### The objectives
$$L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \ldots, u_{i-1}; \Theta)$$

$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \, L_1(\mathcal{C}), \qquad \lambda = 0.5$$

$L_2$ is the supervised task loss; $L_1$ on the task data is the auxiliary LM loss.

#### Model and data
12-layer decoder-only Transformer with masked self-attention, 768-dim states, 12 heads, 3072-dim feed-forward; **learned** position embeddings; GELU; 40,000-merge BPE; 512-token contexts. Pre-trained on **BooksCorpus** (7,000+ unpublished books — long stretches of contiguous text).

#### Results that mattered
State of the art on **9 of the 12** tasks studied, including absolute gains of **8.9%** on commonsense reasoning (Story Cloze), **5.7%** on question answering (RACE) and **1.5%** on textual entailment (MultiNLI).

#### Limitations → what came next
- **Left-to-right only:** a token never sees what follows → [BERT](#deep-bert) (October 2018) pre-trains bidirectionally and compares itself directly against GPT-1.
- Still needs **fine-tuning per task** → [GPT-3](#deep-gpt-3) asks whether scale alone can remove that step.

#### Exam angle
1. How does GPT-1 differ from ELMo in transfer? *(Fine-tunes the whole network vs. frozen features into a task model.)*
2. Why keep the LM loss during fine-tuning? *(Better generalisation and faster convergence.)*
3. How does it handle multi-sentence tasks without new architecture? *(Traversal-style input transformations with delimiter tokens.)*

**Course lesson:** [5.5 Decoder-only models](05_5_decoder_only.html)
:::

---

## 5. Unifying Objectives: T5 and BART (2019)

> [!lecture] The problem with specialization
> By 2019, the field had **BERT for understanding** and **GPT for generation**. But real NLP tasks need *both*: translation (understand source, generate target), summarization (understand long text, generate short), QA (understand question + context, generate answer).

> [!slides] Two paths to unification
> | Model | Architecture | Objective | Key Insight |
> |-------|--------------|-----------|-------------|
> | **T5** (Raffel et al., 2019) | Encoder–Decoder | **Span corruption** (mask spans, predict them) | **Every NLP task = text in → text out** |
> | **BART** (Lewis et al., 2019) | Bidirectional encoder + autoregressive decoder | **Denoising autoencoder** (corrupt → reconstruct) | BERT-style encoder + GPT-style decoder = best of both |

> [!compiler] Why both exist
> T5 and BART are **peers**, not competitors. T5's span corruption is simple and scales; BART's denoising (token masking, deletion, infilling, sentence permutation, document rotation) is richer. **T5's "text-to-text" framing** is the mental model behind instruction tuning (Ch. 6) and prompting (Ch. 6).

> [!slides] Course connections
> - **Ch. 5, [Lesson 5.4](05_4_bart_t5.html)**: BART vs. T5 — span corruption vs. denoising, sentinel tokens
> - **Ch. 5, [Lesson 5.5](05_5_decoder_only.html)**: Decoder-only line (GPT) — why decoders won for generation

:::deep T5 — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Raffel et al., 2019 · ~5 min
**Paper:** Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li, Liu — *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (JMLR 2020) — [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)

#### The problem
By 2019 there were many transfer-learning recipes — different architectures, objectives, datasets and fine-tuning schemes — and each task had its own output head and format, so they were hard to compare fairly.

#### The core idea
1. **Text-to-text:** cast *every* task as input text → output text, using a task prefix: `translate English to German: …`, `summarize: …`, even regression (STS-B similarity scores are predicted as strings).
2. One **standard encoder–decoder Transformer** and one loss (cross-entropy on the output text) for every task.
3. A large **systematic study**: architectures, pre-training objectives, datasets, transfer strategies and scale, changing one factor at a time.
4. Build **C4** (Colossal Clean Crawled Corpus): about **750 GB** of cleaned Common Crawl English text.
5. Scale the best recipe up to **11B parameters**.

#### Span corruption (the chosen objective)
Drop **15%** of tokens in contiguous spans (mean span length 3). Each dropped span becomes one **unique sentinel** token in the input; the target lists each sentinel followed by the tokens it hides:

| | Text |
|--|--|
| Original | Thank you for inviting me to your party last week. |
| Input | Thank you `<X>` me to your party `<Y>` week. |
| Target | `<X>` for inviting `<Y>` last `<Z>` |

Targets contain only the dropped spans, so they are short — cheaper to train than reconstructing the whole sentence.

#### Results that mattered
State-of-the-art results on many benchmarks covering summarisation, question answering and text classification (including GLUE, SuperGLUE and SQuAD) from one model family and one format.

#### Limitations → what came next
- Encoder–decoder models cost more per generated token than decoder-only ones; the field later standardised on decoders ([GPT-3](#deep-gpt-3)).
- But the **text-to-text framing** became the template for instruction tuning — see the [IT Survey](#deep-it-survey).

#### Exam angle
1. What is a sentinel token? *(A unique placeholder that replaces one dropped span in the input and labels it in the target.)*
2. Why is span corruption cheaper than full reconstruction? *(The target holds only the dropped spans.)*
3. How does T5 handle a regression task? *(It outputs the number as text.)*

**Course lesson:** [5.4 BART vs. T5](05_4_bart_t5.html)
:::

:::deep BART — Denoising Sequence-to-Sequence Pre-training | Lewis et al., 2019 · ~4 min
**Paper:** Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov, Zettlemoyer — *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension* (ACL 2020) — [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)

#### The problem
[BERT](#deep-bert) understands but can't naturally generate; [GPT-1](#deep-gpt-1) generates but only sees left context. Generation tasks such as summarisation need both: read the whole input, then write fluently.

#### The core idea
1. A standard seq2seq Transformer: a **bidirectional encoder** (BERT-like) and an **autoregressive decoder** (GPT-like).
2. Pre-train as a **denoising autoencoder**: corrupt a document with any noising function, then reconstruct the **original** document; the loss is cross-entropy on the original.
3. Because the decoder regenerates everything, the corruption can be **arbitrary** — even changing the length or the order of the text.

#### The noising functions compared
| Noise | What happens |
|--|--|
| Token masking | random tokens → `[MASK]` (as in BERT) |
| Token deletion | random tokens removed; the model must find where |
| **Text infilling** | spans with lengths drawn from Poisson($\lambda = 3$) → a **single** `[MASK]` each (0-length spans too) |
| **Sentence permutation** | sentences shuffled |
| Document rotation | document rotated to start at a random token |

The final model uses **text infilling + sentence permutation**. Infilling forces the model to predict *how many* tokens are missing, not just which ones.

#### Results that mattered
Matches **RoBERTa** on GLUE and SQuAD with comparable training resources, gains **up to 6 ROUGE** on abstractive summarisation, and **+1.1 BLEU** over a back-translation system for machine translation.

#### T5 vs BART at a glance
| | T5 | BART |
|--|--|--|
| Masked span → | a unique sentinel per span | a single `[MASK]` per span |
| Target | only the dropped spans | the whole original text |
| Extra noise | — | sentence shuffling |

#### Limitations → what came next
Both still need **fine-tuning for each task**. [GPT-3](#deep-gpt-3) asks whether a big enough model can skip that step altogether.

#### Exam angle
1. Why does text infilling teach length prediction? *(A whole span, of unknown length, collapses to one `[MASK]`.)*
2. Why can BART use arbitrary corruptions? *(The decoder regenerates the full original, so input and output needn't align.)*
3. Which noising combination did BART adopt? *(Text infilling + sentence permutation.)*

**Course lesson:** [5.4 BART vs. T5](05_4_bart_t5.html)
:::

---

## 6. Scale and In-Context Learning: GPT-3 (2020)

> [!lecture] The moment everything changed
> GPT-3 (Brown et al., 2020) scaled the decoder-only line to **175B parameters**. The surprise wasn't just better perplexity — it was **in-context learning (ICL)**: give the model a few examples in the prompt, and it performs the task *without any weight updates*.

> [!slides] What GPT-3 actually showed
> | Capability | Before GPT-3 | With GPT-3 |
> |------------|--------------|------------|
> | New task | Fine-tune (gradient steps) | Few-shot prompt (0 gradient steps) |
> | Task specification | Labelled dataset per task | A few demonstrations in the prompt |
> | Scaling | Gains mostly from fine-tuning | Few-shot ability improves with model size |

> [!compiler] The emergence debate
> **Does ICL "emerge" at scale?** This is **debated, not settled**:
> - *Scale view*: abilities such as few-shot learning appear to grow sharply as models get larger (GPT-3 reported few-shot performance improving with size, more so than zero-shot).
> - *Metric-artifact view* (Schaeffer et al., 2023): apparent "jumps" come from non-linear or discontinuous metrics such as exact match; smooth metrics show gradual improvement.
> - *Data view*: ICL may depend on what is in the training distribution (diverse, task-like text) — which is exactly what PICL (Section 10) exploits.
>
> **The honest answer**: scale, data, training distribution and metric choice all interact. **Ch. 5, [Lesson 5.6](05_6_scaling_laws.html)** covers scaling laws and the emergence discussion.

> [!compiler] Course connections
> - **Ch. 5, [Lesson 5.5](05_5_decoder_only.html)**: GPT-3, few-shot ICL, the decoder-only line
> - **Ch. 6, [Lesson 6.3](06_3_prompting_cot.html)**: Prompting and chain-of-thought
> - **Ch. 11, [Lesson 11.1](11_1_reasoning.html)**: Reasoning — CoT, self-consistency, ToT, GoT, PAL

:::deep GPT-3 — Language Models are Few-Shot Learners | Brown et al., 2020 · ~5 min
**Paper:** Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal, et al. — *Language Models are Few-Shot Learners* (NeurIPS 2020) — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

#### The problem
Fine-tuning needs a labelled dataset for every task, can latch onto spurious patterns in that data, and is unlike how people learn — from an instruction and a few examples. Can a language model learn a task **from its prompt alone**?

#### The core idea
1. **Scale** the decoder-only recipe of GPT-2 to **175B parameters** (alternating dense and locally banded sparse attention layers).
2. Evaluate **without any gradient updates**, giving the task only in the prompt (see the table below).
3. Train **eight sizes** from 125M to 175B to see how each ability changes with scale.

| Setting | What the prompt contains |
|--|--|
| Zero-shot | a task description only |
| One-shot | description + one demonstration |
| Few-shot | description + as many demonstrations as fit in the 2048-token context (typically 10–100) |

#### What a few-shot prompt looks like
```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
cheese =>
```
The model continues with the answer. The "learning" happens entirely inside the forward pass — this is **in-context learning (ICL)**.

#### Model and data
| | GPT-3 175B |
|--|--|
| Layers | 96 |
| $d_{\text{model}}$ | 12,288 |
| Heads | 96 (dimension 128 each) |
| Context | 2048 tokens |
| Training tokens | 300B |

#### Results that mattered
Strong zero-, one- and few-shot performance across translation, question answering, cloze tasks and on-the-fly reasoning, sometimes **competitive with fine-tuned state-of-the-art** systems. Few-shot performance improves with model size **more steeply** than zero-shot. Human evaluators found it hard to tell GPT-3's generated news articles from human-written ones.

#### Limitations → what came next
- Weak on some tasks (for example certain natural-language-inference and reading-comprehension datasets).
- Doesn't reliably **follow instructions** — it continues patterns → [InstructGPT](#deep-instructgpt).
- ICL is a **by-product** of next-token prediction, never trained directly → [PICL](#deep-picl).
- Enormous training and serving cost → the efficiency line ([LoRA](#deep-lora), [FlashAttention](#deep-flashattention)).

#### Exam angle
1. Define in-context learning. *(Performing a task from instructions/demonstrations in the prompt, with no weight updates.)*
2. Zero- vs one- vs few-shot? *(Number of demonstrations given in the context.)*
3. What scaling trend did GPT-3 report for few-shot learning? *(Larger models make more effective use of in-context examples.)*

**Course lessons:** [5.5 Decoder-only models](05_5_decoder_only.html), [5.6 Scaling laws](05_6_scaling_laws.html)
:::

---

## 7. Position Encoding: The Problem That Wouldn't Go Away (2021)

> [!lecture] The hidden bottleneck
> Self-attention has **no built-in notion of order** — shuffle the input tokens and, without position information, the model cannot tell. The original Transformer used **sinusoidal absolute positional encodings**. But absolute PEs:
> - Don't generalize well beyond the trained length
> - Don't capture *relative* relationships directly
> - Make extrapolating to longer sequences hard

> [!slides] The position encoding evolution
> | Approach | Idea | Limitation |
> |----------|------|------------|
> | Absolute (sinusoidal) | Fixed sinusoids added to embeddings | Weak extrapolation |
> | Learned | Trainable table | Fixed max length |
> | Relative (Shaw et al., Transformer-XL, T5) | Bias by relative distance | Extra complexity / clipping |
> | **RoPE** (Su et al., 2021) | **Rotate Q/K vectors** by position-dependent angle | **Relative position via rotation** |

> [!compiler] Why RoPE won
> RoPE (Rotary Position Embedding) treats pairs of Q and K dimensions as complex numbers and **rotates them by position-dependent angles**. The dot product then depends *only on the relative position* (m−n). No absolute position table is needed, and it is now a standard choice in modern open LLMs. The **Position Survey** (Dufter et al., 2021) is the map of this whole design space.

> [!slides] Course connections
> - **Ch. 3, [Lesson 3.4](03_4_positional_encoding.html)**: Positional encoding — absolute vs. relative trade-offs
> - **Ch. 3, [Lesson 3.10](03_10_rope.html)**: RoPE derivation — rotation matrices, relative position, complex numbers
> - **Ch. 11, [Lesson 11.2](11_2_long_context.html)**: Long context — ALiBi, Position Interpolation, YaRN, Ring Attention

:::deep Position Survey — Position Information in Transformers: An Overview | Dufter, Schmitt, Schütze, 2021 · ~4 min
**Paper:** Dufter, Schmitt, Schütze — *Position Information in Transformers: An Overview* (2021) — [arXiv:2102.11090](https://arxiv.org/abs/2102.11090)

#### The problem
Self-attention is **order-blind**, so every Transformer needs some way to inject position. By 2021 there were **over 30** different position models, described in incompatible terms, with no shared map to compare them on.

#### The core idea
The survey doesn't propose a new method. It gives a **common vocabulary** — a set of axes on which any position model can be placed — and reviews the methods along them.

#### The axes
| Axis | Options |
|--|--|
| **Reference point** | **absolute** (where is this token?) vs **relative** (how far apart are two tokens?) — or both |
| **How position is injected** | **adding position embeddings** to the input (APE) vs **modifying the attention matrix** (MAM) with position-dependent terms |
| **Where it is integrated** | only at the input vs in every attention layer |
| **Parameterisation** | learned vs fixed (e.g. sinusoidal) |

It also groups methods by setting: sequential text, graphs, decoders, cross-lingual models, and analysis studies.

#### Placing familiar methods on the axes
| Method | Reference | Injection | Where |
|--|--|--|--|
| Sinusoidal ([Transformer](#deep-transformer)) | absolute | added to input, fixed | input |
| Learned table ([BERT](#deep-bert), [GPT-1](#deep-gpt-1)) | absolute | added to input, learned | input |
| Relative bias ([T5](#deep-t5)) | relative | attention matrix, learned | every layer |
| [RoPE](#deep-rope) | relative (via rotation) | multiplies Q and K | every layer |

#### What it concludes
It deliberately **doesn't crown a winner**. It reports evidence that relative encodings help on long sequences while absolute encodings can suit classification, and it lists open problems — above all the lack of a controlled, like-for-like empirical comparison.

#### Limitations → what came next
A map, not a benchmark. [RoPE](#deep-rope) became the practical answer most modern LLMs adopted.

#### Exam angle
1. What is the difference between APE and MAM? *(Add position vectors to inputs once vs. change attention scores with position-dependent terms.)*
2. Why might relative encodings generalise better to longer inputs? *(They depend on distances, which recur at every length, not on absolute indices never seen in training.)*
3. Classify RoPE on the survey's axes. *(Relative; injected into Q and K; applied in every layer; fixed, not learned.)*

**Course lesson:** [3.4 Positional encoding](03_4_positional_encoding.html)
:::

:::deep RoPE — RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al., 2021 · ~5 min
**Paper:** Su, Lu, Pan, Murtadha, Wen, Liu — *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

#### The problem
Absolute encodings are **added** to the token embeddings, so position gets tangled with content and doesn't directly express the distance between tokens. Relative schemes add bias terms to the attention scores, which complicates the architecture and doesn't fit efficient linear attention.

#### The core idea
Find a position encoding $f$ so that the query–key dot product depends **only on the content and the relative offset**:

$$\langle f_q(\mathbf{x}_m, m),\; f_k(\mathbf{x}_n, n)\rangle = g(\mathbf{x}_m, \mathbf{x}_n, m - n)$$

The solution: **rotate** the query and key vectors by an angle proportional to their position.

#### How the rotation works
1. Split each $d$-dimensional query/key into $d/2$ **pairs** of dimensions.
2. Rotate pair $i$ at position $m$ by angle $m\theta_i$, with $\theta_i = 10000^{-2(i-1)/d}$ — fast rotation for early pairs, slow for late ones.
3. Then $\mathbf{q}_m = R_m W_q \mathbf{x}_m$ and $\mathbf{k}_n = R_n W_k \mathbf{x}_n$, and since rotations compose, $R_m^\top R_n = R_{n-m}$:

$$\mathbf{q}_m^\top \mathbf{k}_n = \mathbf{x}_m^\top W_q^\top R_{n-m} W_k \mathbf{x}_n$$

Absolute position goes in; only **relative** position comes out of the score.

#### In practice
- Applied to **Q and K in every attention layer** — not to V, and nothing is added to the embeddings.
- Implemented element-wise (multiply by $\cos m\theta$ and $\sin m\theta$ and swap pairs), with no extra parameters.

#### Properties the paper claims
Flexibility in **sequence length**, **decaying** inter-token dependency as relative distance grows, and compatibility with **linear self-attention**.

#### Limitations → what came next
Going far beyond the trained context length still degrades, motivating position interpolation and YaRN (course lesson 11.2). [SWAT](#deep-swat) combines RoPE with ALiBi-style biases.

#### Exam angle
1. Why does the score depend only on $m - n$? *(Rotations compose: $R_m^\top R_n = R_{n-m}$.)*
2. Which vectors are rotated? *(Queries and keys only, in every layer.)*
3. Why different frequencies $\theta_i$ per pair? *(Like sinusoidal encodings: fast pairs resolve nearby offsets, slow pairs encode long distances.)*

**Course lessons:** [3.10 RoPE derivation](03_10_rope.html), [11.2 Long context](11_2_long_context.html)
:::

---

## 8. Efficiency Breakthroughs: LoRA (2021) → FlashAttention (2022)

> [!lecture] The compute wall
> Full fine-tuning of a large model stores weights, gradients, and optimizer states (Adam keeps two moments per parameter). **LoRA** (Hu et al., 2021) changed the economics: freeze the base model, inject **low-rank adapters** ΔW = BA (r ≪ d). For GPT-3 175B the paper reports roughly **10,000× fewer trainable parameters** than full fine-tuning.

> [!slides] LoRA arithmetic
> | | Full fine-tune | LoRA |
> |--|----------------|------|
> | Trainable params | 100% | well under 1% (about 0.01% for GPT-3 175B in the paper); depends on r and which matrices are adapted |
> | Storage per task | Full model copy | Small adapter (MBs) |
> | Inference latency | — | None extra: BA can be merged into W |

> [!compiler] FlashAttention: the IO-aware breakthrough
> Attention is largely **memory-bound**, not compute-bound. FlashAttention (Dao et al., 2022) tiles Q, K, V to fit in **on-chip SRAM**, uses **online softmax** (running max + sum), and recomputes in the backward pass. **Exact same output, far fewer HBM reads/writes, memory linear in sequence length.**

> [!compiler] Course connections
> - **Ch. 7, [Lesson 7.2](07_2_adapters_lora.html)**: LoRA, adapters — init (B=0), merge() with zero added latency
> - **Ch. 7, [Lesson 7.1](07_1_soft_prefix_tuning.html)**: Soft prompts and prefix tuning
> - **Ch. 7, [Lesson 7.3](07_3_quantization.html)**: Quantization (INT8, NF4, GPTQ, AWQ)
> - **Ch. 7, [Lesson 7.4](07_4_qlora_prune_distill.html)**: QLoRA — 4-bit NF4 + double quantization + paged optimizers
> - **Ch. 4, [Lesson 4.4](04_4_flash_attention.html)**: FlashAttention — tiling, online softmax derivation, backward recomputation

:::deep LoRA — Low-Rank Adaptation of Large Language Models | Hu et al., 2021 · ~5 min
**Paper:** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022) — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

#### The problem
Fully fine-tuning GPT-3 175B means storing a **175B-parameter copy per task**. Adapter layers add inference latency; prefix tuning is hard to optimise and uses up part of the context.

#### The hypothesis
The **change** in weights needed to adapt a pre-trained model has a **low intrinsic rank** — you don't need a full $d \times k$ matrix to express it.

#### The core idea
1. **Freeze** the pre-trained weight $W_0$.
2. Learn the update as a product of two thin matrices, $\Delta W = BA$, with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ and rank $r \ll \min(d, k)$:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

3. Initialise $A$ with random Gaussian values and $B$ with **zeros**, so $\Delta W = 0$ at the start — training begins exactly at the pre-trained model.
4. $\alpha / r$ scales the update, so changing $r$ needs less re-tuning of the learning rate.
5. For deployment, **merge**: $W = W_0 + BA$. No extra layers, so **no added inference latency**. Switch tasks by subtracting one $BA$ and adding another.

In most experiments LoRA is applied only to the attention projections $W_q$ and $W_v$.

#### The arithmetic (illustration)
For one $12288 \times 12288$ projection (GPT-3's width): full fine-tuning trains about **151M** values; LoRA with $r = 4$ trains $2 \times 12288 \times 4 \approx$ **98K** — roughly 1,500× fewer for that matrix.

#### Results that mattered
On GPT-3 175B, **10,000× fewer trainable parameters** and **3× less GPU memory** than Adam fine-tuning, with quality **on par with or better than** full fine-tuning on RoBERTa, DeBERTa, GPT-2 and GPT-3.

#### Limitations → what came next
- The frozen base model must still sit in GPU memory at full precision → **QLoRA** quantises it to 4 bits (see §11).
- Batching requests that need *different* adapters in one forward pass is not straightforward if the weights are merged.

#### Exam angle
1. Why initialise $B$ to zero? *(So $\Delta W = 0$ and training starts from the unchanged pre-trained model.)*
2. Why does LoRA add no inference latency? *(BA can be merged into $W_0$.)*
3. What does $r$ control? *(The rank, hence the capacity and size of the update.)*

**Course lesson:** [7.2 Adapters and LoRA](07_2_adapters_lora.html)
:::

:::deep FlashAttention — Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al., 2022 · ~5 min
**Paper:** Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

#### The problem
Standard attention writes the full $N \times N$ score matrix $S$ and probability matrix $P$ to the GPU's large but slow **HBM**, then reads them back. On modern GPUs these memory transfers, not the arithmetic, dominate the runtime. Approximate attention methods cut FLOPs but often **don't speed up wall-clock time**.

#### The core idea: be IO-aware
A GPU has a small, very fast on-chip **SRAM** and a large, slower **HBM**. Arrange the computation so the $N \times N$ matrices **never touch HBM**.

1. **Tiling:** split Q, K, V into blocks that fit in SRAM; compute attention block by block.
2. **Online softmax:** softmax needs the whole row, so keep a running **max** $m$ and running **sum** $\ell$ per row, and rescale the partial output whenever a new block raises the max.
3. **Recompute in the backward pass:** store only the output $O$ and the softmax statistics $(m, \ell)$; rebuild $S$ and $P$ block by block during backprop. More FLOPs, far less memory traffic — and faster overall.
4. **One fused CUDA kernel** for the whole thing.

#### The online-softmax update
For a new block of scores $s$ with values $V_b$, starting from the previous $(m, \ell, O)$:

$$m' = \max(m, \max s), \qquad \ell' = e^{m - m'}\ell + \textstyle\sum e^{s - m'}$$

$$O' = \frac{e^{m - m'}\,\ell\, O + e^{s - m'} V_b}{\ell'}$$

The result is **exactly** the softmax over the full row — no approximation.

#### IO complexity
Standard attention: $\Theta(Nd + N^2)$ HBM accesses. FlashAttention: $\Theta(N^2 d^2 / M)$, where $M$ is the SRAM size — many times fewer for typical head dimensions. Extra memory is **linear** in $N$.

#### Results that mattered
**15%** faster end-to-end BERT-large training than the MLPerf 1.1 record, **3×** on GPT-2 (sequence length 1K), **2.4×** on Long Range Arena. Longer contexts became trainable: the first Transformers to beat chance on Path-X (**61.4%**, 16K tokens) and Path-256 (**63.1%**, 64K). A block-sparse variant is faster still.

#### Limitations → what came next
- Still $O(N^2)$ **FLOPs** — it makes exact attention fast, not cheaper in arithmetic.
- Tied to GPU kernel engineering; later versions re-tuned it for newer hardware.
- Complementary levers shrink other costs: [GQA](#deep-gqa) (KV-cache size), [SWAT](#deep-swat) (window).

#### Exam angle
1. Why is FlashAttention faster despite recomputation? *(Attention is memory-bound; it trades cheap FLOPs for expensive HBM traffic.)*
2. Is it approximate? *(No — the output is exactly standard attention.)*
3. What does online softmax keep per row? *(The running max and the running sum of exponentials.)*

**Course lesson:** [4.4 FlashAttention](04_4_flash_attention.html)
:::

---

## 9. Scale to Alignment: GPT-3 → InstructGPT → DPO (2020–2023)

> [!lecture] The alignment problem
> GPT-3 could do few-shot ICL, but it **didn't follow instructions reliably** — it would continue the pattern, ignore the request, or hallucinate. The model had *capability* but not *intent alignment*.

> [!slides] The alignment pipeline
> ```
> Base Model (GPT-3)
>      ↓ SFT on (instruction, output) pairs
> Instruction-Tuned Model
>      ↓ Reward Model trained on human preferences (Bradley-Terry)
>      ↓ PPO with KL penalty (keep near reference)
> InstructGPT
> ```

> [!compiler] DPO: RLHF without the RL
> DPO (Rafailov et al., 2023) optimizes the **same KL-regularized objective** in closed form. The optimal policy is π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β). Plug it into Bradley-Terry and the intractable partition function **cancels**. DPO trains **offline on preference pairs** with an implicit reward — no separate reward model, no PPO loop, no sampling during training.

> [!slides] Course connections
> - **Ch. 6, [Lesson 6.1](06_1_instruction_tuning.html)**: Instruction tuning (SFT)
> - **Ch. 6, [Lesson 6.4](06_4_reward_model.html)**: Reward models, Bradley-Terry, log-sigmoid loss
> - **Ch. 6, [Lesson 6.5](06_5_rlhf_ppo.html)**: PPO loop — REINFORCE, advantage, PPO-Clip
> - **Ch. 6, [Lesson 6.6](06_6_dpo.html)**: DPO derivation, implicit reward, DPO vs PPO (models in memory, reward model, KL term)
> - **Ch. 7, [Lesson 7.2](07_2_adapters_lora.html)**: LoRA, which makes alignment cheap (LoRA + DPO)

:::deep InstructGPT — Training Language Models to Follow Instructions with Human Feedback | Ouyang et al., 2022 · ~5 min
**Paper:** Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, et al. — *Training language models to follow instructions with human feedback* (NeurIPS 2022) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

#### The problem
Next-token prediction on internet text is not the same objective as *"follow the user's instruction helpfully, truthfully and harmlessly"*. GPT-3 was capable but **misaligned**: it continued patterns, made things up, and could produce toxic text.

#### The core idea: three steps
1. **Supervised fine-tuning (SFT):** labelers write demonstrations of the desired behaviour for prompts (many from the OpenAI API); fine-tune GPT-3 on them.
2. **Reward model (RM):** sample several outputs per prompt; labelers **rank** them; train a model to score outputs so preferred ones score higher.
3. **Reinforcement learning:** optimise the SFT policy against the RM with **PPO**, while a KL penalty keeps it close to the SFT model.

#### The objectives
Reward model, on every preferred/rejected pair $(y_w, y_l)$ from a labeler's ranking:

$$\mathcal{L}_{\text{RM}} = -\,\mathbb{E}\left[\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)\right]$$

RL objective (**PPO-ptx**):

$$\mathbb{E}\left[r_\theta(x,y) - \beta \log \frac{\pi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)}\right] + \gamma\, \mathbb{E}_{x \sim \text{pretrain}}\left[\log \pi^{\text{RL}}(x)\right]$$

- The **KL term** ($\beta$) stops the policy drifting into text that games the reward model.
- The **pre-training term** ($\gamma$) mixes in the original LM objective to limit the "alignment tax" — regressions on standard NLP benchmarks.

Why rankings? Comparing outputs is easier and more consistent for people than assigning absolute scores, and a single ranking yields many training pairs.

#### Results that mattered
- Outputs from the **1.3B** InstructGPT were **preferred over the 175B GPT-3**, despite 100× fewer parameters.
- On TruthfulQA it gave truthful and informative answers **about twice as often** as GPT-3.
- Closed-domain hallucination fell from **41% to 21%**.
- About **25% fewer toxic outputs** than GPT-3 when prompted to be respectful.

#### Limitations → what came next
- Expensive human labelling, and the RM can be **reward-hacked**.
- PPO is finicky and memory-hungry: policy, reference, reward and value models all in play.
- → [DPO](#deep-dpo) reaches the same objective without a reward model or RL loop.

#### Exam angle
1. What are the three stages of RLHF? *(SFT → reward model from rankings → PPO with a KL penalty.)*
2. What is the KL penalty for? *(Keep the policy near the SFT model; prevent reward hacking.)*
3. What does PPO-ptx add? *(Pre-training log-likelihood updates to reduce the alignment tax.)*

**Course lessons:** [6.1 Instruction tuning](06_1_instruction_tuning.html), [6.4 Reward models](06_4_reward_model.html), [6.5 RLHF with PPO](06_5_rlhf_ppo.html)
:::

:::deep DPO — Direct Preference Optimization | Rafailov et al., 2023 · ~5 min
**Paper:** Rafailov, Sharma, Mitchell, Ermon, Manning, Finn — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (NeurIPS 2023) — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

#### The problem
RLHF works but is complex: train a separate reward model, then run an unstable RL loop that **samples from the policy during training** and keeps several large models in memory.

#### The derivation in four steps
1. RLHF maximises reward under a KL constraint:

$$\max_{\pi}\; \mathbb{E}\left[r(x,y)\right] - \beta\, \mathrm{KL}\big(\pi(y\mid x)\,\Vert\,\pi_{\text{ref}}(y\mid x)\big)$$

2. That objective has a closed-form optimum:

$$\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, e^{\,r(x,y)/\beta}$$

3. Rearrange to write the **reward in terms of the policy**:

$$r(x,y) = \beta \log \frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$$

4. Plug into the Bradley–Terry preference model $p(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$. Both responses share the same prompt $x$, so the intractable $\beta \log Z(x)$ **cancels**.

#### The DPO loss
$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

The **implicit reward** is $\hat r(x,y) = \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$. The gradient raises the likelihood of $y_w$ and lowers that of $y_l$, weighted by how wrongly the implicit reward currently orders the pair.

#### PPO-RLHF vs DPO
| | PPO-based RLHF | DPO |
|--|--|--|
| Reward model | trained separately | implicit in the policy |
| Sampling during training | yes | no — fixed preference pairs |
| Models involved | policy, reference, reward, value | policy, reference |
| Training | RL loop | one classification-style loss |

#### Results that mattered
Better control of sentiment than PPO-based RLHF, and matching or better response quality on summarisation and single-turn dialogue — with a much simpler pipeline.

#### Limitations → what came next
- **Offline:** learns only from the fixed preference pairs, with no exploration of new responses.
- Sensitive to $\beta$ and to the quality of the preference data.
- → Cheap alignment recipes (LoRA + DPO) and the wider toolkit in the [IT Survey](#deep-it-survey).

#### Exam angle
1. Why does the partition function $Z(x)$ cancel? *(Bradley–Terry uses a reward difference for two responses to the same prompt.)*
2. What is DPO's implicit reward? *($\beta \log \pi_\theta / \pi_{\text{ref}}$.)*
3. What does $\beta$ control? *(How far the policy may move from the reference model.)*

**Course lesson:** [6.6 DPO](06_6_dpo.html)
:::

---

## 10. Explicit ICL Training: PICL (2023)

> [!lecture] The ICL gap
> GPT-3 showed ICL *emerges* — but it is a by-product of next-word prediction, not something the model is explicitly trained for. What if we **pre-train explicitly for ICL**?

> [!lecture] PICL's insight
> PICL (Gu et al., 2023, *Pre-Training to Learn In Context*) observes that plain-text documents already contain **"intrinsic tasks"** — e.g. a review paragraph implicitly performs sentiment analysis. PICL trains a **retriever** (contrastively) so that paragraphs sharing the same intrinsic task lie close together. For each paragraph, it **retrieves k paragraphs with the same intrinsic task and concatenates them in front as demonstrations**, producing ICL-shaped training instances. The model is then pre-trained on these with the **simple language-modeling objective** (mixed with ordinary LM on plain text to keep generality). It is **not** a meta-learning algorithm: no bi-level optimisation, no special architecture — just *curated* next-token prediction.

> [!compiler] Why this works
> ICL requires the model to *recognise a task from examples and apply it*. If every training instance is "several demonstrations of one task, then another instance," then next-token prediction **forces** the model to infer the task from context. The paper reports that PICL beats several baselines and can outperform models with nearly 4× the parameters on its evaluations.

> [!slides] Course connections
> - **Ch. 5, [Lesson 5.5](05_5_decoder_only.html)**: GPT-3 and few-shot ICL — the behaviour PICL tries to train explicitly
> - **Ch. 6, [Lesson 6.3](06_3_prompting_cot.html)**: Prompting — how ICL is used in practice

:::deep PICL — Pre-Training to Learn in Context | Gu, Dong, Wei, Huang, 2023 · ~4 min
**Paper:** Gu, Dong, Wei, Huang — *Pre-Training to Learn in Context* (ACL 2023) — [arXiv:2305.09137](https://arxiv.org/abs/2305.09137)

#### The problem
[GPT-3](#deep-gpt-3)'s in-context learning is a **side effect** of next-token prediction: ordinary documents rarely look like "several examples of a task, then a new case". Instruction-tuning data does, but it is small and covers limited tasks. Can plain text be rearranged so pre-training teaches ICL directly?

#### The core idea
1. **Intrinsic tasks:** many paragraphs implicitly perform a task — a movie review does sentiment analysis, a Q&A post does question answering.
2. Train a **retriever** that puts paragraphs with the **same intrinsic task** close together.
3. For each paragraph, **retrieve $k$ others with the same intrinsic task** and place them in front of it as demonstrations.
4. Pre-train on these instances with the **plain language-modelling loss**, mixed with ordinary text so general ability is kept.

No meta-learning, no special architecture — only *what the model reads next to what* changes.

#### The retriever
- Encoder: **RoBERTa-base**; similarity = dot product of mean-pooled representations; search with FAISS.
- Trained **contrastively** on 37 labelled tasks with prompt templates:
  - **positive:** another example of the same task
  - **easy negatives:** examples of different tasks
  - **hard negatives:** same prompt *format* but a different task — so the retriever can't match on surface wording

#### Building the corpus
About **80M paragraphs (~30 GB)** from OpenWebText, Wikipedia and BookCorpus. Each paragraph gets its **k = 20** nearest same-task paragraphs, concatenated up to 1,024 tokens; a perplexity-based filter (using GPT-2 Large) removes low-quality instances.

#### Results that mattered
A **770M** GPT-2 Large model further pre-trained with PICL outperforms baselines on seven text-classification datasets and on **105** Super-NaturalInstructions tasks — and beats a **2.7B** model (nearly 4× the parameters) on the latter (ROUGE-L **37.6 vs 37.3**).

#### Limitations → what came next
- Depends on how well the retriever recognises intrinsic tasks.
- Shown on sub-billion to ~1.5B models; gains over larger models are small.
- → Instruction tuning makes task-following explicit with curated data — mapped by the [IT Survey](#deep-it-survey).

#### Exam angle
1. What is an intrinsic task? *(The task a plain-text paragraph implicitly performs, such as sentiment in a review.)*
2. Why hard negatives in retriever training? *(To stop it matching surface format instead of the underlying task.)*
3. Is PICL meta-learning? *(No — ordinary next-token prediction on retrieved, task-grouped text.)*

**Course lessons:** [5.5 GPT-3 and few-shot ICL](05_5_decoder_only.html), [6.3 Prompting](06_3_prompting_cot.html)
:::

---

## 11. The Alignment Ecosystem: DPO, IT Survey, LoRA/QLoRA (2023)

> [!lecture] Mapping the landscape
> By 2023, alignment had a **full toolkit**:
> - **DPO** (Rafailov et al.): offline, no reward model, implicit reward
> - **Instruction Tuning Survey** (Zhang et al.): the complete map — datasets, models, efficient tuning, critiques
> - **LoRA/QLoRA**: cheap adaptation for alignment

> [!compiler] Key Survey insights
> - **Three data sources**: human-crafted (FLAN, SuperNI), distillation from stronger LLMs (Self-Instruct, Evol-Instruct), preference data
> - **LoRA + DPO** = cheap alignment (train LoRA adapters with the DPO loss)
> - **LIMA** (Zhou et al.): about 1,000 high-quality examples go a long way — *alignment mostly teaches style and format, not new knowledge*
> - **QLoRA** (Dettmers et al.): 4-bit NF4 + double quantization + paged optimizers = fine-tune a 65B model on a single 48GB GPU

> [!slides] Course connections
> - **Ch. 6, [Lesson 6.2](06_2_synthetic_instructions.html)**: Self-Instruct, Evol-Instruct, LIMA
> - **Ch. 6, [Lesson 6.3](06_3_prompting_cot.html)**: POSIX sensitivity metric (prompt robustness)
> - **Ch. 7, [Lesson 7.4](07_4_qlora_prune_distill.html)**: QLoRA — NF4, double quantization, paged optimizers; distillation (word-level vs sequence-level)

*DPO's walkthrough sits in §9, next to its derivation: [Deep dive: DPO](#deep-dpo).*

:::deep IT Survey — Instruction Tuning for Large Language Models: A Survey | Zhang et al., 2023 · ~4 min
**Paper:** Zhang, Dong, Li, Zhang, Sun, Wang, Li, Hu, Zhang, Wu, Wang — *Instruction Tuning for Large Language Models: A Survey* (2023, revised through 2025) — [arXiv:2308.10792](https://arxiv.org/abs/2308.10792)

#### What instruction tuning is
The survey defines it as **further training an LLM on (instruction, output) pairs in a supervised way** — it uses "instruction tuning" and SFT interchangeably. The goal: close the gap between next-word prediction and *following human instructions*.

#### How the survey is organised
| Part | Covers |
|--|--|
| Methodology | building instruction datasets; the SFT procedure |
| Datasets | human-crafted, distilled from stronger LLMs, self-improvement, reasoning data |
| Instruction-tuned models | InstructGPT, Flan-T5, BLOOMZ, Alpaca, Vicuna, WizardLM, Guanaco, Tülu, … |
| Multimodal & domain-specific | e.g. LLaVA, InstructBLIP; dialogue, IE, medical, code |
| Efficient tuning | LoRA, QLoRA, HINT, LOMO, delta-tuning |
| Evaluation & critique | what IT really changes; SFT vs RLHF/DPO |

#### Ways to get instruction data
1. **Human-crafted:** convert existing annotated NLP datasets into instructions with templates.
2. **Distillation:** have a stronger LLM generate instructions and answers (e.g. Self-Instruct-style pipelines, Alpaca).
3. **Self-improvement:** the model generates and refines its own training data.

#### Landmark datasets it describes
| Dataset | Scale |
|--|--|
| Natural Instructions | 193K instances from 61 NLP tasks |
| Super-NaturalInstructions | 1,616 tasks, 5M instances |
| P3 | 170 English datasets, 2,052 prompts |
| xP3 | 16 task types in 46 languages |
| Flan 2021 | 62 NLP benchmarks |
| LIMA | 1K carefully curated examples |

#### The critiques
- SFT mainly improves tasks **well covered by its training data**.
- It may learn **surface patterns and style** more than new skills or knowledge — the "superficial alignment" view behind LIMA's small-data result.

#### Why it matters for the journey
It connects the alignment line ([InstructGPT](#deep-instructgpt) → [DPO](#deep-dpo)) with the efficiency line ([LoRA](#deep-lora) → QLoRA): the modern recipe is curated instruction data + SFT, then preference tuning, often on low-rank adapters.

#### Exam angle
1. Define instruction tuning. *(Supervised training on (instruction, output) pairs.)*
2. Name two ways to build instruction data. *(Templating existing datasets; distillation from a stronger LLM.)*
3. What is the main critique of SFT? *(Gains concentrate on well-covered tasks; it may mostly teach style and format.)*

**Course lessons:** [6.1 Instruction tuning](06_1_instruction_tuning.html), [6.2 Synthetic instructions](06_2_synthetic_instructions.html), [7.4 QLoRA](07_4_qlora_prune_distill.html)
:::

---

## 12. Attention Efficiency: GQA, FlashAttention, SWAT → GLM-5 (2022–2026)

> [!lecture] The attention bottleneck
> Standard attention costs **O(n²d)** time and an **O(n²)** score matrix, and decoding needs a **KV cache that grows linearly with n** and with the number of K/V heads. Several levers attack different parts of this:

> [!slides] Efficiency levers
> | Lever | Mechanism | Win |
> |-------|-----------|-----|
> | **KV cache** | Cache K/V of past tokens; compute only the new token's Q | Avoids recomputing the past at every decode step |
> | **GQA** (Ainslie et al., 2023) | Query heads split into G groups; each group **shares one K/V head** (1 < G < h) | Near-MQA speed and small cache, close to MHA quality |
> | **FlashAttention** (Dao et al., 2022) | Tiling + online softmax (exact) | Same FLOPs, far fewer HBM reads/writes |
> | **SWAT** (Fu et al., 2025) | Sliding window attention trained with **sigmoid** (not softmax) + **balanced ALiBi** + **RoPE** | Makes the sliding window **trainable**; linear-cost inference |

> [!compiler] GLM-5: The modern MoE stack
> GLM-5 (Zhipu AI + Tsinghua, arXiv:2602.15763, v2 dated 24 Feb 2026, titled *From Vibe Coding to Agentic Engineering*) reports:
> - **744B-parameter MoE** (40B active parameters)
> - **DSA (DeepSeek Sparse Attention)** — adopted to cut training and inference cost for long context
> - **Asynchronous RL infrastructure** — decouples generation from training to speed up post-training, plus asynchronous agent RL algorithms
> - **Agentic engineering** focus — long-horizon, real-world software engineering tasks

> [!slides] Course connections
> - **Ch. 4, [Lesson 4.1](04_1_kv_cache.html)**: KV cache — matmul counting, cache size arithmetic
> - **Ch. 4, [Lesson 4.2](04_2_mqa_gqa.html)**: MQA/GQA — parameter counting and cache size
> - **Ch. 4, [Lesson 4.3](04_3_sliding_window.html)**: Sliding window attention and SWAT
> - **Ch. 4, [Lesson 4.4](04_4_flash_attention.html)**: FlashAttention — online softmax, backward recomputation

*FlashAttention's walkthrough sits in §8: [Deep dive: FlashAttention](#deep-flashattention).*

:::deep GQA — Grouped-Query Attention | Ainslie et al., 2023 · ~4 min
**Paper:** Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (EMNLP 2023) — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

#### The problem
Autoregressive decoding is **memory-bandwidth bound**: every new token must load the whole KV cache. Multi-query attention (MQA, one shared K/V head) shrinks the cache but can **lose quality** and train less stably — and training a separate MQA model from scratch is expensive.

#### The core idea
1. **Grouped-query attention:** split the $h$ query heads into $G$ groups; each group shares **one** key head and **one** value head.
2. It interpolates between the two extremes: **GQA-1 = MQA**, **GQA-$h$ = standard multi-head attention (MHA)**.
3. **Uptraining** instead of training from scratch: convert an existing MHA checkpoint by **mean-pooling** the K and V projection heads within each group, then continue pre-training for just **5%** of the original compute.

#### The KV-cache arithmetic
Per token per layer, the cache holds $2 \times (\text{K/V heads}) \times d_{\text{head}}$ values:

| | K/V heads | Cache vs MHA |
|--|--|--|
| MHA | $h$ | 1× |
| GQA-$G$ | $G$ | $G/h$ |
| MQA | 1 | $1/h$ |

Example: $h = 64$ query heads with $G = 8$ groups → an **8× smaller** KV cache.

#### Results that mattered
An uptrained **T5-XXL** with GQA-8 reaches quality **close to MHA** at inference speed **comparable to MQA**, for 5% of the original pre-training compute.

#### Limitations → what came next
- The cache still grows **linearly with sequence length** → a sliding window caps it ([SWAT](#deep-swat)).
- It saves memory traffic, not attention FLOPs → pair with [FlashAttention](#deep-flashattention).

#### Exam angle
1. MHA vs MQA vs GQA? *($h$, 1 or $G$ shared K/V heads.)*
2. How is an MHA checkpoint converted? *(Mean-pool K/V heads within each group, then uptrain on ~5% of the original compute.)*
3. By how much does GQA shrink the KV cache? *(By a factor of $h/G$.)*

**Course lessons:** [4.1 KV cache](04_1_kv_cache.html), [4.2 MQA/GQA](04_2_mqa_gqa.html)
:::

:::deep SWAT — Sliding Window Attention Training | Fu et al., 2025 · ~4 min
**Paper:** Fu, Song, Wang, Wu, Zheng, Zhang, Xu, Wei, Xu, Zhao — *Sliding Window Attention Training for Efficient Large Language Models* (2025) — [arXiv:2502.18845](https://arxiv.org/abs/2502.18845)

#### The problem
**Sliding-window attention** (each token attends only to the previous $\omega$ tokens) gives linear-cost inference and a fixed-size cache, but models trained this way with softmax tend to underperform. The paper attributes much of the inefficiency to the **attention-sink** phenomenon caused by the **high variance of softmax**: forced to sum to 1, the weights pile onto a few tokens and the rest of the window is under-used.

#### The core idea
1. **Sigmoid instead of softmax.** Each score is squashed independently, with no competition to sum to 1:

$$\mathrm{Attn}(Q,K,V) = \sigma\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

so information across the whole window can be kept and compressed.

2. **Balanced ALiBi.** Dropping softmax also drops some implicit positional signal, so add a distance bias $s \cdot (m - n)$ inside the window, with **half the heads** using negative slopes and **half** positive ones: $s_k = \mp 2^{-k}$ for $k = 1, \ldots, h/2$. Different heads then emphasise different ranges of the window.

3. **RoPE** on queries and keys as well ([RoPE](#deep-rope)).

4. **Train with the window**, so the model learns to carry older context forward through its hidden states; stacking layers widens the effective reach beyond $\omega$.

#### Setup
Models of **340M** and **760M** parameters, trained on FineWeb-Edu with 4,096-token sequences, compared with Transformer++, RetNet, GLA, Mamba, DeltaNet, TTT, Gated DeltaNet and Titans.

#### Results that mattered
At both sizes the best SWAT configuration has the **highest average** on the paper's benchmark suite, narrowly ahead of the strongest baseline (Titans) — at 760M about **51.9 vs 51.6**, versus **48.7** for Transformer++. Gains are real but small.

#### Limitations → what came next
- **Not exact full attention**: anything outside the window must survive in compressed form.
- Evidence so far is at sub-billion scale with modest margins.
- → Production models also pursue sparse attention, e.g. DSA in [GLM-5](#deep-glm-5).

#### Exam angle
1. Why replace softmax with sigmoid? *(Softmax's sum-to-1 competition causes attention sinks; sigmoid scores are independent.)*
2. What does "balanced" ALiBi mean? *(Half the heads get negative slopes, half positive.)*
3. What is the inference cost of sliding-window attention? *(Linear in sequence length, with a fixed-size cache.)*

**Course lesson:** [4.3 Sliding window and SWAT](04_3_sliding_window.html)
:::

:::deep GLM-5 — From Vibe Coding to Agentic Engineering | GLM-5 Team (Zhipu AI & Tsinghua), 2026 · ~4 min
**Paper:** GLM-5 Team, Zhipu AI & Tsinghua University — *GLM-5: from Vibe Coding to Agentic Engineering* (February 2026) — [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)

#### The goal
Move from **"vibe coding"** — generating snippets on request — to **agentic engineering**: long-horizon, multi-step software tasks where the model plans, uses tools and works through real codebases. That needs long context at affordable cost, and post-training on long agent episodes.

#### The architecture: sparse in two directions
| | GLM-5 |
|--|--|
| Total parameters | **744B** |
| Active per token | **40B** |
| Experts | 256 |
| Layers | 80 |
| Context | extended from 4K to about **200K** during mid-training |
| Pre-training data | roughly 27–28.5T tokens (the paper reports both figures) |

- **Sparse experts (MoE):** only a small fraction of the weights runs per token — the line from the [1991 gate](#deep-adaptive-moe) and [2017 top-k routing](#deep-sparsely-gated-moe).
- **Sparse attention (DSA, DeepSeek Sparse Attention):** instead of attending to every previous token ($O(L^2)$), each query attends to a **dynamically selected** fine-grained subset — essential at 200K context.

#### How DSA was added
Not trained from scratch: starting from a dense-attention base, a short **dense warm-up** (1,000 steps) is followed by **sparse adaptation** on about 20B tokens, so the model learns to work with the sparse selection while keeping long-context quality.

#### Post-training: asynchronous RL
Agent episodes vary wildly in length; a synchronous RL loop would leave GPUs idle while waiting for the slowest rollout. GLM-5's RL infrastructure (**slime**) **decouples generation from training** to keep GPUs busy, and the paper introduces asynchronous agent-RL algorithms on top — the descendant of [InstructGPT](#deep-instructgpt)'s PPO stage.

#### Results reported
About **20% better on average** than its predecessor GLM-4.7 across the paper's agentic, coding and reasoning benchmarks (including SWE-bench Verified, Terminal-Bench 2.0, BrowseComp and Humanity's Last Exam), **comparable to Claude Opus 4.5 and GPT-5.2 (xhigh)** and ahead of Gemini 3 Pro on that average.

#### Limitations
A technical report: the comparisons are the authors' own, and training at this scale is out of reach for most labs.

#### Exam angle
1. What do "744B total, 40B active" mean? *(All experts hold 744B parameters; routing runs only about 40B for each token.)*
2. Why sparse attention for agentic work? *(Long contexts make dense $O(L^2)$ attention too costly.)*
3. Why asynchronous RL? *(Rollouts of different lengths would leave synchronous training idle; decoupling keeps GPUs busy.)*
:::

---

## 13. The Complete Progression Graph

> [!slides] One-page mental map
> ```
> 1991: Adaptive MoE (concept)
>      ↓
> 2017: Transformer + sparsely-gated MoE (architecture + sparse compute)
>      ↓
> 2018: ELMo → BERT + GPT-1 (contextual fork: encoder vs decoder)
>      ↓
> 2019: T5 + BART (unification)
>      ↓
> 2020: GPT-3 (scale → ICL; "emergence" debated)
>      ↓
> 2021: RoPE + Position Survey (position encoding) + LoRA (efficiency)
>      ↓
> 2022: InstructGPT (RLHF) + FlashAttention (IO-aware)
>      ↓
> 2023: PICL (explicit ICL) + DPO (offline RLHF) + GQA + IT Survey
>      ↓
> 2025: SWAT (sliding window, trained properly)
>      ↓
> 2026: GLM-5 (MoE + DSA + async RL, agentic engineering)
> ```

---

## 14. How to Use This Map for Exams

:::exam The exam cheat codes
| If asked... | Answer with this paper + one sentence |
|-------------|----------------------------------------|
| "How did we get contextual embeddings?" | **ELMo** — biLSTM layer-weighted sum, not static |
| "What's the difference between BERT and GPT?" | BERT = bidirectional encoder (MLM); GPT = decoder-only (next-token) |
| "What is MoE and why does it scale?" | **MoE 2017** — sparse gating activates only top-k experts per example |
| "Where did conditional computation start?" | **Adaptive MoE 1991** — Jacobs/Jordan/Nowlan/Hinton, experts + gate |
| "How does RoPE encode position?" | **RoFormer** — rotates Q/K vectors; dot product depends on relative distance |
| "What is ICL and where did it come from?" | **GPT-3** — 175B params, few-shot prompting; **PICL** trains it explicitly via intrinsic tasks |
| "How do you make a chat model from base LLM?" | **InstructGPT** — SFT on (instruction, output) → Reward Model → PPO |
| "What are the 3 ways to get instruction data?" | **IT Survey** — human-crafted, distilled from stronger LLMs, preference data |
| "How to fine-tune 65B on one GPU?" | **QLoRA** (IT Survey) — 4-bit NF4 + LoRA adapters |
| "Does IT teach understanding or format?" | **LIMA / IT Survey** — largely style/format, limited new knowledge |
| "What does GQA change?" | **GQA** — query-head groups share one K/V head; smaller KV cache |
| "How to make sliding-window attention trainable?" | **SWAT** — sigmoid instead of softmax + balanced ALiBi + RoPE |
| "What is GLM-5?" | 744B MoE (40B active), DSA sparse attention, async RL, agentic engineering |
| "How does DPO avoid RL?" | Implicit reward = β log(π_θ/π_ref); partition function cancels in BT |

:::

---

## 15. The Papers in Your Vault — Quick Reference

> [!slides] All 18 papers, in progression order
> | # | Paper | Year | Link | Core Idea |
> |---|-------|------|------|-----------|
> | 1 | Adaptive MoE | 1991 | [Adaptive MoE (Jacobs et al., 1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) | Experts + gate = conditional computation |
> | 2 | MoE 2017 | 2017 | [Sparsely-Gated MoE (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538) | Sparsely-gated top-k MoE, >1000× capacity |
> | 3 | ELMo | 2018 | [ELMo (Peters et al., 2018)](https://arxiv.org/abs/1802.05365) | Contextual vectors via biLSTM |
> | 4 | GPT-1 | 2018 | [GPT-1 (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Decoder-only, next-token LM |
> | 5 | BERT | 2018 | [BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) | Bidirectional encoder, MLM |
> | 6 | T5 | 2019 | [T5 (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683) | Text-to-text, span corruption |
> | 7 | BART | 2019 | [BART (Lewis et al., 2019)](https://arxiv.org/abs/1910.13461) | Denoising seq2seq |
> | 8 | GPT-3 | 2020 | [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) | 175B, few-shot ICL |
> | 9 | Position Survey | 2021 | [Position Survey (Dufter et al., 2021)](https://arxiv.org/abs/2102.11090) | Position encoding survey |
> | 10 | RoPE | 2021 | [RoFormer/RoPE (Su et al., 2021)](https://arxiv.org/abs/2104.09864) | Rotary Position Embedding |
> | 11 | LoRA | 2021 | [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) | Low-rank adaptation ΔW=BA |
> | 12 | InstructGPT | 2022 | [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) | RLHF = SFT → RM → PPO |
> | 13 | PICL | 2023 | [PICL (Gu et al., 2023)](https://arxiv.org/abs/2305.09137) | Intrinsic tasks + LM objective |
> | 14 | DPO | 2023 | [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) | Offline preference optimization |
> | 15 | GQA | 2023 | [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) | Grouped-query attention |
> | 16 | IT Survey | 2023 | [IT Survey (Zhang et al., 2023)](https://arxiv.org/abs/2308.10792) | Complete IT landscape |
> | 17 | SWAT | 2025 | [SWAT (Fu et al., 2025)](https://arxiv.org/abs/2502.18845) | Sigmoid + balanced ALiBi + RoPE |
> | 18 | GLM-5 | 2026 | [GLM-5 (Zhipu AI, 2026)](https://arxiv.org/abs/2602.15763) | 744B MoE, DSA, async RL |

> [!compiler] Background papers (not in vault but essential)
> - **Transformer** (Vaswani et al., 2017) — root architecture — [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
> - **FlashAttention** (Dao et al., 2022) — IO-aware exact attention — [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

---

## 16. Suggested Reading Order for Your Exam

:::tip Optimal path through the papers
1. **ELMo → BERT → GPT-1** (representation & pre-training paradigms)
2. **T5 → BART** (unified objectives, encoder-decoder)
3. **Adaptive MoE 1991 → MoE 2017** (efficiency lineage)
4. **GPT-3 → PICL** (ICL: shown at scale → trained explicitly)
5. **Position Survey → RoPE** (how Transformers know position)
6. **InstructGPT → DPO → IT Survey** (alignment & IT ecosystem)
7. **LoRA → GQA → SWAT** (efficiency stack; QLoRA and FlashAttention come via the IT Survey and Ch. 4)
8. **GLM-5** (modern synthesis: MoE + sparse attention + async RL)

:::

---

## 17. Quick Navigation: Link to Course Lessons

:::note Jump to any topic in the course
| Topic | Chapter & Lesson |
|-------|------------------|
| **Transformer architecture** | Ch. 3, Lessons [3.1](03_1_why_remove_recurrence.html)–[3.10](03_10_rope.html) |
| **Positional encoding** | Ch. 3, [Lesson 3.4](03_4_positional_encoding.html) |
| **RoPE derivation** | Ch. 3, [Lesson 3.10](03_10_rope.html) |
| **KV cache, MQA, GQA** | Ch. 4, Lessons [4.1](04_1_kv_cache.html)–[4.2](04_2_mqa_gqa.html) |
| **SWAT, sliding window** | Ch. 4, [Lesson 4.3](04_3_sliding_window.html) |
| **FlashAttention, online softmax** | Ch. 4, [Lesson 4.4](04_4_flash_attention.html) |
| **ELMo, BERT MLM, [CLS], QA** | Ch. 5, Lessons [5.1](05_1_elmo.html)–[5.3](05_3_bert_cls_qa.html) |
| **BART, T5** | Ch. 5, [Lesson 5.4](05_4_bart_t5.html) |
| **GPT training, decoder-only, ICL** | Ch. 5, [Lesson 5.5](05_5_decoder_only.html) |
| **Scaling laws, emergence** | Ch. 5, [Lesson 5.6](05_6_scaling_laws.html) |
| **Instruction tuning, FLAN, Self-Instruct, LIMA** | Ch. 6, Lessons [6.1](06_1_instruction_tuning.html)–[6.2](06_2_synthetic_instructions.html) |
| **Prompting, chain-of-thought, POSIX sensitivity** | Ch. 6, [Lesson 6.3](06_3_prompting_cot.html) |
| **RLHF, PPO, DPO** | Ch. 6, Lessons [6.4](06_4_reward_model.html)–[6.6](06_6_dpo.html) |
| **Soft prompts, prefix tuning** | Ch. 7, [Lesson 7.1](07_1_soft_prefix_tuning.html) |
| **Adapters, LoRA** | Ch. 7, [Lesson 7.2](07_2_adapters_lora.html) |
| **Quantization** | Ch. 7, [Lesson 7.3](07_3_quantization.html) |
| **QLoRA, pruning, distillation** | Ch. 7, [Lesson 7.4](07_4_qlora_prune_distill.html) |
| **RAG, tools, agents, knowledge graphs** | Ch. 8, Lessons [8.1](08_1_rag.html)–[8.3](08_3_knowledge_graphs.html) |
| **Multilingual, ViT, CLIP, VLM/VLA** | Ch. 9, Lessons [9.1](09_1_multilingual.html)–[9.3](09_3_vlm_vla.html) |
| **Bias, toxicity, hallucination** | Ch. 10, Lessons [10.1](10_1_bias.html)–[10.2](10_2_toxicity_hallucination.html) |
| **Reasoning, long context, model editing (ROME, MEMIT), self-evolution, interpretability** | Ch. 11, Lessons [11.1](11_1_reasoning.html)–[11.5](11_5_interpretability.html) |

:::

---

## 18. Final Thought: The Story Isn't Over

> [!lecture] Closing
> The papers in your vault trace a single thread: **how to make language models more capable, more efficient, and more aligned**. The thread runs from *conditional computation* (1991) through *attention* (2017), *contextual representations* (2018), *unified objectives* (2019), *scale and ICL* (2020), *position encoding* (2021), *alignment* (2022–2023), *explicit ICL* (2023), *attention efficiency* (2023–2025), to *agentic MoE* (2026).
>
> But the story doesn't end at GLM-5. The next papers will ask: **How do we make models reason reliably? How do we make them safe at superhuman capability? How do we make them collaborate with humans and each other?**
>
> Your exam tests whether you understand *how we got here*. Your career will be about *where we go next*.

---

:::key
**Key Takeaways**
1. Every paper solves a specific problem left by its predecessors — read the "problem statement" first
2. The field progresses by **alternating capability leaps** (Transformer, GPT-3) and **efficiency/alignment fixes** (LoRA, RLHF, FlashAttention, RoPE)
3. "Emergence" is debated — scale, data, distribution and metric choice interact
4. Alignment is not one technique (RLHF) but an ecosystem (SFT, RM, PPO, DPO, LoRA, QLoRA, constitutional AI)
5. The modern stack (GLM-5) = MoE + sparse attention + async RL + agentic engineering
:::

---

## 19. Quick Check

:::q Q1 · Which paper introduced the gating network + experts idea that modern MoE builds on?
:::ans
**Adaptive Mixtures of Local Experts (Jacobs et al., 1991)** — the conceptual ancestor of sparse MoE. Shazeer et al. (2017) made it practical with a sparsely-gated layer.
:::
:::

:::q Q2 · What is the exact mechanism of RoPE?
:::ans
RoPE treats pairs of Q and K dimensions as complex numbers and rotates them by position-dependent angles θᵢ = 10000^(-2i/d). The dot product then depends only on relative position (m−n). No absolute position table is needed.
:::
:::

:::q Q3 · How does PICL train for in-context learning, and is it meta-learning?
:::ans
PICL is **not** meta-learning. A contrastively-trained retriever finds paragraphs that share the same **intrinsic task**; these are concatenated as demonstrations before a target paragraph, and the model is pre-trained on them with the **ordinary language-modeling objective** (next-token prediction).
:::
:::

:::q Q4 · Name the levers that shrink the cost of attention at inference, and what each saves.
:::ans
**KV cache** avoids recomputing past K/V; **GQA** shares K/V heads across query groups (smaller cache); **FlashAttention** is exact but IO-aware (fewer HBM reads/writes, same FLOPs); **SWAT** restricts attention to a trained sliding window (linear-cost inference, but not exact full attention).
:::
:::

:::q Q5 · What are the headline ingredients of GLM-5?
:::ans
GLM-5 (arXiv 2602.15763, Feb 2026) reports a **744B-parameter MoE (40B active)**, **DSA (DeepSeek Sparse Attention)** for long-context efficiency, an **asynchronous RL** infrastructure for post-training, and a focus on **agentic engineering**.
:::
:::
