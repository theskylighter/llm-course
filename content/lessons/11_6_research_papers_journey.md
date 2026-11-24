**Research Papers Journey: From Conditional Computation to Agentic MoE** — the intellectual arc that connects every paper in your vault into one coherent story.

<<SVG:research_papers_timeline>>

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
