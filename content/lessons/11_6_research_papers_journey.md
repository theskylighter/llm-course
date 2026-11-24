**Research Papers Journey: From Conditional Computation to Agentic MoE** — the intellectual arc that connects every paper in your vault into one coherent story.

---

## 1. The Problem: Why Read Papers at All?

> [!lecture] Opening
> You have implemented attention, you have fine-tuned LoRA adapters, you have built RAG pipelines. But when a new paper drops — *Mixture of Experts*, *Rotary Position Embeddings*, *Direct Preference Optimization* — do you understand **why it exists**, **what problem it solves**, and **how it connects to what you already know**? This lesson is the map that turns a pile of PDFs into a single, navigable landscape.

> [!compiler] How this lesson fits the course
> Every chapter so far has taught you a *technique*: attention (Ch. 3), pretraining (Ch. 5), instruction tuning (Ch. 6), PEFT (Ch. 7), RAG (Ch. 8), multimodality (Ch. 9). This lesson stitches them together by showing the **papers that introduced each idea** and **why each paper had to exist**. Think of it as the "director's commentary" for the entire course.

---

## 2. The Seed: Conditional Computation (1991)

> [!lecture] The idea that started it all
> Before Transformers, before GPUs, before "large" meant billions of parameters — **Jacobs, Jordan, Nowlan, and Hinton (1991)** asked: *What if a neural network could decide which subset of its parameters to use for each input?*
>
> They proposed **Adaptive Mixtures of Local Experts**: a set of specialist networks ("experts") and a **gating network** that chooses which expert handles each input. Only the chosen expert's weights are updated. This is **conditional computation** — the radical idea that *not every parameter needs to participate in every forward pass*.

> [!slides] Why this matters for everything that follows
> | 1991 Idea | 2017 Realization | 2024 Production |
> |-----------|------------------|-----------------|
> | Experts + gating network | Top-k sparse gating (Shazeer et al.) | GLM-5: 744B-param MoE |
> | Conditional computation | Practical on GPUs (Shazeer et al.) | DSA + async RL |
> | "Not every weight for every input" | Only top-k experts active per token | DSA + async RL |

> [!compiler] The thread connecting 1991 → 2017 → 2026
> The 1991 paper was **conceptual** — no GPUs, no Transformers. The 2017 paper (Shazeer et al., *Outrageously Large Neural Networks*) made it practical: **top-k sparse gating** over thousands of experts, **>1000× model capacity with only minor loss in computational efficiency**, up to 137B params. GLM-5 (2026, Zhipu AI + Tsinghua) puts this in a modern stack: **744B parameters**, **DeepSeek Sparse Attention (DSA)**, **asynchronous RL**, agentic engineering. The 1991 idea finally runs at planetary scale.

---

## 3. The Architecture: Attention Is All You Need (2017)

> [!lecture] The paper that changed everything
> *Vaswani et al. (2017)* did not just propose a new architecture — they **unified** the encoder (understanding) and decoder (generation) into a single mechanism: **self-attention**. Every paper in your vault either **uses** this architecture, **modifies** it, or **optimizes** it.

> [!slides] The Transformer family tree
> ```
> Transformer (2017)
>    ├── Encoder stack → BERT (2018) → RoBERTa, DeBERTa...
>    ├── Decoder stack → GPT-1 → GPT-2 → GPT-3 → LLaMA...
>    ├── Encoder–Decoder → T5 (2019) → BART, mT5...
>    └── Modifications → RoPE, ALiBi, FlashAttention, GQA, SWAT, MoE layers
> ```

> [!compiler] Not in vault but essential
> The Transformer paper is **not in your vault PDFs** — but every other paper in your vault either cites it or builds on it. We treat it as the **root node** of the entire progression graph.

---

## 4. Contextual Representations: ELMo → BERT → GPT-1 (2018)

> [!lecture] The problem with static embeddings
> Word2Vec and GloVe give **one vector per word** — "bank" has the same vector in "river bank" and "bank account." ELMo (Peters et al., 2018) proved **contextual embeddings** work: run a **biLSTM** over the sentence, take a **weighted sum of layer hidden states**, and the representation changes with context.

> [!slides] Why ELMo wasn't the final answer
> | Breakthrough | Limitation |
> |--------------|------------|
> | Contextual vectors | biLSTM = sequential (slow, not parallelizable) |
> | Layer-weighted sum | No standard way to use in downstream tasks |
> | Proved context matters | Architecture not scalable |

> [!compiler] The fork in the road
> ELMo's limitation — **biLSTMs don't parallelize** — forced the field to ask: *How do we get contextual representations with parallel computation?*
>
> Two answers emerged **simultaneously in 2018**:
>
> 1. **BERT** (Devlin et al.): **Transformer encoder** + **Masked Language Modeling (MLM)** → bidirectional, parallelizable, great for *understanding* tasks
> 2. **GPT-1** (Radford et al.): **Transformer decoder** + **next-token prediction** → autoregressive, parallelizable at train time, great for *generation*

> [!slides] The 2018 fork: Understanding vs. Generation
> | | BERT (Encoder) | GPT-1 (Decoder) |
> |--|----------------|-----------------|
> | **Attention** | Bidirectional (full context) | Causal (left-to-right) |
> | **Objective** | MLM (predict masked tokens) | Next-token LM |
> | **Strength** | Understanding, classification | Generation, completion |
> | **Can't do** | Generate free text | See future context |

> [!compiler] Course connection
> - **Ch. 3, Lesson 3.7–3.8**: BERT's MLM objective, [CLS] token, fine-tuning heads
> - **Ch. 3, Lesson 3.9**: GPT training — teacher forcing, causal mask, next-token prediction

---

## 5. Unifying Objectives: T5 and BART (2019)

> [!lecture] The problem with specialization
> By 2019, the field had **BERT for understanding** and **GPT for generation**. But real NLP tasks need *both*: translation (understand source, generate target), summarization (understand long text, generate short), QA (understand question + context, generate answer).

> [!slides] Two paths to unification
> | Model | Architecture | Objective | Key Insight |
> |-------|--------------|-----------|-------------|
> | **T5** (Raffel et al., 2019) | Encoder–Decoder | **Span corruption** (mask spans, predict them) | **Every NLP task = text in → text out** |
> | **BART** (Lewis et al., 2019) | Encoder (BERT) + Decoder (GPT) | **Denoising autoencoder** (corrupt → reconstruct) | BERT encoder + GPT decoder = best of both |

> [!compiler] Why both exist
> T5 and BART are **peers**, not competitors. T5's span corruption is simpler and scales; BART's denoising (token masking, deletion, permutation, rotation) is richer. The field uses both. **T5's "text-to-text" framing** is the mental model that enables instruction tuning (Ch. 6) and prompt engineering (Ch. 6).

> [!slides] Course connections
> - **Ch. 5, Lesson 5.4**: BART vs. T5 — span corruption vs. denoising, sentinel tokens
> - **Ch. 5, Lesson 5.5**: Decoder-only line (GPT, LLaMA) — why decoders won for generation

---

## 6. Emergence at Scale: GPT-3 (2020)

> [!lecture] The moment everything changed
> GPT-3 (Brown et al., 2020) scaled the decoder-only line to **175B parameters**. The surprise wasn't better perplexity — it was **in-context learning (ICL)**: give the model a few examples in the prompt, and it performs the task *without any weight updates*.

> [!slides] What GPT-3 actually showed
> | Capability | Before GPT-3 | With GPT-3 |
> |------------|--------------|------------|
> | New task | Fine-tune (gradient steps) | Few-shot prompt (0 gradient steps) |
> | New language | Fine-tune on parallel data | Translate via prompt |
> | Reasoning | Poor | Chain-of-thought helps |

> [!compiler] The emergence debate
> **Does ICL "emerge" at scale?** The literature is divided:
> - *Scale hypothesis*: ICL appears abruptly at ~100B params (Brown et al.)
> - *Data quality hypothesis*: Emergence is a metric artifact (Schaeffer et al., 2023) — non-linear metrics (exact match) create "jumps"; smooth metrics (edit distance, Brier score) show gradual improvement
> - *Data distribution hypothesis*: ICL appears when training data contains diverse task-like patterns
>
> **The honest answer**: We don't fully know. Scale, data quality, training distribution, and metric choice all interact. **Ch. 5, Lesson 5.6** covers the Chinchilla scaling laws and the "emergence is a metric artifact" paper.

> [!compiler] Course connections
> - **Ch. 5, Lesson 5.5**: GPT-3, few-shot ICL, the decoder-only line
> - **Ch. 6, Lesson 6.3**: Chain-of-thought prompting — the reasoning unlock
> - **Ch. 11, Lesson 11.1**: Reasoning — CoT, self-consistency, ToT, GoT, PAL

---

## 7. Position Encoding: The Problem That Wouldn't Go Away (2021)

> [!lecture] The hidden bottleneck
> Transformers are **permutation-equivariant** — shuffle the input tokens, the output shuffles identically. The model has **no notion of order** unless we inject it. The original Transformer used **sinusoidal absolute positional encodings**. But absolute PEs:
> - Don't generalize beyond trained length
> - Don't capture *relative* relationships naturally
> - Break when you extrapolate to longer sequences

> [!slides] The position encoding evolution
> | Approach | Idea | Limitation |
> |----------|------|------------|
> | Absolute (sinusoidal) | Fixed sinusoids | No extrapolation |
> | Learned | Trainable table | Fixed max length |
> | Relative (Shaw et al., Transformer-XL, T5) | Bias by relative distance | Clipping, complexity |
> | **RoPE** (Su et al., 2021) | **Rotate Q/K vectors** by position-dependent angle | **Relative position via rotation**; extrapolates |

> [!compiler] Why RoPE won
> RoPE (Rotary Position Embedding) treats Q and K vectors as complex numbers and **rotates them by position-dependent angles**. The dot product then depends *only on relative position* (m−n). This is **relative position via rotation** — no absolute positions needed, extrapolates to longer context, and is now standard in LLaMA, GPT-Neo, PaLM, Qwen, etc.

> [!slides] Course connections
> - **Ch. 3, Lesson 3.10**: RoPE derivation — rotation matrices, relative position, complex numbers
> - **Ch. 11, Lesson 11.2**: Long context — ALiBi, Position Interpolation, YaRN, Ring Attention
> - **Ch. 3, Lesson 3.10**: Position Survey (Dufter et al., 2021) — absolute vs. relative trade-offs

---

## 8. Efficiency Breakthroughs: LoRA (2021) → FlashAttention (2022)

> [!lecture] The compute wall
> By 2021, fine-tuning GPT-3 (175B) required **>1000 GPUs** — full fine-tuning stores weights, gradients, and optimizer states (Adam: 2× fp32 moments). **LoRA** (Hu et al., 2021) changed the economics: freeze the base model, inject **low-rank adapters** ΔW = BA (r ≪ d). **~10,000× fewer trainable params** vs full fine-tuning.

> [!slides] LoRA arithmetic
> | Model | Full Fine-tune | LoRA (r=4) |
> |-------|----------------|------------|
> | GPT-3 175B | 175B params | ~0.01% |
> | 7B | 7B | ~0.1% |
> | Storage/task | Full copy | ~MB |

> [!compiler] FlashAttention: the IO-aware breakthrough
> Attention is **memory-bound**, not compute-bound. FlashAttention (Dao et al., 2022) tiles Q, K, V to fit in **on-chip SRAM**, uses **online softmax** (incremental max + sum), and recomputes in backward pass. **Exact same output, fewer HBM reads/writes, linear memory.**

> [!compiler] Course connections
> - **Ch. 7, Lesson 7.2**: LoRA, adapters, prefix tuning, prompt tuning
> - **Ch. 7, Lesson 7.4**: QLoRA — 4-bit NF4 + double quantization + paged optimizers
> - **Ch. 7, Lesson 7.3**: Quantization (INT8, NF4, GPTQ, AWQ)
> - **Ch. 7, Lesson 7.3**: FlashAttention — online softmax, tiling, backward recomputation
> - **Ch. 7, Lesson 7.2**: LoRA init (B=0), merge(), mergeable = zero latency

---

## 9. Scale to Alignment: GPT-3 → InstructGPT → DPO (2020–2023)

> [!lecture] The alignment problem
> GPT-3 could do few-shot ICL, but it **didn't follow instructions reliably** — it would continue the pattern, refuse, or hallucinate. The model had *capability* but not *intent alignment*.

> [!slides] The alignment pipeline
> ```
> Base Model (GPT-3)
>      ↓ SFT on (instruction, output) pairs
> Instruction-Tuned Model
>      ↓ Reward Model trained on human preferences (Bradley-Terry)
>      ↓ PPO with KL penalty (keep near reference)
> InstructGPT / ChatGPT
> ```

> [!compiler] DPO: RLHF without the RL
> DPO (Rafailov et al., 2023) optimizes the **same KL-regularized objective** in closed form. The optimal policy is π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β). Plug into Bradley-Terry, the intractable partition function **cancels**. DPO trains **offline on preference pairs** with an implicit reward — no reward model, no PPO loop, no sampling during training.

> [!slides] Course connections
> - **Ch. 6, Lesson 6.4**: Reward models, Bradley-Terry, log-sigmoid loss
> - **Ch. 6, Lesson 6.5**: PPO loop — REINFORCE, advantage, PPO-Clip
> - **Ch. 6, Lesson 6.6**: DPO derivation, implicit reward, offline training
> - **Ch. 6, Lesson 6.6**: DPO vs PPO — models in memory, reward model, KL term
> - **Ch. 7, Lesson 7.1**: PEFT for alignment (LoRA + DPO)

---

## 10. Explicit ICL Training: PICL (2023)

> [!lecture] The ICL gap
> GPT-3 showed ICL *emerges* — but it's **accidental**, not trained. What if we **pre-train explicitly for ICL**?

> [!lecture] PICL's insight
> PICL (Gu et al., 2023) mines **"intrinsic tasks"** from plain text: contiguous spans where the next tokens are predictable from context (e.g., "The capital of France is \_\_\_"). Each span becomes an **intrinsic task** — the prefix is the prompt, the suffix is the target. Pre-train on these with the **ordinary LM objective** (next-token prediction). No meta-learning, no special architecture — just **curated next-token prediction**.

> [!compiler] Why this works
> ICL requires the model to *recognize a task from examples and apply it*. By training on **millions of intrinsic tasks** (question-answer, translation, summarization patterns naturally occurring in text), the model learns the **skill of in-context learning itself** as a byproduct of next-token prediction.

> [!slides] Course connections
> - **Ch. 6, Lesson 6.2**: Self-Instruct — synthetic instruction data generation
> - **Ch. 6, Lesson 6.2**: LIMA — "alignment teaches style, not knowledge"
> - **Ch. 6, Lesson 6.6**: POSIX sensitivity metric

---

## 10. The Alignment Ecosystem: DPO, IT Survey, LoRA/QLoRA (2023)

> [!lecture] Mapping the landscape
> By 2023, alignment had a **full toolkit**:
> - **DPO** (Rafailov et al.): offline, no reward model, implicit reward
> - **Instruction Tuning Survey** (Zhang et al.): the complete map — datasets, models, LoRA/QLoRA, LIMA critique
> - **LoRA/QLoRA**: cheap adaptation for alignment

> [!compiler] Key Survey insights
> - **Three data sources**: human-crafted (FLAN, SuperNI), distillation (Self-Instruct, Evol-Instruct), preference data
> - **LoRA + DPO** = cheap alignment (train LoRA adapters with DPO)
> - **LIMA** (Zhou et al.): 1,000 high-quality examples ≈ full IT dataset — *alignment teaches style, not knowledge*
> - **QLoRA** (Dettmers et al.): 4-bit NF4 + double quantization + paged optimizers = 65B on 1 GPU

> [!slides] Course connections
> - **Ch. 6, Lesson 6.2**: Self-Instruct, Evol-Instruct, LIMA
> - **Ch. 6, Lesson 6.3**: POSIX sensitivity metric (prompt robustness)
> - **Ch. 7, Lesson 7.4**: QLoRA — NF4, double quantization, paged optimizers
> - **Ch. 7, Lesson 7.4**: Distillation — word-level KD vs sequence-level (beam search)

---

## 11. Attention Efficiency: GQA, SWAT, FlashAttention → GLM-5 (2022–2026)

> [!lecture] The attention bottleneck
> Standard attention is **O(n²d)** time, **O(n²)** memory (KV cache). By 2022, three orthogonal solutions emerged:

> [!slides] Three efficiency levers
> | Lever | Paper | Mechanism | Win |
> |-------|-------|-----------|-----|
| **KV Cache** | — | Cache K/V, compute only new Q | O(n²) → O(n) decode |
> **GQA** (Ainslie et al., 2023) | Grouped-Query Attention | Share K/V heads across query groups | h KV heads for h query heads → near-MQA speed, MHA quality |
| **FlashAttention** (Dao et al., 2022) | IO-aware exact attention | Tiling + online softmax | Same FLOPs, far fewer HBM reads/writes |
| **SWAT** (Fu et al., 2025) | Sliding Window Attention Training | Sigmoid + balanced ALiBi + RoPE | Trainable sliding window → linear inference |

> [!compiler] GLM-5: The modern MoE stack
> GLM-5 (Feb 2026, Zhipu AI + Tsinghua, arXiv:2602.15763) puts it all together: **744B-parameter MoE**, **DeepSeek Sparse Attention (DSA)** for long context, **asynchronous RL** infrastructure, **agentic engineering**. It combines:
> - MoE (sparse compute)
> - DSA (sparse attention for long context)
> - Async RL (post-training)
> - Agentic infrastructure (tool use, planning)

> [!slides] Course connections
> - **Ch. 4, Lesson 4.1**: KV Cache — matmul counting, cache size arithmetic
> - **Ch. 4, Lesson 4.2**: MQA/GQA — parameter counting (MHA 288, MQA 168, GQA 216/192)
> - **Ch. 4, Lesson 4.4**: FlashAttention — online softmax derivation, backward recomputation
> - **Ch. 4, Lesson 4.3**: SWAT — sigmoid + ALiBi + RoPE, linear inference
> - **Ch. 7, Lesson 7.3**: FlashAttention internals — tiling, online softmax, backward recomputation

---

## 12. The Complete Progression Graph

> [!slides] One-page mental map
> ```
> 1991: Adaptive MoE (concept)
>      ↓
> 2017: Transformer + MoE 2017 (architecture + sparse compute)
>      ↓
> 2018: ELMo → BERT + GPT-1 (contextual fork: encoder vs decoder)
>      ↓
> 2019: T5 + BART (unification: encoder-decoder)
>      ↓
> 2020: GPT-3 (scale → ICL emergence, debated)
>      ↓
> 2021: RoPE + Position Survey (position solved) + LoRA (efficiency)
>      ↓
> 2022: InstructGPT (RLHF) + FlashAttention (IO-aware) + PICL (explicit ICL)
>      ↓
> 2023: DPO (offline RLHF) + GQA + IT Survey + LoRA/QLoRA (ecosystem)
>      ↓
> 2025: SWAT (sliding window trained right)
>      ↓
> 2026: GLM-5 (MoE + DSA + async RL = agentic production model)
> ```

---

## 13. How to Use This Map for Exams

> [!exam] The exam cheat codes
> | If asked... | Answer with this paper + one sentence |
> |-------------|----------------------------------------|
> | "How did we get contextual embeddings?" | **ELMo** — biLSTM layer-weighted sum, not static |
> | "What's the difference between BERT and GPT?" | BERT = bidirectional encoder (MLM); GPT = decoder-only (next-token) |
> | "What is MoE and why does it scale?" | **MoE 2017** — sparse gating activates only top-k experts per token |
> | "Where did conditional computation start?" | **Adaptive MoE 1991** — Jacobs/Jordan/Nowlan/Hinton, experts + gate |
> | "How does RoPE encode position?" | **RoFormer** — rotates Q/K vectors; dot product depends on relative distance |
> | "What is ICL and where did it come from?" | **GPT-3** — 175B params, few-shot emerges; **PICL** trains it explicitly |
> | "How do you make a chat model from base LLM?" | **InstructGPT** — SFT on (instruction, output) → Reward Model → PPO |
> | "What are the 3 ways to get instruction data?" | **IT Survey** — human-crafted, distillation (GPT-4), self-improvement |
> | "How to fine-tune 65B on one GPU?" | **QLoRA** (IT Survey) — 4-bit NF4 + LoRA adapters |
> | "Does IT teach understanding or format?" | **LIMA / IT Survey** — mostly format/style copying, limited reasoning |
> | "How to make sliding-window attention trainable?" | **SWAT** — sigmoid instead of softmax + balanced ALiBi + RoPE |
> | "What is GLM-5?" | 744B MoE, DSA sparse attention, async RL, agentic engineering |
> | "What is QLoRA?" | 4-bit NF4 + double quantization + paged optimizers + LoRA |
> | "How does DPO avoid RL?" | Implicit reward = β log(π_θ/π_ref); partition function cancels in BT |

---

## 14. The Papers in Your Vault — Quick Reference

> [!slides] All 18 PDFs, in progression order
> | # | Paper | Year | Vault Filename | Core Idea |
> |---|-------|------|--------------|-----------|
> | 1 | Adaptive MoE | 1991 | `AdaptiveMoE_Jacobs1991.pdf` | Experts + gate = conditional computation |
> | 2 | MoE 2017 | 2017 | `MoE_1701.06538.pdf` | Sparse top-k gating, >1000× capacity |
> | 3 | ELMo | 2018 | `ELMo_1802.05365.pdf` | Contextual vectors via biLSTM |
> | 4 | BERT | 2018 | `BERT_1810.04805.pdf` | Bidirectional encoder, MLM |
> | 4 | GPT-1 | 2018 | `GPT-1_1806.01552.pdf` | Decoder-only, next-token LM |
> | 5 | T5 | 2019 | `T5_1910.10683.pdf` | Text-to-text, span corruption |
> | 5 | BART | 2019 | `BART_1910.13461.pdf` | Denoising seq2seq, BERT enc + GPT dec |
> | 6 | GPT-3 | 2020 | `GPT3_ICL_2005.14165.pdf` | 175B, few-shot ICL |
> | 7 | Position Survey | 2021 | `Position_Information_Transformers_2102.11090.pdf` | Position encoding survey |
> | 7 | RoPE | 2021 | `RoFormer_2104.09864.pdf` | Rotary Position Embedding |
> | 7 | LoRA | 2021 | `LoRA_2106.09685.pdf` | Low-rank adaptation ΔW=BA |
> | 8 | InstructGPT | 2022 | `InstructGPT_2203.02155.pdf` | RLHF = SFT → RM → PPO |
> | 8 | PICL | 2023 | `PICL_2305.09137.pdf` | Intrinsic tasks + LM objective |
> | 8 | IT Survey | 2023 | `IT_Survey_2308.10792.pdf` | Complete IT landscape |
> | 8 | DPO | 2023 | `DPO_2305.18290.pdf` | Offline preference optimization |
> | 8 | GQA | 2023 | `GQA_2305.13245.pdf` | Grouped-query attention |
> | 9 | Sliding Window Attention | 2025 | `SlidingWindowAttention_2502.18845.pdf` | Sigmoid + ALiBi + RoPE |
> | 10 | GLM-5 | 2026 | `GLM-5_2602.15763.pdf` | 744B MoE, DSA, async RL |

> [!compiler] Background papers (not in vault but essential)
> - **Transformer** (Vaswani et al., 2017) — root architecture
> - **FlashAttention** (Dao et al., 2022) — IO-aware exact attention

---

## 15. Suggested Reading Order for Your Exam

> [!tip] Optimal path through the papers
> 1. **ELMo → BERT → GPT-1** (representation & pre-training paradigms)
> 2. **T5 → BART** (unified objectives, encoder-decoder)
> 3. **Adaptive MoE 1991 → MoE 2017** (efficiency lineage)
> 4. **GPT-3 → PICL** (ICL: shown at scale → trained explicitly)
> 5. **Position Survey → RoPE** (how Transformers know position)
> 6. **InstructGPT → DPO → IT Survey** (alignment & IT ecosystem)
> 7. **LoRA → QLoRA → FlashAttention → GQA → SWAT** (efficiency stack)
> 8. **GLM-5** (modern synthesis: MoE + sparse attention + async RL)

---

## 16. Quick Navigation: Link to Course Lessons

> [!note] Jump to any topic in the course
> | Topic | Chapter & Lesson |
> |-------|------------------|
> | **Transformer architecture** | Ch. 3, Lessons 3.1–3.10 |
> | **RoPE derivation** | Ch. 3, Lesson 3.10 |
> | **BERT MLM, [CLS], QA** | Ch. 5, Lessons 5.2–5.3 |
> | **GPT training, teacher forcing** | Ch. 5, Lesson 5.5 |
> | **Scaling laws, Chinchilla** | Ch. 5, Lesson 5.6 |
> | **Instruction tuning, FLAN, Self-Instruct** | Ch. 6, Lessons 6.1–6.2 |
> | **Chain-of-thought, ToT, GoT** | Ch. 6, Lesson 6.3 / Ch. 11, Lesson 11.1 |
> | **RLHF, PPO, DPO** | Ch. 6, Lessons 6.4–6.6 |
> | **POSIX sensitivity** | Ch. 6, Lesson 6.3 |
> | **LoRA, adapters, prefix tuning** | Ch. 7, Lessons 7.1–7.2 |
> | **QLoRA, NF4, double quantization** | Ch. 7, Lesson 7.4 |
> | **KV cache, MQA, GQA** | Ch. 4, Lessons 4.1–4.2 |
> | **FlashAttention, online softmax** | Ch. 4, Lesson 4.4 / Ch. 7, Lesson 7.3 |
> | **SWAT, sliding window** | Ch. 4, Lesson 4.3 |
> | **RAG, tools, agents** | Ch. 8, Lessons 8.1–8.3 |
> | **Multilingual, ViT, CLIP** | Ch. 9, Lessons 9.1–9.2 |
> | **Bias, hallucination, constitutional AI** | Ch. 10, Lessons 10.1–10.2 |
> | **Reasoning, long context, editing** | Ch. 11, Lessons 11.1–11.5 |
> | **Model editing (ROME, MEMIT)** | Ch. 11, Lesson 11.3 |
> | **Interpretability, residual stream** | Ch. 11, Lesson 11.5 |

---

## 17. Final Thought: The Story Isn't Over

> [!lecture] Closing

---

## 17. Final Thought: The Story Isn't Over

> [!lecture] Closing
> The papers in your vault trace a single thread: **how to make language models more capable, more efficient, and more aligned**. The thread runs from *conditional computation* (1991) through *attention* (2017), *contextual representations* (2018), *unified objectives* (2019), *emergence at scale* (2020), *position encoding* (2021), *alignment* (2022), *explicit ICL* (2023), *attention efficiency* (2024–2025), to *agentic MoE* (2026).
>
> But the story doesn't end at GLM-5. The next papers will ask: **How do we make models reason reliably? How do we make them safe at superhuman capability? How do we make them collaborate with humans and each other?**
>
> Your exam tests whether you understand *how we got here*. Your career will be about *where we go next*.

---

:::key
**Key Takeaways**
1. Every paper solves a specific problem left by its predecessors — read the "problem statement" first
2. The field progresses by **alternating capability leaps** (Transformer, GPT-3) and **efficiency/alignment fixes** (LoRA, RLHF, FlashAttention, RoPE)
3. "Emergence" is not magic — it's scale + data + distribution + metric choice interacting
4. Alignment is not one technique (RLHF) but an ecosystem (SFT, RM, PPO, DPO, LoRA, QLoRA, constitutional AI)
5. The modern stack (GLM-5) = MoE + sparse attention + async RL + agentic infra
:::

---

## 18. Quick Check

:::q Q1 · Which paper introduced the gating network + experts idea that modern MoE builds on?
:::ans
**Adaptive Mixtures of Local Experts (Jacobs et al., 1991)** — the conceptual ancestor of all sparse MoE.
:::
:::

:::q Q2 · What is the exact mechanism of RoPE?
:::ans
RoPE treats Q and K vectors as complex numbers and rotates them by position-dependent angles θᵢ = 10000^(-2i/d). The dot product then depends only on relative position (m−n). No absolute positions; extrapolates.
:::
:::

:::q Q3 · What is the key difference between PICL and "meta-learning"?
:::ans
PICL does **not** use meta-learning. It mines **intrinsic tasks** (predictable next-token spans) from plain text and pre-trains on them with the **ordinary LM objective** (next-token prediction). The skill of ICL emerges as a byproduct.
:::
:::

:::q Q4 · What is the efficiency stack for modern LLM decoding?
:::ans
KV Cache → GQA (share K/V heads) → FlashAttention (IO-aware kernels) → SWAT (sliding window) → Ring Attention (shard across devices). Together: O(n) decode, linear memory, exact attention.
:::
:::

:::q Q5 · What does GLM-5 combine that no previous paper did?
:::ans
GLM-5 (Feb 2026) unifies: **744B-param MoE** + **DeepSeek Sparse Attention (DSA)** for long context + **asynchronous RL** infrastructure + **agentic engineering** (tool use, planning). First production MoE with sparse attention + async RL + agentic infra.
:::
:::
