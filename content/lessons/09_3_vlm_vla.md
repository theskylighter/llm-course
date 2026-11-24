**Vision–Language Models (VLMs) and Vision–Language–Action Models (VLAs)** form the frontier where language understanding meets visual perception and robotic control. This lesson answers the foundational questions: what they are, how they differ from plain LLMs, how they evolved, and where they are headed.

---

## 1. What Is a Vision–Language Model (VLM)?

> [!lecture] The core idea
> A **Vision–Language Model** is a single neural network that ingests **both images and text** and produces **text** (or structured outputs). It learns a **joint representation space** where visual concepts and linguistic tokens align. Unlike a pure LLM (text-in, text-out) or a pure vision model (image-in, label-out), a VLM can *ground* language in visual evidence and *describe* visual scenes in language.

**Inputs:**
- One or more images (or video frames)
- A text prompt / question / instruction

**Outputs:**
- Free-form text (caption, answer, reasoning)
- Structured output (bounding boxes, segmentation masks, JSON)
- In some architectures: discrete action tokens (VLA)

**Why it matters:**
Traditional pipelines chain a vision encoder -> discrete labels -> LLM. This loses information (the LLM never sees the raw pixels). A VLM fuses modalities *early*, preserving fine-grained visual detail for reasoning.

---

## 2. How VLMs Differ from Other LLMs

| Dimension | Plain LLM (e.g., GPT-3, LLaMA) | Vision–Language Model (e.g., LLaVA, GPT-4V, Qwen-VL) |
|-----------|--------------------------------|--------------------------------------------------------|
| **Input modality** | Text tokens only | Image patches + text tokens (interleaved) |
| **Tokenizer** | Text-only (BPE / SentencePiece) | Image patches -> linear projection -> *visual tokens* + text tokens |
| **Architecture** | Decoder-only Transformer | Vision encoder + projection + LLM backbone (frozen or fine-tuned) |
| **Training data** | Text corpora (web, code, books) | Image-text pairs (captions, VQA, OCR, detection) + interleaved documents |
| **Capabilities** | Text generation, reasoning, coding | Visual QA, captioning, OCR, grounding, visual reasoning, multimodal dialogue |
| **Failure modes** | Hallucination, no visual grounding | Hallucination *plus* visual misinterpretation, object hallucination |

**Key architectural difference:**
A VLM adds a **vision encoder** (ViT, CLIP-ViT, SigLIP) + a **projection layer** (MLP or Q-Former) that maps image embeddings into the LLM's embedding space. The LLM then treats visual tokens exactly like text tokens.

---

## 3. Input / Output of a VLM

```
Input:  [IMAGE]  +  "What is the breed of the dog in this photo?"
Output: "That is a Golden Retriever. Its golden coat, floppy ears, and friendly expression are characteristic of the breed."
```

**Concrete tensor view:**

```
# Image preprocessing
image = load("photo.jpg")                    # (H, W, 3)
patches = patchify(image, patch_size=14)     # (N_patches, 14, 14, 3)
tokens  = vit_encoder(patches)               # (N_patches, D)  -- visual tokens
tokens  = projection(tokens)                 # (N_patches, D_llm)

# Text preprocessing
prompt  = "Describe this image:"
text_toks = tokenizer(prompt)                # (L_text,)

# Concatenate (interleaving varies by model)
input_ids = concat([image_token, tokens, text_toks])  # (N_patches + L_text, D_llm)

# Forward through LLM
logits = llm(input_ids)                      # (seq_len, vocab_size)
```

**Output formats:**
- **Free text** (most common)
- **Structured** -- bounding boxes (`[x1,y1,x2,y2]`), segmentation masks, JSON schemas
- **Discrete actions** (VLA) -- robot joint positions, gripper open/close, navigation waypoints

---

## 4. Commonly Used VLMs (2023-2025 Landscape)

| Model | Vision Encoder | LLM Backbone | Fusion Strategy | Key Trait |
|-------|----------------|--------------|-----------------|-----------|
| **CLIP** (2021) | ViT-B/32, RN50 | -- (dual encoder) | Contrastive (image-text) | Zero-shot classification, retrieval |
| **BLIP / BLIP-2** | ViT-G/14 | OPT / Flan-T5 | Q-Former (learned queries) | Strong captioning, VQA |
| **Flamingo** | NFNet-F6 | Chinchilla | Perceiver Resampler + gated cross-attn | Few-shot, interleaved |
| **LLaVA / LLaVA-1.5** | CLIP-ViT-L/336px | Vicuna / LLaMA-2 | Linear projection (MLP) | Open-source, strong reasoning |
| **LLaVA-Next / LLaVA-OneVision** | SigLIP / CLIP | LLaMA-3 / Qwen2 | MLP + pooling | Video, high-res, multi-image |
| **Qwen-VL / Qwen2-VL** | Custom ViT | Qwen / Qwen2 | Learned projection + RoPE-2D | Strong OCR, multilingual |
| **InternVL / InternVL-2** | InternViT-6B | InternLM2 | Pixel-shuffle + MLP | Dynamic resolution, strong benchmarks |
| **CogVLM / CogVLM2** | ViT-L | LLaMA-2 / LLaMA-3 | Visual expert module | Strong visual grounding |
| **Idefics / Idefics2** | SigLIP | Mistral / LLaMA | Perceiver + cross-attn | Open, strong few-shot |
| **Phi-3-Vision** | Custom | Phi-3-mini | Learned projection | Tiny (4.2B), on-device |
| **GPT-4V / GPT-4o** | Proprietary | GPT-4 | Proprietary | Best-in-class reasoning |
| **Gemini 1.5 / 2.0** | Proprietary | Gemini | Native multimodal | Long context (1M+ tokens), video |

**Open-source leaders (2025):** LLaVA-OneVision, Qwen2-VL, InternVL-2, CogVLM2, Idefics2, Phi-3-Vision.

---

## 5. Types of VLMs (Taxonomy)

| Type | Description | Examples |
|------|-------------|----------|
| **Dual-encoder (Contrastive)** | Separate image & text encoders; aligned via contrastive loss. No generative decoder. | CLIP, ALIGN, SigLIP, ECLIPSE |
| **Encoder-Decoder (Generative)** | Vision encoder -> projection -> text decoder (T5, BART, LLaMA). Generates text. | BLIP-2, LLaVA, InstructBLIP, Qwen-VL |
| **Decoder-only (Unified)** | Single Transformer decoder consumes interleaved image+text tokens. | Flamingo, LLaVA (LLaMA backbone), GPT-4V, Gemini, Qwen-VL |
| **Frozen LLM + Adapter** | Vision encoder + small trainable adapter (Q-Former, Perceiver, MLP) + frozen LLM. | BLIP-2, LLaVA, LLaVA-1.5, InstructBLIP |
| **Native Multimodal (Early Fusion)** | Single Transformer trained from scratch on interleaved image+text. | GPT-4o, Gemini, Chameleon, Emu3 |
| **Specialized** | OCR-heavy, video, 3D, medical, UI understanding | Qwen-VL (OCR), Video-LLaVA, LLaVA-NeXT-Video, LLaVA-3D, Ferret |

---

## 6. Timeline of VLM Development

```
2018  ViT (Vision Transformer)           - pure vision, no language
2019  ViLBERT, LXMERT                    - early encoder-decoder VLMs
2020  UNITER, Oscar, VinVL               - large-scale pretraining
2021  CLIP (OpenAI)                      - dual-encoder, 400M pairs, zero-shot
2021  DALL-E                             - text-to-image (not VLM but related)
2022  Flamingo (DeepMind)                - few-shot, frozen LLM + Perceiver
2022  BLIP-2 (Salesforce)                - Q-Former, frozen LLM, SOTA VQA
2023  LLaVA (Liu et al.)                 - open-source, linear projection, Vicuna
2023  InstructBLIP                       - instruction-tuned BLIP-2
2023  GPT-4V / Gemini (announced)        - proprietary, native multimodal
2023  Qwen-VL, CogVLM, InternVL          - open-source, strong OCR/reasoning
2024  LLaVA-1.5 / LLaVA-Next             - higher res, multi-image, video
2024  LLaVA-OneVision, LLaVA-NeXT-Video - unified image/video, any resolution
2024  Idefics2, Phi-3-Vision, CogVLM2   - stronger open models, smaller
2024  GPT-4o, Gemini 1.5/2.0             - native, long context, real-time
2024  Molmo, Pixtral, Aria               - fully open weights + data
2025  VLA emergence (see below)          - vision + language -> robot actions
```

**Key shifts:**
1. **Dual-encoder -> Generative** (CLIP -> BLIP/Flamingo)
2. **Frozen LLM + tiny adapter** (BLIP-2, LLaVA) -> democratized training
3. **Native multimodal** (GPT-4o, Gemini) -> early fusion from scratch
4. **VLA** -- adding *action* head -> robotics

---

## 7. Vision-Language-Action Models (VLAs)

> [!compiler] The newest frontier
> A **VLA** extends a VLM with an **action head** that outputs *robot control commands* (joint positions, end-effector pose, gripper state, navigation velocity). It treats actions as *tokens* in the same autoregressive sequence as text.

**Core idea:**
`(image, instruction) -> VLM backbone -> (text reasoning, action tokens)`
The same model that *describes* the scene also *acts* in it.

**Training data:**
- Robot demonstrations (teleoperated, scripted, human videos)
- Web-scale image-text (for visual semantics)
- Simulation (Isaac Gym, MuJoCo, ManiSkill) for scale
- Cross-embodiment datasets (Open X-Embodiment, BridgeData, DROID)

**Architecture variants:**

| Model | Vision | Language | Action Head | Embodiment |
|-------|--------|----------|-------------|------------|
| **RT-1 / RT-2** (Google) | ViT / CLIP | PaLM-E / PaLM-2 | Discretized 256-bin joints | Mobile manipulator |
| **Octo** | SigLIP | LLaMA-2 | Diffusion / autoregressive | Multi-embodiment |
| **OpenVLA** | SigLIP | LLaMA-2 / Phi-3 | Autoregressive 7B | Franka, WidowX |
| **pi-zero** | SigLIP | LLaMA-3 | Diffusion + autoregressive | Bimanual, mobile |
| **GR-1 / GR-2** | ViT | BERT / LLaMA | Generative (GPT-style) | Generalist |
| **OctoPi** | SigLIP | Phi-3 | Diffusion | Low-cost arms |

**Input / Output of a VLA:**

```
Input:  [IMAGE: robot wrist cam] + "Pick up the red apple and place it in the bowl."
Output: [TEXT: "I see a red apple on the table. Moving gripper to grasp..."]
        [ACTION: [joint1, joint2, ..., joint7, gripper_open=0.0]]  -- 50 Hz
```

**Why VLAs matter:**
Traditional robotics: perception -> planning -> control (modular, brittle).
VLA: end-to-end learned policy, *grounded in language*, generalizes across tasks/objects.

---

## 8. How VLAs Differ from VLMs

| Aspect | VLM | VLA |
|--------|-----|-----|
| **Output** | Text (description, answer) | Text + **Action tokens** |
| **Training** | Image-text pairs, VQA, captioning | Robot demos + web image-text |
| **Action space** | None | Discretized joints / diffusion / waypoints |
| **Embodiment** | None (disembodied) | Specific robot or multi-embodiment |
| **Control loop** | Open-loop (single forward pass) | Closed-loop (50-100 Hz, recurrent) |
| **Safety** | Hallucination = wrong answer | Hallucination = robot crash / injury |
| **Eval metric** | VQA accuracy, CIDEr, BLEU | Success rate, task completion, safety |

---

## 9. Common VLM/VLA Architectures Compared

```
+-------------------------------------------------------------+
|                    VLM ARCHITECTURE FAMILIES                 |
+------------------+------------------+------------------+---------------+
|  Dual-Encoder    |  Encoder-Decoder |  Decoder-only    |  Native Multi |
|  (CLIP, SigLIP)  |  (BLIP-2, LLaVA) |  (Flamingo,      |  (GPT-4o,     |
|                  |                  |  LLaVA, GPT-4V)  |  Gemini)      |
+------------------+------------------+------------------+---------------+
| Image Encoder    | Image Encoder    | Image Encoder    | Single Tower  |
| Text Encoder     | Projection       | Projection       | (ViT + LLM    |
| Contrastive Loss | LLM Decoder      | LLM Decoder      |  interleaved) |
+------------------+------------------+------------------+---------------+
| Retrieval,       | Captioning, VQA, | Interleaved      | All of above  |
| Zero-shot cls    | Reasoning        | dialogue, tools  | + native video|
+------------------+------------------+------------------+---------------+

+-------------------------------------------------------------+
|                    VLA = VLM + ACTION HEAD                    |
+-------------------------------------------------------------+
|  VLM Backbone (frozen or fine-tuned)  ->  Action Tokenizer  ->  Robot    |
|  (Vision + Language)                  (Discretize / Diffuse)  (50-100Hz)|
+-------------------------------------------------------------+
```

---

## 10. Key Challenges & Open Questions

| Challenge | VLM | VLA |
|-----------|-----|-----|
| **Hallucination** | Object hallucination, wrong attributes | Action hallucination -> unsafe motion |
| **Long video** | Token budget, temporal reasoning | Real-time 50Hz, history compression |
| **Data quality** | Noisy web pairs, caption noise | Demonstration noise, embodiment gap |
| **Embodiment gap** | N/A | Sim-to-real, morphology transfer |
| **Safety** | Toxic text, bias | Physical harm, runaway actions |
| **Eval** | Benchmarks saturated (VQAv2, OK-VQA) | Real-world success rate, sim-to-real gap |

---

## 11. Quick Check

:::q Q1 - What is the fundamental architectural difference between a plain LLM and a VLM?
:::ans
A VLM adds a **vision encoder** (e.g., ViT) + a **projection layer** that maps image embeddings into the LLM's token embedding space, allowing the model to process interleaved image and text tokens.
:::
:::

:::q Q2 - What does a VLA output that a VLM does not?
:::ans
**Action tokens** - discretized joint positions, gripper state, or diffusion parameters - emitted at control frequency (50-100 Hz) alongside text.
:::
:::

:::q Q3 - Which VLM architecture uses a *dual encoder* with contrastive loss and has *no generative decoder*?
:::ans
CLIP / SigLIP / ALIGN - dual-encoder (image + text) trained with InfoNCE contrastive loss.
:::
:::

:::q Q4 - What is the "Q-Former" in BLIP-2?
:::ans
A lightweight **Querying Transformer** that extracts a fixed small set of visual query embeddings from the vision encoder, which are then fed to the frozen LLM. It bridges the modality gap with few trainable parameters.
:::
:::

:::q Q5 - Why is "early fusion" (native multimodal) considered stronger than "late fusion" (dual-encoder)?
:::ans
Early fusion lets the model attend *cross-modally at every layer*, enabling fine-grained reasoning (e.g., "the red car left of the tree"). Dual-encoders only align global embeddings, losing fine-grained alignment.
:::
:::

---

## 12. Going Deeper

- **Papers to read:**
  - *CLIP* (Radford et al., 2021) - contrastive dual-encoder
  - *Flamingo* (Alayrac et al., 2022) - few-shot, frozen LLM + Perceiver
  - *BLIP-2* (Li et al., 2023) - Q-Former
  - *LLaVA* (Liu et al., 2023) - linear projection + Vicuna
  - *LLaVA-OneVision* (2024) - unified image/video, any resolution
  - *RT-2* (Brohan et al., 2023) - VLA, web-scale pretraining + robot data
  - *OpenVLA* (2024) - open 7B VLA, LLaMA-2 + SigLIP
  - *pi-zero* (2024) - diffusion + autoregressive policy, bimanual
  - *Molmo* (2024) - fully open weights + data, strong pointing

- **Benchmarks:** VQAv2, GQA, OK-VQA, TextVQA, POPE (hallucination), MM-Vet, MMBench, SEED-Bench, LLaVA-Bench, RealWorldQA, Calvin, BridgeData, Open X-Embodiment.

- **Open-source repos:** LLaVA, LLaVA-OneVision, Qwen2-VL, InternVL, Idefics2, OpenVLA, Octo, pi-zero, Molmo.