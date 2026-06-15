# su26-ai301-contribution

# Contribution 1: [New Model]: HiDream-O1-Image

**Contribution Number:** 1  
**Student:** SOKOUDJOU LEOPOLD  
**Issue:** https://github.com/vllm-project/vllm-omni/issues/3733  
**Status:** Phase II Complete

---

## Why I Chose This Issue

I chose this issue because I am proficient in python and related frameworks. Moreover, this project requires working with an image generative model which is a domain I am currently interested in. Upon completion, I hope to understand the architecture of this model.

---

## Understanding the Issue

### Problem Description

Issue #3733 asks vLLM-Omni to add support for HiDream-ai/HiDream-O1-Image, a text-to-image model. Right now, vLLM-Omni simply can't load or run this model — there's no entry for it in the model registry and no code that knows its architecture.

### Expected Behavior

A user should be able to run Omni(model="HiDream-ai/HiDream-O1-Image") (offline) or vllm serve HiDream-ai/HiDream-O1-Image --omni (online) and get text-to-image generation working.

### Current Behavior

A user can not have access to the HiDream-O1-Image model because it is not supported.

### Affected Components

#### 1. Existing Repo Architecture & Infrastructure (Where to add the code)
   
##### New Directory to Create
vllm_omni/diffusion/models/hidream_o1_image/

__init__.py: For exposing the pipeline and registration classes.

hidream_o1_image_transformer.py: Holds the core ported neural network architecture.

pipeline_hidream_o1_image.py: Implements the scheduling and pixel-space denoising loop.

fm_solvers_unipc.py: Holds the ported custom scheduler.

model_index.json: Configuration mapping (_class_name: "HiDreamO1ImagePipeline").

##### Existing Core Registration Points
vllm_omni/diffusion/registry.py: You must append your new pipeline class here to make it discoverable by the framework engine.

#### 2. In-Repo References & Templates (What to copy/adapt)
Rather than building from scratch, you will be copying design patterns and code structures from these existing components:

vllm_omni/diffusion/models/sensenova_u1/

Involvement: This is your primary architectural template. It provides the implementation pattern for an LLM backbone serving as a diffusion denoiser.

What to extract: Look here to see how to wrap the text decoder blocks, configure tensor-parallel layers (QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear), and structure the pipeline.

vllm_omni/diffusion/attention/layer.py

Involvement: This contains the non-paged Attention layer class used across all diffusion configurations in vllm-omni. You will use this to enforce the custom causal/bidirectional sequence layout instead of stock vLLM paged attention.

.claude/skills/add-diffusion-model/references/custom-model-patterns.md

Involvement: Provides "BAGEL Pattern 2" (Custom safetensors at root). This dictates how to structure the load_weights() script to ingest a sharded weight index directly from the repository root (subfolder=None) and remap Hugging Face weight names to your custom tensor-parallel layer names.

---

## Reproduction Process
No Reproduction Process was needed because this is a new model we want vLLM-Omni to support. It is not a bug.

---

## Solution Approach

### Analysis

vLLM-Omni currently has no way to load or run HiDream-ai/HiDream-O1-Image. The root cause is simply that the model's architecture is unknown to the framework: there's no entry in vllm_omni/diffusion/registry.py's _DIFFUSION_MODELS, and no module under vllm_omni/diffusion/models/ that implements its forward pass or weight loading.

What makes this non-trivial is that HiDream-O1-Image isn't a standard diffusers DiT+VAE pipeline — it's a Qwen3-VL-8B vision-language model repurposed as a pixel-space image denoiser. There's no VAE and no separate text encoder; the image is split into 32x32 patches, treated as extra tokens appended to the text prompt, and a single LLM forward pass (with mixed causal text attention + bidirectional image attention) predicts the denoising velocity at each of ~50 flow-matching steps. The weights ship as a single non-diffusers safetensors set at the repo root.

### Proposed Solution

 Add a new self-contained diffusion pipeline module, vllm_omni/diffusion/models/hidream_o1_image/, following the existing "Adding a Diffusion Model" pattern and registered in registry.py. Concretely:
 
- Transformer (hidream_o1_image_transformer.py): port the Qwen3-VL backbone (text decoder + vision tower, DeepStack injection, mrope) plus the three small custom modules (BottleneckPatchEmbed, TimestepEmbedder, FinalLayer), using SenseNova-U1 as the structural template since it's the closest existing "LLM-as-denoiser" model—including its TP-aware linears and its use of vllm_omni.diffusion.attention.layer.Attention with AttentionMetadata(attn_mask=...) for the mixed causal/bidirectional mask.
- Scheduler: port FlowUniPCMultistepScheduler near-verbatim from the reference repo.
- Pipeline (pipeline_hidream_o1_image.py): implement forward(req) -> DiffusionOutput — prompt/patch tokenization, the denoising loop (forward + CFG + scheduler step, repeated), and un-patchify back to pixels. Scope v1 to text-to-image only.
- Weight loading: use BAGEL's "Pattern 2" (weights_sources at repo root, subfolder=None, custom load_weights() name remapping against model.safetensors.index.json).
- Examples/tests/docs: offline + online serving examples, an L4 e2e test, and a supported_models.md entry.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** 

Issue #3733 asks vLLM-Omni to add support for HiDream-ai/HiDream-O1-Image, a text-to-image model
that's currently near the top of the open-weights leaderboards. Right now, vLLM-Omni simply can't load or run this model — there's no entry for it in the model registry and no code that knows its architecture.

What should happen instead: a user should be able to run Omni(model="HiDream-ai/HiDream-O1-Image") (offline) or vllm serve HiDream-ai/HiDream-O1-Image --omni (online) and get text-to-image generation working, on par with the official reference implementation.

What I learned the model actually is: it's a Qwen3-VL-8B vision-language model repurposed as an image generator. It denoises images directly in pixel space — no VAE, no separate text encoder. The image is split into 32x32 patches, treated like extra "tokens" appended to the prompt; on each denoising step the model runs one forward pass over (text tokens + noisy image-patch tokens + a timestep token), where text tokens attend causally and image tokens attend bidirectionally to everything. The output prediction is fed through a flow-matching scheduler to produce the next, less-noisy version of the image. Repeat ~50 times, reassemble patches into pixels — done.
 

**Match:**

The closest existing pattern in this codebase is SenseNova-U1 (vllm_omni/diffusion/models/sensenova_u1/) — it's the same core idea: an LLM backbone (Qwen3) does the denoising itself, rather than a separate DiT. I'll use its __init__.py / pipeline_sensenova_u1.py / sensenova_u1_transformer.py as the structural template, including its tensor-parallel-aware linear layers (QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear) and its custom pre/post-process functions.

For weight loading specifically, BAGEL (.claude/skills/add-diffusion-model/references/custom-model-patterns.md, "Pattern 2: custom safetensors at root") is the match — both models ship one safetensors set at the repo root with non-diffusers naming, loaded via weights_sources (subfolder=None) + a custom load_weights() remapper.

For the overall workflow, the official vLLM-Omni "Adding a Diffusion Model" guide applies directly (single-stage pipeline under vllm_omni/diffusion/models/, registered in vllm_omni/diffusion/registry.py's _DIFFUSION_MODELS). The separate "Adding an Omni-Modality Model" guide does not apply — that one is for multi-stage models like Qwen3-Omni (thinker/talker/code2wav), and HiDream-O1-Image is a single self-contained pipeline.

**Plan:**

New directory: vllm_omni/diffusion/models/hidream_o1_image/

 1. Scaffold: __init__.py, empty hidream_o1_image_transformer.py and
 pipeline_hidream_o1_image.py, plus the registry entry in vllm_omni/diffusion/registry.py
 (_DIFFUSION_MODELS["HiDreamO1ImagePipeline"] = ("hidream_o1_image", "pipeline_hidream_o1_image",
 "HiDreamO1ImagePipeline")).
 2. Transformer (hidream_o1_image_transformer.py): port the Qwen3-VL vision tower + text decoder
 (weight names, DeepStack visual injection, mrope), replacing attention with
 vllm_omni.diffusion.attention.layer.Attention (causal=False, mixed mask via
 AttentionMetadata(attn_mask=...), QKV shape [batch, seq, num_heads, head_dim], role="self").
 Swap in TP-aware linears per sensenova_u1_transformer.py. Add the 3 small new modules from the
 reference (BottleneckPatchEmbed/x_embedder, TimestepEmbedder/t_embedder1, FinalLayer/
 final_layer2) essentially unchanged. Implement the mixed causal/bidirectional _forward_generation
 path (4D additive mask variant).
 3. Scheduler: port FlowUniPCMultistepScheduler from the reference models/fm_solvers_unipc.py
 (near-verbatim — it's a diffusers-style scheduler, default for the "full" 50-step model).
 4. Pipeline (pipeline_hidream_o1_image.py): nn.Module with forward(req) -> DiffusionOutput.
 Port prompt/token construction (build_t2i_text_sample, get_rope_index_fix_point) and the
 denoising loop (patchify -> N steps of forward+CFG+scheduler step -> un-patchify). Scope to
 text-to-image only for v1 — defer reference-image conditioning/editing/personalization and the
 "dev" variant/schedulers.
 5. Weight loading: weights_sources at repo root (subfolder=None, BAGEL Pattern 2), custom
 load_weights() remapping model.safetensors.index.json names to the ported module's parameter
 names.
 6. Examples/tests/docs:
   - examples/offline_inference/hidream_o1_image/end2end.py + README
   - examples/online_serving/hidream_o1_image/ (run_server.sh + openai_chat_client.py + README)
   - L4 functionality test, e.g. tests/e2e/online_serving/test_hidream_o1_image_expansion.py
   - docs/models/supported_models.md entry, recipes/HiDream/HiDream-O1-Image.md recipe, and
 acceleration-table updates (docs/user_guide/diffusion_acceleration.md,
 docs/user_guide/diffusion/parallelism_acceleration.md, etc.) once parallelism lands.
 7. Parallelism (follow-up PRs, not this one): TP for the backbone first (reuse SenseNova-U1's
 pattern), then SP/CFG-parallel/cache-dit as stretch goals.

**Implement:**

 (Phase III — placeholder)
 - Branch: TBD (e.g. feat/hidream-o1-image)
 - PR: TBD
   
**Review:**

Before opening the PR, I'll run the precheck-pr skill (referenced from CONTRIBUTING.md), which checks against vllm-project/vllm-omni conventions:
 - PR title format: must use one of the project's bracket prefixes. For this change, that's
 [Model] Add HiDream-O1-Image support (the [Model] prefix requires naming the model).
 - PR categorization: this is both a "New Model" and "Diffusion Model" PR (new files under
 vllm_omni/diffusion/models/hidream_o1_image/ + registry changes), so the union of both checklists
 in references/checklists.md applies — registry/config correctness, no dead code, accuracy of any
 claims, etc.
 - Self-review checklist (from the diffusion-model guide's PR checklist):
   - ✅ Transformer, pipeline, registry entries, __init__.py exports all present
   - ✅ Example script(s) under examples/
   - ✅ Test file under tests/e2e/
   - ✅ docs/models/supported_models.md updated
   - ✅ Acceleration tables updated if parallelism/caching is included (not in v1)
   - PR description includes hardware/driver/CUDA versions used for testing, per the guide's "Performance/Speed Check" step.

**Evaluate:**

 - Correctness: run the new offline example against a downloaded checkpoint with a fixed
 prompt/seed, and compare the output image to the reference inference.py run with the same
 prompt/seed (no diffusers baseline exists for this model, so the original repo's script is the ground
 truth).
 - Automated test: add an L4 functionality test per docs/contributing/ci/CI_5levels.md (e.g.
 tests/e2e/online_serving/test_hidream_o1_image_expansion.py, modeled on the existing
 test_hidream_i1_full_expansion.py / SenseNova-U1 e2e tests) — confirms the model loads, registers
 correctly, and produces a valid image end-to-end via the server.
 - Sanity check: if/when TP is added in a follow-up, verify TP>1 output matches TP=1 output for the
 same seed.

---

## Testing Strategy

### Unit Tests

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week [X] Progress

[What you built this week, challenges faced, decisions made]

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files modified:** [List]
- **Key commits:** [Links to important commits]
- **Approach decisions:** [Why you chose certain approaches]

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**

- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
