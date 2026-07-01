# su26-ai301-contribution

# Contribution 1: [New Model]: HiDream-O1-Image

**Contribution Number:** 1  
**Student:** SOKOUDJOU LEOPOLD  
**Issue:** https://github.com/vllm-project/vllm-omni/issues/3733  
**Status:** Phase IV In-Progress

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

**init**.py: For exposing the pipeline and registration classes.

hidream_o1_image_transformer.py: Holds the core ported neural network architecture.

pipeline_hidream_o1_image.py: Implements the scheduling and pixel-space denoising loop.

fm_solvers_unipc.py: Holds the ported custom scheduler.

model_index.json: Configuration mapping (\_class_name: "HiDreamO1ImagePipeline").

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

vLLM-Omni currently has no way to load or run HiDream-ai/HiDream-O1-Image. The root cause is simply that the model's architecture is unknown to the framework: there's no entry in vllm_omni/diffusion/registry.py's \_DIFFUSION_MODELS, and no module under vllm_omni/diffusion/models/ that implements its forward pass or weight loading.

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

The closest existing pattern in this codebase is SenseNova-U1 (vllm_omni/diffusion/models/sensenova_u1/) — it's the same core idea: an LLM backbone (Qwen3) does the denoising itself, rather than a separate DiT. I'll use its **init**.py / pipeline_sensenova_u1.py / sensenova_u1_transformer.py as the structural template, including its tensor-parallel-aware linear layers (QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear) and its custom pre/post-process functions.

For weight loading specifically, BAGEL (.claude/skills/add-diffusion-model/references/custom-model-patterns.md, "Pattern 2: custom safetensors at root") is the match — both models ship one safetensors set at the repo root with non-diffusers naming, loaded via weights_sources (subfolder=None) + a custom load_weights() remapper.

For the overall workflow, the official vLLM-Omni "Adding a Diffusion Model" guide applies directly (single-stage pipeline under vllm_omni/diffusion/models/, registered in vllm_omni/diffusion/registry.py's \_DIFFUSION_MODELS). The separate "Adding an Omni-Modality Model" guide does not apply — that one is for multi-stage models like Qwen3-Omni (thinker/talker/code2wav), and HiDream-O1-Image is a single self-contained pipeline.

**Plan:**

New directory: vllm_omni/diffusion/models/hidream_o1_image/

1.  Scaffold: **init**.py, empty hidream_o1_image_transformer.py and
    pipeline_hidream_o1_image.py, plus the registry entry in vllm_omni/diffusion/registry.py
    (\_DIFFUSION_MODELS["HiDreamO1ImagePipeline"] = ("hidream_o1_image", "pipeline_hidream_o1_image",
    "HiDreamO1ImagePipeline")).
2.  Transformer (hidream_o1_image_transformer.py): port the Qwen3-VL vision tower + text decoder
    (weight names, DeepStack visual injection, mrope), replacing attention with
    vllm_omni.diffusion.attention.layer.Attention (causal=False, mixed mask via
    AttentionMetadata(attn_mask=...), QKV shape [batch, seq, num_heads, head_dim], role="self").
    Swap in TP-aware linears per sensenova_u1_transformer.py. Add the 3 small new modules from the
    reference (BottleneckPatchEmbed/x_embedder, TimestepEmbedder/t_embedder1, FinalLayer/
    final_layer2) essentially unchanged. Implement the mixed causal/bidirectional \_forward_generation
    path (4D additive mask variant).
3.  Scheduler: port FlowUniPCMultistepScheduler from the reference models/fm_solvers_unipc.py
    (near-verbatim — it's a diffusers-style scheduler, default for the "full" 50-step model).
4.  Pipeline (pipeline_hidream_o1_image.py): nn.Module with forward(req) -> DiffusionOutput.
    Port prompt/token construction (build_t2i_text_sample, get_rope_index_fix_point) and the
    denoising loop (patchify -> N steps of forward+CFG+scheduler step -> un-patchify). Scope to
    text-to-image only for v1 — defer reference-image conditioning/editing/personalization and the
    "dev" variant/schedulers.
5.  Weight loading: weights_sources at repo root (subfolder=None, BAGEL Pattern 2), custom
    load_weights() remapping model.safetensors.index.json names to the ported module's parameter
    names.
6.  Examples/tests/docs:

- examples/offline_inference/hidream_o1_image/end2end.py + README
- examples/online_serving/hidream_o1_image/ (run_server.sh + openai_chat_client.py + README)
- L4 functionality test, e.g. tests/e2e/online_serving/test_hidream_o1_image_expansion.py
- docs/models/supported_models.md entry, recipes/HiDream/HiDream-O1-Image.md recipe, and
  acceleration-table updates (docs/user_guide/diffusion_acceleration.md,
  docs/user_guide/diffusion/parallelism_acceleration.md, etc.) once parallelism lands.

7.  Parallelism (follow-up PRs, not this one): TP for the backbone first (reuse SenseNova-U1's
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
  - ✅ Transformer, pipeline, registry entries, **init**.py exports all present
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

### Week 3 Progress

[What you built this week, challenges faced, decisions made]

#### Implementation Progress

Built the first phase of HiDream-O1-Image support (Path B — custom/private-repo model, no diffusers pipeline, no VAE) in vllm_omni/diffusion/models/hidream_o1_image/:

utils_hidream_o1.py — ported the position-id logic (get_rope_index_fix_point, 3D MRoPE with the model's "fix_point" anchoring for image spans), resolution snapping, patchify/depatchify (raw pixel patches, no VAE), and a function that builds two parallel attention-masking representations (a dense mask + full_attn_spans) from the same token-type data.
qwen3_vl_uit_transformer.py — the backbone: imports the vanilla parts of HF's Qwen3-VL directly, and only writes the genuinely new pieces (BottleneckPatchEmbed, TimestepEmbedder, FinalLayer) plus a rewired attention layer/decoder that uses vLLM-Omni's Attention + AttentionMetadata instead of the reference repo's hand-rolled two-pass flash-attention trick.
pipeline_hidream_o1_image.py — the orchestration: builds the packed text+image sequence, runs the denoise loop with CFG support, remaps the checkpoint's weight names, exposes the model to vLLM-Omni.
registry.py — registered as HiDreamO1ImagePipeline (kept distinct from the existing, unrelated HiDreamImagePipeline/HiDream-I1 entry).
Example script + README under examples/offline_inference/hidream_o1_image/, unit tests under tests/diffusion/models/hidream_o1_image/.
Key commit: 9d10a8e8 — "[Model] Add HiDream-O1-Image text-to-image support" on leopold/Add-HiDream-O1-Image.

#### Challenges Faced

The model's architecture doesn't look like anything else in the repo: it's one Qwen3-VL-based transformer that diffuses directly on raw pixels (no VAE, no separate text encoder), where text tokens need causal attention and image/timestep tokens need bidirectional attention in the same forward pass. I de-risked this before writing code by reading the reference repo's source directly (via curl/WebFetch against the GitHub repo) and tracing vLLM-Omni's existing piecewise_attn/AttentionMetadata machinery line-by-line, which confirmed it was an exact fit — that saved a lot of guessing.

The harder part was the real bugs that only showed up once I ran it on an actual GPU against the real checkpoint:

A hardcoded module dimension (bottleneck_dim=768) that was wrong — found instantly by checking the checkpoint's actual tensor shapes before guessing.
A self-inflicted ordering bug (self.scheduler assigned after the denoise loop instead of before it).
The most subtle one: a missing noise_scale=8.0 multiplier on the initial noise. Without it, the output didn't crash — it just slowly collapsed to flat gray over the course of the denoise loop, which only became obvious by logging per-step latent statistics and comparing 4-step vs. 28-step runs.
There was also an environment quirk: this is a shared multi-GPU box, and two of the four GPUs were silently reserved by another user (visible only as a generic "CUDA device busy" error, not in nvidia-smi's usage stats) — found by testing each GPU index individually.

#### Testing Strategy

Two layers:

Unit tests (test_packed_sequence.py, 9 cases) — test the masking/position-id/patchify logic in isolation from model weights: confirms text never attends into the image span, image/timestep positions attend to everything, the position-id anchoring actually jumps away from the naive placement, and multi-span detection works (needed for the next phase's multi-reference-image case).
Real end-to-end validation on GPU — downloaded the actual HiDream-O1-Image-Dev checkpoint and verified: 100% of 758 parameters load with zero shape mismatches; a full denoise loop runs without NaNs; the output is a genuinely coherent, prompt-matching image (not just "doesn't crash"); both the no-CFG and CFG (2-branch combine) code paths work; and it all runs correctly through the real Omni() engine (multiprocess worker, scheduler, clean shutdown), not just a manual bypass script.

#### Branch Link

https://github.com/SOKOUDJOU-LEOPOLD/vllm-omni/tree/leopold/Add-HiDream-O1-Image

### Week 4 Progress

#### Implementation Progress:

Phase 2 of HiDream-O1-Image integration is complete, adding image editing (1 reference image) and multi-reference subject personalization (2–10+ reference images) on top of the Phase 1 text-to-image foundation. The core addition is a dual-pathway conditioning system where reference images flow through two independent paths simultaneously: a VLM semantic path (ref images → Qwen3-VL vision tower → patch embeddings + DeepStack intermediate features injected into early decoder layers) and a pixel-patch path (ref images patchified at adaptive resolution and concatenated with the target noise as token-type=2 bidirectional tokens). The pipeline gained \_build_edit_sample() for constructing packed sequences with K references, SupportImageInput protocol support so the engine routes image inputs correctly, and a vision-tower embed cache so the expensive VLM encoding runs exactly once per request regardless of denoising step count or CFG branches.

#### Challenges Faced:

Three bugs were discovered and fixed during implementation. First, forward_generation() had an erroneous torch.cat(image_embeds, dim=0) call — get_image_features() already returns a single merged tensor (pooler_output), not a list, so the cat raised a runtime error; removed it. Second, \_deepstack_process() was called inside the text decoder's forward loop but never defined — it was latent in Phase 1 because T2I always has pixel_values=None so the branch never executed; added the static method. Third, the Phase 2 test additions used Image.new(...) but PIL wasn't imported at that point in the test file; fixed by adding from PIL import Image as PILImage locally. A subtler design challenge was caching vision embeddings across denoising steps without changing predict_noise()'s return type (which must stay torch.Tensor for CFGParallelMixin.\_wrap() compatibility) — solved by passing a mutable dict \_embed_storage as a side-channel kwarg that predict_noise() fills on step 0 without touching its return value.

#### Testing Strategy:

23 CPU-only unit tests across three files, all marked core_model + cpu (no weights or GPU required):

test_packed_sequence.py — extended with TestAdaptiveRefMaxSize (4 tests: single ref = full resolution, two refs smaller, monotonic decrease across K, minimum always positive), TestPreprocessRefPatches (3 tests: output shape, two-ref concatenation, normalization to [-1, 1] verified with solid white/black images), and TestEditSampleTokenTypes (4 tests: vinput_mask is True only at type=1 target positions, tms+target+ref pixels form one contiguous gen span, ref pixel rows attend to the full sequence, text rows are causally blocked from gen positions).
test_embed_cache.py (new) — 3 tests simulating the N-step denoising loop with a fake forward function: vision tower called exactly once across 4 steps, \_embed_storage populated after step 0, T2I path (no refs) never invokes the vision tower.

#### Branch Link:

https://github.com/SOKOUDJOU-LEOPOLD/vllm-omni/tree/leopold/Add-HiDream-O1-Image

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
