# RefChar V21 VAE-only reference-conditioned generation

## Design

RefChar V21 uses Qwen Image Edit Plus as a reference-conditioned generator:

```text
target prompt
    │
    ▼
Qwen2.5-VL (text only, refchar_generative_v1)
    │ text embeddings
    ├──────────────────────────┐
    │                          ▼
reference image ── VAE ── reference latent tokens ── MMDiT ── target image

reference image ──X── Qwen2.5-VL visual tower
```

The VAE path supplies character identity and appearance. The text path specifies
the target pose, action, camera, composition, environment, lighting, and style.
This is not pure T2I and not local image editing; it is
**reference-conditioned generation**.

V21 training starts from the RefChar V17 Stage1 transformer. V20 Stage3/4/5 are
not V21 training bases.

## Prompt template

The exact `refchar_generative_v1` system prompt is:

```text
Generate a new image conditioned on a separately provided character reference. Use the reference to preserve the character's identity and relevant appearance details. Treat the user's prompt as the target image specification, following its pose, action, camera view, composition, environment, lighting, and style.
```

This combines the useful identity-preservation prior of an edit prompt with the
target-image specification semantics of a generative prompt. It never asks the
text encoder to inspect an image that it does not receive.

## Framework configuration

Set the model options as follows:

```yaml
model:
  arch: qwen_image_edit_plus
  model_kwargs:
    text_conditioning:
      mode: vae_only
      template: refchar_generative_v1
```

Available modes:

- `dual`: native Qwen Image Edit behavior; the reference enters both Qwen2.5-VL
  and the VAE. This remains the default for backward compatibility.
- `vae_only`: the reference enters only the VAE.

Built-in VAE-only templates:

- `edit`
- `generative`
- `refchar_generative_v1`

A custom system prompt is also supported:

```yaml
model_kwargs:
  text_conditioning:
    mode: vae_only
    template: custom_name
    system_prompt: >-
      Your custom system prompt.
```

The implementation:

1. Dynamically computes the system-prefix token count instead of hardcoding the
   native templates' 34/64-token offsets.
2. Releases the Qwen2.5-VL visual tower after pipeline construction in
   VAE-only mode.
3. Leaves reference-image VAE encoding and MMDiT token concatenation unchanged.
4. Includes the mode, template name, and system-prompt SHA-256 digest in the
   text embedding cache namespace.
5. Preserves historical Dual cache keys and behavior.

## Dataset preparation

The helper converts RefChar WebDataset shards into AI Toolkit's folder layout:

```bash
source /venv/main/bin/activate
python scripts/prepare_refchar_v21_dataset.py \
  --shards /workspace/fusal-refchar-v20/shards \
  --output /workspace/refchar_v21/dataset
```

Output:

```text
/workspace/refchar_v21/dataset/target  # target JPG and matching caption TXT
/workspace/refchar_v21/dataset/ref0    # reference JPG
```

The script verifies that target, source, and caption keys match and rejects
size-mismatched existing files.

If the prompt template changes, old text embedding caches must not be reused.
The cache namespace change handles this automatically for framework-created
caches.

## Training

Example:

```text
config/examples/train_refchar_v21_vae_only_qwen_image_edit_96gb.yaml
```

Start training:

```bash
source /venv/main/bin/activate
python run.py config/examples/train_refchar_v21_vae_only_qwen_image_edit_96gb.yaml
```

The example uses:

- V17 Stage1 transformer
- VAE-only `refchar_generative_v1`
- DoRA rank/alpha 64/64
- 12 target-name patterns that match 840 Qwen transformer Linear modules
- 720 resolution
- 47,778 steps for 15,926 pairs × 3 epochs

The V17 archive used during development contains only the transformer. In that
case, set `extras_name_or_path` to a compatible full Qwen Image Edit pipeline
containing the tokenizer, text encoder, VAE, and processor. This does not replace
the transformer selected by `name_or_path`.

Run a short copied smoke configuration before starting a long training job.

## Inference

Use the repository entry point:

```bash
source /venv/main/bin/activate

python scripts/infer_refchar_v21.py \
  --base /path/to/v17-stage1-or-v21-merged-base \
  --extras /path/to/full-qwen-image-edit-pipeline \
  --dora /path/to/refchar_v21_checkpoint.safetensors \
  --reference /path/to/reference.jpg \
  --prompt-file /path/to/prompt.txt \
  --output /path/to/result.png \
  --width 1280 \
  --height 720 \
  --steps 25 \
  --seed 42 \
  --cfg 4
```

The script fixes the runtime conditioning to:

```text
mode = vae_only
template = refchar_generative_v1
DoRA target scope = all 840 V21 Linear modules
```

It writes `result.png.json` alongside the image with the base, extras, adapter,
template, reference, prompt, dimensions, steps, seed, CFG, and elapsed time.

For a fully merged V21 pipeline:

- point `--base` and `--extras` to the full merged directory;
- omit `--dora` to avoid applying the adapter twice;
- keep the patched framework and VAE-only template settings.

Merging weights does not serialize the Python prompt-conditioning behavior.

## Deployment checklist

Training and inference must agree on:

- the exact `refchar_generative_v1` text;
- dynamic prefix trimming;
- no reference image input to Qwen2.5-VL;
- reference image input to the VAE;
- the 12 all-layer DoRA target patterns;
- DoRA rank/alpha 64/64.

For an unmerged deployment, move all of:

- this patched AI Toolkit revision;
- the V17 Stage1 base;
- compatible full-pipeline extras;
- the V21 DoRA checkpoint;
- `scripts/infer_refchar_v21.py`.

## Validation performed

- Python compilation and YAML parsing passed.
- Default Dual and VAE-only configuration initialization passed.
- Template changes produce distinct text embedding cache namespaces.
- The 12 target patterns match exactly 840 Linear modules on V17 Stage1.
- A V17 Stage1 VAE-only smoke inference ran end-to-end with the visual input
  disabled and produced a valid RGB PNG.
