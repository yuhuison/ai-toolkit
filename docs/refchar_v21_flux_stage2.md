# RefChar V21 Flux Stage2 reference routing

Stage2 trains reference-conditioned generation after the completed Stage1.5
T2I run. It preserves the effective Stage1.5 model as a fixed merged base and
emits a separate Stage2 LoRA.

## Model path

```text
FLUX.2 Klein 9B
  + community NSFW LoKr @ 0.7
  + v20 Stage1 T2I LoKr @ 1.0
  + V21 Stage1.5-10000 all-layer T2I LoRA @ 1.0
  + V21 Stage2 attention-only reference LoRA
```

FLUX.2 reference images use the VAE path: reference latent tokens are appended
to the target latent sequence. Qwen3 encodes text only.

## Adapter scope

Stage2 uses a conventional rank/alpha 32/32 LoRA. It targets `img_attn` and
`txt_attn` inside all eight double blocks:

- image QKV and output projection;
- text QKV and output projection;
- 32 Linear modules;
- 12,582,912 trainable parameters.

MLPs and all single blocks stay frozen. The narrower scope is intentional:
Stage1.5 already learned style and generation quality, while Stage2 needs to
learn routing among reference, target-image, and text tokens.

## Mixed dataset

- I2I: 9,978 balanced pairs after excluding 22 near-duplicate source/targets.
- T2I rehearsal: 2,494 samples, stratified by setting label and SFW/NSFW.
- T2I share: 19.997% of 12,472 sample exposures per epoch.
- Reference controls: white-letterboxed to a fixed 720x720 so batch-two
  reference token sequences have identical lengths.

FLUX.2 uses 16-pixel bucket divisibility. With aspect buckets, an epoch has
5,013 I2I batches and 1,265 T2I batches. The three-epoch run therefore uses
18,834 optimizer steps.

## Schedule

- 720 resolution, batch 2;
- gradient checkpointing enabled;
- AdamW8bit, LR 3e-5, zero weight decay;
- constant scheduler with 300 warmup steps;
- shifted flow-matching timesteps;
- bf16;
- sample every 500 steps and save every 1,000 steps.

The executable configuration is:

`config/examples/train_refchar_v21_flux_stage2_ref_lora32_720.yaml`

For inference, load the Stage1.5-equivalent merged transformer and apply the
Stage2 LoRA at multiplier 1.0. Loading Stage2 directly on a bare Klein model is
not equivalent to the training stack.
