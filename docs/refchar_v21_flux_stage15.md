# RefChar V21 Flux Stage1.5

This run adds a pure-T2I, all-transformer LoRA after the small v20 Stage1
style LoKr and before the future reference-character I2I stage.

## Training stack

```text
FLUX.2 Klein base
  + community NSFW LoKr at 0.7
  + v20 Stage1 T2I LoKr at 1.0
  + Stage1.5 T2I LoRA (this run)
```

The two LoKr adapters are frozen and merged into the training checkpoint.
Stage1.5 produces only a new LoRA.

## Data and schedule

- Dataset: the RefChar setting-balanced 10k target images
- Captions: pure T2I captions; no source/reference image or I2I instruction
- Distribution: 5,603 NSFW and 4,397 SFW
- Resolution: 720 with aspect-ratio buckets
- Batch size: 2
- Steps: 10,000, equal to two full dataset epochs
- LoRA: rank 32, alpha 32
- Scope: all Linear modules in all FLUX.2 double and single blocks
- Actual scope: 112 modules, 82,837,504 trainable parameters
- Optimizer: AdamW8bit, LR 5e-5, weight decay 0
- Scheduler: constant with 200 warmup steps
- Timestep sampling: shift
- Precision: bf16

The executable configuration is
`config/examples/train_refchar_v21_flux_stage15_t2i_lora32_720.yaml`.

## Custom text encoder

`Flux2KleinModel.load_te()` honors `model.te_name_or_path`. This permits a
local uncensored Qwen3 checkpoint and its tokenizer to be used without changing
the FLUX.2 transformer or VAE source.

## Inference compatibility

The Stage1.5 LoRA must be applied to the same effective base used for training:
community NSFW LoKr at 0.7 plus v20 Stage1 LoKr at 1.0. Those adapters may be
merged or loaded separately; the resulting Stage1.5 LoRA is the same.
