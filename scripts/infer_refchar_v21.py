#!/usr/bin/env python3
"""Inference entry point for RefChar V21 VAE-only reference generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


AI_TOOLKIT_DIR = Path(__file__).resolve().parents[1]
if str(AI_TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(AI_TOOLKIT_DIR))

from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig  # noqa: E402
from toolkit.lora_special import LoRASpecialNetwork  # noqa: E402
from toolkit.util.get_model import get_model_class  # noqa: E402


DEFAULT_BASE = Path(
    "/workspace/fusal-refchar-v17-stage1/stage2_qwen2511/merged_base/"
    "qwen_image_edit_2511_nsfw07_stage1v17s09"
)
DEFAULT_EXTRAS = Path("/workspace/fusal-refchar-image-preview")
ALL_LAYER_TARGETS = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mlp.net",
    "txt_mlp.net",
    "img_mod",
    "txt_mod",
]


def attach_dora(model, checkpoint: Path, rank: int, scale: float) -> None:
    network_config = NetworkConfig(
        type="dora",
        linear=rank,
        linear_alpha=rank,
        network_kwargs={"only_if_contains": ALL_LAYER_TARGETS},
    )
    network_kwargs = dict(network_config.network_kwargs)
    if hasattr(model, "target_lora_modules"):
        network_kwargs["target_lin_modules"] = model.target_lora_modules

    network = LoRASpecialNetwork(
        text_encoder=model.text_encoder,
        unet=model.get_model_to_train(),
        lora_dim=network_config.linear,
        multiplier=scale,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        conv_lora_dim=network_config.conv,
        conv_alpha=network_config.conv_alpha,
        is_sdxl=model.model_config.is_xl or model.model_config.is_ssd,
        is_v2=model.model_config.is_v2,
        is_v3=model.model_config.is_v3,
        is_pixart=model.model_config.is_pixart,
        is_auraflow=model.model_config.is_auraflow,
        is_flux=model.model_config.is_flux,
        is_lumina2=model.model_config.is_lumina2,
        is_ssd=model.model_config.is_ssd,
        is_vega=model.model_config.is_vega,
        dropout=network_config.dropout,
        use_text_encoder_1=model.model_config.use_text_encoder_1,
        use_text_encoder_2=model.model_config.use_text_encoder_2,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_transformer=model.is_transformer,
        base_model=model,
        **network_kwargs,
    )
    network.force_to(model.device_torch, dtype=torch.float32)
    model.network = network
    network._update_torch_multiplier()
    network.apply_to(
        model.text_encoder,
        model.get_model_to_train(),
        apply_text_encoder=False,
        apply_unet=True,
    )
    network.load_weights(str(checkpoint))
    network.eval()
    network.multiplier = scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--extras", type=Path, default=DEFAULT_EXTRAS)
    parser.add_argument(
        "--dora",
        type=Path,
        help="V21 DoRA checkpoint. Omit only for framework smoke tests.",
    )
    parser.add_argument("--dora-rank", type=int, default=64)
    parser.add_argument("--dora-scale", type=float, default=1.0)
    parser.add_argument("--reference", type=Path, required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--do-cfg-norm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = (
        args.prompt
        if args.prompt is not None
        else args.prompt_file.read_text(encoding="utf-8").strip()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(
        name_or_path=str(args.base),
        extras_name_or_path=str(args.extras),
        arch="qwen_image_edit_plus",
        dtype="bf16",
        quantize=False,
        quantize_te=False,
        low_vram=False,
        model_kwargs={
            "match_target_res": False,
            "text_conditioning": {
                "mode": "vae_only",
                "template": "refchar_generative_v1",
            },
        },
    )
    model_class = get_model_class(model_config)
    model = model_class(
        device="cuda:0",
        model_config=model_config,
        dtype="bf16",
        noise_scheduler=model_class.get_train_scheduler(),
    )
    model.load_model()

    if args.dora is not None:
        attach_dora(model, args.dora, args.dora_rank, args.dora_scale)

    config = GenerateImageConfig(
        prompt=prompt,
        negative_prompt="",
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        network_multiplier=args.dora_scale,
        output_path=str(args.output),
        output_ext=args.output.suffix.lstrip(".") or "png",
        ctrl_img=str(args.reference),
        do_cfg_norm=args.do_cfg_norm,
    )
    started = time.time()
    with torch.inference_mode():
        model.generate_images([config], sampler="flowmatch")
    elapsed = time.time() - started

    metadata = {
        "architecture": "qwen_image_edit_plus",
        "text_conditioning": {
            "mode": "vae_only",
            "template": "refchar_generative_v1",
        },
        "base": str(args.base),
        "extras": str(args.extras),
        "dora": str(args.dora) if args.dora is not None else None,
        "dora_rank": args.dora_rank if args.dora is not None else None,
        "dora_scale": args.dora_scale if args.dora is not None else None,
        "reference": str(args.reference),
        "prompt": prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "cfg": args.cfg,
        "do_cfg_norm": args.do_cfg_norm,
        "elapsed_seconds": elapsed,
        "output": str(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.output} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
