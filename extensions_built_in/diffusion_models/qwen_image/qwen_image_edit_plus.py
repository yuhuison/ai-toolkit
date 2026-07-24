import hashlib
import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
import yaml
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
)
from optimum.quanto import QTensor, freeze
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from toolkit import train_tools
from toolkit.accelerator import get_accelerator, unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import get_qtype, quantize, quantize_model

from .qwen_image import QwenImageModel

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

try:
    from .qwen_image_pipelines import QwenImageEditPlusCustomPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        VAE_IMAGE_SIZE,
    )
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing 'pip uninstall diffusers' and then 'pip install -r requirements.txt'"
    )


TEXT_ONLY_SYSTEM_PROMPTS = {
    "edit": (
        "Describe the key features of the input image (color, shape, size, texture, "
        "objects, background), then explain how the user's text instruction should "
        "alter or modify the image. Generate a new image that meets the user's "
        "requirements while maintaining consistency with the original input where "
        "appropriate."
    ),
    "generative": (
        "Describe the image by detailing the color, shape, size, texture, quantity, "
        "text, spatial relationships of the objects and background:"
    ),
    "refchar_generative_v1": (
        "Generate a new image conditioned on a separately provided character reference. "
        "Use the reference to preserve the character's identity and relevant appearance "
        "details. Treat the user's prompt as the target image specification, following "
        "its pose, action, camera view, composition, environment, lighting, and style."
    ),
}


class QwenImageEditPlusModel(QwenImageModel):
    arch = "qwen_image_edit_plus"
    _qwen_image_keep_visual = True
    _qwen_pipeline = QwenImageEditPlusCustomPipeline

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["QwenImageTransformer2DModel"]

        text_conditioning = self.model_config.model_kwargs.get(
            "text_conditioning", {}
        )
        if text_conditioning is None:
            text_conditioning = {}
        if not isinstance(text_conditioning, dict):
            raise ValueError("model_kwargs.text_conditioning must be a mapping")

        self.text_conditioning_mode = text_conditioning.get("mode", "dual")
        if self.text_conditioning_mode not in {"dual", "vae_only"}:
            raise ValueError(
                "model_kwargs.text_conditioning.mode must be 'dual' or 'vae_only'"
            )

        default_template = (
            "edit"
            if self.text_conditioning_mode == "dual"
            else "refchar_generative_v1"
        )
        self.text_conditioning_template = text_conditioning.get(
            "template", default_template
        )
        self.text_conditioning_system_prompt = text_conditioning.get(
            "system_prompt"
        )

        if self.text_conditioning_mode == "dual":
            if (
                self.text_conditioning_template != "edit"
                or self.text_conditioning_system_prompt is not None
            ):
                raise ValueError(
                    "Dual conditioning uses the native Qwen Image Edit template. "
                    "Set mode='vae_only' to select a text-only template."
                )
        elif self.text_conditioning_system_prompt is None:
            if self.text_conditioning_template not in TEXT_ONLY_SYSTEM_PROMPTS:
                available = ", ".join(sorted(TEXT_ONLY_SYSTEM_PROMPTS))
                raise ValueError(
                    "Unknown VAE-only text template "
                    f"'{self.text_conditioning_template}'. Available: {available}; "
                    "or provide model_kwargs.text_conditioning.system_prompt."
                )
            self.text_conditioning_system_prompt = TEXT_ONLY_SYSTEM_PROMPTS[
                self.text_conditioning_template
            ]
        elif not isinstance(self.text_conditioning_system_prompt, str):
            raise ValueError(
                "model_kwargs.text_conditioning.system_prompt must be a string"
            )

        # set true for models that encode control image into text embeddings
        self.encode_control_in_text_embeddings = (
            self.text_conditioning_mode == "dual"
        )
        # control images will come in as a list for encoding some things if true
        self.has_multiple_control_images = True
        # do not resize control images
        self.use_raw_control_images = True

    @property
    def text_embedding_space_version(self):
        if self.text_conditioning_mode == "dual":
            # Preserve the historical cache key for backward compatibility.
            return self.arch
        prompt_hash = hashlib.sha256(
            self.text_conditioning_system_prompt.encode("utf-8")
        ).hexdigest()[:12]
        return (
            f"{self.arch}:{self.text_conditioning_mode}:"
            f"{self.text_conditioning_template}:{prompt_hash}"
        )

    def load_model(self):
        super().load_model()
        if self.text_conditioning_mode == "vae_only":
            # QwenImageEditPlusModel keeps the visual tower while constructing the
            # native pipeline. It is not used in VAE-only mode, so release it after
            # construction. The lightweight processor remains registered because
            # diffusers requires that pipeline component, but it is never invoked
            # when precomputed prompt embeddings are supplied.
            text_encoder = self.text_encoder[0]
            if getattr(text_encoder.model, "visual", None) is not None:
                text_encoder.model.visual = None
                flush()
            self.print_and_status_update(
                "Qwen text conditioning: VAE-only "
                f"({self.text_conditioning_template}); VL image input disabled"
            )
        else:
            self.print_and_status_update(
                "Qwen text conditioning: dual path (VL image + VAE reference)"
            )

    def get_generation_pipeline(self):
        scheduler = QwenImageModel.get_train_scheduler()

        pipeline: QwenImageEditPlusCustomPipeline = QwenImageEditPlusCustomPipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            processor=self.processor,
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: QwenImageEditPlusCustomPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        control_img_list = []
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        elif gen_config.ctrl_img_1 is not None:
            control_img = Image.open(gen_config.ctrl_img_1)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)

        if gen_config.ctrl_img_2 is not None:
            control_img = Image.open(gen_config.ctrl_img_2)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_3 is not None:
            control_img = Image.open(gen_config.ctrl_img_3)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)

        # flush for low vram if we are doing that
        # flush_between_steps = self.model_config.low_vram
        flush_between_steps = False

        # Fix a bug in diffusers/torch
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if flush_between_steps:
                flush()
            latents = callback_kwargs["latents"]

            return {"latents": latents}

        if self.model_config.low_vram:
            # set vae to tile decode
            pipeline.vae.enable_tiling()

        img = pipeline(
            image=control_img_list,
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_embeds_mask=conditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.int64
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.int64
            ),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            true_cfg_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            do_cfg_norm=gen_config.do_cfg_norm,
            **extra,
        ).images[0]

        if self.model_config.low_vram:
            # restore no tiling
            pipeline.vae.disable_tiling()

        return img

    def condition_noisy_latents(
        self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"
    ):
        # we get the control image from the batch
        return latents.detach()

    def _get_text_only_prompt_embeds(self, prompt: List[str]) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        if not isinstance(prompt, list):
            prompt = [prompt]

        system_prompt = self.text_conditioning_system_prompt.strip()
        prefix = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n"

        # Compute this from the actual tokenizer instead of copying the native
        # templates' fixed 34/64-token offsets. This keeps custom templates safe.
        drop_idx = len(
            self.pipeline.tokenizer(
                prefix, add_special_tokens=False
            ).input_ids
        )
        text = [f"{prefix}{item}{suffix}" for item in prompt]
        tokenizer_max_length = getattr(
            self.pipeline, "tokenizer_max_length", 1024
        )
        tokens = self.pipeline.tokenizer(
            text,
            max_length=tokenizer_max_length + drop_idx,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device_torch)

        outputs = self.pipeline.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        bool_mask = tokens.attention_mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_hidden_states = torch.split(
            selected, valid_lengths.tolist(), dim=0
        )
        split_hidden_states = [item[drop_idx:] for item in split_hidden_states]

        attn_mask_list = [
            torch.ones(item.size(0), dtype=torch.long, device=item.device)
            for item in split_hidden_states
        ]
        max_seq_len = max(item.size(0) for item in split_hidden_states)
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [
                        item,
                        item.new_zeros(
                            max_seq_len - item.size(0), item.size(1)
                        ),
                    ]
                )
                for item in split_hidden_states
            ]
        )
        prompt_embeds_mask = torch.stack(
            [
                torch.cat(
                    [
                        item,
                        item.new_zeros(max_seq_len - item.size(0)),
                    ]
                )
                for item in attn_mask_list
            ]
        )

        pe = PromptEmbeds(
            prompt_embeds.to(
                dtype=self.torch_dtype, device=self.device_torch
            )
        )
        pe.attention_mask = prompt_embeds_mask
        return pe

    def get_prompt_embeds(self, prompt: List, control_images=None) -> PromptEmbeds:
        # todo handle not caching text encoder
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        if self.text_conditioning_mode == "vae_only":
            return self._get_text_only_prompt_embeds(prompt)
            
        if control_images is None:
            raise ValueError("Missing control images for QwenImageEditPlusModel")
        
        if not isinstance(control_images, list):
            control_images = [control_images]
        
        # expects a list of list of control images List[List[Tensor]] where each item corresponds to a batch item, 
        # and each item in the inner list corresponds to a control image for that batch item.
        # for single image/caching, it may come in as just List[Tensor], so we handle that case by wrapping it in another list
        if not isinstance(control_images[0], list):
            control_images = [control_images]
        
        if len(prompt) != len(control_images):
            raise ValueError("Number of prompts must match number of control image sets")
        
        prompt_embeds_list = []
        prompt_embeds_mask_list = []
        
        for b in range(len(prompt)):
            batch_control_images = control_images[b]

            for i in range(len(batch_control_images)):
                if len(batch_control_images[i].shape) == 3:
                    batch_control_images[i] = batch_control_images[i].unsqueeze(0)
                # control images are 0 - 1 scale, shape (bs, ch, height, width)
                ratio = batch_control_images[i].shape[2] / batch_control_images[i].shape[3]
                height = math.sqrt(CONDITION_IMAGE_SIZE * ratio)
                width = height / ratio

                width = round(width / 32) * 32
                height = round(height / 32) * 32

                batch_control_images[i] = F.interpolate(
                    batch_control_images[i], size=(height, width), mode="bilinear"
                )

            prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
                prompt,
                image=batch_control_images,
                device=self.device_torch,
                num_images_per_prompt=1,
            )
            # diffusers >=0.37 returns None when all tokens are valid (no padding)
            if prompt_embeds_mask is None:
                prompt_embeds_mask = torch.ones(
                    prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.int64
                )
            prompt_embeds_list.append(prompt_embeds)
            prompt_embeds_mask_list.append(prompt_embeds_mask)
        pe = PromptEmbeds(torch.cat(prompt_embeds_list, dim=0))
        pe.attention_mask = torch.cat(prompt_embeds_mask_list, dim=0)
        return pe

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        with torch.no_grad():
            batch_size, num_channels_latents, height, width = latent_model_input.shape
            if self.vae.device != self.device_torch:
                self.vae.to(self.device_torch)
            
            control_image_res = VAE_IMAGE_SIZE
            if self.model_config.model_kwargs.get("match_target_res", False):
                # use the current target size to set the control image res
                control_image_res = height * self.pipeline.vae_scale_factor * width * self.pipeline.vae_scale_factor

            # pack image tokens
            latent_model_input = latent_model_input.view(
                batch_size, num_channels_latents, height // 2, 2, width // 2, 2
            )
            latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
            latent_model_input = latent_model_input.reshape(
                batch_size, (height // 2) * (width // 2), num_channels_latents * 4
            )

            raw_packed_latents = latent_model_input

            img_h2, img_w2 = height // 2, width // 2

            # build distinct instances per batch item, per mamad8
            img_shapes = [[(1, img_h2, img_w2)] for _ in range(batch_size)]

            # pack controls
            if batch is None:
                raise ValueError("Batch is required for QwenImageEditPlusModel")

            # split the latents into batch items so we can concat the controls
            packed_latents_list = torch.chunk(latent_model_input, batch_size, dim=0)
            packed_latents_with_controls_list = []
            
            batch_control_tensor_list = batch.control_tensor_list
            if batch_control_tensor_list is None and batch.control_tensor is not None:
                batch_control_tensor_list = []
                for b in range(batch_size):
                    batch_control_tensor_list.append(batch.control_tensor[b : b + 1])

            if batch_control_tensor_list is not None:
                b = 0
                for control_tensor_list in batch_control_tensor_list:
                    # control tensor list is a list of tensors for this batch item
                    controls = []
                    # pack control
                    for control_img in control_tensor_list:
                        # control images are 0 - 1 scale, shape (1, ch, height, width)
                        control_img = control_img.to(
                            self.device_torch, dtype=self.torch_dtype
                        )
                        # if it is only 3 dim, add batch dim
                        if len(control_img.shape) == 3:
                            control_img = control_img.unsqueeze(0)
                        ratio = control_img.shape[2] / control_img.shape[3]
                        c_height = math.sqrt(control_image_res * ratio)
                        c_width = c_height / ratio

                        c_width = round(c_width / 32) * 32
                        c_height = round(c_height / 32) * 32

                        control_img = F.interpolate(
                            control_img, size=(c_height, c_width), mode="bilinear"
                        )

                        # scale to -1 to 1
                        control_img = control_img * 2 - 1

                        control_latent = self.encode_images(
                            control_img,
                            device=self.device_torch,
                            dtype=self.torch_dtype,
                        )

                        clb, cl_num_channels_latents, cl_height, cl_width = (
                            control_latent.shape
                        )

                        control = control_latent.view(
                            1,
                            cl_num_channels_latents,
                            cl_height // 2,
                            2,
                            cl_width // 2,
                            2,
                        )
                        control = control.permute(0, 2, 4, 1, 3, 5)
                        control = control.reshape(
                            1,
                            (cl_height // 2) * (cl_width // 2),
                            num_channels_latents * 4,
                        )

                        img_shapes[b].append((1, cl_height // 2, cl_width // 2))
                        controls.append(control)

                    # stack controls on dim 1
                    control = torch.cat(controls, dim=1).to(
                        packed_latents_list[b].device,
                        dtype=packed_latents_list[b].dtype,
                    )
                    # concat with latents
                    packed_latents_with_control = torch.cat(
                        [packed_latents_list[b], control], dim=1
                    )

                    packed_latents_with_controls_list.append(
                        packed_latents_with_control
                    )

                    b += 1

                latent_model_input = torch.cat(packed_latents_with_controls_list, dim=0)

            prompt_embeds_mask = text_embeddings.attention_mask.to(
                self.device_torch, dtype=torch.int64
            )
            enc_hs = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(
                self.device_torch, self.torch_dtype
            ).detach(),
            timestep=(timestep / 1000).detach(),
            guidance=None,
            encoder_hidden_states=enc_hs.detach(),
            encoder_hidden_states_mask=prompt_embeds_mask.detach(),
            img_shapes=img_shapes,
            return_dict=False,
            **kwargs,
        )[0]

        noise_pred = noise_pred[:, : raw_packed_latents.size(1)]

        # unpack
        noise_pred = noise_pred.view(
            batch_size, height // 2, width // 2, num_channels_latents, 2, 2
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred
