"""
controlnet_pipeline.py
Universal Sketch-to-Realistic Image Synthesizer powered by ControlNet (Scribble)
and Stable Diffusion 1.5, optimized for 4GB VRAM GPUs (NVIDIA GTX 1650 / 1660).
"""

import time
import threading
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import torch

_INFERENCE_LOCK = threading.Lock()

try:
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


STYLE_PRESETS = {
    "Color Sketch": {
        "label": "🎨 Color Sketch (Colored Pencil / Marker)",
        "prompt_prefix": "vibrant colored sketch of",
        "prompt_suffix": "detailed colored pencil sketch, artistic colored line drawing, rich vivid colors, clean white background, single subject",
        "negative_prompt": "photograph, photorealistic, 3d render, extra objects, duplicate, multiple objects, split image, collage, cluttered background, complex scenery, messy lines, watermark, text"
    },
    "Cartoon / 2D Animation": {
        "label": "🎬 Cartoon / 2D Animation",
        "prompt_prefix": "cute 2d cartoon illustration of",
        "prompt_suffix": "clean bold outlines, vibrant flat colors, smooth vector cartoon art, solid white background, single subject, high quality animated style",
        "negative_prompt": "photograph, photorealistic, realistic 3d, extra objects, duplicate, multiple objects, collage, split view, cluttered background, noisy, grainy, watermark, text"
    },
    "Watercolor Painting": {
        "label": "🖌️ Watercolor Painting",
        "prompt_prefix": "soft watercolor painting of",
        "prompt_suffix": "colorful watercolor sketch, elegant ink outlines, clean white paper background, single subject, artistic masterpiece",
        "negative_prompt": "photograph, 3d render, photorealistic, extra objects, multiple items, duplicate, complex background, text, watermark"
    },
    "Comic / Anime": {
        "label": "✏️ Comic / Anime Art",
        "prompt_prefix": "anime comic art style illustration of",
        "prompt_suffix": "bold ink lines, vibrant cel shading, dynamic anime art, clean plain background, single subject",
        "negative_prompt": "photograph, realistic photo, extra objects, duplicate, multiple objects, collage, split screen, messy background, text"
    },
    "Photorealistic (Isolated)": {
        "label": "📸 Photorealistic (Isolated)",
        "prompt_prefix": "professional studio photograph of",
        "prompt_suffix": "centered, single object, isolated on solid white studio background, 8k resolution, sharp focus, clean lighting",
        "negative_prompt": "cartoon, sketch, drawing, extra objects, duplicate, multiple objects, collage, split screen, street background, outdoor clutter, bad proportions"
    },
    "Custom / Prompt Only": {
        "label": "🪄 Custom / Prompt Only",
        "prompt_prefix": "",
        "prompt_suffix": "",
        "negative_prompt": "low quality, blurry, distorted, extra objects, multiple items, duplicate, collage, watermark"
    }
}


def build_styled_prompt(
    prompt: str,
    style_key: str = "Color Sketch",
    isolate_subject: bool = True
) -> Tuple[str, str]:
    preset = None
    for k, v in STYLE_PRESETS.items():
        if k.lower() in style_key.lower() or style_key.lower() in k.lower() or v.get("label", "").lower() == style_key.lower():
            preset = v
            break
    if preset is None:
        preset = STYLE_PRESETS["Color Sketch"]

    clean_p = prompt.strip()

    if "custom" in style_key.lower():
        styled_prompt = clean_p
    else:
        parts = []
        if preset["prompt_prefix"]:
            parts.append(f"{preset['prompt_prefix']} {clean_p}")
        else:
            parts.append(clean_p)
        if preset["prompt_suffix"]:
            parts.append(preset["prompt_suffix"])
        styled_prompt = ", ".join(parts)

    if isolate_subject and "custom" not in style_key.lower():
        styled_prompt += ", centered, single isolated subject, clean plain white background, no extra items, uncluttered"

    neg_parts = [preset["negative_prompt"]]
    if isolate_subject:
        neg_parts.append("extra objects, duplicate, multiple objects, split image, collage, cluttered background, complex scenery, people, landscape, out of frame")

    all_neg = []
    seen = set()
    for segment in ", ".join(neg_parts).split(","):
        seg = segment.strip().lower()
        if seg and seg not in seen:
            seen.add(seg)
            all_neg.append(seg)
    full_negative = ", ".join(all_neg)

    return styled_prompt, full_negative


def preprocess_sketch_for_controlnet(
    image: Image.Image,
    target_size: int = 512,
    invert: bool = True
) -> Image.Image:
    gray = image.convert("L")
    gray = gray.resize((target_size, target_size), Image.Resampling.BICUBIC)

    if invert:
        np_gray = np.array(gray)
        if np_gray.mean() > 127:
            gray = ImageOps.invert(gray)

    np_gray = np.array(gray)
    np_gray[np_gray < 15] = 0
    np_gray[np_gray >= 15] = np.clip(np_gray[np_gray >= 15] * 1.5, 0, 255).astype(np.uint8)

    return Image.fromarray(np_gray).convert("RGB")


class UniversalControlNetSynthesizer:
    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-scribble"
    ):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers and transformers must be installed.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        needs_fp32 = False
        if self.device == "cuda":
            dev_name = torch.cuda.get_device_name(0).lower()
            if ("1650" in dev_name or "1660" in dev_name) and "rtx" not in dev_name:
                needs_fp32 = True

        self.dtype = torch.float32 if (self.device == "cpu" or needs_fp32) else torch.float16

        print(f"[CONTROLNET] Initializing ControlNet Scribble on device: {self.device} (dtype: {self.dtype})...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=self.dtype
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None
        )
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

        self.scheduler_config = dict(self.pipe.scheduler.config)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.scheduler_config)

        if self.device == "cuda":
            if needs_fp32:
                self.pipe.enable_sequential_cpu_offload()
            else:
                self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()
            torch.cuda.empty_cache()

        print(f"[CONTROLNET] Pipeline successfully loaded and memory-optimized (needs_fp32={needs_fp32}).")

    def generate(
        self,
        sketch_image: Image.Image,
        prompt: str,
        style: str = "Color Sketch",
        isolate_subject: bool = True,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 12,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.10,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image, float]:
        start_time = time.perf_counter()

        scribble_img = preprocess_sketch_for_controlnet(sketch_image, target_size=512, invert=True)

        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        styled_prompt, default_neg = build_styled_prompt(
            prompt=prompt,
            style_key=style,
            isolate_subject=isolate_subject
        )
        final_negative = negative_prompt if (negative_prompt and negative_prompt.strip()) else default_neg

        print(f"[CONTROLNET] Generating with Style='{style}', Isolate={isolate_subject}, CondScale={controlnet_conditioning_scale}")
        print(f"[CONTROLNET] Final Prompt: '{styled_prompt}'")

        with _INFERENCE_LOCK:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.scheduler_config)

            with torch.inference_mode():
                result = self.pipe(
                    prompt=styled_prompt,
                    negative_prompt=final_negative,
                    image=scribble_img,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator
                )

        elapsed = time.perf_counter() - start_time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result.images[0], scribble_img, elapsed
