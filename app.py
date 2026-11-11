import os
import io
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch

from models import UNetGenerator
from dataset import tensor_to_numpy_image


st.set_page_config(
    page_title="Sketch2Real | Universal Doodle Synthesizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #9C27B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-gpu { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
    .badge-cpu { background-color: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
    .badge-mode { background-color: #ede7f6; color: #512da8; border: 1px solid #d1c4e9; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_controlnet_engine():
    from controlnet_pipeline import UniversalControlNetSynthesizer
    return UniversalControlNetSynthesizer()


@st.cache_resource(show_spinner=False)
def load_pix2pix_generator(checkpoint_path: str, device_str: str) -> Tuple[Optional[UNetGenerator], str]:
    device = torch.device(device_str)
    model = UNetGenerator(in_channels=3, out_channels=3).to(device)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        return None, f"Checkpoint not found at '{checkpoint_path}'."

    try:
        checkpoint = torch.load(ckpt_file, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            epoch = checkpoint.get("epoch", "N/A")
            status = f"Loaded weights from epoch {epoch}"
        else:
            model.load_state_dict(checkpoint)
            status = "Loaded direct weights"
        model.eval()
        return model, status
    except Exception as e:
        return None, f"Error: {e}"


def preprocess_canvas_image(
    image_array: np.ndarray,
    target_size: int = 512,
    background_image: Optional[Image.Image] = None
) -> Tuple[Image.Image, torch.Tensor]:
    if image_array.shape[2] == 4:
        alpha = image_array[:, :, 3:] / 255.0
        rgb = image_array[:, :, :3].astype(np.float32)
        if background_image is not None:
            bg_resized = background_image.resize((image_array.shape[1], image_array.shape[0])).convert("RGB")
            bg_array = np.array(bg_resized, dtype=np.float32)
        else:
            bg_array = np.ones_like(rgb, dtype=np.float32) * 255.0
        blended = (alpha * rgb + (1.0 - alpha) * bg_array).astype(np.uint8)
    else:
        blended = image_array[:, :, :3].astype(np.uint8)

    pil_img = Image.fromarray(blended).convert("RGB")
    pil_resized = pil_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

    np_img = np.array(pil_resized.resize((256, 256)), dtype=np.float32) / 255.0
    norm_img = (np_img - 0.5) / 0.5
    tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()

    return pil_resized, tensor


def is_canvas_blank(image_array: Optional[np.ndarray], has_template: bool) -> bool:
    if has_template:
        return False
    if image_array is None:
        return True
    if image_array.shape[2] == 4:
        return bool((image_array[:, :, 3] > 0).sum() < 30)
    return bool(image_array.std() < 1.0)


def generate_synthetic_sketch(category: str) -> Image.Image:
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    cat = category.lower()

    if "cycle" in cat or "bicycle" in cat:
        cv2.circle(img, (140, 360), 75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(img, (370, 360), 75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (140, 360), (230, 260), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (230, 260), (330, 260), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (230, 260), (250, 360), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (250, 360), (370, 360), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (330, 260), (370, 360), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (330, 260), (340, 220), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (320, 220), (360, 220), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (230, 260), (220, 230), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (200, 230), (240, 230), (0, 0, 0), 6, cv2.LINE_AA)
    elif "pen" in cat:
        pts = np.array([[120, 380], [360, 140], [390, 170], [150, 410]], np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 0), 4, cv2.LINE_AA)
        nib = np.array([[120, 380], [80, 420], [150, 410]], np.int32)
        cv2.fillPoly(img, [nib], (0, 0, 0))
        cv2.line(img, (330, 150), (280, 200), (0, 0, 0), 4, cv2.LINE_AA)
    elif "tire" in cat:
        cv2.circle(img, (256, 256), 160, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.circle(img, (256, 256), 110, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.circle(img, (256, 256), 40, (0, 0, 0), 4, cv2.LINE_AA)
        for angle in range(0, 360, 30):
            rad = np.deg2rad(angle)
            x1, y1 = int(256 + 140 * np.cos(rad)), int(256 + 140 * np.sin(rad))
            x2, y2 = int(256 + 160 * np.cos(rad)), int(256 + 160 * np.sin(rad))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (256, 330), (200, 35), 0, 0, 180, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (56, 330), (456, 330), (0, 0, 0), 5, cv2.LINE_AA)
        pts = np.array([[70, 330], [120, 230], [240, 220], [340, 260], [450, 330]], np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (120, 230), (150, 160), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (150, 160), (240, 160), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (240, 160), (240, 220), (0, 0, 0), 5, cv2.LINE_AA)

    return Image.fromarray(img)


def main():
    st.markdown('<div class="main-title">🎨 Sketch2Real: Universal Doodle Synthesizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Draw <b>anything</b>—a bicycle, sneaker, car, cartoon character, or object—and '
        'generate a <b>Color Sketch</b>, <b>Cartoon illustration</b>, or <b>Realistic render</b> adhering strictly to your sketch without extra clutter.</div>',
        unsafe_allow_html=True
    )

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    device_str = "cuda" if has_cuda else "cpu"

    with st.sidebar:
        st.header("⚙️ Engine & Hardware")
        if has_cuda:
            st.markdown(f'<span class="status-badge badge-gpu">⚡ GPU: {device_name} (4GB VRAM)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-cpu">⚠️ Device: CPU</span>', unsafe_allow_html=True)

        st.markdown("---")
        mode = st.radio(
            "Select Synthesis Engine",
            [
                "🌟 Universal AI (ControlNet): Draw Anything",
                "👟 Local Pix2Pix GAN: Shoes (Ultra-Fast)"
            ],
            index=0
        )

        st.markdown("---")
        st.subheader("Canvas Settings")
        drawing_mode = st.selectbox("Tool", ("freedraw", "line", "rect", "transform"), index=0)
        stroke_width = st.slider("Stroke Thickness", min_value=2, max_value=24, value=5, step=1)
        stroke_color = st.color_picker("Stroke Color", "#000000")

        st.markdown("---")
        st.subheader("Doodle Templates")
        template_choice = st.selectbox("Quick Preset", ("None", "Bicycle", "Pen", "Tire", "Sneaker"))

    if "Universal AI" in mode:
        st.markdown('<span class="status-badge badge-mode">Mode: Universal ControlNet (Draw Anything)</span>', unsafe_allow_html=True)
        st.caption("Draw any object on the canvas, describe it in the prompt box, and AI will turn your sketch lines into a realistic image.")

        col_prompt, col_chips = st.columns([3, 2])
        with col_prompt:
            prompt_input = st.text_input(
                "Describe what you drew:",
                value="a modern bicycle",
                placeholder="e.g., a mountain bicycle, a red sports car, a luxury fountain pen, a cute cat..."
            )

        with col_chips:
            st.markdown("<small><b>Quick Category Prompts:</b></small>", unsafe_allow_html=True)
            chip_cols = st.columns(4)
            if chip_cols[0].button("🚲 Cycle"):
                prompt_input = "a racing bicycle"
            if chip_cols[1].button("🖊️ Pen"):
                prompt_input = "a luxury golden fountain pen"
            if chip_cols[2].button("🛞 Tire"):
                prompt_input = "a car tire"
            if chip_cols[3].button("👟 Sneaker"):
                prompt_input = "a stylish sneaker shoe"

        col_style, col_isolate = st.columns([3, 2])
        with col_style:
            style_choice = st.selectbox(
                "🎨 Artistic Style",
                (
                    "Color Sketch",
                    "Cartoon / 2D Animation",
                    "Watercolor Painting",
                    "Comic / Anime",
                    "Photorealistic (Isolated)",
                    "Custom / Prompt Only"
                ),
                index=0,
                help="Select 'Color Sketch' or 'Cartoon' to transform your doodle into colorful sketch or cartoon art as-is."
            )

        with col_isolate:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            isolate_subject = st.checkbox(
                "🎯 Keep sketch subject only (No extra objects/scenes)",
                value=True,
                help="Prevents the AI from adding background landscapes, extra objects, or clutter. Ensures the output strictly renders your doodle as-is."
            )

        with st.sidebar.expander("🛠️ ControlNet Parameters (Advanced)", expanded=False):
            steps = st.slider("Inference Steps", min_value=8, max_value=25, value=12, step=1, help="12 steps gives fast, high-quality results on 4GB VRAM.")
            conditioning_scale = st.slider("Sketch Adherence (ControlNet Scale)", min_value=0.5, max_value=1.5, value=1.10, step=0.05, help="Higher values (1.05 - 1.20) adhere strictly to your doodle lines so no extra parts or objects are added.")
            guidance_scale = st.slider("Prompt Creativity (CFG Scale)", min_value=4.0, max_value=12.0, value=7.5, step=0.5)
            seed = st.number_input("Random Seed (-1 for random)", value=-1, step=1)

        col_canvas, col_result = st.columns([1, 1], gap="large")

        initial_bg = None
        if template_choice != "None":
            initial_bg = generate_synthetic_sketch(template_choice)

        with col_canvas:
            st.markdown("### ✏️ 1. Draw Your Doodle")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#FFFFFF",
                background_image=initial_bg,
                update_streamlit=True,
                height=340,
                width=340,
                drawing_mode=drawing_mode,
                key="universal_canvas"
            )

            with st.expander("📁 Or Upload a Doodle Image"):
                uploaded_file = st.file_uploader("Upload sketch", type=["png", "jpg", "jpeg"], key="universal_uploader")

            btn_generate = st.button("🚀 Synthesize Artwork", type="primary", use_container_width=True)

        with col_result:
            st.markdown(f"### 🎨 2. Rendered Artwork ({style_choice})")

            pil_sketch = None
            if uploaded_file is not None:
                pil_sketch = Image.open(uploaded_file).convert("RGB").resize((512, 512))
            elif canvas_result.image_data is not None:
                pil_sketch, _ = preprocess_canvas_image(
                    canvas_result.image_data,
                    target_size=512,
                    background_image=initial_bg
                )

            if btn_generate:
                if uploaded_file is None and is_canvas_blank(canvas_result.image_data, template_choice != "None"):
                    st.warning("⚠️ The canvas is blank! Please draw a doodle or choose a preset template first.")
                elif pil_sketch is None:
                    st.warning("Please draw a sketch on the canvas first!")
                else:
                    with st.spinner(f"Generating {style_choice} with ControlNet (GTX 1650 FP32 NaN-safe mode)..."):
                        synthesizer = load_controlnet_engine()
                        seed_val = int(seed) if seed >= 0 else None
                        try:
                            output_image, scribble_map, latency = synthesizer.generate(
                                sketch_image=pil_sketch,
                                prompt=prompt_input,
                                style=style_choice,
                                isolate_subject=isolate_subject,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=conditioning_scale,
                                seed=seed_val
                            )
                        except Exception as e:
                            st.error(f"❌ Error during generation: {e}")
                            output_image = None

                    if output_image is not None:
                        st.image(output_image, caption=f"Generated Output ({style_choice} • {latency:.1f}s)", use_column_width=True)

                        buf = io.BytesIO()
                        output_image.save(buf, format="PNG")
                        st.download_button(
                            label="💾 Download Rendered Image",
                            data=buf.getvalue(),
                            file_name=f"sketch2real_{prompt_input[:20].replace(' ', '_')}.png",
                            mime="image/png",
                            use_container_width=True
                        )

                        with st.expander("🔍 View Processed Scribble Conditioning Map"):
                            st.image(scribble_map, caption="Normalized Scribble Input Seen by ControlNet", width=250)
            else:
                st.info("👈 Draw your doodle on the canvas, select **'Color Sketch'** or **'Cartoon'**, and click **'🚀 Synthesize Artwork'**.")

    else:
        st.markdown('<span class="status-badge badge-mode">Mode: Local Pix2Pix GAN (Trained on Berkeley Shoes)</span>', unsafe_allow_html=True)
        st.caption("Runs your custom 8-layer U-Net Pix2Pix model for instant shoe generation (~15ms).")

        ckpt_path = "checkpoints/generator_latest.pth"
        model_pix, msg_pix = load_pix2pix_generator(ckpt_path, device_str)

        col_canvas, col_result = st.columns([1, 1], gap="large")

        initial_bg_pix = None
        if template_choice != "None":
            initial_bg_pix = generate_synthetic_sketch("sneaker")

        with col_canvas:
            st.markdown("### ✏️ 1. Draw Shoe Doodle")
            canvas_pix = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#FFFFFF",
                background_image=initial_bg_pix,
                update_streamlit=True,
                height=300,
                width=300,
                drawing_mode=drawing_mode,
                key="pix2pix_canvas"
            )
            btn_pix = st.button("🚀 Render Shoe (Pix2Pix)", type="primary", use_container_width=True)

        with col_result:
            st.markdown("### 👟 2. Rendered Shoe Product")
            if btn_pix:
                if model_pix is None:
                    st.warning(f"Pix2Pix model checkpoint not ready: {msg_pix}. Train first using `python train.py`.")
                elif is_canvas_blank(canvas_pix.image_data, template_choice != "None"):
                    st.warning("⚠️ Please draw a shoe doodle on the canvas or pick a preset template first!")
                elif canvas_pix.image_data is not None:
                    _, tensor_in = preprocess_canvas_image(
                        canvas_pix.image_data,
                        target_size=256,
                        background_image=initial_bg_pix
                    )
                    tensor_in = tensor_in.to(torch.device(device_str))

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out_fake = model_pix(tensor_in)
                        out_np = tensor_to_numpy_image(out_fake)
                    dt_ms = (time.perf_counter() - t0) * 1000.0

                    st.image(out_np, caption=f"Pix2Pix Footwear Render ({dt_ms:.1f} ms)", use_column_width=True)


if __name__ == "__main__":
    main()
