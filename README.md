# 🎨 Sketch2Real: Universal Doodle-to-Artwork Synthesizer

Convert **any hand-drawn doodle**—bicycles, cars, sneakers, pens, watches, animals, or characters—into stylized **Color Sketches**, **Cartoons**, **Watercolor Paintings**, or **Photorealistic Images** adhering strictly to your sketch lines without hallucinating extra objects or background clutter.

Engineered and optimized from the ground up for consumer **4GB VRAM GPUs (NVIDIA GeForce GTX 1650 / 1660 series)** as well as higher-end cards.

---

## 🌟 Key Highlights & Features

### 1. 🎨 Universal ControlNet Synthesizer (Draw Anything)
- **Direct Doodle Conditioning:** Powered by `lllyasviel/sd-controlnet-scribble` and `runwayml/stable-diffusion-v1-5`.
- **Multiple Artistic Styles:**
  - 🎨 **Color Sketch (Default):** Vibrant colored pencil / marker sketch with artistic lineart on clean paper.
  - 🎬 **Cartoon / 2D Animation:** Clean bold outlines, vibrant flat colors, and smooth 2D animated vector art.
  - 🖌️ **Watercolor Painting:** Soft watercolor splashes with delicate ink outlines.
  - ✏️ **Comic / Anime Art:** Dynamic cel-shaded anime illustration.
  - 📸 **Photorealistic (Isolated):** Clean studio photography with crisp lighting.
  - 🪄 **Custom / Prompt Only:** Raw user prompt without automatic styling.
- **Strict Sketch Adherence ("Generate As It Is"):** ControlNet conditioning scale defaults to `1.10` to preserve the user's sketch contours without morphing shapes.
- **Subject Isolation ("No Extra Images"):** Dedicated toggle that pins generation to a single centered subject on a plain background, actively suppressing duplicate objects, collage views, or background clutter.

### 2. ⚡ Local Pix2Pix Footwear GAN Mode
- **Instant Inference (~15ms):** Custom 8-layer U-Net Generator with skip connections and 70×70 PatchGAN Discriminator trained locally on the Berkeley `edges2shoes` dataset.

---

## ⚙️ Hardware Engineering for 4GB VRAM (GTX 1650 / 1660)

Running Stable Diffusion and ControlNet on a 4GB GPU like the NVIDIA GeForce GTX 1650 poses two major technical hurdles:

| Challenge | Problem | Sketch2Real Solution |
| :--- | :--- | :--- |
| **FP16 NaN Black Images** | GTX 1650 (Turing TU117) lacks Tensor Cores. IEEE half-precision (`torch.float16`) overflows in UNet/attention, turning latents into `NaN` (resulting in a solid black image). | Automatic GPU detection switches GTX 1650/1660 cards to **`torch.float32` full precision**, eliminating `NaN` latents completely. |
| **4GB VRAM Limit & OOM** | Running full FP32 precision with standard model offloading exceeds the 3.63 GiB usable VRAM, causing `CUDA out of memory`. | Uses **`pipe.enable_sequential_cpu_offload()`** and attention slicing to stream model layers to the GPU on demand. Peak VRAM remains **under 1.5 GB**. |
| **Scheduler State Corruption** | Streamlit multi-threading or rapid double-clicking corrupts `UniPCMultistepScheduler` internal buffers (`AssertionError: assert self.this_order > 0`). | Global **`_INFERENCE_LOCK`** serializes execution, and the scheduler is cleanly re-instantiated fresh from configuration on every run. |

---

## 📁 Project Structure

```text
.
├── app.py                      # Streamlit interactive UI (Drawing canvas, styles, parameters)
├── controlnet_pipeline.py      # ControlNet Scribble engine optimized for GTX 1650 (FP32, sequential offload)
├── dataset.py                  # PyTorch Dataset & DataLoader for paired Pix2Pix data
├── download_dataset.py         # Resilient dataset downloader with resume & progress tracking
├── train.py                    # Pix2Pix GAN training script with Automatic Mixed Precision (AMP)
├── models/
│   ├── __init__.py             # Model exports
│   ├── generator.py            # 8-layer U-Net Generator with skip connections
│   └── discriminator.py        # 70x70 PatchGAN Discriminator
├── checkpoints/                # Saved weights (generator_latest.pth)
├── samples/                    # Visual progress snapshots from training
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules for checkpoints, datasets, and cache
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10+ and NVIDIA drivers installed:

```bash
# Clone the repository
git clone <repo-url>
cd "GenAI project"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Interface

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser:
1. **Draw** any doodle on the canvas (or choose a quick preset like *Bicycle*, *Pen*, *Tire*, or *Sneaker*).
2. **Select Style:** Pick **`🎨 Color Sketch`** or **`🎬 Cartoon / 2D Animation`**.
3. **Describe Subject:** Type a concise description (e.g. `a bicycle`, `a sports car`, `a cute cat`).
4. Click **`🚀 Synthesize Artwork`**.

---

## 👟 Training Pix2Pix Locally (Optional)

To train or fine-tune the custom 8-layer U-Net on Berkeley's `edges2shoes` dataset:

```bash
# 1. Download and extract the dataset (~2.1 GB)
python download_dataset.py

# 2. Start training with memory optimizations (Batch Size: 2 for 4GB VRAM)
python train.py --batch_size 2 --epochs 15 --amp
```

Checkpoints will be saved automatically to `checkpoints/generator_latest.pth`, and preview comparisons will be written to `samples/`.

---

## 🛠️ Controls & Parameters

| Setting | Recommended Value | Description |
| :--- | :--- | :--- |
| **Artistic Style** | `Color Sketch` / `Cartoon` | Defines the aesthetic rendering mode. |
| **Keep Sketch Subject Only** | `Enabled` (Checked) | Restricts generation to the drawn subject on a plain background, preventing extraneous clutter or secondary objects. |
| **Sketch Adherence** | `1.05 - 1.20` | How strictly the AI adheres to doodle stroke positions and contours. |
| **Inference Steps** | `10 - 14` | The UniPC scheduler converges in 10–14 steps (~25s on GTX 1650). |
| **Prompt Creativity (CFG)**| `7.0 - 8.0` | Balances prompt adherence against doodle geometry fidelity. |

---

## 📄 License
