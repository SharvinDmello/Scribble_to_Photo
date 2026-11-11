import os
import random
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class Edges2ShoesDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        img_size: int = 256,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.root_dir = Path(root_dir).resolve()
        self.is_train = is_train
        self.img_size = img_size

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory '{self.root_dir}' does not exist. "
                f"Please run `python download_dataset.py` first."
            )

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = sorted([
            p for p in self.root_dir.iterdir()
            if p.suffix.lower() in valid_extensions and p.is_file()
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No image files found in '{self.root_dir}'. "
                f"Ensure images are placed directly inside this folder."
            )

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_transforms(
        self,
        sketch: Image.Image,
        real: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_train:
            jitter_size = int(self.img_size * 1.12)
            sketch = TF.resize(sketch, [jitter_size, jitter_size], interpolation=TF.InterpolationMode.BICUBIC)
            real = TF.resize(real, [jitter_size, jitter_size], interpolation=TF.InterpolationMode.BICUBIC)

            i, j, h, w = transforms.RandomCrop.get_params(
                sketch, output_size=(self.img_size, self.img_size)
            )
            sketch = TF.crop(sketch, i, j, h, w)
            real = TF.crop(real, i, j, h, w)

            if random.random() > 0.5:
                sketch = TF.hflip(sketch)
                real = TF.hflip(real)
        else:
            sketch = TF.resize(sketch, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BICUBIC)
            real = TF.resize(real, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BICUBIC)

        sketch_tensor = TF.to_tensor(sketch)
        real_tensor = TF.to_tensor(real)

        sketch_tensor = TF.normalize(sketch_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        real_tensor = TF.normalize(real_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return sketch_tensor, real_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as combined_img:
            combined_img = combined_img.convert("RGB")
            w, h = combined_img.size

            midpoint = w // 2
            sketch_img = combined_img.crop((0, 0, midpoint, h))
            real_img = combined_img.crop((midpoint, 0, w, h))

        sketch_tensor, real_tensor = self._apply_transforms(sketch_img, real_img)
        return sketch_tensor, real_tensor


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor * 0.5) + 0.5, 0.0, 1.0)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    denorm = denormalize(tensor)
    if denorm.dim() == 4:
        denorm = denorm[0]
    np_img = denorm.detach().cpu().permute(1, 2, 0).numpy()
    np_img = (np_img * 255.0).round().astype(np.uint8)
    return np_img


def get_dataloader(
    dataset_dir: str,
    batch_size: int = 2,
    is_train: bool = True,
    img_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
    max_samples: Optional[int] = None
) -> DataLoader:
    dataset = Edges2ShoesDataset(
        root_dir=dataset_dir,
        is_train=is_train,
        img_size=img_size,
        max_samples=max_samples
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=is_train
    )
    return loader


if __name__ == "__main__":
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/edges2shoes/val"
    if os.path.exists(test_dir):
        print(f"[TEST] Loading dataset from: {test_dir}")
        loader = get_dataloader(test_dir, batch_size=2, is_train=False, max_samples=4)
        for i, (sketches, reals) in enumerate(loader):
            print(f"Batch {i}: Sketches shape={sketches.shape}, dtype={sketches.dtype}, "
                  f"min={sketches.min():.2f}, max={sketches.max():.2f}")
            print(f"Batch {i}: Reals shape={reals.shape}, dtype={reals.dtype}, "
                  f"min={reals.min():.2f}, max={reals.max():.2f}")
            break
        print("[TEST] Dataset loaded successfully!")
    else:
        print(f"[INFO] Dataset directory '{test_dir}' not found yet. Run `python download_dataset.py` to fetch data.")
