#!/usr/bin/env python3
import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import get_dataloader, denormalize
from models import UNetGenerator, PatchGANDiscriminator, init_weights


def get_lr_scheduler(optimizer, start_decay_epoch: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < start_decay_epoch:
            return 1.0
        decay_epochs = max(1, total_epochs - start_decay_epoch)
        return max(0.0, 1.0 - (epoch - start_decay_epoch) / float(decay_epochs))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    net: nn.Module,
    optimizer: Adam,
    epoch: int,
    filepath: Path,
    loss: Optional[float] = None
):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(payload, filepath)


def load_checkpoint(filepath: Path, net: nn.Module, optimizer: Optional[Adam] = None):
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location="cpu")
    if "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0)
    else:
        net.load_state_dict(checkpoint)
        return 0


def generate_visual_samples(
    generator: nn.Module,
    val_loader,
    device: torch.device,
    save_path: Path,
    num_samples: int = 4
):
    generator.eval()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for sketches, reals in val_loader:
            sketches = sketches.to(device)
            reals = reals.to(device)

            sketches = sketches[:num_samples]
            reals = reals[:num_samples]

            fakes = generator(sketches)

            sketches_vis = denormalize(sketches)
            reals_vis = denormalize(reals)
            fakes_vis = denormalize(fakes)

            comparison = []
            for i in range(sketches.size(0)):
                comparison.extend([sketches_vis[i], reals_vis[i], fakes_vis[i]])

            save_image(comparison, save_path, nrow=3, padding=4, normalize=False)
            break

    generator.train()


def train(args):
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[SETUP] Using device: {device}")

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"[SETUP] GPU Model: {gpu_name} ({gpu_vram:.0f} MiB VRAM)")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    train_dir = Path(args.dataset_dir) / "train"
    val_dir = Path(args.dataset_dir) / "val"

    if not train_dir.exists():
        print(f"[ERROR] Training dataset not found at {train_dir}")
        print("Please run `python download_dataset.py` to download edges2shoes.")
        sys.exit(1)

    print(f"[SETUP] Loading dataset from: {train_dir}")
    train_loader = get_dataloader(
        dataset_dir=str(train_dir),
        batch_size=args.batch_size,
        is_train=True,
        img_size=args.resolution,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )

    val_loader = None
    if val_dir.exists():
        val_loader = get_dataloader(
            dataset_dir=str(val_dir),
            batch_size=args.batch_size,
            is_train=False,
            img_size=args.resolution,
            num_workers=args.num_workers,
            max_samples=16
        )

    netG = UNetGenerator(in_channels=3, out_channels=3).to(device)
    netD = PatchGANDiscriminator(in_channels=6).to(device)

    init_weights(netG, init_type="normal", init_gain=0.02)
    init_weights(netD, init_type="normal", init_gain=0.02)

    optimizer_G = Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    use_amp = use_cuda and args.amp
    scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[SETUP] Automatic Mixed Precision (AMP): {'ENABLED' if use_amp else 'DISABLED'}")

    scheduler_G = get_lr_scheduler(optimizer_G, args.decay_start_epoch, args.epochs)
    scheduler_D = get_lr_scheduler(optimizer_D, args.decay_start_epoch, args.epochs)

    ckpt_dir = Path(args.checkpoint_dir)
    sample_dir = Path(args.sample_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume:
        gen_latest = ckpt_dir / "generator_latest.pth"
        disc_latest = ckpt_dir / "discriminator_latest.pth"
        if gen_latest.exists() and disc_latest.exists():
            print(f"[SETUP] Resuming from checkpoint: {gen_latest}")
            start_epoch = load_checkpoint(gen_latest, netG, optimizer_G) + 1
            load_checkpoint(disc_latest, netD, optimizer_D)
            print(f"[SETUP] Resumed at epoch {start_epoch}")
        else:
            print("[WARN] Checkpoints not found to resume. Starting from scratch.")

    print("\n" + "=" * 60)
    print("      Sketch2Real: Pix2Pix GAN Training Starting")
    print(f"      Batch Size: {args.batch_size} | Resolution: {args.resolution}x{args.resolution}")
    print(f"      L1 Lambda: {args.l1_lambda} | Base LR: {args.lr}")
    print(f"      Epochs: {args.epochs} | AMP: {use_amp}")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        netG.train()
        netD.train()

        running_d_loss = 0.0
        running_g_loss = 0.0
        running_g_gan = 0.0
        running_g_l1 = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch:03d}/{args.epochs:03d}]",
            unit="batch",
            leave=True
        )

        for step, (sketches, reals) in enumerate(pbar):
            sketches = sketches.to(device, non_blocking=True)
            reals = reals.to(device, non_blocking=True)

            optimizer_D.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                fakes = netG(sketches)

                pred_real = netD(sketches, reals)
                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

                pred_fake = netD(sketches, fakes.detach())
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

                loss_D = 0.5 * (loss_D_real + loss_D_fake)

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            optimizer_G.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred_fake_for_G = netD(sketches, fakes)
                loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

                loss_G_L1 = criterion_L1(fakes, reals) * args.l1_lambda

                loss_G = loss_G_GAN + loss_G_L1

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            running_d_loss += loss_D.item()
            running_g_loss += loss_G.item()
            running_g_gan += loss_G_GAN.item()
            running_g_l1 += loss_G_L1.item()

            pbar.set_postfix({
                "G_loss": f"{loss_G.item():.4f}",
                "D_loss": f"{loss_D.item():.4f}",
                "G_L1": f"{loss_G_L1.item():.2f}",
                "G_GAN": f"{loss_G_GAN.item():.3f}"
            })

        scheduler_G.step()
        scheduler_D.step()

        num_batches = len(train_loader)
        avg_g_loss = running_g_loss / num_batches
        avg_d_loss = running_d_loss / num_batches
        print(f"\n[EPOCH {epoch:03d} SUMMARY] Avg G_Loss: {avg_g_loss:.4f} | Avg D_Loss: {avg_d_loss:.4f} | LR: {scheduler_G.get_last_lr()[0]:.6f}")

        save_checkpoint(netG, optimizer_G, epoch, ckpt_dir / "generator_latest.pth", avg_g_loss)
        save_checkpoint(netD, optimizer_D, epoch, ckpt_dir / "discriminator_latest.pth", avg_d_loss)

        if epoch % args.save_freq == 0:
            save_checkpoint(netG, optimizer_G, epoch, ckpt_dir / f"generator_epoch_{epoch:03d}.pth", avg_g_loss)

        if val_loader is not None and (epoch % args.sample_freq == 0 or epoch == 1):
            sample_img_path = sample_dir / f"epoch_{epoch:03d}.png"
            generate_visual_samples(netG, val_loader, device, sample_img_path)
            print(f"[SAMPLE] Saved visual comparison to: {sample_img_path}")

    print("\n[COMPLETE] Training successfully finished!")
    print(f"[COMPLETE] Latest generator checkpoint saved at: {ckpt_dir / 'generator_latest.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketch2Real: Pix2Pix GAN Training Script")

    parser.add_argument("--dataset_dir", type=str, default="datasets/edges2shoes", help="Path to edges2shoes dataset.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--sample_dir", type=str, default="samples", help="Directory to save preview images.")

    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size (2 recommended for 4GB VRAM).")
    parser.add_argument("--resolution", type=int, default=256, help="Square image resolution.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Initial Adam learning rate.")
    parser.add_argument("--l1_lambda", type=float, default=100.0, help="Weight for L1 pixel reconstruction loss.")
    parser.add_argument("--epochs", type=int, default=15, help="Total training epochs.")
    parser.add_argument("--decay_start_epoch", type=int, default=10, help="Epoch to start linear LR decay.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes.")

    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (saves VRAM).")
    parser.add_argument("--no_amp", action="store_false", dest="amp", help="Disable Automatic Mixed Precision.")
    parser.add_argument("--resume", action="store_true", help="Resume training from generator_latest.pth.")
    parser.add_argument("--save_freq", type=int, default=5, help="Save numbered epoch checkpoint frequency.")
    parser.add_argument("--sample_freq", type=int, default=1, help="Visual sample generation frequency in epochs.")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap training dataset size for rapid testing.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")

    args = parser.parse_args()
    train(args)
