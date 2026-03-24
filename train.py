#!/usr/bin/env python3
"""
Training script for HGVFI model on Vimeo90K dataset.
Trains on tri_trainlist.txt, validates on tri_testlist.txt.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from hgvfi import (
    HintGuidedVFI,
    CharbonnierLoss,
    load_vimeo_triplet,
    compress_frame_h264,
    downsample_hint,
    numpy_to_tensor,
    tensor_to_numpy,
    compute_metrics,
)


class Vimeo90KDataset(Dataset):
    """PyTorch Dataset for Vimeo90K triplets with augmentation"""

    def __init__(self, image_paths, crop_size=256, use_augmentation=True):
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im1_path, im2_path, im3_path = self.image_paths[idx]

        # Helper: load compressed frame (check cache first, then online compress)
        def _load_compressed(im_path):
            cached_path = im_path.replace('.png', '_h264.png')
            if os.path.exists(cached_path):
                return np.array(Image.open(cached_path)).astype(np.float32) / 255.0
            else:
                # Load original and compress online
                frame = np.array(Image.open(im_path)).astype(np.float32) / 255.0
                return compress_frame_h264(frame, crf=23)

        # Load images (from cache or online compression)
        frame0_c = _load_compressed(im1_path)
        frame_gt_c = _load_compressed(im2_path)
        frame2_c = _load_compressed(im3_path)

        # Create hint (4x downsample of compressed gt)
        hint = downsample_hint(frame_gt_c, factor=4)

        # Augmentation: random crop
        if self.use_augmentation and self.crop_size:
            h, w = frame0_c.shape[:2]
            if h > self.crop_size and w > self.crop_size:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
                frame0_c = frame0_c[y:y+self.crop_size, x:x+self.crop_size]
                frame_gt_c = frame_gt_c[y:y+self.crop_size, x:x+self.crop_size]
                frame2_c = frame2_c[y:y+self.crop_size, x:x+self.crop_size]
                # Hint crop: 4x smaller coordinates
                hint_y, hint_x = y // 4, x // 4
                hint_size = self.crop_size // 4
                hint = hint[hint_y:hint_y+hint_size, hint_x:hint_x+hint_size]

        # Augmentation: horizontal flip
        if self.use_augmentation and random.random() < 0.5:
            frame0_c = frame0_c[:, ::-1].copy()
            frame_gt_c = frame_gt_c[:, ::-1].copy()
            frame2_c = frame2_c[:, ::-1].copy()
            hint = hint[:, ::-1].copy()

        # Augmentation: temporal reverse (swap frame0 and frame2)
        if self.use_augmentation and random.random() < 0.5:
            frame0_c, frame2_c = frame2_c.copy(), frame0_c.copy()

        # Convert to tensors
        frame0_t = numpy_to_tensor(frame0_c).squeeze(0)
        frame2_t = numpy_to_tensor(frame2_c).squeeze(0)
        hint_t = numpy_to_tensor(hint).squeeze(0)
        frame_gt_t = numpy_to_tensor(frame_gt_c).squeeze(0)

        return {
            'frame0': frame0_t,
            'frame2': frame2_t,
            'hint': hint_t,
            'gt': frame_gt_t,
        }


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        frame0 = batch['frame0'].to(device)
        frame2 = batch['frame2'].to(device)
        hint = batch['hint'].to(device)
        gt = batch['gt'].to(device)

        # Forward pass
        output = model(frame0, frame2, hint)
        loss = loss_fn(output, gt)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")

    return total_loss / num_batches


def validate(model, dataset, device, num_samples=100):
    """Validate model"""
    model.eval()
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for idx, (im1_path, im2_path, im3_path) in enumerate(dataset[:num_samples]):
            if idx % 100 == 0 and idx > 0:
                print(f"  Validating: {idx}/{min(num_samples, len(dataset))}")

            # Load and prepare
            frame0 = np.array(Image.open(im1_path)).astype(np.float32) / 255.0
            frame_gt = np.array(Image.open(im2_path)).astype(np.float32) / 255.0
            frame2 = np.array(Image.open(im3_path)).astype(np.float32) / 255.0

            frame0_c = compress_frame_h264(frame0, crf=23)
            frame2_c = compress_frame_h264(frame2, crf=23)
            frame_gt_c = compress_frame_h264(frame_gt, crf=23)

            hint = downsample_hint(frame_gt_c, factor=4)

            f0_t = numpy_to_tensor(frame0_c).to(device)
            f2_t = numpy_to_tensor(frame2_c).to(device)
            hint_t = numpy_to_tensor(hint).to(device)

            # Forward pass
            output = model(f0_t, f2_t, hint_t)
            output_np = tensor_to_numpy(output)

            # Metrics
            psnr, ssim = compute_metrics(frame_gt_c, output_np)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    return avg_psnr, avg_ssim


def main():
    # TESTING: Set LIMIT=None for full training, or use 1/10/100 for quick tests
    LIMIT = None  # Change to 1, 10, 100, 1000 for testing

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    dataset_root = "/Users/rsaran/Projects/ivp/vimeo_triplet"
    train_list = os.path.join(dataset_root, "tri_trainlist.txt")
    test_list = os.path.join(dataset_root, "tri_testlist.txt")

    # Load dataset - CORRECT SPLIT
    print("Loading Vimeo90K training dataset...")
    train_paths = load_vimeo_triplet(dataset_root, train_list, limit=LIMIT)
    print(f"Loaded {len(train_paths)} training triplets")

    print("Loading Vimeo90K validation dataset...")
    val_paths = load_vimeo_triplet(dataset_root, test_list, limit=LIMIT)
    print(f"Loaded {len(val_paths)} validation triplets\n")

    # Create datasets with augmentation
    train_dataset = Vimeo90KDataset(train_paths, crop_size=256, use_augmentation=True)
    val_dataset = val_paths  # Raw paths for validation (no augmentation in validate())

    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    # Create model
    model = HintGuidedVFI().to(device)

    # Use AdamW + CosineAnnealingLR (proper training config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    loss_fn = CharbonnierLoss(eps=1e-6)

    print(f"Training with {len(train_paths)} samples")
    print(f"Validation with {len(val_paths)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Training loop - fewer epochs for testing
    num_epochs = 3 if LIMIT else 30
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(model, dataloader, optimizer, loss_fn, device)
        history['train_loss'].append(train_loss)
        print(f"  Average Loss: {train_loss:.6f}")

        # Step scheduler
        scheduler.step()

        # Validate (subset to speed up)
        num_val_samples = min(200, len(val_dataset))
        val_psnr, val_ssim = validate(model, val_dataset, device, num_samples=num_val_samples)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        print(f"  Val PSNR: {val_psnr:.2f} dB, Val SSIM: {val_ssim:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = f'/Users/rsaran/Projects/ivp/hgvfi_epoch_{epoch + 1}.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: hgvfi_epoch_{epoch + 1}.pt")

    # Save final model
    torch.save(model.state_dict(), '/Users/rsaran/Projects/ivp/hgvfi_model.pt')
    print("\n" + "="*50)
    print("Final model saved to hgvfi_model.pt")
    print("="*50)

    # Save history
    with open('/Users/rsaran/Projects/ivp/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining history saved to training_history.json")

    # Final results
    print("\nFinal Results:")
    print(f"  Best PSNR: {max(history['val_psnr']):.2f} dB")
    print(f"  Best SSIM: {max(history['val_ssim']):.4f}")


if __name__ == "__main__":
    main()
