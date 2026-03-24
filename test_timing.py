#!/usr/bin/env python3
"""
Quick timing test: Train for 30 epochs with 10 samples
"""

import os
import json
import time
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

        def _load_compressed(im_path):
            cached_path = im_path.replace('.png', '_h264.png')
            if os.path.exists(cached_path):
                return np.array(Image.open(cached_path)).astype(np.float32) / 255.0
            else:
                frame = np.array(Image.open(im_path)).astype(np.float32) / 255.0
                return compress_frame_h264(frame, crf=23)

        frame0_c = _load_compressed(im1_path)
        frame_gt_c = _load_compressed(im2_path)
        frame2_c = _load_compressed(im3_path)

        hint = downsample_hint(frame_gt_c, factor=4)

        if self.use_augmentation and self.crop_size:
            h, w = frame0_c.shape[:2]
            if h > self.crop_size and w > self.crop_size:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
                frame0_c = frame0_c[y:y+self.crop_size, x:x+self.crop_size]
                frame_gt_c = frame_gt_c[y:y+self.crop_size, x:x+self.crop_size]
                frame2_c = frame2_c[y:y+self.crop_size, x:x+self.crop_size]
                hint_y, hint_x = y // 4, x // 4
                hint_size = self.crop_size // 4
                hint = hint[hint_y:hint_y+hint_size, hint_x:hint_x+hint_size]

        if self.use_augmentation and random.random() < 0.5:
            frame0_c = frame0_c[:, ::-1].copy()
            frame_gt_c = frame_gt_c[:, ::-1].copy()
            frame2_c = frame2_c[:, ::-1].copy()
            hint = hint[:, ::-1].copy()

        if self.use_augmentation and random.random() < 0.5:
            frame0_c, frame2_c = frame2_c.copy(), frame0_c.copy()

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

        output = model(frame0, frame2, hint)
        loss = loss_fn(output, gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    LIMIT = 1000  # Use 1000 samples for timing test
    NUM_EPOCHS = 30

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    dataset_root = "/Users/rsaran/Projects/ivp/vimeo_triplet"
    train_list = os.path.join(dataset_root, "tri_trainlist.txt")

    print("Loading Vimeo90K training dataset (10 samples)...")
    train_paths = load_vimeo_triplet(dataset_root, train_list, limit=LIMIT)
    print(f"Loaded {len(train_paths)} training triplets\n")

    train_dataset = Vimeo90KDataset(train_paths, crop_size=256, use_augmentation=True)

    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for timing accuracy
        persistent_workers=False
    )

    model = HintGuidedVFI().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    loss_fn = CharbonnierLoss(eps=1e-6)

    print(f"Training config:")
    print(f"  Samples: {len(train_paths)}")
    print(f"  Batch size: 4")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        epoch_start = time.time()
        train_loss = train_epoch(model, dataloader, optimizer, loss_fn, device)
        epoch_time = time.time() - epoch_start

        scheduler.step()

        print(f"  Loss: {train_loss:.6f}, Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("TIMING RESULTS")
    print("="*60)
    print(f"Total time for 30 epochs with 10 samples: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average time per epoch: {total_time/NUM_EPOCHS:.2f}s")

    # Scaling calculation
    samples_full = 51312  # Vimeo90K training set size
    time_per_sample = total_time / (LIMIT * NUM_EPOCHS)
    estimated_full_time = time_per_sample * samples_full * NUM_EPOCHS

    print("\n" + "="*60)
    print("EXTRAPOLATION TO FULL DATASET")
    print("="*60)
    print(f"Time per sample per epoch: {time_per_sample:.6f}s")
    print(f"Estimated time for 30 epochs with {samples_full} samples:")
    print(f"  {estimated_full_time:.0f} seconds")
    print(f"  {estimated_full_time/60:.1f} minutes")
    print(f"  {estimated_full_time/3600:.1f} hours")

    torch.save(model.state_dict(), '/Users/rsaran/Projects/ivp/test_model.pt')
    print("\nTest model saved.")


if __name__ == "__main__":
    main()
