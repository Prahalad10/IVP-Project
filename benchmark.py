#!/usr/bin/env python3
"""
HGVFI Benchmark: Comprehensive comparison of interpolation methods
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from hgvfi import HintGuidedVFI, load_vimeo_triplet, compress_frame_h264, downsample_hint
from hgvfi import numpy_to_tensor, tensor_to_numpy, compute_metrics


class SimpleLinearInterpolation:
    """Baseline: Linear interpolation between two frames"""

    def __call__(self, frame0, frame2):
        return (frame0 + frame2) / 2.0


class BicubicInterpolation:
    """Baseline: Actual bicubic interpolation using PyTorch"""

    def __call__(self, frame0, frame2):
        # Blend first
        avg = (frame0 + frame2) / 2.0

        # Convert to tensor for bicubic processing
        t = torch.from_numpy(avg.transpose(2, 0, 1)).unsqueeze(0).float()
        h, w = t.shape[2:]

        # Downsample then upsample via bicubic (simulates compression artifacts)
        down = F.interpolate(t, scale_factor=0.5, mode='bicubic', align_corners=False)
        up = F.interpolate(down, size=(h, w), mode='bicubic', align_corners=False)

        return up.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


class BicubicHintInterpolation:
    """Baseline: Bicubic 4x upsample of the hint frame (key comparison)"""

    def __call__(self, hint):
        """
        Args:
            hint: hint frame at 1/4 resolution
        Returns:
            upsampled hint at full resolution
        """
        t = torch.from_numpy(hint.transpose(2, 0, 1)).unsqueeze(0).float()

        # 4x bicubic upsample to recover full resolution
        up = F.interpolate(t, scale_factor=4, mode='bicubic', align_corners=False)

        return up.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


def evaluate_baseline(method_name, dataset, num_samples=100):
    """Evaluate baseline methods"""
    psnr_scores = []
    ssim_scores = []

    if method_name == "Linear":
        interpolator = SimpleLinearInterpolation()
    elif method_name == "Bicubic":
        interpolator = BicubicInterpolation()
    elif method_name == "BicubicHint":
        interpolator = BicubicHintInterpolation()
    else:
        raise ValueError(f"Unknown baseline method: {method_name}")

    print(f"\nEvaluating {method_name} Interpolation...")

    for idx, (im1_path, im2_path, im3_path) in enumerate(dataset[:num_samples]):
        # Load images
        frame0 = np.array(Image.open(im1_path)).astype(np.float32) / 255.0
        frame_gt = np.array(Image.open(im2_path)).astype(np.float32) / 255.0
        frame2 = np.array(Image.open(im3_path)).astype(np.float32) / 255.0

        # Compress
        frame0_compressed = compress_frame_h264(frame0, crf=23)
        frame2_compressed = compress_frame_h264(frame2, crf=23)
        frame_gt_compressed = compress_frame_h264(frame_gt, crf=23)

        # Interpolate
        if method_name == "BicubicHint":
            # For BicubicHint, we need the hint frame
            hint = downsample_hint(frame_gt_compressed, factor=4)
            output = interpolator(hint)
        else:
            output = interpolator(frame0_compressed, frame2_compressed)

        # Compute metrics
        psnr, ssim = compute_metrics(frame_gt_compressed, output)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{num_samples}] PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)

    return {
        'psnr_mean': avg_psnr,
        'psnr_std': std_psnr,
        'ssim_mean': avg_ssim,
        'ssim_std': std_ssim,
        'psnr_scores': psnr_scores,
        'ssim_scores': ssim_scores,
    }


def evaluate_neural(model, dataset, device, num_samples=100):
    """Evaluate neural VFI model"""
    model.eval()
    psnr_scores = []
    ssim_scores = []

    print(f"\nEvaluating HGVFI Model...")

    with torch.no_grad():
        for idx, (im1_path, im2_path, im3_path) in enumerate(dataset[:num_samples]):
            # Load images
            frame0 = np.array(Image.open(im1_path)).astype(np.float32) / 255.0
            frame_gt = np.array(Image.open(im2_path)).astype(np.float32) / 255.0
            frame2 = np.array(Image.open(im3_path)).astype(np.float32) / 255.0

            # Compress
            frame0_compressed = compress_frame_h264(frame0, crf=23)
            frame2_compressed = compress_frame_h264(frame2, crf=23)
            frame_gt_compressed = compress_frame_h264(frame_gt, crf=23)

            # Create hint
            hint = downsample_hint(frame_gt_compressed, factor=4)

            # Convert to tensors
            f0_tensor = numpy_to_tensor(frame0_compressed).to(device)
            f2_tensor = numpy_to_tensor(frame2_compressed).to(device)
            hint_tensor = numpy_to_tensor(hint).to(device)

            # Interpolate
            output = model(f0_tensor, f2_tensor, hint_tensor)
            output_np = tensor_to_numpy(output)

            # Compute metrics
            psnr, ssim = compute_metrics(frame_gt_compressed, output_np)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            if (idx + 1) % 50 == 0:
                print(f"  [{idx + 1}/{num_samples}] PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    std_psnr = np.std(psnr_scores)
    std_ssim = np.std(ssim_scores)

    return {
        'psnr_mean': avg_psnr,
        'psnr_std': std_psnr,
        'ssim_mean': avg_ssim,
        'ssim_std': std_ssim,
        'psnr_scores': psnr_scores,
        'ssim_scores': ssim_scores,
    }


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_root = "/Users/rsaran/Projects/ivp/vimeo_triplet"
    test_list = os.path.join(dataset_root, "tri_testlist.txt")

    # Load dataset
    print("Loading Vimeo90K test dataset...")
    dataset = load_vimeo_triplet(dataset_root, test_list, limit=None)
    print(f"Loaded {len(dataset)} triplets")

    # Load trained HGVFI model
    model = HintGuidedVFI().to(device)
    if os.path.exists('/Users/rsaran/Projects/ivp/hgvfi_model.pt'):
        model.load_state_dict(torch.load('/Users/rsaran/Projects/ivp/hgvfi_model.pt', map_location=device))
        print("Loaded pre-trained model")
    else:
        print("Using untrained model (no checkpoint found)")

    # Evaluate all methods
    results = {}
    num_samples = 200

    print("\n" + "="*70)
    print("BASELINE METHODS")
    print("="*70)

    for method in ["Linear", "Bicubic", "BicubicHint"]:
        results[method] = evaluate_baseline(method, dataset, num_samples=num_samples)

    print("\n" + "="*70)
    print("NEURAL METHODS")
    print("="*70)

    results["HGVFI"] = evaluate_neural(model, dataset, device, num_samples=num_samples)

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Method':<15} {'PSNR (dB)':<20} {'SSIM':<20}")
    print("-" * 70)

    for method, metrics in results.items():
        psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}"
        ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}"
        print(f"{method:<15} {psnr_str:<20} {ssim_str:<20}")

    print("="*70)

    # Calculate improvements vs Linear
    linear_psnr = results["Linear"]['psnr_mean']
    linear_ssim = results["Linear"]['ssim_mean']

    print(f"\nImprovements vs Linear Interpolation:")
    for method in ["BicubicHint", "HGVFI"]:
        if method in results:
            method_psnr = results[method]['psnr_mean']
            method_ssim = results[method]['ssim_mean']
            psnr_gain = method_psnr - linear_psnr
            ssim_gain = (method_ssim - linear_ssim) / linear_ssim * 100
            print(f"  {method}: PSNR +{psnr_gain:.2f} dB, SSIM +{ssim_gain:.1f}%")

    # Save detailed results
    results_to_save = {}
    for method, metrics in results.items():
        results_to_save[method] = {
            'psnr_mean': float(metrics['psnr_mean']),
            'psnr_std': float(metrics['psnr_std']),
            'ssim_mean': float(metrics['ssim_mean']),
            'ssim_std': float(metrics['ssim_std']),
        }

    with open('/Users/rsaran/Projects/ivp/benchmark_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print("\nBenchmark results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
