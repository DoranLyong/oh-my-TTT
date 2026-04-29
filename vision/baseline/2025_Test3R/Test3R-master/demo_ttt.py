#!/usr/bin/env python3
"""
Test3R TTT (Test-Time Training) Demo Script
Usage: python demo_ttt.py --images demo/data/1.png demo/data/2.png demo/data/3.png
"""

import os
import sys
import argparse
import torch
import numpy as np

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference, inference_ttt
from dust3r.image_pairs import make_pairs, make_pairs_tri
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

torch.backends.cuda.matmul.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser("Test3R TTT Demo", add_help=True)
    parser.add_argument(
        "--images",
        nargs='+',
        default=["demo/data/1.png", "demo/data/2.png", "demo/data/3.png"],
        help="Input image paths (minimum 3 images for TTT)"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--model_path", type=str, default="checkpoints/dust3r_vit_large",
                        help="Path to model checkpoint directory or HuggingFace model name")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224])
    parser.add_argument("--output_dir", type=str, default="./demo_output", help="Output directory")

    # TTT parameters
    parser.add_argument("--epoches", type=int, default=1, help="TTT epochs")
    parser.add_argument("--lr", type=float, default=0.00001, help="TTT learning rate")
    parser.add_argument("--accum_iter", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--prompt_size", type=int, default=32, help="Prompt size for TTT")

    # Comparison mode
    parser.add_argument("--compare", action="store_true", help="Compare with/without TTT")

    return parser


def run_inference(model, imgs, device, use_ttt=False, args=None):
    """Run inference with or without TTT"""

    if use_ttt and hasattr(model, 'use_prompt') and model.use_prompt:
        print("\n>> Running TTT (Test-Time Training)...")
        pairs_tri, idx_tri = make_pairs_tri(imgs, prefilter=None, symmetrize=True)
        print(f"   Created {len(pairs_tri)} triplet pairs for TTT")

        model = inference_ttt(
            pairs_tri, idx_tri, model, device,
            batch_size=1,
            epoches=args.epoches,
            lr=args.lr,
            accum_iter=args.accum_iter,
            verbose=True
        )
        model.eval()
        print(">> TTT completed!")

    # Standard inference
    print("\n>> Running inference...")
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f"   Created {len(pairs)} pairs for inference")

    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=True)

    # Global alignment
    print("\n>> Running global alignment...")
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=True)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)
        print(f"   Final alignment loss: {loss}")

    return scene, model


def save_pointcloud(scene, output_path, name="pointcloud"):
    """Save point cloud as PLY file"""
    try:
        import open3d as o3d

        pts3d = scene.get_pts3d()
        imgs = scene.imgs
        masks = scene.get_masks()

        pts_all = []
        colors_all = []

        for i, (pts, img, mask) in enumerate(zip(pts3d, imgs, masks)):
            pts_np = pts.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            pts_masked = pts_np[mask_np]
            colors_masked = img[mask_np]

            pts_all.append(pts_masked)
            colors_all.append(colors_masked)

        pts_all = np.concatenate(pts_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_all.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors_all.reshape(-1, 3))

        ply_path = os.path.join(output_path, f"{name}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f">> Saved point cloud to {ply_path}")
        return True

    except ImportError:
        print(">> Warning: open3d not installed, skipping point cloud save")
        return False


def main(args):
    # Validate inputs
    if len(args.images) < 3:
        print("Warning: TTT requires at least 3 images. Adding duplicate if needed.")
        while len(args.images) < 3:
            args.images.append(args.images[-1])

    # Check if images exist
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Error: Image not found: {img_path}")
            sys.exit(1)

    print(f"=" * 60)
    print(f"Test3R TTT Demo")
    print(f"=" * 60)
    print(f"Images: {args.images}")
    print(f"Device: {args.device}")
    print(f"Model path: {args.model_path}")
    print(f"TTT params: epoches={args.epoches}, lr={args.lr}, prompt_size={args.prompt_size}")
    print(f"=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model with TTT support
    print("\n>> Loading model with TTT support...")
    model = AsymmetricCroCo3DStereo.from_pretrained(
        args.model_path,
        use_prompt=True,
        prompt_size=args.prompt_size
    ).to(args.device)
    print(f"   Model loaded from: {args.model_path}")
    print(f"   use_prompt: {model.use_prompt}")

    # Load images
    print(f"\n>> Loading {len(args.images)} images...")
    imgs = load_images(args.images, size=args.image_size, verbose=True)
    for i, img in enumerate(imgs):
        img['idx'] = i
    print(f"   Loaded {len(imgs)} images")

    if args.compare:
        # Run without TTT first
        print("\n" + "=" * 60)
        print("Running WITHOUT TTT (baseline DUSt3R)")
        print("=" * 60)

        # Need to reload model without prompt for fair comparison
        model_baseline = AsymmetricCroCo3DStereo.from_pretrained(
            args.model_path,
            use_prompt=False
        ).to(args.device)
        model_baseline.eval()

        scene_baseline, _ = run_inference(model_baseline, imgs, args.device, use_ttt=False, args=args)
        save_pointcloud(scene_baseline, args.output_dir, name="pointcloud_baseline")

        del model_baseline
        torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print("Running WITH TTT (Test3R)")
        print("=" * 60)

    # Run with TTT
    scene_ttt, model = run_inference(model, imgs, args.device, use_ttt=True, args=args)
    save_pointcloud(scene_ttt, args.output_dir, name="pointcloud_ttt")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
