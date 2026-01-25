"""
Dataset Visualization and Testing Script

Tests datasets and creates visualizations including:
- Input (context) images
- Target images
- 3D camera pose visualizations with coordinate frames

Usage:
    python -m datasets.visualize_dataset --dataset mipnerf360 --num_scenes 3
    python -m datasets.visualize_dataset --dataset tnt --num_scenes 2
    python -m datasets.visualize_dataset --dataset acid --num_scenes 1
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.dataset import build_dataset, DatasetCfg
from datasets.dataset_mipnerf360 import MipNeRF360Dataset
from datasets.dataset_tnt import TanksAndTemplesDataset
from datasets.dataset_acid import AcidDataset
from datasets.dataset_re10k import Re10kDataset


def save_image_tensor(img_tensor: torch.Tensor, output_path: Path):
    """
    Save a torch image tensor [3, H, W] to disk.
    
    Args:
        img_tensor: Image tensor in [0, 1] range, shape [3, H, W]
        output_path: Path to save the image
    """
    # Convert to numpy and transpose to [H, W, 3]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    # Clip to [0, 1] range
    img_np = np.clip(img_np, 0, 1)
    
    # Save using matplotlib
    plt.imsave(str(output_path), img_np)
    plt.close()


def extract_camera_pose(extrinsics: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera position and orientation from extrinsics matrix.
    
    Args:
        extrinsics: [4, 4] world-to-camera transformation matrix
        
    Returns:
        position: [3] camera position in world coordinates
        rotation: [3, 3] camera rotation matrix (world-to-camera)
    """
    # Extrinsics is world-to-camera: [R_w2c | t_w2c; 0 0 0 1]
    # Camera position in world: c_w = -R_w2c^T @ t_w2c = -R_c2w @ t_w2c
    R_w2c = extrinsics[:3, :3].cpu().numpy()
    t_w2c = extrinsics[:3, 3].cpu().numpy()
    
    # Convert to camera-to-world for visualization
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    
    return t_c2w, R_c2w


def visualize_camera_poses(
    context_extrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
    output_path: Path,
    scene_name: str,
):
    """
    Visualize camera poses in 3D space with coordinate frames.
    
    Args:
        context_extrinsics: [N_context, 4, 4] context camera extrinsics
        target_extrinsics: [N_target, 4, 4] target camera extrinsics
        output_path: Path to save the visualization
        scene_name: Name of the scene
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera positions and orientations
    context_positions = []
    context_rotations = []
    for i in range(context_extrinsics.shape[0]):
        pos, rot = extract_camera_pose(context_extrinsics[i])
        context_positions.append(pos)
        context_rotations.append(rot)
    
    target_positions = []
    target_rotations = []
    for i in range(target_extrinsics.shape[0]):
        pos, rot = extract_camera_pose(target_extrinsics[i])
        target_positions.append(pos)
        target_rotations.append(rot)
    
    context_positions = np.array(context_positions)
    target_positions = np.array(target_positions)
    
    # Plot camera positions
    if len(context_positions) > 0:
        ax.scatter(
            context_positions[:, 0],
            context_positions[:, 1],
            context_positions[:, 2],
            c='blue',
            s=100,
            label='Context cameras',
            alpha=0.7,
        )
    
    if len(target_positions) > 0:
        ax.scatter(
            target_positions[:, 0],
            target_positions[:, 1],
            target_positions[:, 2],
            c='red',
            s=100,
            label='Target cameras',
            alpha=0.7,
        )
    
    # Draw coordinate frames for cameras
    frame_length = 0.1 * np.max([
        np.max(np.abs(context_positions)) if len(context_positions) > 0 else 0,
        np.max(np.abs(target_positions)) if len(target_positions) > 0 else 0,
    ]) if len(context_positions) > 0 or len(target_positions) > 0 else 1.0
    
    # Draw context camera frames
    for i, (pos, rot) in enumerate(zip(context_positions, context_rotations)):
        # X axis (red)
        x_end = pos + rot[:, 0] * frame_length
        ax.plot([pos[0], x_end[0]], [pos[1], x_end[1]], [pos[2], x_end[2]], 'r-', linewidth=2)
        # Y axis (green)
        y_end = pos + rot[:, 1] * frame_length
        ax.plot([pos[0], y_end[0]], [pos[1], y_end[1]], [pos[2], y_end[2]], 'g-', linewidth=2)
        # Z axis (blue) - points towards scene
        z_end = pos - rot[:, 2] * frame_length  # Negative Z (camera looks along -Z)
        ax.plot([pos[0], z_end[0]], [pos[1], z_end[1]], [pos[2], z_end[2]], 'b-', linewidth=2)
    
    # Draw target camera frames
    for i, (pos, rot) in enumerate(zip(target_positions, target_rotations)):
        # X axis (red)
        x_end = pos + rot[:, 0] * frame_length
        ax.plot([pos[0], x_end[0]], [pos[1], x_end[1]], [pos[2], x_end[2]], 'r--', linewidth=1.5, alpha=0.6)
        # Y axis (green)
        y_end = pos + rot[:, 1] * frame_length
        ax.plot([pos[0], y_end[0]], [pos[1], y_end[1]], [pos[2], y_end[2]], 'g--', linewidth=1.5, alpha=0.6)
        # Z axis (blue)
        z_end = pos - rot[:, 2] * frame_length
        ax.plot([pos[0], z_end[0]], [pos[1], z_end[1]], [pos[2], z_end[2]], 'b--', linewidth=1.5, alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Camera Poses: {scene_name}', fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Set equal aspect ratio
    all_positions = np.vstack([context_positions, target_positions]) if len(context_positions) > 0 and len(target_positions) > 0 else (
        context_positions if len(context_positions) > 0 else target_positions
    )
    if len(all_positions) > 0:
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min(),
        ]).max() / 2.0
        
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add coordinate frame legend
    legend_elements = [
        mpatches.Patch(color='red', label='X axis'),
        mpatches.Patch(color='green', label='Y axis'),
        mpatches.Patch(color='blue', label='Z axis (viewing direction)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_dataset(
    dataset_name: str,
    data_root: str,
    output_dir: Path,
    num_scenes: int = 3,
    num_input_views: int = 2,
    num_target_views: int = 4,
    target_image_size: Tuple[int, int] = (256, 256),
):
    """
    Visualize dataset batches and save outputs.
    
    Args:
        dataset_name: Name of dataset ('mipnerf360', 'tnt', 'acid', 're10k')
        data_root: Root directory of dataset
        output_dir: Directory to save visualizations
        num_scenes: Number of scenes to visualize
        num_input_views: Number of input views
        num_target_views: Number of target views
        target_image_size: Target image size (H, W)
    """
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    context_dir = output_dir / "context_images"
    target_dir = output_dir / "target_images"
    poses_dir = output_dir / "camera_poses"
    
    context_dir.mkdir(exist_ok=True)
    target_dir.mkdir(exist_ok=True)
    poses_dir.mkdir(exist_ok=True)
    
    # Build dataset
    print(f"\n{'='*60}")
    print(f"Visualizing {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Handle target_image_size - can be int or tuple
    # For DatasetCfg, we need int, but datasets can accept tuple
    if isinstance(target_image_size, tuple):
        img_size = target_image_size[0]  # Use first element for config
    else:
        img_size = target_image_size
    
    # Build dataset directly to handle tuple target_image_size properly
    try:
        if dataset_name == "mipnerf360":
            from datasets.dataset_mipnerf360 import MipNeRF360Dataset
            dataset = MipNeRF360Dataset(
                data_root=data_root,
                stage="test",
                num_input_views=num_input_views,
                num_target_views=num_target_views,
                target_image_size=target_image_size,
                max_train_steps=0,
            )
        elif dataset_name == "tnt":
            from datasets.dataset_tnt import TanksAndTemplesDataset
            dataset = TanksAndTemplesDataset(
                data_root=data_root,
                stage="test",
                num_input_views=num_input_views,
                num_target_views=num_target_views,
                target_image_size=img_size,  # TNT uses int
                max_train_steps=0,
            )
        elif dataset_name == "acid":
            from datasets.dataset_acid import AcidDataset
            dataset = AcidDataset(
                data_root=data_root,
                stage="train",
                num_input_views=num_input_views,
                num_target_views=num_target_views,
                target_image_size=target_image_size,  # ACID uses tuple
                max_train_steps=0,
            )
        elif dataset_name == "re10k":
            from datasets.dataset_re10k import Re10kDataset
            dataset = Re10kDataset(
                data_root=data_root,
                stage="train",
                num_input_views=num_input_views,
                num_target_views=num_target_views,
                target_image_size=target_image_size,  # RE10K uses tuple
                max_train_steps=0,
            )
        else:
            # Fallback to build_dataset
            config = DatasetCfg(
                name=dataset_name,
                data_root=data_root,
                stage="test" if dataset_name in ["mipnerf360", "tnt"] else "train",
                num_input_views=num_input_views,
                num_target_views=num_target_views,
                target_image_size=img_size,
                max_train_steps=0,
            )
            dataset = build_dataset(config)
    except Exception as e:
        print(f"❌ Error building dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"✓ Dataset created successfully")
    print(f"✓ Output directory: {output_dir}")
    
    # Process scenes
    scene_count = 0
    for batch_idx, batch in enumerate(dataset):
        if scene_count >= num_scenes:
            break
        
        scene_name = batch['scene_key']
        print(f"\n📁 Processing scene: {scene_name}")
        
        # Create scene directories
        scene_context_dir = context_dir / scene_name
        scene_target_dir = target_dir / scene_name
        scene_context_dir.mkdir(exist_ok=True)
        scene_target_dir.mkdir(exist_ok=True)
        
        # Save context images
        context_images = batch['context']['images']  # [N, 3, H, W]
        print(f"  Saving {context_images.shape[0]} context images...")
        for i in range(context_images.shape[0]):
            img_path = scene_context_dir / f"context_{i:03d}.png"
            save_image_tensor(context_images[i], img_path)
        
        # Save target images
        target_images = batch['target']['images']  # [N, 3, H, W]
        print(f"  Saving {target_images.shape[0]} target images...")
        for i in range(target_images.shape[0]):
            img_path = scene_target_dir / f"target_{i:03d}.png"
            save_image_tensor(target_images[i], img_path)
        
        # Visualize camera poses
        print(f"  Creating camera pose visualization...")
        context_extrinsics = batch['context']['extrinsics']  # [N, 4, 4]
        target_extrinsics = batch['target']['extrinsics']  # [N, 4, 4]
        
        pose_path = poses_dir / f"{scene_name}_camera_poses.png"
        visualize_camera_poses(
            context_extrinsics,
            target_extrinsics,
            pose_path,
            scene_name,
        )
        
        # Print statistics
        print(f"  ✓ Context cameras: {context_extrinsics.shape[0]}")
        print(f"  ✓ Target cameras: {target_extrinsics.shape[0]}")
        print(f"  ✓ Images saved to: {scene_context_dir} and {scene_target_dir}")
        print(f"  ✓ Pose visualization: {pose_path}")
        
        scene_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Visualization complete!")
    print(f"   Processed {scene_count} scenes")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize and test datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mipnerf360", "tnt", "acid", "re10k"],
        help="Dataset name",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root directory (default: dataset-specific)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=3,
        help="Number of scenes to visualize",
    )
    parser.add_argument(
        "--num_input_views",
        type=int,
        default=2,
        help="Number of input/context views",
    )
    parser.add_argument(
        "--num_target_views",
        type=int,
        default=4,
        help="Number of target views",
    )
    parser.add_argument(
        "--target_image_size",
        type=int,
        default=256,
        help="Target image size (square)",
    )
    
    args = parser.parse_args()
    
    # Set default data roots
    if args.data_root is None:
        defaults = {
            "mipnerf360": "/workspace/re10kvol/mipnerf360",
            "tnt": "/workspace/re10kvol/tnt",
            "acid": "/workspace/re10kvol/acid",
            "re10k": "/workspace/re10kvol/re10k",
        }
        args.data_root = defaults[args.dataset]
    
    output_dir = Path(args.output_dir) / args.dataset
    
    visualize_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        output_dir=output_dir,
        num_scenes=args.num_scenes,
        num_input_views=args.num_input_views,
        num_target_views=args.num_target_views,
        target_image_size=(args.target_image_size, args.target_image_size),
    )


if __name__ == "__main__":
    main()
