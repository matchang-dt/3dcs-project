"""
Coordinate System Validation Test Script

This script tests all datasets to ensure they operate under the same coordinate system.
It produces visualizations for:
1. Camera pose plots showing coordinate frames
2. Image crops from context and target views
3. Intrinsics output (normalized vs denormalized)
4. Coordinate system validation metrics

Memory optimizations:
- Uses a single iterator per dataset (reused across scenes) to prevent memory accumulation
- Moves tensors to CPU before processing (CPU-only safe, no GPU required)
- Explicitly deletes large objects and calls gc.collect() after each scene
- Closes matplotlib figures with plt.close('all') to free memory
- Default num_scenes is 1 to avoid OOM issues (especially for large datasets like Tanks & Temples)

CPU-only operation:
- All GPU operations are safely guarded with torch.cuda.is_available() checks
- Works perfectly on systems without GPU

Output directory structure:
output/
  dataset_name/
    scene_XXX/
      camera_poses.png          - 3D visualization of all camera poses
      context_images/           - Context view images
      target_images/            - Target view images
      intrinsics_context.txt    - Context intrinsics (normalized and denormalized)
      intrinsics_target.txt     - Target intrinsics (normalized and denormalized)
      coordinate_validation.txt - Validation metrics
"""

import argparse
import sys
import os
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.dataset import DATASETS, DatasetCfg
from datasets.view_sampler.view_sampler import ViewSet
from datasets.shims.norm_shim import (
    normalize_intrinsics, 
    denormalize_intrinsics,
    already_normalized
)


def clear_gpu_cache():
    """Safely clear GPU cache if CUDA is available, otherwise do nothing."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_image_tensor(img_tensor: torch.Tensor, output_path: Path):
    """
    Save a torch image tensor [3, H, W] or [H, W, 3] to disk.
    
    Args:
        img_tensor: Image tensor in [0, 1] range
        output_path: Path to save the image
    """
    # Handle both [3, H, W] and [H, W, 3] formats
    if img_tensor.dim() == 3:
        if img_tensor.shape[0] == 3:
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img_tensor.cpu().numpy()
    else:
        img_np = img_tensor.cpu().numpy()
    
    # Clip to [0, 1] range
    img_np = np.clip(img_np, 0, 1)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save using matplotlib
    plt.imsave(str(output_path), img_np)
    plt.close('all')  # Close all figures to free memory
    del img_np  # Explicitly delete numpy array


def extract_camera_pose(extrinsics: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera position and orientation from extrinsics matrix.
    
    Args:
        extrinsics: [4, 4] world-to-camera transformation matrix
        
    Returns:
        position: [3] camera position in world coordinates (c2w)
        rotation: [3, 3] camera rotation matrix (c2w)
    """
    # Extrinsics is world-to-camera: [R_w2c | t_w2c]
    R_w2c = extrinsics[:3, :3].cpu().numpy()
    t_w2c = extrinsics[:3, 3].cpu().numpy()
    
    # Convert to camera-to-world for visualization
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    
    return t_c2w, R_c2w


def draw_camera_frustum(ax, position: np.ndarray, rotation: np.ndarray, 
                        color: str, scale: float = 0.3, label: str = None):
    """
    Draw a camera frustum in 3D space.
    
    Args:
        ax: Matplotlib 3D axis
        position: [3] camera position in world coordinates
        rotation: [3, 3] camera rotation matrix (c2w)
        color: Color for the frustum
        scale: Scale factor for the frustum size
        label: Label for the camera
    """
    # Define camera frustum vertices in camera coordinates
    # Camera looks down -Z axis, with Y pointing down and X pointing right
    frustum_points = np.array([
        [0, 0, 0],           # Camera center
        [-1, -1, 1],         # Top-left
        [1, -1, 1],          # Top-right
        [1, 1, 1],           # Bottom-right
        [-1, 1, 1],          # Bottom-left
    ]) * scale
    
    # Transform to world coordinates
    world_points = (rotation @ frustum_points.T).T + position
    
    # Draw frustum edges
    center = world_points[0]
    for i in range(1, 5):
        ax.plot([center[0], world_points[i][0]], 
                [center[1], world_points[i][1]], 
                [center[2], world_points[i][2]], 
                color=color, linewidth=1.5, alpha=0.7)
    
    # Draw frustum rectangle
    for i in range(1, 5):
        next_i = (i % 4) + 1
        ax.plot([world_points[i][0], world_points[next_i][0]], 
                [world_points[i][1], world_points[next_i][1]], 
                [world_points[i][2], world_points[next_i][2]], 
                color=color, linewidth=1.5, alpha=0.7)
    
    # Draw coordinate axes (RGB = XYZ)
    axis_length = scale * 0.5
    # X axis (red)
    x_axis = rotation @ np.array([axis_length, 0, 0])
    ax.plot([position[0], position[0] + x_axis[0]], 
            [position[1], position[1] + x_axis[1]], 
            [position[2], position[2] + x_axis[2]], 
            'r-', linewidth=2, alpha=0.8)
    
    # Y axis (green)
    y_axis = rotation @ np.array([0, axis_length, 0])
    ax.plot([position[0], position[0] + y_axis[0]], 
            [position[1], position[1] + y_axis[1]], 
            [position[2], position[2] + y_axis[2]], 
            'g-', linewidth=2, alpha=0.8)
    
    # Z axis (blue) - viewing direction
    z_axis = rotation @ np.array([0, 0, axis_length])
    ax.plot([position[0], position[0] + z_axis[0]], 
            [position[1], position[1] + z_axis[1]], 
            [position[2], position[2] + z_axis[2]], 
            'b-', linewidth=2, alpha=0.8)
    
    # Plot camera center
    ax.scatter(*position, color=color, s=100, alpha=0.8, label=label)


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
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract and draw context cameras
    for i in range(context_extrinsics.shape[0]):
        pos, rot = extract_camera_pose(context_extrinsics[i])
        label = f"Context {i}" if i == 0 else None
        draw_camera_frustum(ax, pos, rot, 'blue', scale=0.3, label=label)
    
    # Extract and draw target cameras
    for i in range(target_extrinsics.shape[0]):
        pos, rot = extract_camera_pose(target_extrinsics[i])
        label = f"Target {i}" if i == 0 else None
        draw_camera_frustum(ax, pos, rot, 'red', scale=0.3, label=label)
    
    # Draw world coordinate frame at origin
    origin = np.array([0, 0, 0])
    axis_length = 0.5
    ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=3, label='World X')
    ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=3, label='World Y')
    ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=3, label='World Z')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera Poses for {scene_name}\n(Blue=Context, Red=Target)')
    
    # Set equal aspect ratio
    all_positions = []
    for i in range(context_extrinsics.shape[0]):
        pos, _ = extract_camera_pose(context_extrinsics[i])
        all_positions.append(pos)
    for i in range(target_extrinsics.shape[0]):
        pos, _ = extract_camera_pose(target_extrinsics[i])
        all_positions.append(pos)
    
    if all_positions:
        all_positions = np.array(all_positions)
        max_range = np.max(np.abs(all_positions)) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close('all')  # Close all figures to free memory
    del fig, ax  # Explicitly delete figure and axis


def save_intrinsics_info(
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    output_path: Path,
    view_type: str
):
    """
    Save intrinsics information to a text file.
    
    Args:
        intrinsics: [N, 3, 3] camera intrinsics
        image_size: (H, W) image size
        output_path: Path to save the text file
        view_type: "context" or "target"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"=== {view_type.upper()} VIEW INTRINSICS ===\n")
        f.write(f"Image Size: {image_size[0]}x{image_size[1]} (HxW)\n")
        f.write(f"Number of views: {intrinsics.shape[0]}\n\n")
        
        for i in range(intrinsics.shape[0]):
            K = intrinsics[i].cpu().numpy()
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            f.write(f"--- View {i} ---\n")
            f.write(f"Intrinsics Matrix:\n")
            f.write(f"  {K[0, 0]:.6f}  {K[0, 1]:.6f}  {K[0, 2]:.6f}\n")
            f.write(f"  {K[1, 0]:.6f}  {K[1, 1]:.6f}  {K[1, 2]:.6f}\n")
            f.write(f"  {K[2, 0]:.6f}  {K[2, 1]:.6f}  {K[2, 2]:.6f}\n")
            f.write(f"fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}\n")
            
            # Check if normalized
            is_norm = already_normalized(intrinsics[i:i+1], "intrinsics")
            f.write(f"Is Normalized: {is_norm}\n")
            
            # If normalized, show denormalized version
            if is_norm:
                K_denorm = denormalize_intrinsics(intrinsics[i], image_size).cpu().numpy()
                fx_d, fy_d = K_denorm[0, 0], K_denorm[1, 1]
                cx_d, cy_d = K_denorm[0, 2], K_denorm[1, 2]
                f.write(f"Denormalized: fx={fx_d:.6f}, fy={fy_d:.6f}, cx={cx_d:.6f}, cy={cy_d:.6f}\n")
            else:
                # If not normalized, show normalized version
                K_norm = normalize_intrinsics(intrinsics[i], image_size).cpu().numpy()
                fx_n, fy_n = K_norm[0, 0], K_norm[1, 1]
                cx_n, cy_n = K_norm[0, 2], K_norm[1, 2]
                f.write(f"Normalized: fx={fx_n:.6f}, fy={fy_n:.6f}, cx={cx_n:.6f}, cy={cy_n:.6f}\n")
            
            f.write("\n")


def validate_coordinate_system(
    context_views: ViewSet,
    target_views: ViewSet,
    image_size: Tuple[int, int],
    output_path: Path,
    dataset_name: str
):
    """
    Validate that the coordinate system is consistent.
    
    Args:
        context_views: Context ViewSet
        target_views: Target ViewSet
        image_size: (H, W) image size
        output_path: Path to save validation results
        dataset_name: Name of the dataset
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"=== COORDINATE SYSTEM VALIDATION ===\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Image Size: {image_size[0]}x{image_size[1]} (HxW)\n\n")
        
        # Check extrinsics properties
        f.write("--- EXTRINSICS VALIDATION ---\n")
        
        # Check if extrinsics are valid transformation matrices
        all_extrinsics = torch.cat([context_views.extrinsics, target_views.extrinsics], dim=0)
        for i, ext in enumerate(all_extrinsics):
            det = torch.det(ext[:3, :3])
            is_orthogonal = torch.allclose(
                ext[:3, :3] @ ext[:3, :3].T, 
                torch.eye(3, device=ext.device),
                atol=1e-4
            )
            last_row_correct = torch.allclose(
                ext[3, :], 
                torch.tensor([0., 0., 0., 1.], device=ext.device),
                atol=1e-6
            )
            
            view_type = "Context" if i < len(context_views.extrinsics) else "Target"
            view_idx = i if i < len(context_views.extrinsics) else i - len(context_views.extrinsics)
            
            f.write(f"{view_type} View {view_idx}:\n")
            f.write(f"  Rotation determinant: {det:.6f} (should be ±1)\n")
            f.write(f"  Is orthogonal: {is_orthogonal}\n")
            f.write(f"  Last row correct [0,0,0,1]: {last_row_correct}\n")
            
            # Extract camera position
            pos, rot = extract_camera_pose(ext)
            f.write(f"  Camera position (world): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n")
            
            # Check viewing direction (negative Z axis in camera coords)
            view_dir = rot @ np.array([0, 0, 1])  # Camera looks down +Z in c2w
            f.write(f"  Viewing direction: [{view_dir[0]:.4f}, {view_dir[1]:.4f}, {view_dir[2]:.4f}]\n\n")
        
        # Check intrinsics properties
        f.write("\n--- INTRINSICS VALIDATION ---\n")
        
        # Context intrinsics
        f.write("Context Views:\n")
        for i, K in enumerate(context_views.intrinsics):
            fx, fy = K[0, 0].item(), K[1, 1].item()
            cx, cy = K[0, 2].item(), K[1, 2].item()
            
            is_norm = already_normalized(K.unsqueeze(0), "intrinsics")
            
            f.write(f"  View {i}: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}\n")
            f.write(f"    Normalized: {is_norm}\n")
            
            if is_norm:
                f.write(f"    cx/W={cx:.4f} (should be ~0.5 for centered), cy/H={cy:.4f}\n")
            else:
                f.write(f"    cx={cx:.1f} (should be ~{image_size[1]/2:.1f}), cy={cy:.1f} (should be ~{image_size[0]/2:.1f})\n")
        
        f.write("\nTarget Views:\n")
        for i, K in enumerate(target_views.intrinsics):
            fx, fy = K[0, 0].item(), K[1, 1].item()
            cx, cy = K[0, 2].item(), K[1, 2].item()
            
            is_norm = already_normalized(K.unsqueeze(0), "intrinsics")
            
            f.write(f"  View {i}: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}\n")
            f.write(f"    Normalized: {is_norm}\n")
            
            if is_norm:
                f.write(f"    cx/W={cx:.4f} (should be ~0.5 for centered), cy/H={cy:.4f}\n")
            else:
                f.write(f"    cx={cx:.1f} (should be ~{image_size[1]/2:.1f}), cy={cy:.1f} (should be ~{image_size[0]/2:.1f})\n")
        
        # Image properties
        f.write("\n--- IMAGE VALIDATION ---\n")
        if context_views.images is not None:
            f.write(f"Context images shape: {context_views.images.shape}\n")
            f.write(f"  Min value: {context_views.images.min():.4f}\n")
            f.write(f"  Max value: {context_views.images.max():.4f}\n")
            f.write(f"  Mean value: {context_views.images.mean():.4f}\n")
            
        if target_views.images is not None:
            f.write(f"Target images shape: {target_views.images.shape}\n")
            f.write(f"  Min value: {target_views.images.min():.4f}\n")
            f.write(f"  Max value: {target_views.images.max():.4f}\n")
            f.write(f"  Mean value: {target_views.images.mean():.4f}\n")
        
        # Coordinate system summary
        f.write("\n--- COORDINATE SYSTEM SUMMARY ---\n")
        f.write("Expected conventions:\n")
        f.write("  - Extrinsics: [4x4] world-to-camera transformation matrix\n")
        f.write("  - Rotation: Orthogonal matrix with det(R) = ±1\n")
        f.write("  - Camera looks down NEGATIVE Z axis in camera coordinates\n")
        f.write("  - X axis points right, Y axis points down, Z axis points forward\n")
        f.write("  - Intrinsics: Can be normalized [0,1] or pixel coordinates\n")
        f.write("  - Images: [0,1] range, shape [3, H, W] or [H, W, 3]\n")


def process_dataset_sample(
    dataset_iterator,
    dataset_name: str,
    output_dir: Path,
    scene_idx: int,
    num_scenes: int
):
    """
    Process a single sample from a dataset iterator and save visualizations.
    
    Args:
        dataset_iterator: Iterator over the dataset (reused across calls)
        dataset_name: Name of the dataset
        output_dir: Output directory
        scene_idx: Index of the scene being processed
        num_scenes: Total number of scenes to process
    """
    try:
        # Get a sample from the dataset iterator
        sample = next(dataset_iterator)
        
        # Handle the nested dict format returned by datasets
        context_data = sample['context']
        target_data = sample['target']
        
        # Move tensors to CPU to save GPU memory (if any are on GPU)
        # This is safe even if tensors are already on CPU
        def to_cpu_if_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().detach()  # detach() to break any gradient tracking
            return x
        
        # Create ViewSet objects from the data (move to CPU to save memory)
        context_views = ViewSet(
            extrinsics=to_cpu_if_tensor(context_data['extrinsics']),
            intrinsics=to_cpu_if_tensor(context_data['intrinsics']),
            images=to_cpu_if_tensor(context_data['images']) if context_data['images'] is not None else None
        )
        target_views = ViewSet(
            extrinsics=to_cpu_if_tensor(target_data['extrinsics']),
            intrinsics=to_cpu_if_tensor(target_data['intrinsics']),
            images=to_cpu_if_tensor(target_data['images']) if target_data['images'] is not None else None
        )
        
        # Determine image size
        if context_views.images is not None:
            if context_views.images.dim() == 4:  # [N, 3, H, W]
                image_size = (context_views.images.shape[2], context_views.images.shape[3])
            else:  # [N, H, W, 3]
                image_size = (context_views.images.shape[1], context_views.images.shape[2])
        elif target_views.images is not None:
            if target_views.images.dim() == 4:
                image_size = (target_views.images.shape[2], target_views.images.shape[3])
            else:
                image_size = (target_views.images.shape[1], target_views.images.shape[2])
        else:
            image_size = (256, 256)  # Default
        
        # Create scene output directory
        scene_dir = output_dir / dataset_name / f"scene_{scene_idx:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Processing scene {scene_idx+1}/{num_scenes} for {dataset_name}")
        print(f"    Image size: {image_size}")
        print(f"    Context views: {context_views.extrinsics.shape[0]}")
        print(f"    Target views: {target_views.extrinsics.shape[0]}")
        
        # 1. Visualize camera poses
        visualize_camera_poses(
            context_views.extrinsics,
            target_views.extrinsics,
            scene_dir / "camera_poses.png",
            f"{dataset_name}_scene_{scene_idx}"
        )
        
        # 2. Save context images
        if context_views.images is not None:
            context_img_dir = scene_dir / "context_images"
            context_img_dir.mkdir(exist_ok=True)
            
            for i in range(context_views.images.shape[0]):
                img = context_views.images[i]
                # Handle both [3, H, W] and [H, W, 3] formats
                if img.dim() == 3 and img.shape[0] == 3:
                    # Already in [3, H, W] format
                    pass
                elif img.dim() == 3:
                    # Convert [H, W, 3] to [3, H, W]
                    img = img.permute(2, 0, 1)
                
                save_image_tensor(img, context_img_dir / f"view_{i:02d}.png")
        
        # 3. Save target images
        if target_views.images is not None:
            target_img_dir = scene_dir / "target_images"
            target_img_dir.mkdir(exist_ok=True)
            
            for i in range(target_views.images.shape[0]):
                img = target_views.images[i]
                # Handle both [3, H, W] and [H, W, 3] formats
                if img.dim() == 3 and img.shape[0] == 3:
                    pass
                elif img.dim() == 3:
                    img = img.permute(2, 0, 1)
                
                save_image_tensor(img, target_img_dir / f"view_{i:02d}.png")
        
        # 4. Save intrinsics info
        save_intrinsics_info(
            context_views.intrinsics,
            image_size,
            scene_dir / "intrinsics_context.txt",
            "context"
        )
        
        save_intrinsics_info(
            target_views.intrinsics,
            image_size,
            scene_dir / "intrinsics_target.txt",
            "target"
        )
        
        # 5. Validate coordinate system
        validate_coordinate_system(
            context_views,
            target_views,
            image_size,
            scene_dir / "coordinate_validation.txt",
            dataset_name
        )
        
        print(f"    ✓ Saved to {scene_dir}")
        
        # Explicitly free memory
        del context_views, target_views, context_data, target_data, sample
        if 'img' in locals():
            del img
        clear_gpu_cache()
        gc.collect()
        
    except StopIteration:
        print(f"  ⚠ No more scenes available for {dataset_name} (requested {num_scenes}, got {scene_idx})")
        raise
    except Exception as e:
        print(f"  ✗ Error processing scene {scene_idx} for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error too
        clear_gpu_cache()
        gc.collect()


def test_all_datasets(
    output_dir: Path,
    num_scenes: int = 1,
    datasets_to_test: Optional[List[str]] = None
):
    """
    Test all available datasets or a subset.
    
    Args:
        output_dir: Output directory for results
        num_scenes: Number of scenes to test per dataset
        datasets_to_test: List of dataset names to test (None = all)
    """
    if datasets_to_test is None:
        datasets_to_test = list(DATASETS.keys())
    
    print(f"Testing datasets: {datasets_to_test}")
    print(f"Number of scenes per dataset: {num_scenes}")
    if num_scenes > 1:
        print(f"⚠ WARNING: Processing {num_scenes} scenes per dataset may cause OOM issues!")
    print(f"Output directory: {output_dir}")
    print(f"GPU available: {torch.cuda.is_available()}\n")
    
    # Common test configuration
    test_config = {
        "stage": "train",
        "num_input_views": 2,
        "num_target_views": 4,
        "target_image_size": (256, 256),
        "max_train_steps": 300000,
    }
    
    # Dataset-specific data roots (update these paths as needed)
    dataset_roots = {
        "acid": "/workspace/re10kvol/acid",
        "re10k": "/workspace/re10kvol/re10k",
        "mipnerf360": "/workspace/re10kvol/mipnerf360",
        "tnt": "/workspace/re10kvol/tnt",
        "deepblending": "/workspace/re10kvol/deepblending",
        "dtu": "/workspace/re10kvol/dtu",
    }
    
    results = {}
    
    for dataset_name in datasets_to_test:
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Check if dataset is available
            if dataset_name not in DATASETS:
                print(f"  ✗ Dataset {dataset_name} not found in registry")
                continue
            
            # Get data root
            data_root = dataset_roots.get(dataset_name)
            if data_root is None:
                print(f"  ✗ Data root not specified for {dataset_name}")
                continue
            
            if not Path(data_root).exists():
                print(f"  ✗ Data root does not exist: {data_root}")
                continue
            
            # Create dataset
            dataset_class = DATASETS[dataset_name]
            dataset = dataset_class(
                data_root=data_root,
                **test_config
            )
            
            print(f"  ✓ Dataset initialized: {dataset_name}")
            
            # Create a single iterator for all scenes (prevents memory accumulation)
            dataset_iterator = iter(dataset)
            
            # Process scenes using the same iterator (only 1 scene by default to avoid OOM)
            scenes_processed = 0
            try:
                for scene_idx in range(num_scenes):
                    process_dataset_sample(
                        dataset_iterator,
                        dataset_name,
                        output_dir,
                        scene_idx,
                        num_scenes
                    )
                    scenes_processed += 1
                
                results[dataset_name] = f"SUCCESS ({scenes_processed} scene(s))"
            except StopIteration:
                # Ran out of scenes, but that's okay
                results[dataset_name] = f"SUCCESS (partial - {scenes_processed} scene(s) processed)"
            
            # Clean up dataset and iterator
            del dataset_iterator, dataset
            clear_gpu_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Error testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = f"FAILED: {str(e)}"
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for dataset_name, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {dataset_name}: {status}")
    
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Test coordinate systems across all datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./coordinate_system_test_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=1,
        help="Number of scenes to test per dataset (default: 1 to avoid OOM)"
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=20,
        help="Limit number of views per scene to save memory (default: 20)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific datasets to test (e.g., acid re10k mipnerf360)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_all_datasets(
        output_dir=output_dir,
        num_scenes=args.num_scenes,
        datasets_to_test=args.datasets
    )


if __name__ == "__main__":
    main()

