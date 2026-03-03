#!/usr/bin/env python3
"""
Generate HSI classification maps from a trained MambaVision checkpoint.

Supports dataset-specific colormaps (HongHu, Longkou, Salinas) and reuses the
same PCA + patch extraction pipeline as the training code, but WITHOUT
re-running training.

Example usage:

  # Longkou (path points to the folder containing model_best.pth.tar)
  python -m generate_hsi_maps \\
      --dataset longkou \\
      --checkpoint /mnt/D/MambaVision/output/train/prompt_Long/20251127-201251-mamba_vision_T-15/model_best.pth.tar \\
      --data_dir /mnt/D/MambaVision/mambavision/Data \\
      --use-prompt

  # Salinas
  python -m mambavision.generate_hsi_maps \\
      --dataset salinas \\
      --checkpoint /home/ilvesbenaissa/MambaVision/output/train/prompt_salinas10/20251120-202503-mamba_vision_T-15/model_best.pth.tar \\
      --data_dir /mnt/D/MambaVision/mambavision/Data \\
      --use-prompt
"""

import argparse
import os
from typing import Tuple, Callable

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt

from mambavision.utils.datasets import (
    _discover_hsi_in_dir,
    apply_pca,
    create_image_cubes,
    HyperspectralDataset,
)
from mambavision.models.mamba_vision import mamba_vision_T


# -------------------------------------------------------------------------
# Dataset-specific colormaps
# -------------------------------------------------------------------------

def list_to_colormap_honghu(x_list: np.ndarray) -> np.ndarray:
    """WHU-Hi-HongHu palette (22 classes incl. background)."""
    # Reuse the palette currently in get_cls_map.py (integer RGB 0-255)
    # Background + 21 classes.
    palette = np.array([
        [0, 0, 0],       # 0 background
        [72, 34, 25],
        [0, 44, 167],
        [240, 138, 6],
        [0, 194, 0],
        [194, 174, 209],
        [120, 196, 255],
        [0, 255, 140],
        [60, 115, 65],
        [204, 255, 204],
        [255, 255, 0],
        [255, 0, 255],
        [63, 0, 161],
        [132, 255, 255],
        [247, 255, 255],
        [188, 255, 127],
        [173, 255, 0],
        [255, 208, 108],
        [255, 122, 0],
        [255, 255, 193],
        [153, 255, 255],
        [121, 133, 121],
        [255, 153, 0],
    ], dtype=np.float32)

    x_int = x_list.astype(np.int64)
    x_int = np.clip(x_int, 0, palette.shape[0] - 1)
    return palette[x_int] / 255.0


def list_to_colormap_longkou(x_list: np.ndarray) -> np.ndarray:
    """Convert class labels to RGB colors for Longkou dataset (corrected palette)."""
    palette = np.array([
        [255, 255, 255],   # 0: Background (white)
        [0, 0, 0],         # 1: Corn (black)
        [128, 0, 128],     # 2: Cotton (purple)
        [0, 255, 0],       # 3: Sesame (green)
        [0, 128, 128],     # 4: Broad-leaf soybean (teal)
        [0, 255, 127],     # 5: Narrow-leaf soybean (spring green)
        [127, 255, 0],     # 6: Rice (chartreuse)
        [255, 255, 0],     # 7: Water (yellow)
        [255, 0, 0],       # 8: Roads and houses (red)
        [128, 128, 128],   # 9: Mixed weed (gray)
    ], dtype=np.float32)

    x_int = x_list.astype(np.int64)
    x_int = np.clip(x_int, 0, palette.shape[0] - 1)
    return palette[x_int] / 255.0


def list_to_colormap_salinas(x_list: np.ndarray) -> np.ndarray:
    """Convert class labels to RGB colors for Salinas dataset (user-corrected palette)."""
    palette = np.array([
        [0, 0, 0],          # 0: Background
        [255, 0, 0],        # 1: Broccoli_green_weeds_1 (Red)
        [0, 255, 0],        # 2: Broccoli_green_weeds_2 (Bright Green)
        [31, 69, 252],        # 3: Fallow (Blue)
        [255, 255, 0],      # 4: BroFallow_rough_plow (Brown)
        [255, 0, 255],      # 5: Fallow_smooth (Magenta)
        [0, 255, 255],      # 6: Stubble (Cyan)
        [139, 69, 19],      # 7: Celery (Yellow)
        [34, 139, 34],      # 8: Grapes_untrained (Forest Green)
        [128, 0, 128],      # 9: Soil_vinyard_develop (Purple)
        [255, 105, 180],    # 10: Corn_senesced_green_weeds (Hot Pink)
        [0, 255, 127],      # 11: Lettuce_romaine_4wk (Spring Green)
        [21, 27, 141],        # 12: Lettuce_romaine_5wk (Navy Blue)
        [160, 82, 45],      # 13: Lettuce_romaine_6wk (Sienna)
        [107, 142, 35],     # 14: Lettuce_romaine_7wk (Olive Drab)
        [102, 102, 204],    # 15: Vineyard_untrained (Slate Blue/Violet)
        [210, 180, 140],    # 16: Vineyard_vertical_trellis (Tan)
    ], dtype=np.float32)

    x_int = x_list.astype(np.int64)
    x_int = np.clip(x_int, 0, palette.shape[0] - 1)
    return palette[x_int] / 255.0


def get_colormap_fn(dataset: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the appropriate colormap function for a dataset name."""
    d = dataset.lower()
    if "longkou" in d:
        return list_to_colormap_longkou
    if "salinas" in d:
        return list_to_colormap_salinas
    if "honghu" in d:
        return list_to_colormap_honghu
    # Fallback: HongHu palette
    return list_to_colormap_honghu


# -------------------------------------------------------------------------
# Classification map reconstruction (copied from get_cls_map.py)
# -------------------------------------------------------------------------

def get_classification_map(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Rebuild 2D classification map from patch-wise predictions.

    This mirrors mambavision.get_cls_map.get_classification_map:
    - y_pred: predictions for all non-background pixels (0-based class indices)
    - y:      original 2D label map with 0 as background, 1..C as classes
    """
    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i, j] = y_pred[k] + 1
                k += 1
    return cls_labels


# -------------------------------------------------------------------------
# Data loading & patch creation
# -------------------------------------------------------------------------

def load_full_hsi(
    data_dir: str,
    dataset_flag: str,
    hsi_bands: int,
    patch_size: int,
    batch_size: int,
) -> Tuple[DataLoader, np.ndarray]:
    """
    Load full HSI cube + GT labels and create a DataLoader over all patches.

    Returns:
        all_loader: DataLoader over all non-background patches
        y_full: original 2D ground-truth labels (with 0 as background, 1..C as classes)
    """
    data_root = os.path.abspath(data_dir)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_dir does not exist: {data_root}")

    # Use same discovery logic as training
    dp, lp, dkey, lkey = _discover_hsi_in_dir(data_root, preferred_dataset=dataset_flag)

    X = sio.loadmat(dp)[dkey]
    y_full = sio.loadmat(lp)[lkey]
    if y_full.ndim > 2:
        y_full = y_full.squeeze()

    # PCA to hsi_bands if needed
    if X.shape[2] != hsi_bands:
        X = apply_pca(X, num_components=hsi_bands)

    # Extract patches for all non-background pixels
    patches, labels = create_image_cubes(
        X, y_full, window_size=patch_size, remove_zero_labels=True
    )

    # [N, H, W, C] -> [N, C, H, W]
    patches = np.transpose(patches, (0, 3, 1, 2))

    all_set = HyperspectralDataset(patches, labels, per_sample_norm=True)
    all_loader = DataLoader(
        all_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 2),
        drop_last=False,
    )

    return all_loader, y_full


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------

def build_mambavision_model(
    checkpoint_path: str,
    num_classes: int,
    hsi_bands: int,
    use_prompt: bool,
    task_classes: int,
    prompt_inject_levels: Tuple[int, ...],
    prompt_mode: str,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate MambaVision-T for HSI and load checkpoint weights.

    NOTE: For checkpoints saved by trainv2_simple.py, the file contains more than
    just the state_dict (argparse.Namespace, etc.). With PyTorch >= 2.6 the
    default torch.load(..., weights_only=True) cannot unpickle that, so we
    explicitly set weights_only=False in a safe way, assuming the checkpoint
    is trusted (local training output).
    """
    model = mamba_vision_T(
        pretrained=False,
        num_classes=num_classes,
        use_hsi_adaptor=True,
        hsi_bands=hsi_bands,
        drop_path_rate=0.0,
        use_kan=False,
        use_prompt=use_prompt,
        task_classes=task_classes,
        prompt_inject_levels=list(prompt_inject_levels),
        prompt_mode=prompt_mode,
    )

    # Load checkpoint; for PyTorch >= 2.6 we must disable weights_only
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only argument
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))

    # Strip 'module.' prefix if saved from DDP
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Load non-strict to be robust to small config differences
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------------------------------
# Map generation
# -------------------------------------------------------------------------

def generate_hsi_maps(args: argparse.Namespace) -> None:
    # MambaVision + mamba_ssm selective_scan currently require CUDA.
    # Running on CPU will trigger errors like "Expected u.is_cuda() to be true".
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but MambaVision's selective_scan op is GPU-only.\n"
            "Please run this script in the same GPU environment you used for training "
            "(where torch.cuda.is_available() is True)."
        )
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Map friendly dataset name -> .mat discovery flag
    dataset_flag_map = {
        "honghu": "WHU_Hi_HongHu",
        "longkou": "WHU_Hi_LongKou",
        "salinas": "Salinas_corrected",
    }
    dataset_key = args.dataset.lower()
    if dataset_key not in dataset_flag_map:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Use one of: honghu, longkou, salinas.")

    dataset_flag = dataset_flag_map[dataset_key]
    print(f"Loading dataset: {dataset_flag} from {args.data_dir}")

    # Load all patches + full GT
    all_loader, y_full = load_full_hsi(
        data_dir=args.data_dir,
        dataset_flag=dataset_flag,
        hsi_bands=args.hsi_bands,
        patch_size=args.hsi_patch,
        batch_size=args.batch_size,
    )

    num_classes = int(y_full.max())
    print(f"Detected {num_classes} foreground classes (labels 1..{num_classes}).")

    # Build and load model
    model = build_mambavision_model(
        checkpoint_path=args.checkpoint,
        num_classes=num_classes,
        hsi_bands=args.hsi_bands,
        use_prompt=args.use_prompt,
        task_classes=args.task_classes,
        prompt_inject_levels=tuple(args.prompt_inject_levels),
        prompt_mode=args.prompt_mode,
        device=device,
    )

    # Run inference over all patches
    print("Running inference over all patches...")
    predictions = []
    with torch.no_grad():
        progress = tqdm(all_loader, desc="Predicting", ncols=100)
        for inputs, _ in progress:
            inputs = inputs.to(device)
            # Per-patch normalization (same as training)
            inputs = (inputs - inputs.mean(dim=(2, 3), keepdim=True)) / (
                inputs.std(dim=(2, 3), keepdim=True) + 1e-8
            )

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)

    predictions = np.array(predictions)

    # Rebuild 2D classification map
    print("Reconstructing classification map...")
    cls_labels = get_classification_map(predictions, y_full)

    # Optional spatial median filtering
    if args.spatial_filter:
        print("Applying 3x3 median filter...")
        cls_labels = ndimage.median_filter(cls_labels, size=3)

    # Colorize prediction & GT
    cmap_fn = get_colormap_fn(args.dataset)
    x = cls_labels.ravel()
    gt = y_full.flatten()

    y_pred_rgb = cmap_fn(x).reshape(y_full.shape[0], y_full.shape[1], 3)
    y_gt_rgb = cmap_fn(gt).reshape(y_full.shape[0], y_full.shape[1], 3)

    os.makedirs(args.output_dir, exist_ok=True)

    # Prediction map
    plt.figure(figsize=(10, 10))
    plt.imshow(y_pred_rgb)
    plt.title(f"Prediction Map ({args.dataset})")
    plt.axis("off")
    plt.tight_layout()
    pred_path = os.path.join(args.output_dir, f"{args.map_name}.png")
    plt.savefig(pred_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Ground truth map
    plt.figure(figsize=(10, 10))
    plt.imshow(y_gt_rgb)
    plt.title(f"Ground Truth Map ({args.dataset})")
    plt.axis("off")
    plt.tight_layout()
    gt_path = os.path.join(args.output_dir, "ground_truth.png")
    plt.savefig(gt_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Difference map (binary error mask)
    diff_map = np.zeros_like(cls_labels)
    H, W = y_full.shape
    for i in range(H):
        for j in range(W):
            if y_full[i, j] != 0:  # ignore background
                diff_map[i, j] = 1 if cls_labels[i, j] != y_full[i, j] else 0

    plt.figure(figsize=(10, 10))
    plt.imshow(diff_map, cmap="coolwarm")
    plt.title("Difference Map (errors = 1)")
    plt.colorbar(label="Incorrect Prediction")
    plt.axis("off")
    plt.tight_layout()
    diff_path = os.path.join(args.output_dir, "difference_map.png")
    plt.savefig(diff_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    print("Saved maps:")
    print(f"  Prediction:   {pred_path}")
    print(f"  Ground truth: {gt_path}")
    print(f"  Difference:   {diff_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HSI classification maps from a trained MambaVision checkpoint."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["honghu", "longkou", "salinas"],
        help="HSI dataset name (controls .mat discovery and colormap).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model_best.pth.tar (or compatible) checkpoint.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./mambavision/Data",
        help="Directory containing the HSI .mat files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="classification_maps",
        help="Directory to save the classification maps.",
    )
    parser.add_argument(
        "--map_name",
        type=str,
        default="prediction_map",
        help="Base filename for the prediction map PNG.",
    )
    parser.add_argument(
        "--spatial_filter",
        action="store_true",
        help="Apply 3x3 median filter to the classification map.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference over patches.",
    )
    parser.add_argument(
        "--hsi_bands",
        type=int,
        default=15,
        help="Number of spectral bands after PCA (must match training).",
    )
    parser.add_argument(
        "--hsi_patch",
        type=int,
        default=15,
        help="Spatial patch size used during training (must match training).",
    )
    # Prompt-related options (match training config as needed)
    parser.add_argument(
        "--use-prompt",
        action="store_true",
        help="Enable prompt integration in the model (must match training).",
    )
    parser.add_argument(
        "--task-classes",
        type=int,
        default=6,
        help="Number of task classes for prompts (if use-prompt).",
    )
    parser.add_argument(
        "--prompt-inject-levels",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Levels at which to inject prompts (if use-prompt).",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="full",
        choices=["full", "visual_only", "text_only"],
        help="Prompt ablation mode (must match training if prompts were used).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_hsi_maps(args)

