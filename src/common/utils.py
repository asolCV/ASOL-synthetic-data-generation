# common/utils.py
from typing import Any, Dict, List, Union
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from math import ceil

# Import config, transforms using absolute paths
from src.common.config import cfg


# --- Plotting ---
def plot_metrics(
    train_losses: List[float],
    val_metrics: List[float],  # Zmieniono nazwę z val_ious na val_metrics
    metric_name: str,  # Dodano argument metric_name
    save_path: Union[str, Path],  # Akceptuje string lub Path
):
    """Plots training loss and a specified validation metric."""
    epochs = range(1, len(train_losses) + 1)
    if not train_losses or not val_metrics:
        print("Warning: Cannot plot metrics, empty loss or validation metric list.")
        return
    if len(train_losses) != len(val_metrics):
        print(
            f"Warning: Mismatched lengths for plotting: "
            f"train_losses ({len(train_losses)}), val_metrics ({len(val_metrics)}). Plotting common range."
        )
        min_len = min(len(train_losses), len(val_metrics))
        epochs = range(1, min_len + 1)
        train_losses = train_losses[:min_len]
        val_metrics = val_metrics[:min_len]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    # Użycie metric_name w tytule, etykiecie osi Y i legendzie
    plt.plot(epochs, val_metrics, "r-", label=f"Validation {metric_name}")
    plt.title(f"Validation {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)  # Użycie metric_name jako etykiety osi Y
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    save_path = Path(save_path)  # Upewnij się, że to obiekt Path
    try:
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving metrics plot to {save_path}: {e}")
    finally:
        plt.close()  # Zamknij figurę, aby zwolnić pamięć, nawet jeśli wystąpi błąd zapisu


# --- Saving Predictions ---
def save_validation_prediction(
    original_image_rgb_tensor, pred_mask_tensor, gt_mask_tensor, save_path
):
    """Saves a visual comparison: Original | Ground Truth | Prediction | Overlay"""
    # Convert tensors to numpy arrays on CPU
    pred_mask = pred_mask_tensor.squeeze().cpu().numpy().astype(np.uint8) * 255
    gt_mask = gt_mask_tensor.squeeze().cpu().numpy().astype(np.uint8) * 255
    # Expecting original_image as C, H, W, RGB, uint8 tensor
    original_image = (
        original_image_rgb_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    )

    gt_mask_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    pred_mask_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

    h, w = original_image.shape[:2]
    if gt_mask_bgr.shape[:2] != (h, w):
        gt_mask_bgr = cv2.resize(gt_mask_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_mask_bgr.shape[:2] != (h, w):
        pred_mask_bgr = cv2.resize(
            pred_mask_bgr, (w, h), interpolation=cv2.INTER_NEAREST
        )

    # Convert original RGB numpy image to BGR for OpenCV functions
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_image_bgr, 0.6, pred_mask_bgr, 0.4, 0)

    combined_viz = np.hstack((original_image_bgr, gt_mask_bgr, pred_mask_bgr, overlay))

    cv2.imwrite(str(save_path), combined_viz)  # cv2.imwrite expects string path


# --- Checkpointing ---
def save_checkpoint(
    state: Dict[str, Any],
    filename: str = "checkpoint.pth.tar",
    checkpoint_dir: Union[
        str, Path
    ] = cfg.CHECKPOINT_DIR,  # Użyj ścieżki z config jako domyślnej
):
    """Saves model and optimizer state."""
    # Użyj przekazanego checkpoint_dir
    checkpoint_dir = Path(checkpoint_dir)  # Upewnij się, że to obiekt Path
    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )  # Upewnij się, że katalog istnieje
    filepath = checkpoint_dir / filename
    print(f"=> Saving checkpoint: {filepath}")
    try:
        torch.save(state, filepath)
    except Exception as e:
        print(f"Error saving checkpoint to {filepath}: {e}")


import torch
from pathlib import Path
from src.common.config import (
    cfg,
)  # Zakładam, że cfg jest tutaj dostępne lub przekazywane inaczej


def load_checkpoint(checkpoint_path, model, optimizer=None, device=cfg.DEVICE):
    checkpoint_path = Path(checkpoint_path)  # Ensure Path object
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return 0, 0.0  # Zwraca wartości domyślne, jeśli plik nie istnieje

    print(f"=> Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        # --- KONIEC ZMIANY ---

        # Sprawdź, czy checkpoint ma oczekiwaną strukturę słownika
        if not isinstance(checkpoint, dict):
            print(
                f"Error: Checkpoint file at {checkpoint_path} does not contain a dictionary."
            )
            # Można by spróbować załadować bezpośrednio, jeśli to tylko state_dict, ale trzymajmy się struktury
            # model.load_state_dict(checkpoint)
            # return 0, 0.0 # Lub zgłosić błąd
            raise TypeError("Checkpoint file is not a dictionary.")

        # Wyodrębnij state_dict - upewnij się, że klucz istnieje
        if "state_dict" not in checkpoint:
            print(
                f"Error: 'state_dict' key not found in checkpoint dictionary at {checkpoint_path}."
            )
            # Jeśli nie ma klucza 'state_dict', nie możemy załadować wag modelu
            raise KeyError("'state_dict' key missing from checkpoint.")
        state_dict = checkpoint["state_dict"]

        # Obsługa prefixu 'module.' z DataParallel
        new_state_dict = {}
        # Sprawdź, czy *wszystkie* klucze zaczynają się od 'module.' - bezpieczniejsze niż sprawdzanie tylko pierwszego
        is_data_parallel = all(k.startswith("module.") for k in state_dict.keys())

        for k, v in state_dict.items():
            name = (
                k[7:] if is_data_parallel and k.startswith("module.") else k
            )  # Usuń `module.` jeśli istnieje
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)  # Ładuj oczyszczony state_dict

        # Bezpieczne pobieranie metadanych z checkpointu
        start_epoch = checkpoint.get("epoch", 0)  # Domyślnie 0, jeśli brakuje klucza
        best_metric = checkpoint.get(
            "best_metric", 0.0
        )  # Domyślnie 0.0, jeśli brakuje klucza

        # Ładowanie stanu optymalizatora, jeśli jest wymagany i dostępny
        if optimizer and "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("   Optimizer state loaded.")
            # Lepiej łapać bardziej ogólny wyjątek lub specyficzne, które mogą wystąpić
            except Exception as e:
                print(
                    f"   Warning: Could not load optimizer state: {e}. Optimizer might be reinitialized or in its initial state."
                )
        elif optimizer:
            print("   Optimizer state not found in checkpoint.")

        print(
            f"   Loaded model from epoch {start_epoch}. Best recorded metric: {best_metric:.4f}"
        )
        return start_epoch, best_metric

    except (
        FileNotFoundError
    ):  # Ten catch jest technicznie redundantny przez wcześniejsze sprawdzenie exists(), ale nie szkodzi
        print(
            f"Error: Checkpoint file not found at {checkpoint_path} (should have been caught earlier)."
        )
        raise  # Zgłoś ponownie
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        raise e


# --- Metrics ---
def calculate_iou(pred_logits, target_mask, num_classes, smooth=1e-6):
    """Calculates mean IoU for semantic segmentation on full masks"""
    pred_logits = pred_logits.detach().cpu()
    target_mask = target_mask.detach().cpu()

    if num_classes == 1:
        pred_prob = torch.sigmoid(pred_logits)
        pred_classes = (pred_prob > 0.5).squeeze(0).long()
    else:
        pred_classes = torch.argmax(pred_logits, dim=0).long()

    iou_list = []
    # Calculate IoU for foreground and optionally background
    target_class_indices = (
        range(num_classes) if num_classes > 1 else [0, 1]
    )  # Include background (0) for binary

    for cls in target_class_indices:
        pred_inds = pred_classes == cls
        target_inds = target_mask == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        # Only consider class if it's present in either prediction or target
        if union > 0:
            iou = (intersection + smooth) / (union + smooth)
            iou_list.append(iou)
        # Handle case where class '0' (background) is perfectly predicted as absent
        elif cls == 0 and intersection == 0 and union == 0:
            iou_list.append(1.0)

    if not iou_list:
        # If both pred and target are empty (union=0 for all classes)
        return 1.0 if target_mask.sum() == 0 and pred_classes.sum() == 0 else 0.0
    else:
        return np.mean(iou_list)


# --- Sliding Window Inference ---
def sliding_window_predict(
    model, image_tensor, tile_size, overlap, device, num_classes, batch_size=4
):
    """Performs prediction on a large image tensor using sliding window."""
    model.eval()  # Ensure model is in eval mode
    C, H, W = image_tensor.shape  # Expects [C, H, W]
    th = tw = tile_size
    stride = tile_size - overlap

    if overlap >= tile_size:
        raise ValueError("Overlap must be smaller than tile size")

    pred_map = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    norm_map = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    num_tiles_y = ceil(max(1, H - overlap) / stride)
    num_tiles_x = ceil(max(1, W - overlap) / stride)

    with torch.inference_mode():
        tile_batch_tensors = []
        coord_batch = []
        padding_batch = []

        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                y_start = i * stride
                x_start = j * stride
                y_end = min(y_start + th, H)
                x_end = min(x_start + tw, W)

                tile = image_tensor[:, y_start:y_end, x_start:x_end]

                dy = th - tile.shape[1]  # Note: shape[1] is Height for C,H,W tensor
                dx = tw - tile.shape[2]  # Note: shape[2] is Width for C,H,W tensor
                padding = (0, dx, 0, dy)  # left, right, top, bottom for F.pad
                if dy > 0 or dx > 0:
                    tile = torch.nn.functional.pad(
                        tile, padding, mode="constant", value=0
                    )

                tile_batch_tensors.append(tile)
                coord_batch.append((y_start, x_start, y_end, x_end))
                padding_batch.append((dy, dx))

                if len(tile_batch_tensors) == batch_size or (
                    i == num_tiles_y - 1 and j == num_tiles_x - 1
                ):
                    batch_tensor = torch.stack(tile_batch_tensors, dim=0).to(device)

                    batch_preds = model(batch_tensor)

                    for k in range(len(tile_batch_tensors)):
                        pred_tile_logits = batch_preds[k]
                        y_s, x_s, y_e, x_e = coord_batch[k]
                        pad_y, pad_x = padding_batch[k]

                        effective_h = th - pad_y
                        effective_w = tw - pad_x
                        pred_tile_unpadded = pred_tile_logits[
                            :, :effective_h, :effective_w
                        ]

                        pred_map[:, y_s:y_e, x_s:x_e] += pred_tile_unpadded
                        norm_map[:, y_s:y_e, x_s:x_e] += 1

                    tile_batch_tensors = []
                    coord_batch = []
                    padding_batch = []

    norm_map[norm_map == 0] = 1.0
    final_pred_map_logits = pred_map / norm_map

    # model.train() # Caller should handle setting model back to train if needed
    return final_pred_map_logits
