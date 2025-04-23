# common/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import os  # Keep for cpu_count usage if needed in Config

# Local imports using absolute paths from project root or relative paths if structured correctly
# Assuming 'src' is the project root directory containing 'common', 'models', etc.
# Adjust imports based on your exact project structure if necessary.
# If train.py is inside 'common', relative imports might work like:
# from .config import cfg
# from .datasets import TileTrainingDataset, FullImageDataset
# from ..models.unet import UNet # Navigate up one level then into models
# from . import transforms as T
# from . import utils
# Using absolute paths based on a common source root is often more robust:
from src.common.config import cfg
from src.common.datasets import (
    TileTrainingDataset,
    FullImageDataset,
)  # Ensure TileTrainingDataset has the jitter/no-padding logic
from src.models.unet import UNet
from src.common import transforms as T
from src.common import utils


# Global variable for tqdm description in train_one_epoch
# Alternatively, pass epoch number as an argument
current_epoch = 0


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn,
    device: str,
):
    """
    Runs a single training epoch using tiled data.

    Args:
        loader: DataLoader for the training set (yielding batches of tiles).
        model: The neural network model to train.
        optimizer: The optimizer for updating model weights.
        loss_fn: The loss function.
        device: The device to run computations on ('cuda' or 'cpu').

    Returns:
        The average training loss for the epoch.
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    loop = tqdm(
        loader, desc=f"Epoch {current_epoch}/{cfg.NUM_EPOCHS} Training", leave=False
    )
    total_loss = 0.0
    batches_processed = 0

    for batch_idx, batch in enumerate(loop):
        # Move data and targets to the configured device
        data = batch["image"].to(device=device)
        targets = batch["mask"].to(device=device)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients // Typo 'baskward' corrected to 'backward'
        optimizer.step()  # Update model weights

        total_loss += loss.item()  # Accumulate loss (use .item() to get scalar value)
        batches_processed += 1
        loop.set_postfix(
            loss=loss.item()
        )  # Update tqdm progress bar with current batch loss

    # Calculate average loss over all batches
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    return avg_loss


def validate_and_visualize(
    loader: DataLoader, model: nn.Module, device: str, epoch_num: int
):
    """
    Performs validation on full images using sliding window inference and saves some predictions.

    Args:
        loader: DataLoader for the validation set (yielding full images, batch_size=1).
        model: The neural network model (should be the same instance as used for training).
        device: The device to run computations on ('cuda' or 'cpu').
        epoch_num: The current epoch number (used for saving filenames).

    Returns:
        The average Intersection over Union (IoU) across the validation set.
    """
    print("\nRunning Sliding Window Validation...")
    model.eval()  # Set model to evaluation mode (disables dropout, uses running stats for batch norm)
    total_iou = 0.0
    num_samples = 0

    # Use torch.inference_mode() for efficiency (disables gradient calculation)
    with torch.inference_mode():
        for batch_idx, batch in enumerate(
            tqdm(loader, desc="Validation Progress", leave=False)
        ):
            # Data comes from FullImageDataset
            image_tensor = batch["image_tensor"].to(
                device
            )  # Expected shape: [1, C, H, W]
            gt_mask = batch["gt_mask"][0].to(
                device
            )  # Expected shape: [H, W] (remove batch dim)
            original_image_rgb_tensor = batch["original_image"][
                0
            ]  # Expected shape: [C, H, W] (RGB for saving)
            image_path = Path(batch["image_path"][0])  # Get the image path string

            # Perform prediction using sliding window on the single full image tensor
            # Note: image_tensor needs to be [C, H, W] for the utility function
            pred_logits_map = utils.sliding_window_predict(
                model=model,
                image_tensor=image_tensor.squeeze(
                    0
                ),  # Remove batch dimension -> [C, H, W]
                tile_size=cfg.INFERENCE_TILE_SIZE,
                overlap=cfg.INFERENCE_OVERLAP,
                device=device,
                num_classes=cfg.NUM_CLASSES,
                batch_size=cfg.BATCH_SIZE,  # Batch size for processing *tiles* during inference
                # Consider a separate cfg.INFERENCE_BATCH_SIZE if memory is tight
            )  # Expected output shape: [Num_Classes, H, W] (logits)

            # Calculate IoU for this image
            # Ensure calculate_iou handles logits correctly (e.g., applies sigmoid/argmax internally if needed)
            iou = utils.calculate_iou(pred_logits_map, gt_mask, cfg.NUM_CLASSES)
            total_iou += iou
            num_samples += 1

            # Save a few validation predictions for visual inspection
            if cfg.SAVE_VALIDATION_PREDS and batch_idx < cfg.NUM_VALIDATION_SAVE:
                # Convert logits to a binary/class mask for visualization
                if cfg.NUM_CLASSES == 1:
                    # Binary case: Apply sigmoid and threshold
                    pred_mask_viz = (
                        torch.sigmoid(pred_logits_map) > 0.5
                    ).long()  # Output: [1, H, W]
                else:
                    # Multi-class case: Apply argmax
                    pred_mask_viz = torch.argmax(
                        pred_logits_map, dim=0, keepdim=True
                    ).long()  # Output: [1, H, W]

                # Define the output path for the visualization
                save_path = (
                    cfg.VALIDATION_PRED_DIR
                    / f"{image_path.stem}_epoch_{epoch_num}_val.png"
                )
                # Call the utility function to save the overlay
                utils.save_validation_prediction(
                    original_image_rgb_tensor=original_image_rgb_tensor,  # Should be [C, H, W] Tensor
                    pred_mask_tensor=pred_mask_viz,  # Should be [1, H, W] Tensor
                    gt_mask_tensor=gt_mask.unsqueeze(
                        0
                    ),  # Add channel dim -> [1, H, W] Tensor
                    save_path=save_path,
                )

    # Calculate the average IoU over all validation samples
    avg_iou = total_iou / num_samples if num_samples > 0 else 0.0

    # Optional: Clear GPU cache after validation (might help if memory is very constrained)
    if device == "cuda":
        torch.cuda.empty_cache()

    # Note: The caller (main loop) should set the model back to train() mode if needed
    return avg_iou


def main():
    global current_epoch  # Allow modification of the global epoch number for tqdm

    print(f"--- Configuration ---")
    print(f"Project Root: {cfg.PROJECT_ROOT}")
    print(f"Image Dir: {cfg.IMAGE_DIR}")
    print(f"Mask Dir: {cfg.MASK_DIR}")
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    print(
        f"Device: {cfg.DEVICE}, Workers: {cfg.NUM_WORKERS}, Pin Memory: {cfg.PIN_MEMORY}"
    )
    print(
        f"Train Tile: {cfg.TILE_SIZE}x{cfg.TILE_SIZE}, Stride: {cfg.TRAIN_TILE_STRIDE}"
    )
    # Include jitter if you added it to config or dataset init
    # print(f"Train Tile Jitter: {cfg.TILE_JITTER}")
    print(
        f"Inference Tile: {cfg.INFERENCE_TILE_SIZE}x{cfg.INFERENCE_TILE_SIZE}, Overlap: {cfg.INFERENCE_OVERLAP}%"
    )
    print(
        f"Batch Size: {cfg.BATCH_SIZE}, Epochs: {cfg.NUM_EPOCHS}, LR: {cfg.LEARNING_RATE}"
    )
    print(f"Num Classes: {cfg.NUM_CLASSES}, Input Channels: {cfg.INPUT_CHANNELS}")
    print(f"Seed: {cfg.SEED}, Validation Split: {cfg.VALIDATION_SPLIT}")
    print(f"---------------------")

    # --- Seed Everything for Reproducibility ---
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if cfg.DEVICE == "cuda":
        torch.cuda.manual_seed(cfg.SEED)
        # Optional: These might slightly hurt performance but increase reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Prepare Data ---
    print("Scanning for image/mask pairs (assuming identical filenames)...")
    image_files = []
    mask_files = []
    # Define common image extensions to look for
    image_extensions = ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files_scanned = 0
    files_matched = 0

    for ext in image_extensions:
        for img_path in cfg.IMAGE_DIR.glob(ext):
            files_scanned += 1
            # Construct the expected corresponding mask path
            expected_mask_path = (
                cfg.MASK_DIR / img_path.with_suffix(".png").name
            )  # Assumes mask has the *same name and .png extension*
            # Check if the corresponding mask file actually exists
            if expected_mask_path.is_file():
                image_files.append(img_path)
                mask_files.append(expected_mask_path)
                files_matched += 1
            else:
                print(
                    f"Warning: Mask file not found for image '{img_path.name}' at '{expected_mask_path}'. Skipping this image."
                )

    if not image_files:
        raise ValueError(
            f"No image files found with corresponding masks in {cfg.IMAGE_DIR} and {cfg.MASK_DIR}"
        )

    print(
        f"Found {files_matched} matching image/mask pairs out of {files_scanned} files scanned in {cfg.IMAGE_DIR}."
    )
    assert len(image_files) == len(
        mask_files
    ), "Mismatch between image and mask file counts after filtering."

    # --- Split Data ---
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_files,  # List of valid image paths
        mask_files,  # Corresponding list of mask paths
        test_size=cfg.VALIDATION_SPLIT,
        random_state=cfg.SEED,
        shuffle=True,  # Shuffle before splitting
    )
    print(
        f"Splitting data: {len(train_img_paths)} train, {len(val_img_paths)} validation images."
    )

    # --- Create Datasets and DataLoaders ---
    # Define transformations
    # Ensure get_minimal_transforms includes normalization and ToTensor
    # Ideally, define separate train/val transforms if augmentation is used only in training
    train_transform = T.get_minimal_transforms()  # Add augmentations here if needed
    val_transform = T.get_minimal_transforms()  # Usually just normalization + ToTensor

    print("Initializing datasets...")
    # Use the TileTrainingDataset (with jitter/no-padding) for training
    train_dataset = TileTrainingDataset(
        image_paths=train_img_paths,
        mask_paths=train_mask_paths,
        tile_size=cfg.TILE_SIZE,
        stride=cfg.TRAIN_TILE_STRIDE,
        # Add jitter_max here if implemented in TileTrainingDataset
        # jitter_max=cfg.TILE_JITTER, # Example if added to config
        transform=train_transform,
    )
    # Use FullImageDataset for validation
    val_dataset = FullImageDataset(
        image_paths=val_img_paths,
        mask_paths=val_mask_paths,
        transform=val_transform,  # Apply validation transforms (e.g., normalization)
    )

    print("Initializing dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,  # Shuffle training data each epoch
        # num_workers=cfg.NUM_WORKERS,
        # pin_memory=cfg.PIN_MEMORY,  # Speeds up CPU->GPU transfer if True
        drop_last=True,  # Drop last incomplete batch if dataset size is not divisible by batch size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # IMPORTANT: Validation batch size must be 1 for full image processing logic
        shuffle=False,  # No need to shuffle validation data
        # num_workers=cfg.NUM_WORKERS,
        # pin_memory=cfg.PIN_MEMORY,
    )

    # --- Model, Loss, Optimizer, Scheduler ---
    print(f"Initializing model: {cfg.MODEL_NAME.upper()}")
    # Ensure UNet arguments match your model definition
    model = UNet(n_channels=cfg.INPUT_CHANNELS, n_classes=cfg.NUM_CLASSES).to(
        cfg.DEVICE
    )

    # Select loss function based on the number of classes
    if cfg.NUM_CLASSES == 1:
        # For binary segmentation (1 output channel), use BCEWithLogitsLoss
        # Target masks should be float [0, 1] and shape [B, 1, H, W]
        loss_fn = nn.BCEWithLogitsLoss()
        print(
            "Using Binary Cross Entropy with Logits Loss (output layer should have 1 channel, targets should be float [0,1])"
        )
    else:
        # For multi-class segmentation (N output channels), use CrossEntropyLoss
        # Target masks should be long integers [0, N-1] and shape [B, H, W]
        loss_fn = nn.CrossEntropyLoss()
        print(
            "Using Cross Entropy Loss (output layer should have N channels, targets should be long [0..N-1])"
        )

    # Choose an optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    # Learning rate scheduler (e.g., reduce LR if validation metric plateaus)
    # 'max' mode because we want to maximize IoU
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=5, verbose=True
    )

    # --- Load Checkpoint (Optional) ---
    start_epoch = 0
    best_val_iou = 0.0  # Initialize best metric score
    # Ensure CHECKPOINT_DIR exists (should be handled by cfg.create_dirs())
    if cfg.CHECKPOINT_DIR.exists():
        # Attempt to load the *best* model first, then the *latest* checkpoint
        best_chkpt_pattern = f"best_model_epoch_*.pth.tar"
        latest_chkpt_pattern = f"checkpoint_epoch_*.pth.tar"

        # Find checkpoints, sort them (assuming filenames allow chronological sorting)
        best_checkpoints = sorted(cfg.CHECKPOINT_DIR.glob(best_chkpt_pattern))
        latest_checkpoints = sorted(cfg.CHECKPOINT_DIR.glob(latest_chkpt_pattern))

        load_path = None
        if best_checkpoints:
            # Load the most recent 'best' checkpoint
            load_path = best_checkpoints[-1]
            print(f"Found best checkpoint: {load_path.name}")
        elif latest_checkpoints:
            # If no 'best' found, load the most recent regular checkpoint
            load_path = latest_checkpoints[-1]
            print(f"Found latest checkpoint: {load_path.name}")
        else:
            print("No existing checkpoints found.")

        if load_path:
            print(f"Attempting to load checkpoint: {load_path}...")
            try:
                # Use the utility function to load state
                start_epoch, best_val_iou = utils.load_checkpoint(
                    checkpoint_path=load_path,
                    model=model,
                    optimizer=optimizer,  # Pass optimizer to potentially resume its state
                    device=cfg.DEVICE,
                )
                print(
                    f"Checkpoint loaded successfully. Resuming from epoch {start_epoch + 1}. Best previous IoU: {best_val_iou:.4f}"
                )
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                start_epoch = 0
                best_val_iou = 0.0
    else:
        print("Checkpoint directory not found. Starting training from scratch.")

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    training_start_time = time.time()
    train_loss_history = []
    val_iou_history = []

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        current_epoch = epoch + 1  # Use 1-based indexing for logging
        epoch_start_time = time.time()
        print("-" * 30)
        print(f"Epoch {current_epoch}/{cfg.NUM_EPOCHS}")
        current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate
        print(f"Learning Rate: {current_lr:.6e}")

        # --- Training Step ---
        # No need to set model.train() here, train_one_epoch does it
        train_loss = train_one_epoch(
            train_loader, model, optimizer, loss_fn, cfg.DEVICE
        )
        train_loss_history.append(train_loss)
        print(f"Epoch {current_epoch} Avg Training Loss: {train_loss:.4f}")

        # --- Validation Step ---
        # No need to set model.eval() here, validate_and_visualize does it
        current_val_iou = validate_and_visualize(
            val_loader, model, cfg.DEVICE, current_epoch  # Pass epoch number for saving
        )
        val_iou_history.append(current_val_iou)
        print(f"Epoch {current_epoch} Validation Mean IoU: {current_val_iou:.4f}")

        # --- Update Learning Rate Scheduler ---
        # Scheduler step is based on the validation metric (IoU in this case)
        scheduler.step(current_val_iou)

        # --- Save Checkpoint ---
        is_best = current_val_iou > best_val_iou
        if is_best:
            print(
                f"*** New best validation IoU: {current_val_iou:.4f} (Previous best: {best_val_iou:.4f}) ***"
            )
            best_val_iou = current_val_iou
            # Save the best model checkpoint
            utils.save_checkpoint(
                state={
                    "epoch": current_epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_val_iou,  # Store the metric value that made this the best
                    # Optionally add scheduler state: 'scheduler': scheduler.state_dict()
                },
                # Use a descriptive filename for the best model
                filename=f"best_model_epoch_{current_epoch}_iou_{best_val_iou:.4f}.pth.tar",
                checkpoint_dir=cfg.CHECKPOINT_DIR,  # Pass the directory explicitly
            )

        # Save a regular checkpoint periodically or at the end
        if current_epoch % cfg.CHECKPOINT_FREQ == 0 or epoch == cfg.NUM_EPOCHS - 1:
            utils.save_checkpoint(
                state={
                    "epoch": current_epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_val_iou,  # Also save the current best metric here for reference
                    # Optionally add scheduler state: 'scheduler': scheduler.state_dict()
                },
                # Use a standard filename for regular checkpoints
                filename=f"checkpoint_epoch_{current_epoch}.pth.tar",
                checkpoint_dir=cfg.CHECKPOINT_DIR,
            )

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {current_epoch} completed in {epoch_duration:.2f} seconds.")

        # --- Plot Metrics (Optional: Plot after each epoch) ---
        if len(train_loss_history) > 0 and len(val_iou_history) > 0:
            utils.plot_metrics(
                train_losses=train_loss_history,
                val_metrics=val_iou_history,  # Assuming IoU is the validation metric
                metric_name="Mean IoU",  # Label for the validation metric axis
                save_path=cfg.PLOT_DIR
                / "training_metrics_live.png",  # Overwrite plot each epoch
            )

    # --- End of Training ---
    training_duration = time.time() - training_start_time
    print("\n" + "=" * 30)
    print(f"--- Training Finished ---")
    print(
        f"Total training time: {training_duration / 3600:.2f} hours ({training_duration:.2f} seconds)"
    )
    print(f"Best Validation Mean IoU achieved: {best_val_iou:.4f}")
    # Find the epoch number corresponding to the best IoU if needed
    # best_epoch = val_iou_history.index(max(val_iou_history)) + 1 if val_iou_history else 'N/A'
    # print(f"Best IoU occurred at epoch: {best_epoch}")
    print(f"Checkpoints saved in: {cfg.CHECKPOINT_DIR}")
    print(f"Validation predictions saved in: {cfg.VALIDATION_PRED_DIR}")
    print(f"Plots saved in: {cfg.PLOT_DIR}")
    print("=" * 30)

    # --- Final Plot Saving ---
    # Save the final plot with a different name
    if len(train_loss_history) > 0 and len(val_iou_history) > 0:
        utils.plot_metrics(
            train_losses=train_loss_history,
            val_metrics=val_iou_history,
            metric_name="Mean IoU",
            save_path=cfg.PLOT_DIR / "training_metrics_final.png",
        )


if __name__ == "__main__":
    # Ensure output directories exist before starting
    cfg.create_dirs()  # Call the static method from the imported config object
    main()
