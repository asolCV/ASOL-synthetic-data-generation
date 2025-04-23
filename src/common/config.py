# common/config.py
from pathlib import Path
import torch
import os  # Keep os for cpu_count


class Config:
    # Data Paths - Adjust these to your actual data locations if needed
    # Consider using absolute paths or paths relative to a known root if running scripts from different locations
    PROJECT_ROOT = (
        Path(__file__).resolve().parent.parent.parent
    )  # Assumes config.py is in common/ which is in project root
    DATA_ROOT = PROJECT_ROOT / "datasets"  # Example relative path
    KEYPOINTS_PATH = (
        DATA_ROOT / "keypoints" / "asol-keypoint-4"
    )  # Example relative path
    IMAGE_DIR = KEYPOINTS_PATH / "train"  # Example relative path
    MASK_DIR = KEYPOINTS_PATH / "masks"  # Example relative path
    OUTPUT_DIR = DATA_ROOT / "output"  # Example relative path

    # Model & Training Params
    MODEL_NAME = "unet"
    NUM_CLASSES = 1  # Binary: 1 class (foreground) + background assumed
    INPUT_CHANNELS = 3  # RGB

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = DEVICE == "cuda"  # Simplified boolean check
    NUM_WORKERS = (
        os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0
    )  # Safer default
    SEED = 42

    # --- Tiling Configuration ---
    TILE_SIZE = 128  # Size of the square tiles for training
    TRAIN_TILE_STRIDE = TILE_SIZE // 2  # Example: 50% overlap

    # Training Hyperparameters
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    CHECKPOINT_FREQ = 5
    VALIDATION_SPLIT = 0.15

    # --- Sliding Window Inference/Validation ---
    INFERENCE_TILE_SIZE = 128
    INFERENCE_OVERLAP = 20

    # --- Validation Output ---
    SAVE_VALIDATION_PREDS = True
    NUM_VALIDATION_SAVE = 2

    # --- Derived Paths --- (Defined after OUTPUT_DIR)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    VALIDATION_PRED_DIR = OUTPUT_DIR / "validation_preds"
    PLOT_DIR = OUTPUT_DIR / "plots"

    # --- Ensure Directories Exist ---
    @staticmethod
    def create_dirs():
        """Creates necessary output directories."""
        paths_to_create = [
            Config.OUTPUT_DIR,
            Config.CHECKPOINT_DIR,
            Config.VALIDATION_PRED_DIR,
            Config.PLOT_DIR,
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)


cfg = Config()
# Call the static method to create directories when config is loaded/imported
cfg.create_dirs()
