import os
import glob
import random
import shutil
from pathlib import Path

import cv2  # OpenCV for image processing (pip install opencv-python)
import numpy as np
from ultralytics import YOLO  # (pip install ultralytics)
import yaml  # (pip install PyYAML)

# --- Configuration ---
INPUT_IMAGE_DIR = Path(
    "datasets/keypoints/asol-keypoint.v2i.coco-segmentation/train"
)  # Directory with your original images
INPUT_MASK_DIR = Path(
    "datasets/keypoints/asol-keypoint.v2i.coco-segmentation/masks"
)  # Directory with your original masks
OUTPUT_DATASET_DIR = Path("./yolo_dataset")  # Directory to save YOLO formatted data

# Dataset split configuration
VALIDATION_SPLIT = 0.20  # 20% for validation

# Class configuration
CLASS_NAMES = ["fence"]  # CHANGE THIS if you have different/more class names
# IMPORTANT: If you have multiple classes encoded by different pixel values in masks,
# the 'create_yolo_labels' function needs modification.
# This current version assumes all non-zero pixels in the mask belong to the first class (index 0).
NUM_CLASSES = len(CLASS_NAMES)

# YOLO Training configuration
MODEL_CONFIG = "yolov8n-seg.pt"  # Base model (n, s, m, l, x) -seg
EPOCHS = 2
IMG_SIZE = 640
BATCH_SIZE = 8  # Adjust based on your GPU memory
PROJECT_NAME = "custom_segmentation"
RUN_NAME = "yolov8_seg_run_1"
# --- End Configuration ---

# --- Helper Functions ---


def create_yolo_labels(mask_path, label_path, image_shape):
    """
    Converts a mask image to YOLO segmentation label format and saves it.
    Assumes mask is binary or non-zero pixels represent the single object class.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}")
        return False

    h, w = image_shape[:2]  # Use image shape for normalization
    label_lines = []

    # Find contours - Use RETR_EXTERNAL to find only outer contours
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # print(f"Warning: No contours found in mask {mask_path}")
        # Create an empty file if no objects are present
        Path(label_path).touch()
        return True

    for contour in contours:
        # --- Check if contour is reasonably sized ---
        # This helps filter out tiny noise contours if necessary
        # You might need to adjust the threshold based on your objects
        if cv2.contourArea(contour) < 10:
            continue

        # Normalize contour points
        normalized_contour = contour.astype(np.float32)
        normalized_contour[:, :, 0] /= w  # Normalize x
        normalized_contour[:, :, 1] /= h  # Normalize y

        # YOLO format: class_index x1 y1 x2 y2 ... xn yn
        # Assuming single class with index 0
        class_index = 0
        segment = normalized_contour.flatten().tolist()
        label_line = f"{class_index} " + " ".join(map(str, segment))
        label_lines.append(label_line)

    # Save the label file
    if label_lines:
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))
        return True
    else:
        # Create an empty file if all contours were filtered out
        Path(label_path).touch()
        # print(f"Warning: All contours filtered out for mask {mask_path}")
        return True


def prepare_yolo_dataset():
    """
    Organizes images and generates YOLO labels, splitting into train/val sets.
    """
    print("Preparing YOLO dataset...")

    # Create output directories
    img_train_dir = OUTPUT_DATASET_DIR / "images" / "train"
    img_val_dir = OUTPUT_DATASET_DIR / "images" / "val"
    lbl_train_dir = OUTPUT_DATASET_DIR / "labels" / "train"
    lbl_val_dir = OUTPUT_DATASET_DIR / "labels" / "val"

    for dir_path in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Find all mask files (assuming they dictate the dataset items)
    mask_files = (
        list(INPUT_MASK_DIR.glob("*.png"))
        + list(INPUT_MASK_DIR.glob("*.jpg"))
        + list(INPUT_MASK_DIR.glob("*.jpeg"))
    )
    if not mask_files:
        print(f"Error: No masks found in {INPUT_MASK_DIR}")
        return False

    print(f"Found {len(mask_files)} mask files.")
    random.shuffle(mask_files)  # Shuffle for random split

    split_index = int(len(mask_files) * (1 - VALIDATION_SPLIT))
    train_masks = mask_files[:split_index]
    val_masks = mask_files[split_index:]

    print(f"Splitting dataset: {len(train_masks)} train, {len(val_masks)} validation.")

    processed_count = 0
    skipped_count = 0

    # Process Training Data
    print("\nProcessing Training Data...")
    for mask_path in train_masks:
        base_name = mask_path.stem  # Name without extension
        # Assume image has the same name but possibly different extension
        # Look for common image extensions
        img_path = None
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            potential_matches = list(
                INPUT_IMAGE_DIR.glob(f"{base_name}{ext.lstrip('*')}")
            )
            if potential_matches:
                img_path = potential_matches[0]
                break

        if not img_path or not img_path.exists():
            print(
                f"Warning: Corresponding image for mask {mask_path.name} not found in {INPUT_IMAGE_DIR}. Skipping."
            )
            skipped_count += 1
            continue

        # Read image to get dimensions for normalization
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            skipped_count += 1
            continue

        # Define output paths
        output_img_path = img_train_dir / img_path.name
        output_label_path = lbl_train_dir / f"{base_name}.txt"

        # Copy image
        shutil.copy(str(img_path), str(output_img_path))

        # Create label file
        if create_yolo_labels(mask_path, output_label_path, img.shape):
            processed_count += 1
        else:
            print(f"Warning: Failed to process mask {mask_path.name}. Skipping.")
            # Clean up potentially copied image if label creation failed fundamentally
            if output_img_path.exists():
                output_img_path.unlink()
            skipped_count += 1
        print(f"Processed {processed_count}/{len(mask_files)}", end="\r")

    # Process Validation Data
    print("\nProcessing Validation Data...")
    for mask_path in val_masks:
        base_name = mask_path.stem
        img_path = None
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            potential_matches = list(
                INPUT_IMAGE_DIR.glob(f"{base_name}{ext.lstrip('*')}")
            )
            if potential_matches:
                img_path = potential_matches[0]
                break

        if not img_path or not img_path.exists():
            print(
                f"Warning: Corresponding image for mask {mask_path.name} not found in {INPUT_IMAGE_DIR}. Skipping."
            )
            skipped_count += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            skipped_count += 1
            continue

        output_img_path = img_val_dir / img_path.name
        output_label_path = lbl_val_dir / f"{base_name}.txt"

        shutil.copy(str(img_path), str(output_img_path))

        if create_yolo_labels(mask_path, output_label_path, img.shape):
            processed_count += 1
        else:
            print(f"Warning: Failed to process mask {mask_path.name}. Skipping.")
            if output_img_path.exists():
                output_img_path.unlink()
            skipped_count += 1
        print(f"Processed {processed_count}/{len(mask_files)}", end="\r")

    print(
        f"\nDataset preparation complete. Total processed: {processed_count}, Skipped: {skipped_count}"
    )
    if processed_count == 0:
        print("Error: No images/masks were successfully processed.")
        return False
    return True


def create_yaml_file():
    """Creates the data.yaml file for YOLO training."""
    print("Creating data.yaml file...")
    data_yaml = {
        "path": str(OUTPUT_DATASET_DIR.resolve()),  # Absolute path to dataset root
        "train": str(Path("images") / "train"),  # Relative path from 'path'
        "val": str(Path("images") / "val"),  # Relative path from 'path'
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    yaml_path = OUTPUT_DATASET_DIR / "data.yaml"
    try:
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=None)
        print(f"data.yaml saved to {yaml_path}")
        return str(yaml_path)
    except Exception as e:
        print(f"Error creating data.yaml: {e}")
        return None


# --- Main Execution ---

if __name__ == "__main__":
    # 1. Check input directories
    if not INPUT_IMAGE_DIR.is_dir():
        print(f"Error: Input image directory not found: {INPUT_IMAGE_DIR}")
        exit()
    if not INPUT_MASK_DIR.is_dir():
        print(f"Error: Input mask directory not found: {INPUT_MASK_DIR}")
        exit()

    # 2. Prepare the dataset (convert masks, split data)
    if not prepare_yolo_dataset():
        print("Failed to prepare dataset. Exiting.")
        exit()

    # 3. Create the data.yaml file
    yaml_file_path = create_yaml_file()
    if not yaml_file_path:
        print("Failed to create data.yaml. Exiting.")
        exit()

    # 4. Train the YOLOv8-seg model
    print("\nStarting YOLOv8 segmentation training...")
    try:
        # Load a pretrained YOLOv8 segmentation model
        model = YOLO(MODEL_CONFIG)

        # Train the model
        results = model.train(
            data=yaml_file_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=RUN_NAME,
            device="cuda:0",
            # You can add more training arguments here if needed
            # e.g., device='cuda:0', patience=10, etc.
        )
        print("Training finished.")
        print(
            f"Results saved to: {results.save_dir}"
        )  # Access the results object for save directory

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback

        traceback.print_exc()
