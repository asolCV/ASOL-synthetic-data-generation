"""
Script to efficiently trim images to the fence (pole) area based on YOLOv8s-seg model inference.

This script uses a trained YOLOv8s-seg model to perform segmentation on images,
identifies pole contours from the segmentation masks, and trims the images
to regions surrounding pairs of pole contours. It saves the trimmed images
to a specified output directory.
"""

from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import torch

from asol_synthetic_data_generation.common.paths import *

# --- Configuration ---
# Modify INPUT_DATA_DIR to point to the directory containing images to trim
# and OUTPUT_TRIMMED_IMAGES_DIR to specify the output directory for trimmed images.
INPUT_DATA_DIR = (
    DATASETS_DIR / "instance-segmentation"
)  # Directory containing images to trim
OUTPUT_TRIMMED_IMAGES_DIR = DATASETS_DIR / "trimmed-images"  # Output directory

# Leave as fixed
POLE_CLASS_NAME = "Pole"  # Class name for poles in the YOLO model

# Modify these parameters as needed
MIN_CONFIDENCE = 0.2  # Minimum confidence threshold for YOLO predictions
MIN_CONTOUR_AREA = 1000  # Minimum contour area to consider
CENTROID_OFFSET_X = 40  # Horizontal padding around pole centroids
VERTICAL_PADDING_TOP = 250  # Padding above the pole
VERTICAL_PADDING_BOTTOM = 50  # Padding below the pole
OUTPUT_IMAGE_SUFFIX = ".png"  # Output image format


def load_yolo_model(model_path: Path) -> YOLO:
    """
    Loads the YOLOv8s-seg model from the specified path.

    Args:
        model_path: Path to the YOLOv8s-seg model file.

    Returns:
        Loaded YOLO model.
    """
    try:
        model = YOLO(str(model_path), task="segmentation")
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise


def get_segmentation_mask(model: YOLO, image_path: Path) -> np.ndarray:
    """
    Performs inference with YOLOv8s-seg model to get segmentation mask for poles.

    Args:
        model: Loaded YOLO model.
        image_path: Path to the input image.

    Returns:
        Binary mask with pole segments.
    """
    # Read image
    image = cv.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR to RGB for YOLO
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Get image dimensions
    h, w = image.shape[:2]

    # Create empty mask with same dimensions as input image
    mask = np.zeros((h, w), dtype=np.uint8)

    # Perform inference
    results = model(image_rgb, conf=MIN_CONFIDENCE)

    # Check if we have any results
    if not results or len(results) == 0:
        print(f"No detections found in image: {image_path.name}")
        return mask

    # Get first result (assuming single image input)
    result = results[0]
    names = result.names

    # Check if masks exist in the results
    if not hasattr(result, "masks") or result.masks is None:
        print(f"No segmentation masks found in image: {image_path.name}")
        return mask

    mask_array = np.zeros(result.orig_shape, dtype=np.int32)

    # Process each mask
    for i, (cls, conf) in enumerate(
        zip(
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        )
    ):
        # Check if this is a pole with sufficient confidence
        if names[int(cls)] == POLE_CLASS_NAME and float(conf) >= MIN_CONFIDENCE:

            cv.fillPoly(mask_array, np.int32(result.masks[i].xy), 255)

    return mask_array


def find_pole_contours(mask_image: np.ndarray, min_area: int) -> list:
    """
    Finds contours of pole pixels in a binary mask image.

    Args:
        mask_image: Binary mask image as a NumPy array.
        min_area: Minimum contour area to consider.

    Returns:
        A list of pole contours that meet the minimum area criteria, sorted by x-coordinate.
    """
    # Ensure mask is not empty
    if np.max(mask_image) == 0:
        return []

    contours, _ = cv.findContours(
        np.uint8(mask_image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    valid_contours = [c for c in contours if cv.contourArea(c) > min_area]
    valid_contours.sort(key=lambda x: cv.boundingRect(x)[0])  # Sort by x-coordinate
    return valid_contours


def trim_and_save_image(
    image_path: Path,
    image: np.ndarray,
    contours: list,
    trimmed_dataset_dir: Path,
    idx: int,
    h_max: int,
    w_max: int,
):
    """
    Trims a color image based on a pair of pole contours and saves the trimmed image.

    Args:
        image_path: Path to the original image (for naming the output).
        image: The color image to be trimmed as a NumPy array.
        contours: A list of two contours representing a pair of poles.
        trimmed_dataset_dir: Path to the directory where trimmed images will be saved.
        idx: Index of the contour pair (for naming the output).
        h_max: Maximum height of the image (for boundary checks).
        w_max: Maximum width of the image (for boundary checks).
    """
    c1, c2 = contours
    c1_moment = cv.moments(c1)
    c2_moment = cv.moments(c2)

    # Calculate centroids, handling potential division by zero
    c1_centroid_x = (
        int(c1_moment["m10"] / c1_moment["m00"]) if c1_moment["m00"] != 0 else 0
    )
    c2_centroid_x = (
        int(c2_moment["m10"] / c2_moment["m00"]) if c2_moment["m00"] != 0 else 0
    )

    start_x = max(c1_centroid_x - CENTROID_OFFSET_X, 0)
    end_x = min(c2_centroid_x + CENTROID_OFFSET_X, w_max)

    combined_contour = np.concatenate(contours)
    x, y, w, h = cv.boundingRect(combined_contour)

    # Define trimming boundaries with padding
    start_y = max(y - VERTICAL_PADDING_TOP, 0)
    end_y = min(y + h + VERTICAL_PADDING_BOTTOM, h_max)

    # Extract trimmed image region
    trimmed_image = image[start_y:end_y, start_x:end_x]

    # Construct output file name and path
    save_img_name = f"{image_path.stem}_pole_{idx}{OUTPUT_IMAGE_SUFFIX}"
    output_path = trimmed_dataset_dir / save_img_name

    # Save the trimmed image
    cv.imwrite(str(output_path), trimmed_image)
    print(f"Saved trimmed image: {save_img_name}")


def process_image(image_path: Path, model: YOLO, trimmed_dataset_dir: Path):
    """
    Processes a single image using the YOLO model, finds pole contours, and trims the image.

    Args:
        image_path: Path to the input image file.
        model: Loaded YOLO model for segmentation.
        trimmed_dataset_dir: Directory to save trimmed images.
    """
    print(f"\nProcessing image: {image_path.name}")

    # Read the input image
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    h_max, w_max, _ = image.shape  # Get image dimensions

    try:
        # Get segmentation mask from YOLO model
        mask = get_segmentation_mask(model, image_path)

        # Find pole contours
        contours = find_pole_contours(mask, MIN_CONTOUR_AREA)
        print(f"Number of pole contours found: {len(contours)}")

        if len(contours) <= 1:
            print("Info: Less than 2 contours found, skipping trimming for this image.")
            return

        # Process contours in pairs
        for idx, contour_pair in enumerate(zip(contours[:-1], contours[1:])):
            trim_and_save_image(
                image_path, image, contour_pair, trimmed_dataset_dir, idx, h_max, w_max
            )

    except Exception as e:
        print(f"Error processing image {image_path.name}: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function to execute the image trimming script."""

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create trimmed dataset directory if it doesn't exist
    trimmed_dataset_dir = OUTPUT_TRIMMED_IMAGES_DIR
    trimmed_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    model = load_yolo_model(YOLOV8S_SEG_POLE_MODEL_PATH)

    # Set model device
    model.to(device)

    # Get input images
    input_dir = INPUT_DATA_DIR
    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Process each image
    image_extensions = [".jpg", ".jpeg", ".png"]
    for image_path in input_dir.glob("**/*"):
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            process_image(image_path, model, trimmed_dataset_dir)

    print("\nImage trimming process completed.")


if __name__ == "__main__":
    print("hello")
    breakpoint()
    main()
