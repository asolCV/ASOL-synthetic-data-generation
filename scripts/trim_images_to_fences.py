"""
Script to efficiently trim images to the fence (pole) area based on semantic segmentation masks.

This script iterates through segmentation mask images, identifies pole contours,
and trims the corresponding color images to regions surrounding pairs of pole contours.
It saves the trimmed images to a specified output directory.
"""

from pathlib import Path
import cv2 as cv
import numpy as np
import csv

# --- Configuration ---
SEMANTIC_DATA_DIR = Path("dataset/semantic-segmentation-asol")
TRIMMED_DATA_DIR = Path("dataset/trimmed-data/images")
POLE_CLASS_NAME = "Pole"
CLASSES_FILE_PATH = SEMANTIC_DATA_DIR / "train" / "_classes.csv"
MASK_SUFFIX = "_mask"
IMAGE_SUFFIX = ".jpg"
OUTPUT_IMAGE_SUFFIX = ".png"  # Saving trimmed images as PNG for lossless quality
MIN_CONTOUR_AREA = 1000
CENTROID_OFFSET_X = 40
VERTICAL_PADDING_TOP = 250
VERTICAL_PADDING_BOTTOM = 50


def load_classes_dict(classes_file: Path) -> dict[str, int]:
    """
    Loads the class name to pixel value mapping from the classes CSV file.

    Args:
        classes_file: Path to the classes CSV file.

    Returns:
        A dictionary where keys are class names and values are corresponding pixel values.
    """
    classes_dict = {}
    with open(classes_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            classes_dict[row[1].strip()] = int(row[0])
    return classes_dict


def find_pole_contours(mask_image: np.ndarray, pole_value: int, min_area: int) -> list:
    """
    Finds contours of pole pixels in a grayscale mask image.

    Args:
        mask_image: Grayscale mask image as a NumPy array.
        pole_value: Pixel value representing the pole class.
        min_area: Minimum contour area to consider.

    Returns:
        A list of pole contours that meet the minimum area criteria, sorted by x-coordinate.
    """
    pole_pixels = (mask_image == pole_value).astype(np.uint8)  # Extract pole pixels
    contours, _ = cv.findContours(pole_pixels, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv.contourArea(c) > min_area]
    valid_contours.sort(key=lambda x: cv.boundingRect(x)[0])  # Sort by x-coordinate
    return valid_contours


def trim_and_save_image(
    mask_file: Path,
    color_image: np.ndarray,
    contours: list,
    trimmed_dataset_dir: Path,
    idx: int,
    h_max: int,
    w_max: int,
):
    """
    Trims a color image based on a pair of pole contours and saves the trimmed image.

    Args:
        mask_file: Path to the mask file (for naming the output).
        color_image: The color image to be trimmed as a NumPy array.
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
    trimmed_image = color_image[start_y:end_y, start_x:end_x]

    # Construct output file name and path
    save_img_name = mask_file.name.replace(MASK_SUFFIX, f"_{idx}")
    output_path = trimmed_dataset_dir / save_img_name

    # Save the trimmed image
    cv.imwrite(str(output_path), trimmed_image)
    print(f"Saved trimmed image: {save_img_name}")


def process_mask_file(mask_file: Path, classes_dict: dict, trimmed_dataset_dir: Path):
    """
    Processes a single mask file to find pole contours and trim the corresponding color image.

    Args:
        mask_file: Path to the mask file.
        classes_dict: Dictionary mapping class names to pixel values.
        trimmed_dataset_dir: Directory to save trimmed images.
    """
    if (
        not mask_file.is_file()
        or mask_file.suffix != ".png"
        or MASK_SUFFIX not in mask_file.name
    ):
        return  # Skip if not a mask file

    print(f"\nProcessing mask file: {mask_file.name}")
    mask_image = cv.imread(str(mask_file), cv.IMREAD_GRAYSCALE)

    if mask_image is None:
        print(f"Error: Could not read mask image: {mask_file}")
        return

    pole_value = classes_dict.get(POLE_CLASS_NAME)
    if pole_value is None:
        print(f"Error: Pole class '{POLE_CLASS_NAME}' not found in classes dictionary.")
        return

    contours = find_pole_contours(mask_image, pole_value, MIN_CONTOUR_AREA)
    print(f"Number of pole contours found: {len(contours)}")

    if len(contours) <= 1:
        print("Info: Less than 2 contours found, skipping trimming for this image.")
        return

    # Construct path to the corresponding color image
    color_image_path = (
        SEMANTIC_DATA_DIR
        / "train"
        / mask_file.with_suffix(IMAGE_SUFFIX).name.replace(MASK_SUFFIX, "")
    )
    color_image = cv.imread(str(color_image_path))

    if color_image is None:
        print(f"Error: Could not read corresponding color image: {color_image_path}")
        return

    h_max, w_max, _ = color_image.shape  # Get color image dimensions

    # Process contours in pairs
    for idx, contour_pair in enumerate(zip(contours[:-1], contours[1:])):
        trim_and_save_image(
            mask_file, color_image, contour_pair, trimmed_dataset_dir, idx, h_max, w_max
        )


def main():
    """Main function to execute the image trimming script."""

    # Create trimmed dataset directory if it doesn't exist
    trimmed_dataset_dir = TRIMMED_DATA_DIR
    trimmed_dataset_dir.mkdir(parents=True, exist_ok=True)

    classes_dict = load_classes_dict(CLASSES_FILE_PATH)

    mask_dir = SEMANTIC_DATA_DIR / "train"
    if not mask_dir.is_dir():
        print(f"Error: Mask directory not found: {mask_dir}")
        return

    for mask_file in mask_dir.iterdir():
        process_mask_file(mask_file, classes_dict, trimmed_dataset_dir)

    print("\nImage trimming process completed.")


if __name__ == "__main__":
    main()
