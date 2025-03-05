"""
Script to generate segmentation masks from COCO keypoint annotations.

This script reads a COCO JSON annotation file, extracts segmentation keypoints
for 'vertical' and 'horizontal' categories, and generates binary masks
by drawing lines connecting these keypoints. Masks are saved as grayscale images.
"""

import json
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  # Keeping for display, consider removing for production

# --- Configuration ---
ROBOFLOW_PROJECT_DIR = Path("asol-keypoint.v2i.coco-segmentation")
ANNOTATION_FILE_PATH = ROBOFLOW_PROJECT_DIR / "train" / "_annotations.coco.json"
IMAGES_DIR = ROBOFLOW_PROJECT_DIR / "train"
MASK_OUTPUT_DIR = ROBOFLOW_PROJECT_DIR / "masks"
VERTICAL_CATEGORY_ID = 1
HORIZONTAL_CATEGORY_ID = 2
LINE_THICKNESS = 5
MASK_COLOR = 255  # White color for lines in the mask (grayscale)
MASK_SUFFIX = ".jpg"  # Suffix for mask filenames


def load_annotations(annotation_file: Path) -> dict:
    """
    Loads COCO annotations from a JSON file.

    Args:
        annotation_file: Path to the COCO JSON annotation file.

    Returns:
        A dictionary containing the loaded JSON data.
    """
    try:
        with open(annotation_file, mode="r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(
            f"Error decoding JSON in file: {annotation_file}. "
            "Please check if the file is a valid JSON."
        )


def create_image_dict(json_data: dict) -> dict[int, tuple[str, int, int]]:
    """
    Creates a dictionary mapping image IDs to image metadata (filename, height, width).

    Args:
        json_data: Loaded JSON data containing image information.

    Returns:
        A dictionary where keys are image IDs and values are tuples:
        (filename, height, width).
    """
    image_dict = {}
    for image in json_data.get("images", []):
        image_dict[image["id"]] = (
            image["file_name"],
            image["height"],
            image["width"],
        )
    return image_dict


def create_annotations_dict(json_data: dict) -> dict[int, list[dict]]:
    """
    Creates a dictionary mapping image IDs to a list of their annotations.

    Args:
        json_data: Loaded JSON data containing annotation information.

    Returns:
        A dictionary where keys are image IDs and values are lists of annotation dictionaries.
    """
    img_annotations = {}
    for anno in json_data.get("annotations", []):
        image_id = anno["image_id"]
        if image_id not in img_annotations:
            img_annotations[image_id] = []
        img_annotations[image_id].append(anno)
    return img_annotations


def extract_segmentation_keypoints(annotation: dict) -> list[tuple[int, int]]:
    """
    Extracts segmentation keypoints as a list of (x, y) tuples from an annotation.

    Args:
        annotation: An annotation dictionary containing segmentation data.

    Returns:
        A list of (x, y) tuples representing the segmentation keypoints.
    """
    keypoints_flat = annotation["segmentation"][0]
    keypoints = [
        (int(keypoints_flat[i]), int(keypoints_flat[i + 1]))
        for i in range(0, len(keypoints_flat), 2)
    ]
    return keypoints


def process_image_annotations(
    image_id: int,
    file_name: str,
    height: int,
    width: int,
    annotations: list[dict],
    images_dir: Path,
    mask_output_dir: Path,
):
    """
    Processes annotations for a single image, generates a mask, and saves it.

    Args:
        image_id: ID of the image being processed.
        file_name: Filename of the image.
        height: Height of the image.
        width: Width of the image.
        annotations: List of annotations for the image.
        images_dir: Path to the directory containing the images.
        mask_output_dir: Path to the directory to save the generated masks.
    """
    img_path = images_dir / file_name
    img = cv.imread(str(img_path))
    if img is None:
        print(
            f"Warning: Could not read image file: {img_path}. Skipping mask generation."
        )
        return

    mask = np.zeros((height, width, 1), dtype=np.uint8)
    anno_vertical_points = []
    anno_horizontal_points = []

    for anno in annotations:
        keypoints = extract_segmentation_keypoints(anno)
        category_id = anno["category_id"]

        if category_id == VERTICAL_CATEGORY_ID:
            anno_vertical_points.append(keypoints)
        elif category_id == HORIZONTAL_CATEGORY_ID:
            anno_horizontal_points.append(keypoints)
        else:
            raise ValueError(f"Unknown category_id: {category_id}")

    # Connect points within each segmentation for vertical lines
    for p1, p2 in zip(*anno_vertical_points):
        cv.line(mask, p1, p2, MASK_COLOR, LINE_THICKNESS)

    # Connect points within each segmentation for horizontal lines
    for p1, p2 in zip(*anno_horizontal_points):
        cv.line(mask, p1, p2, MASK_COLOR, LINE_THICKNESS)

    mask_filename = Path(file_name).stem + MASK_SUFFIX
    mask_filepath = mask_output_dir / mask_filename
    cv.imwrite(str(mask_filepath), mask)
    print(f"Mask generated and saved: {mask_filepath}")


def main():
    """
    Main function to load annotations, process images, and generate masks.
    """
    print("Starting mask generation from COCO keypoint annotations...")

    mask_output_dir = MASK_OUTPUT_DIR
    mask_output_dir.mkdir(
        parents=True, exist_ok=True
    )  # Create output directory if needed

    try:
        json_data = load_annotations(ANNOTATION_FILE_PATH)
        image_dict = create_image_dict(json_data)
        img_annotations = create_annotations_dict(json_data)

        if not image_dict:
            print("Warning: No images found in the annotation file.")
        if not img_annotations:
            print("Warning: No annotations found in the annotation file.")

        for img_id, (file_name, height, width) in image_dict.items():
            if img_id in img_annotations:
                process_image_annotations(
                    img_id,
                    file_name,
                    height,
                    width,
                    img_annotations[img_id],
                    IMAGES_DIR,
                    mask_output_dir,
                )
            else:
                print(f"Warning: No annotations found for image ID: {img_id}")

        print("\nMask generation process completed.")

    except FileNotFoundError as e:
        print(f"File error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

    # Example to display the last generated mask (for debugging/demonstration)
    # plt.subplots(1, 1)[1].imshow(mask, cmap="gray") # `mask` is not accessible here, needs to be returned from process_image_annotations if you want to display it
    # plt.show()
