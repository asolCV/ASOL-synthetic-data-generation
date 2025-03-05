"""
Script to generate inpainted images and YOLO annotations for all combinations of images and shape masks.

This script iterates through each image and combines it with every shape mask available.
For each combination, it performs inpainting and generates a YOLO bounding box annotation
around the inpainted 'Hole' regions. This results in N * M output images and annotation files,
where N is the number of images and M is the number of shape masks.
Additionally, this script generates a YOLO names file (names.txt) in the YOLO annotations directory.
"""

from pathlib import Path
import cv2 as cv
import numpy as np

# --- Configuration ---
ROBOFLOW_PROJECT_DIR = Path("asol-keypoint.v2i.coco-segmentation")
IMAGES_DIR = ROBOFLOW_PROJECT_DIR / "train"
FENCE_MASKS_DIR = ROBOFLOW_PROJECT_DIR / "masks"
SHAPE_MASKS_DIR = Path("shape_masks")
OUTPUT_INPAINTED_IMAGES_DIR = Path(
    "inpainted_images_combined"
)  # Changed output dir to indicate combinations
OUTPUT_YOLO_ANNOTATIONS_DIR = Path(
    "yolo_annotations_combined"
)  # Changed output dir to indicate combinations
DILATION_KERNEL_SIZE = 3
DILATION_ITERATIONS = 1
INPAINT_RADIUS = 11
YOLO_CLASS_ID = 0  # Assuming 'Hole' class is class ID 0 in YOLO
MASK_RESIZE_INTERPOLATION = cv.INTER_NEAREST  # or cv.INTER_LINEAR for smoother resize
YOLO_CLASS_NAMES = ["Hole"]  # Define class names here


def load_image(image_path: Path) -> np.ndarray:
    """Loads an image from the given path."""
    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")
    return img


def load_mask_grayscale(mask_path: Path) -> np.ndarray:
    """Loads a mask image in grayscale from the given path."""
    mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask file: {mask_path}")
    return mask


def create_mask_from_fence_and_shape(
    img: np.ndarray, fence_mask: np.ndarray, shape_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combines fence mask and shape mask, dilates the fence mask, and performs bitwise_and.
    """
    # Resize shape mask to image dimensions
    resized_shape_mask = cv.resize(
        shape_mask,
        (img.shape[1], img.shape[0]),
        interpolation=MASK_RESIZE_INTERPOLATION,
    )

    # Dilate the fence mask to expand the masked region
    kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    dilated_fence_mask = cv.dilate(fence_mask, kernel, iterations=DILATION_ITERATIONS)

    # Combine dilated fence mask with the resized shape mask using bitwise AND
    combined_mask = cv.bitwise_and(dilated_fence_mask, resized_shape_mask)
    return (
        combined_mask,
        dilated_fence_mask,
    )  # Return both combined and dilated fence mask


def inpaint_image_with_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaints the image using the provided mask."""
    return cv.inpaint(img, mask, inpaintRadius=INPAINT_RADIUS, flags=cv.INPAINT_TELEA)


def calculate_bounding_box(mask: np.ndarray) -> tuple:
    """Calculates a YOLO format bounding box around the masked region."""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # No contours found, no bounding box

    # Find the largest contour (assuming one main 'Hole' area)
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    return x, y, w, h


def convert_bbox_to_yolo(bbox: tuple, image_width: int, image_height: int) -> str:
    """
    Converts bounding box coordinates to YOLO format.
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    with values normalized by image width and height.
    """
    if bbox is None:
        return ""  # No bounding box found

    x, y, w, h = bbox
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    norm_w = w / image_width
    norm_h = h / image_height
    return f"{YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def save_inpainted_image(inpainted_img: np.ndarray, output_path: Path):
    """Saves the inpainted image to the specified path."""
    cv.imwrite(str(output_path), inpainted_img)


def save_yolo_annotation(yolo_annotation: str, output_path: Path):
    """Saves the YOLO annotation string to a text file."""
    with open(output_path, "w") as f:
        f.write(yolo_annotation)


def generate_yolo_names_file(output_annotations_dir: Path, class_names: list[str]):
    """Generates the YOLO names.txt file in the output annotation directory."""
    names_file_path = output_annotations_dir / "names.txt"
    with open(names_file_path, "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"YOLO names file created at: {names_file_path}")


def process_images_and_masks(
    images_dir: Path,
    fence_masks_dir: Path,
    shape_masks_dir: Path,
    output_images_dir: Path,
    output_annotations_dir: Path,
):
    """
    Processes images and masks to perform inpainting and generate YOLO annotations for all combinations.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    # Generate YOLO names file
    generate_yolo_names_file(output_annotations_dir, YOLO_CLASS_NAMES)

    shape_mask_files = [
        f
        for f in shape_masks_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    if not shape_mask_files:
        print(
            f"Warning: No shape mask files found in {shape_masks_dir}. Skipping inpainting."
        )
        return

    image_files = [
        f for f in images_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    if not image_files:
        print(f"Warning: No image files found in {images_dir}. Skipping inpainting.")
        return

    for img_file in image_files:
        try:
            img = load_image(img_file)
            fence_mask = load_mask_grayscale(fence_masks_dir / img_file.name)

            for (
                shape_mask_file
            ) in (
                shape_mask_files
            ):  # Inner loop: iterate through all shape masks for each image
                try:
                    shape_mask = load_mask_grayscale(shape_mask_file)

                    combined_mask, dilated_fence_mask = (
                        create_mask_from_fence_and_shape(img, fence_mask, shape_mask)
                    )
                    inpainted_img = inpaint_image_with_mask(img, combined_mask)
                    bbox = calculate_bounding_box(combined_mask)
                    yolo_annotation = convert_bbox_to_yolo(
                        bbox, img.shape[1], img.shape[0]
                    )

                    # Construct output filenames to be unique for each combination
                    output_image_name = (
                        f"{img_file.stem}_{shape_mask_file.stem}{img_file.suffix}"
                    )
                    output_annotation_name = (
                        f"{img_file.stem}_{shape_mask_file.stem}.txt"
                    )

                    output_image_path = output_images_dir / output_image_name
                    save_inpainted_image(inpainted_img, output_image_path)
                    output_annotation_path = (
                        output_annotations_dir / output_annotation_name
                    )
                    save_yolo_annotation(yolo_annotation, output_annotation_path)

                    print(
                        f"Processed and saved: Image: {output_image_name}, annotation: {output_annotation_name}"
                    )

                except FileNotFoundError as e:
                    print(
                        f"File processing error (inner loop - shape mask {shape_mask_file.name}): {e}"
                    )
                except Exception as e:
                    print(
                        f"Error processing combination (image: {img_file.name}, shape mask: {shape_mask_file.name}): {e}"
                    )

        except FileNotFoundError as e:
            print(f"File processing error (outer loop - image {img_file.name}): {e}")
        except Exception as e:
            print(
                f"An unexpected error occurred while processing image {img_file.name}: {e}"
            )
        break

    print("\nInpainting and YOLO annotation generation for all combinations completed.")


def main():
    """Main function to run the image inpainting and YOLO annotation script for all combinations."""
    print(
        "Starting image inpainting and YOLO annotation generation for all combinations..."
    )

    process_images_and_masks(
        IMAGES_DIR,
        FENCE_MASKS_DIR,
        SHAPE_MASKS_DIR,
        OUTPUT_INPAINTED_IMAGES_DIR,
        OUTPUT_YOLO_ANNOTATIONS_DIR,
    )


if __name__ == "__main__":
    main()
