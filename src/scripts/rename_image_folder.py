"""All images from ComfyUI are saved with the same name as trimmed image file, but with '_001' suffix.
This script renames the folders to match the image names.
"""

from pathlib import Path

img_folder = Path("test_folder")


def rename_images(img_folder: Path):
    """Renames image names to match original image names"""
    for img in img_folder.glob("*.png"):
        new_name = img.with_stem(img.stem.replace("_001", ""))
        img.rename(new_name)


rename_images(img_folder)
