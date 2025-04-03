# common/predict.py
import torch
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

# Local imports using absolute paths
from src.common.config import cfg  # Import default config
from src.models.unet import UNet  # Import model
from src.common import utils  # Import utils
from src.common import transforms as T  # Import transforms alias


def predict_image(
    model, image_path, output_dir, device, tile_size, overlap, num_classes
):
    """Predicts segmentation for a single image using sliding window."""
    print(f"Processing: {image_path.name}")
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Error: Could not read image {image_path.name}. Skipping.")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape[:2]  # H, W

        # --- Preprocessing ---
        transform = T.get_minimal_transforms()
        augmented = transform(image=image_rgb)
        image_tensor = augmented["image"].unsqueeze(0).to(device)  # [1, C, H, W]

        # --- Sliding Window Prediction ---
        model.eval()  # Ensure model is in eval mode
        with torch.inference_mode():
            pred_logits_map = utils.sliding_window_predict(
                model,
                image_tensor.squeeze(0),  # Pass [C, H, W]
                tile_size=tile_size,
                overlap=overlap,
                device=device,
                num_classes=num_classes,
                batch_size=cfg.BATCH_SIZE,  # Use config's batch size or set dedicated inference batch
            )  # Returns logits [Cls, H, W]

        # --- Post-processing ---
        if num_classes == 1:
            pred_prob_map = torch.sigmoid(pred_logits_map)
            pred_mask = (pred_prob_map > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            pred_mask = (
                torch.argmax(pred_logits_map, dim=0).cpu().numpy().astype(np.uint8)
            )

        if pred_mask.shape != original_shape:
            print(
                f"  Warning: Prediction shape {pred_mask.shape} differs from original {original_shape}. Resizing prediction."
            )
            pred_mask = cv2.resize(
                pred_mask,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # --- Save Results ---
        output_mask_path = Path(output_dir) / f"{image_path.stem}_mask.png"
        output_overlay_path = Path(output_dir) / f"{image_path.stem}_overlay.png"

        cv2.imwrite(str(output_mask_path), pred_mask * 255)

        pred_mask_bgr = cv2.cvtColor(pred_mask * 255, cv2.COLOR_GRAY2BGR)
        # Use original BGR image for overlay to avoid double conversion
        overlay = cv2.addWeighted(image, 0.6, pred_mask_bgr, 0.4, 0)
        cv2.imwrite(str(output_overlay_path), overlay)

        # print(f"  Saved mask to: {output_mask_path}") # Less verbose during batch processing
        # print(f"  Saved overlay to: {output_overlay_path}")

    except Exception as e:
        print(f"  Error processing {image_path.name}: {e}")
        # import traceback # Uncomment for detailed debugging if needed
        # traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Inference")
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to input image or folder."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Directory to save predictions."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to trained model (.pth.tar).",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=cfg.INFERENCE_TILE_SIZE,
        help="Tile size for sliding window.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=cfg.INFERENCE_OVERLAP,
        help="Overlap for sliding window.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda', 'cpu', or None for auto).",
    )
    args = parser.parse_args()

    # --- Setup ---
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Auto-select
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading model...")
    model = UNet(n_channels=cfg.INPUT_CHANNELS, n_classes=cfg.NUM_CLASSES).to(device)
    try:
        epoch, metric = utils.load_checkpoint(
            args.checkpoint, model, optimizer=None, device=device
        )
        print(f"Loaded checkpoint from epoch {epoch} (Metric: {metric:.4f})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Exiting.")
        return

    model.eval()  # Set to evaluation mode

    # --- Find Images ---
    if args.input.is_file():
        image_paths = [args.input]
    elif args.input.is_dir():
        image_paths = sorted(
            list(args.input.glob("*.jpg")) + list(args.input.glob("*.png"))
        )
    else:
        print(f"Error: Input path {args.input} not found or invalid.")
        return

    if not image_paths:
        print(f"Error: No images found in {args.input}.")
        return
    print(f"Found {len(image_paths)} images to predict.")

    # --- Predict ---
    for img_path in tqdm(image_paths, desc="Predicting Images"):
        predict_image(
            model,
            img_path,
            args.output,
            device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            num_classes=cfg.NUM_CLASSES,
        )

    print("\nPrediction complete.")
    print(f"Results saved in: {args.output}")


if __name__ == "__main__":
    main()
