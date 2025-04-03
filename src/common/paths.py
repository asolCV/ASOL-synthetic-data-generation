from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent

# All datasets are stored in the datasets directory
DATASETS_DIR = PROJECT_DIR / "datasets"

# All model weights are stored in the weights directory
WEIGHTS_DIR = PROJECT_DIR / "weights"

YOLOV8S_SEG_POLE_MODEL_PATH = (
    WEIGHTS_DIR / "segmentation" / "poles" / "yolov8s-seg" / "best.pt"
)
YOLOV8S_SEG_RAZOR_WIRE_MODEL_PATH = (
    WEIGHTS_DIR / "segmentation" / "razor-wire" / "yolov8s-seg" / "best.pt"
)
