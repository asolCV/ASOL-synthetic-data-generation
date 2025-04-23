from pathlib import Path
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt
import cv2

from src.scripts.download_dataset import download_roboflow_dataset


# Konfiguracja Roboflow API
ROBOFLOW_API_KEY = "cRNpwpG5hUaBUszly98K"  # Wstaw tutaj swój klucz API Roboflow
ROBOFLOW_WORKSPACE = "asol-mc75w"
ROBOFLOW_PROJECT = "asol-instance-segmentation"
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = "yolov8"

# Konfiguracja ścieżek za pomocą pathlib
BASE_DIR = Path(".")
DATASET_PATH = BASE_DIR / "roboflow_dataset"
MODEL_PATH = BASE_DIR / "yolov8_segmentation_model"
INFERENCE_IMAGE_PATH = (
    BASE_DIR / "inference_image.jpg"
)  # Zmień na ścieżkę do obrazu testowego, np. 'path/to/your/image.jpg'
TRAINED_MODEL_PATH = (
    MODEL_PATH / "segmentation_model.pt"
)  # Ścieżka do zapisanego modelu

# Upewnij się, że katalogi istnieją
DATASET_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def train_yolov8_segmentation(dataset_path: Path, model_output_path: Path):
    """Trenuje model YOLOv8 do segmentacji semantycznej."""
    # Załaduj pre-trenowany model YOLOv8-seg (np. yolov8s-seg.pt)
    model = YOLO(
        "yolov8s-seg.pt", task="segmentation"
    )  # Możesz zmienić na 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt' dla większych modeli

    # Ścieżka do pliku data.yaml wygenerowanego przez Roboflow
    data_yaml_path = dataset_path / "data.yaml"

    # Trenowanie modelu
    print("Rozpoczęcie trenowania modelu YOLOv8 Segmentation...")
    model.train(
        data=str(data_yaml_path),
        epochs=50,
        imgsz=(1920, 1080),  # Dostosuj epoki i rozmiar obrazu, data musi być str
    )  # Dostosuj epoki i rozmiar obrazu, data musi być str
    print("Trenowanie zakończone.")

    # Zapisz wytrenowany model w formacie .pt (domyślny format YOLOv8)
    best_model_path_pt = Path(
        "runs/segment/train/weights/best.pt"
    )  # Ścieżka do najlepszego modelu po treningu .pt
    saved_model_path_pt = model_output_path / "segmentation_model.pt"
    import shutil

    shutil.copyfile(
        best_model_path_pt, saved_model_path_pt
    )  # Kopiowanie best.pt do docelowej lokalizacji
    print(f"Wytrenowany model zapisany w: {saved_model_path_pt}")
    return saved_model_path_pt  # Zwraca ścieżkę do zapisanego modelu .pt


def perform_inference(model_path: Path, image_path: Path):
    """Dokonuje inferencji na obrazie za pomocą wytrenowanego modelu."""
    model = YOLO(str(model_path))  # Załaduj wytrenowany model, model_path musi być str

    print(f"Dokonywanie inferencji na obrazie: {image_path}")
    results = model(str(image_path))  # Wykonaj inferencję, image_path musi być str

    # Wyświetl wyniki za pomocą supervision
    for result in results:
        detections = sv.Detections.from_yolov8(result)
        if result.masks is not None:  # Sprawdzenie czy maski istnieją
            detections.mask = result.masks.data.cpu().numpy()

        image = cv2.imread(str(image_path))  # cv2.imread akceptuje str lub Path od 3.10
        if image is None:
            print(f"Błąd: Nie można odczytać obrazu z ścieżki: {image_path}")
            return

        box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette.DEFAULT, opacity=0.5)
        labels = [
            f"{model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]

        annotated_image = mask_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.title("Wynik Inferencji Segmentacji Semantycznej")
        plt.axis("off")
        plt.show()


def test_model(model_path_pt: Path, dataset_path: Path):
    """Testuje wytrenowany model na datasetcie walidacyjnym."""
    model = YOLO(
        str(model_path_pt)
    )  # Załaduj model .pt do testowania, model_path_pt musi być str
    data_yaml_path = dataset_path / "data.yaml"
    print("Rozpoczęcie testowania modelu...")
    metrics = model.val(data=str(data_yaml_path))  # data musi być str
    print("Testowanie zakończone.")
    # metrics zawiera różne metryki, np. metrics.box.map50-95, metrics.seg.map50-95 dla segmentacji


def main():
    # 1. Pobierz dataset z Roboflow
    dataset_location = download_roboflow_dataset(
        api_key=ROBOFLOW_API_KEY,
        workspace=ROBOFLOW_WORKSPACE,
        project=ROBOFLOW_PROJECT,
        version=ROBOFLOW_VERSION,
        format=ROBOFLOW_FORMAT,
        output_path=DATASET_PATH,
    )

    # 3. Trenuj model YOLOv8 Segmentation
    trained_model_path_pt = train_yolov8_segmentation(
        dataset_path=dataset_location, model_output_path=MODEL_PATH
    )

    # 4. Wykonaj inferencję na przykładowym obrazie
    if INFERENCE_IMAGE_PATH.exists():  # Sprawdź czy obraz inferencyjny istnieje
        perform_inference(
            model_path=trained_model_path_pt,  # Użyj zapisanego modelu .pt do inferencji
            image_path=INFERENCE_IMAGE_PATH,
        )
    else:
        print(
            f"Obraz inferencyjny nie znaleziony w ścieżce: {INFERENCE_IMAGE_PATH}. Umieść obraz i zaktualizuj ścieżkę."
        )

    # 5. Testuj model na danych walidacyjnych
    test_model(
        model_path_pt=trained_model_path_pt,  # Użyj wytrenowanego modelu .pt do testowania
        dataset_path=dataset_location,
    )

    print("Zakończono wszystkie kroki.")


if __name__ == "__main__":
    main()
