# visualize_simple_dataset.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# --- Zależności Projektu ---
# Załóż, że ten skrypt jest uruchamiany z katalogu,
# gdzie importy ścieżek względnych lub bezwzględnych działają.
try:
    from src.common.config import cfg
    from src.common.datasets import TileTrainingDataset

    # Importuj *faktyczne* transformacje używane w treningu, jeśli to możliwe
    # Jeśli zawierają normalizację, obrazy mogą wyglądać "dziwnie" bez denormalizacji
    # Tutaj użyjemy minimalnych jako przykład
    from src.common import transforms as T

    TRAIN_TRANSFORM = T.get_minimal_transforms()  # LUB transformacje treningowe
except ImportError as e:
    print(f"Nie udało się zaimportować modułów projektu: {e}")
    print("Upewnij się, że skrypt jest w odpowiednim miejscu i ścieżki są poprawne.")
    exit(1)

# --- Konfiguracja Wizualizacji ---
NUM_SAMPLES_PER_TYPE = 3  # Ile przykładów każdego typu (z obiektem / bez obiektu)
MAX_DATASET_CHECKS = 500  # Ile max próbek z datasetu sprawdzić, aby znaleźć przykłady


# --- Funkcja pomocnicza do konwersji tensora na obraz NumPy ---
def tensor_to_numpy_for_display(tensor, is_mask=False):
    """Konwertuje tensor (obraz lub maskę) do formatu NumPy do wyświetlenia."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Oczekiwano tensora, otrzymano {type(tensor)}")

    tensor = tensor.cpu().detach().numpy()

    if is_mask:
        # Maska: Oczekujemy (1, H, W) lub (H, W), float lub long [0, 1]
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # (1, H, W) -> (H, W)
        # Konwertuj do uint8 [0, 255]
        return (
            (tensor * 255).astype(np.uint8)
            if tensor.max() <= 1.0
            else tensor.astype(np.uint8)
        )
    else:
        # Obraz: Oczekujemy (C, H, W), np. float [0, 1] lub po normalizacji
        # Konwertuj do (H, W, C)
        tensor = np.transpose(tensor, (1, 2, 0))  # CHW -> HWC
        # Skaluj do [0, 255] jeśli jest w zakresie [0, 1]
        if tensor.dtype in [np.float32, np.float64, np.float16]:
            if tensor.min() >= 0.0 and tensor.max() <= 1.0:
                tensor = tensor * 255
            else:
                # Jeśli jest znormalizowany (np. wartości ujemne), przytnij do [0, 255]
                # To *nie jest* denormalizacja, obrazy mogą wyglądać źle.
                mean = np.array([0.485, 0.456, 0.406])  # Przykład
                std = np.array([0.229, 0.224, 0.225])  # Przykład
                tensor = (tensor * std + mean) * 255.0
                tensor = np.clip(tensor, 0, 255)

        return tensor.astype(np.uint8)


# --- Główna Logika ---
if __name__ == "__main__":
    print("Uruchamianie prostego skryptu wizualizacji datasetu...")

    # 1. Pobierz ścieżki plików z konfiguracji
    # Zakładamy, że cfg.IMAGE_DIR i cfg.MASK_DIR są poprawnie ustawione
    print(f"Szukanie obrazów w: {cfg.IMAGE_DIR}")
    print(f"Szukanie masek w: {cfg.MASK_DIR}")
    # Proste znajdowanie par (zakładając identyczne nazwy plików)
    image_paths = []
    mask_paths = []
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    all_images = []
    for ext in extensions:
        all_images.extend(list(cfg.IMAGE_DIR.glob(ext)))

    if not all_images:
        print(f"Nie znaleziono obrazów w {cfg.IMAGE_DIR}")
        exit(1)

    for img_p in tqdm(all_images, desc="Weryfikacja par obraz-maska"):
        mask_p = cfg.MASK_DIR / img_p.name
        if mask_p.exists():
            image_paths.append(img_p)
            mask_paths.append(mask_p)

    if not image_paths:
        print(f"Nie znaleziono pasujących par obraz-maska.")
        exit(1)
    print(f"Znaleziono {len(image_paths)} pasujących par.")

    # 2. Utwórz Dataset
    print("Inicjalizacja TileTrainingDataset...")
    try:
        # Użyj parametrów z config.py
        dataset = TileTrainingDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            tile_size=cfg.TILE_SIZE,
            stride=cfg.TRAIN_TILE_STRIDE,
            jitter_max=(
                cfg.TILE_JITTER if hasattr(cfg, "TILE_JITTER") else 5
            ),  # Dodaj TILE_JITTER do config lub ustaw domyślnie
            transform=TRAIN_TRANSFORM,  # Użyj transformacji (np. z ToTensorV2)
        )
        if len(dataset) == 0:
            print("Dataset jest pusty! Sprawdź konfigurację.")
            exit(1)
        print(f"Dataset zainicjalizowany, potencjalna liczba kafelków: {len(dataset)}")
    except Exception as e:
        print(f"Błąd podczas inicjalizacji datasetu: {e}")
        raise  # Rzuć błąd dalej, aby zobaczyć pełny traceback

    # 3. Zbierz próbki
    positive_samples = []  # Kafelki z obiektami
    negative_samples = []  # Kafelki tylko z tłem
    checked_indices = set()

    print(f"Zbieranie {NUM_SAMPLES_PER_TYPE} pozytywnych i negatywnych próbek...")
    pbar = tqdm(total=NUM_SAMPLES_PER_TYPE * 2)
    attempts = 0
    max_attempts = min(
        len(dataset), MAX_DATASET_CHECKS
    )  # Nie sprawdzaj więcej niż jest w datasecie lub limit

    while (
        len(positive_samples) < NUM_SAMPLES_PER_TYPE
        or len(negative_samples) < NUM_SAMPLES_PER_TYPE
    ) and attempts < max_attempts:
        idx = random.randint(0, len(dataset) - 1)
        if idx in checked_indices:
            continue
        checked_indices.add(idx)
        attempts += 1

        try:
            sample = dataset[idx]
            mask = sample["mask"]  # Oczekujemy tensora

            is_positive = torch.any(mask > 0).item()

            if is_positive and len(positive_samples) < NUM_SAMPLES_PER_TYPE:
                positive_samples.append(sample)
                pbar.update(1)
                pbar.set_description(
                    f"Pos: {len(positive_samples)}/{NUM_SAMPLES_PER_TYPE}, Neg: {len(negative_samples)}/{NUM_SAMPLES_PER_TYPE}"
                )
            elif not is_positive and len(negative_samples) < NUM_SAMPLES_PER_TYPE:
                negative_samples.append(sample)
                pbar.update(1)
                pbar.set_description(
                    f"Pos: {len(positive_samples)}/{NUM_SAMPLES_PER_TYPE}, Neg: {len(negative_samples)}/{NUM_SAMPLES_PER_TYPE}"
                )

        except Exception as e:
            print(f"\nBłąd podczas pobierania próbki {idx}: {e}")
            # Kontynuuj szukanie innych próbek
            continue
    pbar.close()

    if not positive_samples and not negative_samples:
        print("Nie udało się zebrać żadnych próbek.")
        exit()

    # 4. Narysuj siatkę
    all_samples = positive_samples + negative_samples
    labels = ["Positive"] * len(positive_samples) + ["Negative"] * len(negative_samples)
    num_rows = len(all_samples)

    if num_rows == 0:
        print("Brak próbek do wyświetlenia.")
        exit()

    fig, axes = plt.subplots(
        num_rows, 2, figsize=(6, 3 * num_rows), squeeze=False
    )  # squeeze=False dla pewności

    print("Rysowanie siatki...")
    for i, (sample, label) in enumerate(zip(all_samples, labels)):
        try:
            img_np = tensor_to_numpy_for_display(sample["image"], is_mask=False)
            mask_np = tensor_to_numpy_for_display(sample["mask"], is_mask=True)

            ax_img = axes[i, 0]
            ax_mask = axes[i, 1]

            ax_img.imshow(img_np)
            ax_img.set_title(f"Sample {i+1} ({label}) - Image")
            ax_img.axis("off")

            ax_mask.imshow(
                mask_np, cmap="gray", vmin=0, vmax=255
            )  # vmin/vmax dla pewności
            ax_mask.set_title(f"Sample {i+1} ({label}) - Mask")
            ax_mask.axis("off")
        except Exception as e:
            print(f"Błąd podczas przetwarzania/rysowania próbki {i}: {e}")
            if "img_np" in locals():
                print(f"  Image shape: {img_np.shape}, dtype: {img_np.dtype}")
            if "mask_np" in locals():
                print(f"  Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
            # Możesz chcieć narysować placeholder w razie błędu
            axes[i, 0].set_title(f"Sample {i+1} - Error")
            axes[i, 1].set_title(f"Sample {i+1} - Error")
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")

    plt.tight_layout()
    plt.suptitle("Dataset Tile Samples", y=1.02)
    plt.show()

    print("Zakończono wizualizację.")
