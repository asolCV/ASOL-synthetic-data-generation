# common/datasets.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random  # Potrzebne do jittera

# from pathlib import Path
from tqdm import tqdm


class TileTrainingDataset(Dataset):
    """
    Dataset for training on tiles extracted from large images with random jitter
    and guaranteed tile size without padding.

    Attributes:
        image_paths (list): List of paths to the input images.
        mask_paths (list): List of paths to the corresponding masks.
        tile_size (int): The dimension (width and height) of the tiles to extract.
        stride (int): The stride used to generate nominal tile starting points.
        jitter_max (int): Maximum pixel offset (positive or negative) for random jitter.
        transform (callable, optional): Optional transform to be applied on a sample.
        tile_coords (list): Pre-calculated list of nominal tile coordinates (img_idx, y, x).
    """

    def __init__(
        self, image_paths, mask_paths, tile_size, stride, jitter_max=5, transform=None
    ):
        """
        Args:
            image_paths (list): List of paths to the input images.
            mask_paths (list): List of paths to the corresponding masks.
            tile_size (int): The dimension (width and height) of the tiles.
            stride (int): The stride for generating nominal tile centers.
            jitter_max (int): Maximum random offset for tile coordinates (default: 5).
            transform (callable, optional): Optional transform for augmentation.
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must be the same.")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if jitter_max < 0:
            raise ValueError("jitter_max cannot be negative.")

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tile_size = tile_size
        self.stride = stride
        self.jitter_max = jitter_max
        self.transform = transform

        # Przechowujemy pełne ścieżki, aby uniknąć wielokrotnego ładowania info o obrazie
        self.image_info = self._get_image_info()
        self.tile_coords = self._create_tile_list()

        if not self.tile_coords:
            raise ValueError(
                "No valid tiles found. Check image sizes, tile_size, and stride. "
                "Ensure images are at least tile_size x tile_size."
            )

    def _get_image_info(self):
        """Reads dimensions of all images."""
        info = []
        print("Reading image dimensions...")
        for img_path in tqdm(self.image_paths, desc="Reading Image Info"):
            # Użyj imfinfo lub odczytaj nagłówek zamiast całego obrazu, jeśli to możliwe i szybsze
            # Tutaj dla prostoty odczytamy obraz
            img = cv2.imread(
                str(img_path), cv2.IMREAD_UNCHANGED
            )  # Użyj IMREAD_UNCHANGED dla pewności
            if img is None:
                print(
                    f"Warning: Could not read image {img_path} for dimensions. Skipping."
                )
                continue
            h, w = img.shape[:2]
            if h < self.tile_size or w < self.tile_size:
                print(
                    f"Warning: Image {img_path} ({h}x{w}) is smaller than tile_size "
                    f"({self.tile_size}x{self.tile_size}). Skipping this image for tiling."
                )
                continue
            info.append({"path": img_path, "height": h, "width": w})
            del img  # Zwolnij pamięć
        return info

    def _create_tile_list(self):
        """
        Pre-calculates the nominal coordinates of all potential tiles based on stride.
        Stores (image_index, nominal_y, nominal_x).
        """
        tile_coords = []
        print("Creating tile index...")
        # Używamy teraz self.image_info, które zawiera tylko obrazy >= tile_size
        for i, info in enumerate(tqdm(self.image_info, desc="Indexing Tiles")):
            h, w = info["height"], info["width"]

            # Generuj nominalne koordynaty lewego górnego rogu kafelka
            # Iterujemy tak, by potencjalny kafelek zaczynał się w obrazie
            for y in range(0, h, self.stride):
                for x in range(0, w, self.stride):
                    # Sprawdzenie, czy początek jest w granicach obrazu
                    # (Koniec nie musi być, poradzimy sobie z tym w __getitem__)
                    if y < h and x < w:
                        # Zapisujemy indeks obrazu w self.image_info oraz nominalne y, x
                        tile_coords.append(
                            (i, y, x)
                        )  # Zmieniono na indeks zamiast ścieżek

        print(f"Found {len(tile_coords)} potential training tile starting points.")
        return tile_coords

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        # 1. Pobierz informacje o nominalnym kafelku
        image_idx, y_nominal, x_nominal = self.tile_coords[idx]
        img_info = self.image_info[image_idx]
        img_path = img_info["path"]
        # Znajdź odpowiednią ścieżkę maski (zakładamy tę samą kolejność jak w image_paths)
        # Jeśli image_paths i self.image_info mogą mieć różną kolejność/długość,
        # potrzebny będzie bardziej złożony mechanizm mapowania
        original_img_idx = self.image_paths.index(img_path)  # Znajdź oryginalny index
        mask_path = self.mask_paths[original_img_idx]

        h, w = img_info["height"], img_info["width"]

        # 2. Zastosuj Jitter
        offset_y = random.randint(-self.jitter_max, self.jitter_max)
        offset_x = random.randint(-self.jitter_max, self.jitter_max)
        y_jittered = y_nominal + offset_y
        x_jittered = x_nominal + offset_x

        # 3. Oblicz *ostateczne* koordynaty startowe (y_start, x_start)
        #    gwarantujące rozmiar tile_size x tile_size bez wychodzenia poza obraz

        # Najpierw ogranicz jittered coords, aby były nieujemne
        y_jittered = max(0, y_jittered)
        x_jittered = max(0, x_jittered)

        # Teraz oblicz y_start: jeśli kafelek wychodzi poza dolną krawędź, przesuń go w górę
        if y_jittered + self.tile_size > h:
            y_start = h - self.tile_size
        else:
            y_start = y_jittered

        # Analogicznie dla x_start: jeśli kafelek wychodzi poza prawą krawędź, przesuń go w lewo
        if x_jittered + self.tile_size > w:
            x_start = w - self.tile_size
        else:
            x_start = x_jittered

        # Upewnij się, że startowe koordynaty nie są negatywne (choć max(0, ...) powinno to załatwić)
        y_start = max(0, y_start)
        x_start = max(0, x_start)

        # Końcowe koordynaty
        y_end = y_start + self.tile_size
        x_end = x_start + self.tile_size

        # 4. Wczytaj *pełny* obraz i maskę (optymalizacja: można wczytać tylko potrzebny fragment, ale to bardziej skomplikowane)
        #    Wczytywanie całego obrazu jest prostsze, ale może być wolniejsze dla bardzo dużych obrazów.
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Ten błąd nie powinien się zdarzyć, bo sprawdziliśmy w _get_image_info
            raise IOError(f"Could not read image tile source: {img_path}")
        if mask is None:
            # Maskę też warto sprawdzić wcześniej, jeśli to możliwe
            raise IOError(f"Could not read mask tile source: {mask_path}")

        # 5. Wytnij kafelek o gwarantowanym rozmiarze tile_size x tile_size
        #    Assert dla pewności, że wycinamy poprawny rozmiar
        assert (
            0 <= y_start < y_end <= h
        ), f"Y coords error: y_start={y_start}, y_end={y_end}, h={h}"
        assert (
            0 <= x_start < x_end <= w
        ), f"X coords error: x_start={x_start}, x_end={x_end}, w={w}"

        img_tile = image[y_start:y_end, x_start:x_end]
        mask_tile = mask[y_start:y_end, x_start:x_end]

        # Sprawdźmy rozmiar po wycięciu - powinien być już poprawny
        if img_tile.shape[0] != self.tile_size or img_tile.shape[1] != self.tile_size:
            raise RuntimeError(
                f"Tile cutting error for {img_path} at ({y_start},{x_start}). "
                f"Expected ({self.tile_size},{self.tile_size}), got {img_tile.shape[:2]}. "
                f"h={h}, w={w}, y_end={y_end}, x_end={x_end}"
            )
        if mask_tile.shape[0] != self.tile_size or mask_tile.shape[1] != self.tile_size:
            raise RuntimeError(
                f"Mask tile cutting error for {mask_path} at ({y_start},{x_start}). "
                f"Expected ({self.tile_size},{self.tile_size}), got {mask_tile.shape[:2]}."
            )

        # 6. Przetwarzanie maski i obrazu
        # Konwersja maski do binarnej float32 (0.0 lub 1.0)
        mask_tile = (mask_tile > 0).astype(np.float32)

        # Konwersja obrazu do RGB (ważne dla transformacji i modelu)
        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_BGR2RGB)

        # 7. Zastosuj transformacje (augmentacje, normalizacje)
        if self.transform:
            # Transformacje powinny oczekiwać obrazu jako np.array HWC i maski jako np.array CHW lub HW
            # Dostosuj, jeśli Twoje transformacje oczekują innego formatu wejściowego
            # breakpoint()
            augmented = self.transform(image=img_tile, mask=mask_tile)
            img_tile = augmented["image"]  # Powinien być już tensorem CHW
            mask_tile = augmented["mask"].unsqueeze(
                0
            )  # Powinien być już tensorem CHW lub HW

        if isinstance(img_tile, np.ndarray):
            img_tile = torch.from_numpy(
                img_tile.transpose((2, 0, 1))
            ).float()  # HWC -> CHW

        # Jeśli maska jest nadal NumPy HW, konwertuj do Tensora (1, H, W)
        if isinstance(mask_tile, np.ndarray):
            # Tutaj zostawiamy float jak wcześniej
            if len(mask_tile.shape) == 2:  # Jeśli jest HW
                mask_tile = np.expand_dims(mask_tile, axis=0)  # HW -> 1HW (numpy)
            # Teraz konwertuj do tensora (zakładając float)
            mask_tile = torch.from_numpy(mask_tile).float()  # (1, H, W) Tensor

        return {"image": img_tile, "mask": mask_tile}


# FullImageDataset pozostaje bez zmian, jak w Twoim przykładzie
class FullImageDataset(Dataset):
    """Dataset that loads full images and masks, primarily for validation/prediction."""

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise IOError(f"Could not read image: {img_path}")
        if mask is None:
            raise IOError(f"Could not read mask: {mask_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB early
        gt_mask = (mask > 0).astype(
            np.uint8
        )  # Użyj uint8 dla maski GT, LongTensor później

        normalized_image_tensor = None
        if self.transform:
            # Transformacja dla walidacji zwykle zawiera tylko normalizację i ToTensor
            transformed = self.transform(image=image_rgb)
            normalized_image_tensor = transformed["image"]
        else:
            # Podstawowa konwersja do tensora, jeśli brak transformacji
            print("Warning: No validation transform provided, using basic ToTensor.")
            normalized_image = (image_rgb / 255.0).astype(np.float32)
            normalized_image_tensor = torch.from_numpy(normalized_image).permute(
                2, 0, 1
            )  # HWC to CHW

        return {
            "image_tensor": normalized_image_tensor,  # Tensor CHW znormalizowany
            "original_image": torch.from_numpy(image_rgb).permute(
                2, 0, 1
            ),  # Oryginalny obraz jako Tensor CHW (nieznormalizowany)
            "gt_mask": torch.from_numpy(
                gt_mask
            ).long(),  # Maska jako Tensor HW Long (dla wizualizacji lub metryk)
            "image_path": str(img_path),  # Ścieżka do identyfikacji
        }
