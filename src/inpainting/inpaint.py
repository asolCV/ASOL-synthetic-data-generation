import cv2
import numpy as np
from ultralytics import YOLO
import random


def find_mask_base_center(mask):
    """
    Znajduje centralny punkt podstawy maski (najniższy rząd pikseli).
    (Bez zmian - nadal działa na masce binarnej)

    Args:
        mask (np.array): Binarna maska (0 lub 255) obiektu (wysokość, szerokość).

    Returns:
        tuple: Współrzędne (x, y) środka podstawy lub None, jeśli nie można znaleźć.
               Współrzędne są w formacie (kolumna, wiersz), zgodnym z OpenCV.
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    max_row = np.max(coords[:, 0])
    base_points = coords[coords[:, 0] == max_row]
    if base_points.size > 0:
        center_col = np.mean(base_points[:, 1])
        return (int(center_col), int(max_row))
    else:
        return None


def create_mask_from_polygon(polygon_xy, shape_hw):
    """
    Tworzy maskę binarną na podstawie współrzędnych poligonu.

    Args:
        polygon_xy (np.array): Tablica współrzędnych (x, y) wierzchołków poligonu.
                                Powinny być to współrzędne dla oryginalnego obrazu.
        shape_hw (tuple): Krotka (wysokość, szerokość) docelowej maski.

    Returns:
        np.array: Maska binarna (0 lub 255) typu uint8 o wymiarach shape_hw,
                  lub None jeśli poligon jest pusty/nieprawidłowy.
    """
    if polygon_xy is None or len(polygon_xy) < 3:
        return None  # Potrzebujemy co najmniej 3 punktów do utworzenia poligonu

    mask = np.zeros(shape_hw, dtype=np.uint8)
    # Konwertuj współrzędne na int32, wymagane przez fillPoly
    # Upewnij się, że jest to lista list punktów (nawet jeśli jest tylko jeden poligon)
    pts = np.array([polygon_xy], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return mask


def warp_fence_posts_from_polygons(
    image_path: str,
    model_path: str,
    output_path: str,
    post_class_id: int,
    angle_deg: float = 15.0,
    random_angle_range: float = None,
):
    """
    Wczytuje obraz, wykrywa paliki płotu (używając poligonów), wykrzywia je
    wokół podstawy i zapisuje wynikowy obraz.

    Args:
        image_path (str): Ścieżka do obrazu wejściowego.
        model_path (str): Ścieżka do wytrenowanego modelu YOLO .pt z segmentacją.
        output_path (str): Ścieżka do zapisu obrazu wynikowego.
        post_class_id (int): ID klasy odpowiadającej palikom płotu.
        angle_deg (float): Domyślny kąt obrotu w stopniach (przeciwnie do zegara).
                           Używany, jeśli random_angle_range jest None.
        random_angle_range (float, optional): Jeśli podany, kąt będzie losowany
                                               z zakresu [-range, +range]. Ignoruje angle_deg.
                                               Defaults to None.
    """
    # --- 1. Załaduj model i obraz ---
    try:
        model = YOLO(model_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Błąd: Nie można wczytać obrazu z {image_path}")
            return
        h, w = img.shape[:2]  # Oryginalne wymiary obrazu
        output_img = img.copy()  # Kopia do modyfikacji
    except Exception as e:
        print(f"Błąd podczas ładowania modelu lub obrazu: {e}")
        return

    # --- 2. Przeprowadź inferencję YOLO ---
    try:
        # Używamy domyślnych ustawień, model sam zadba o skalowanie
        # Ważne, że `results[0].masks.xy` zawierać będą współrzędne
        # przeskalowane z powrotem do ORYGINALNEGO rozmiaru obrazu.
        results = model(img)
    except Exception as e:
        print(f"Błąd podczas inferencji YOLO: {e}")
        return

    # --- 3. Przetwarzanie wyników ---
    if not results or not results[0].masks:
        print("Nie wykryto żadnych obiektów z maskami lub model ich nie zwrócił.")
        if output_path:
            cv2.imwrite(output_path, output_img)
        return

    # Sprawdź czy mamy dane poligonów (xy)
    if results[0].masks.xy is None or len(results[0].masks.xy) == 0:
        print("Model zwrócił maski, ale brak danych poligonów (xy).")
        if output_path:
            cv2.imwrite(output_path, output_img)
        return

    polygons_xy = results[0].masks.xy  # Lista poligonów [N][points, 2]
    classes = results[0].boxes.cls.cpu().numpy()  # Klasy dla każdej detekcji

    print(f"Wykryto {len(polygons_xy)} obiektów z poligonami.")

    processed_indices = []

    for i in range(len(polygons_xy)):
        # Sprawdź, czy klasa obiektu to palik płotu
        if int(classes[i]) != post_class_id:
            continue

        print(f"Przetwarzanie palika (poligon) nr {i+1}...")
        polygon = polygons_xy[i]  # Współrzędne (x, y) dla tego palika

        # --- *** NOWOŚĆ: Utwórz maskę z poligonu *** ---
        # Tworzymy maskę binarną o pełnej rozdzielczości na podstawie poligonu
        mask_binary = create_mask_from_polygon(polygon, (h, w))

        if mask_binary is None or np.sum(mask_binary) == 0:
            print(
                f"  - Nie można utworzyć maski z poligonu lub maska jest pusta dla obiektu {i+1}, pomijanie."
            )
            continue
        # --- *** KONIEC NOWOŚCI *** ---

        # --- 4. Znajdź podstawę palika (na masce z poligonu) ---
        base_center = find_mask_base_center(mask_binary)
        if base_center is None:
            print(f"  - Nie można znaleźć podstawy dla obiektu {i+1}, pomijanie.")
            continue
        print(f"  - Znaleziono podstawę w: {base_center}")  # (x, y)

        # --- 5. Wyodrębnij piksele palika (używając maski z poligonu) ---
        post_extracted = cv2.bitwise_and(img, img, mask=mask_binary)

        # --- 6. Wypełnij oryginalną pozycję palika czarnym kolorem ---
        output_img[mask_binary > 0] = [0, 0, 0]  # Czarny BGR

        # --- 7. Przygotuj i wykonaj rotację ---
        current_angle = angle_deg
        if random_angle_range is not None:
            current_angle = random.uniform(-random_angle_range, random_angle_range)
        print(f"  - Obracanie o kąt: {current_angle:.2f} stopni")

        M = cv2.getRotationMatrix2D(center=base_center, angle=current_angle, scale=1.0)

        # Obróć wyodrębniony obraz palika
        rotated_post = cv2.warpAffine(
            post_extracted,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,  # Lepsza jakość dla obrazu
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Obróć również maskę (tę z poligonu), aby wiedzieć, gdzie wkleić
        rotated_mask = cv2.warpAffine(
            mask_binary,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,  # Lepsze dla maski binarnej
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # --- 8. Wklej obrócony palik na obraz wynikowy ---
        # Używamy rotated_mask do określenia, które piksele w output_img
        # powinny zostać zastąpione pikselami z rotated_post.
        output_img[rotated_mask > 0] = rotated_post[rotated_mask > 0]
        processed_indices.append(i)

    print(f"Przetworzono {len(processed_indices)} palików.")

    # --- 9. Zapisz wynik ---
    if output_path:
        try:
            cv2.imwrite(output_path, output_img)
            print(f"Obraz wynikowy zapisano w: {output_path}")
        except Exception as e:
            print(f"Błąd podczas zapisywania obrazu wynikowego: {e}")
    else:
        print("Nie podano ścieżki wyjściowej, obraz nie został zapisany.")


# --- Ustawienia (takie same jak poprzednio) ---
MODEL_FILE = "weights/segmentation/poles/yolov8s-seg/best.pt"
IMAGE_FILE = r"datasets\instance-segmentation\asol-instance-segmentation.v2-raw-dataset.yolov8\train\images\20241025_102238-mp4_frame_4_jpg.rf.249d2aa43bdf345cbf8d6d39f5e3af7b.jpg"
OUTPUT_FILE = "output_polygon.png"  # Zmieniona nazwa pliku wyjściowego
POST_CLASS_INDEX = 0  # <--- ZMIEŃ TUTAJ (indeks klasy palików)

# Kąt obrotu:
ROTATION_ANGLE = 30.0
RANDOM_RANGE = None
# lub
# ROTATION_ANGLE = 0
# RANDOM_RANGE = 25.0

# --- Wywołanie funkcji ---
if __name__ == "__main__":
    warp_fence_posts_from_polygons(  # Wywołanie nowej funkcji
        image_path=IMAGE_FILE,
        model_path=MODEL_FILE,
        output_path=OUTPUT_FILE,
        post_class_id=POST_CLASS_INDEX,
        angle_deg=ROTATION_ANGLE,
        random_angle_range=RANDOM_RANGE,
    )
