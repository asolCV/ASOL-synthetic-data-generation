from pathlib import Path
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import random
import os
import itertools  # Potrzebne do generowania kombinacji

# --- Konfiguracja ---
IMAGES_NO_POLES_DIRPATH = Path("test_folder")
IMAGES_WITH_POLES_DIRPATH = Path("datasets") / "trimmed-images-whole-poles"
OUTPUT_DIRPATH = Path(
    "output_augmented_images_expanded"
)  # Zmieniono nazwę katalogu wyjściowego
MODEL_WEIGHTS_PATH = Path("weights/segmentation/poles/yolov8s-seg/best.pt")

LEFT_POLE_ANGLE_RANGE = (-40, -15)
RIGHT_POLE_ANGLE_RANGE = (15, 40)

YOLO_CONF = 0.2
YOLO_IOU = 0.2
# --------------------

OUTPUT_DIRPATH.mkdir(parents=True, exist_ok=True)

try:
    yolo_model = YOLO(MODEL_WEIGHTS_PATH)
    print(f"Model YOLO loaded successfully from: {MODEL_WEIGHTS_PATH}")
except Exception as e:
    print(
        f"FATAL ERROR: Could not load YOLO model from {MODEL_WEIGHTS_PATH}. Error: {e}"
    )
    exit()


def find_mask_base_center(mask):
    """Znajduje centralny punkt podstawy maski."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    max_row = np.max(coords[:, 0])
    base_points = coords[coords[:, 0] == max_row]
    if base_points.size > 0:
        center_col = np.mean(base_points[:, 1])
        return (int(center_col), int(max_row))
    return None


def create_mask_from_polygon(polygon_xy, shape_hw):
    """Tworzy maskę binarną z poligonu."""
    if polygon_xy is None or len(polygon_xy) < 3:
        return None
    mask = np.zeros(shape_hw, dtype=np.uint8)
    try:
        pts = np.array([polygon_xy], dtype=np.int32)
        cv.fillPoly(mask, pts, 255)
    except Exception as e:
        print(f"Error creating mask with fillPoly: {e}")
        return None
    return mask


def rotate_object(image, mask, base_center, angle, img_shape_wh):
    """Obraca obiekt (piksele i maskę)."""
    w, h = img_shape_wh
    if (
        base_center is None
        or mask is None
        or image is None
        or not (0 <= base_center[0] < w and 0 <= base_center[1] < h)
    ):
        # print(f"  - Warning: Invalid base_center {base_center} for image shape {(w,h)}. Skipping rotation.")
        return None, None
    M = cv.getRotationMatrix2D(center=base_center, angle=angle, scale=1.0)
    rotated_pixels = cv.warpAffine(
        image,
        M,
        (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rotated_mask = cv.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_mask = (rotated_mask > 127).astype(np.uint8) * 255
    return rotated_pixels, rotated_mask


def paste_object(background, obj_pixels, obj_mask):
    """Wkleja obiekt na tło używając maski."""
    if obj_pixels is not None and obj_mask is not None:
        if (
            background.shape[:2] != obj_mask.shape[:2]
            or background.shape[:2] != obj_pixels.shape[:2]
        ):
            # print(f"  - ERROR in paste_object: Shape mismatch! Skipping paste.")
            return
        try:
            indices = obj_mask > 0
            background[indices] = obj_pixels[indices]
        except IndexError as e:
            print(
                f"  - IndexError during pasting! {e}. BG:{background.shape}, MASK:{obj_mask.shape}, PIXELS:{obj_pixels.shape}"
            )
        except Exception as e:
            print(f"  - Unexpected error during pasting: {e}")


def main():
    print(f"Processing images from: {IMAGES_NO_POLES_DIRPATH}")
    print(f"Using corresponding images with poles from: {IMAGES_WITH_POLES_DIRPATH}")
    print(f"Saving augmented images to: {OUTPUT_DIRPATH}")

    processed_count = 0
    skipped_count = 0
    total_files_saved = 0

    image_files = list(IMAGES_NO_POLES_DIRPATH.glob("*.png"))
    if not image_files:
        print(f"Warning: No PNG images found in {IMAGES_NO_POLES_DIRPATH}")
        return
    print(f"Found {len(image_files)} background images to process.")

    for img_no_pole_path in image_files:
        print(f"\nProcessing background: {img_no_pole_path.name}")
        base_filename = img_no_pole_path.stem

        img_with_poles_path = IMAGES_WITH_POLES_DIRPATH / img_no_pole_path.name
        if not img_with_poles_path.exists():
            print(
                f"  - Corresponding image with poles not found: {img_with_poles_path}"
            )
            skipped_count += 1
            continue

        img_no_pole = cv.imread(str(img_no_pole_path))
        img_with_poles = cv.imread(str(img_with_poles_path))

        if img_no_pole is None or img_with_poles is None:
            print(f"  - Failed to load one or both images.")
            skipped_count += 1
            continue

        h_orig, w_orig = img_with_poles.shape[:2]
        h_bg, w_bg = img_no_pole.shape[:2]

        if (h_bg, w_bg) != (h_orig, w_orig):
            # print(f"  - Warning: Dimension mismatch. Resizing background.")
            img_no_pole = cv.resize(
                img_no_pole, (w_orig, h_orig), interpolation=cv.INTER_LINEAR
            )
            if img_no_pole.shape[:2] != (h_orig, w_orig):
                print(f"  - ERROR: Failed to resize background image correctly!")
                skipped_count += 1
                continue
        h, w = h_orig, w_orig

        # --- Detekcja palików ---
        try:
            results_list = yolo_model.predict(
                img_with_poles,
                verbose=False,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                imgsz=(1080, 1920),
            )
        except Exception as e:
            print(f"  - YOLO prediction failed: {e}")
            skipped_count += 1
            continue

        # --- Przetwarzanie wyników detekcji ---
        poles_data = []
        if results_list and results_list[0].masks is not None:
            results = results_list[0]
            polygons_xy = results.masks.xy
            for i, polygon in enumerate(polygons_xy):
                original_pole_mask = create_mask_from_polygon(polygon, (h, w))
                if original_pole_mask is None or np.sum(original_pole_mask) == 0:
                    continue
                base_center = find_mask_base_center(original_pole_mask)
                if base_center is None:
                    continue

                pole_extracted_pixels = cv.bitwise_and(
                    img_with_poles, img_with_poles, mask=original_pole_mask
                )
                is_left = base_center[0] < w // 2
                poles_data.append(
                    {
                        "mask": original_pole_mask,
                        "pixels": pole_extracted_pixels,
                        "base": base_center,
                        "is_left": is_left,
                        "original_index": i,  # Używane do nazw plików kombinacji
                    }
                )
        num_poles = len(poles_data)
        print(f"  - Found {num_poles} valid poles.")

        # --- Generowanie obrazów wyjściowych ---
        output_images_to_save = []  # Lista krotek (obraz, pełna_ścieżka_zapisu)

        # 1. ZAWSZE ZAPISZ CZYSTE TŁO
        bg_path = OUTPUT_DIRPATH / f"{base_filename}_background_only.png"
        output_images_to_save.append((img_no_pole.copy(), str(bg_path)))

        if num_poles > 0:
            # Podziel dane na lewe i prawe dla łatwiejszego dostępu
            left_poles = [p for p in poles_data if p["is_left"]]
            right_poles = [p for p in poles_data if not p["is_left"]]

            # 2. ZAPISZ TYLKO LEWE (ORYGINALNE)
            if left_poles:
                img_left_only = img_no_pole.copy()
                for data in left_poles:
                    paste_object(img_left_only, data["pixels"], data["mask"])
                left_only_path = OUTPUT_DIRPATH / f"{base_filename}_left_poles_only.png"
                output_images_to_save.append((img_left_only, str(left_only_path)))

            # 3. ZAPISZ TYLKO PRAWE (ORYGINALNE)
            if right_poles:
                img_right_only = img_no_pole.copy()
                for data in right_poles:
                    paste_object(img_right_only, data["pixels"], data["mask"])
                right_only_path = (
                    OUTPUT_DIRPATH / f"{base_filename}_right_poles_only.png"
                )
                output_images_to_save.append((img_right_only, str(right_only_path)))

            # 4. GENERUJ KOMBINACJE (JEŚLI WIĘCEJ NIŻ 1 PALIK)
            # Generujemy kombinacje dla k od 1 do N palików
            # Uwaga: To obejmie przypadki "tylko lewe", "tylko prawe" i "wszystkie" (oryginał)
            # jeśli występują jako podzbiory.
            print(f"  - Generating combinations for {num_poles} poles...")
            combination_count = 0
            # Iteruj przez możliwe rozmiary podzbiorów (od 1 do N)
            for k in range(1, num_poles):
                # Generuj wszystkie kombinacje palików o rozmiarze k
                for pole_combination in itertools.combinations(poles_data, k):
                    # Sprawdzenie, czy ta kombinacja to nie są przypadkiem "wszystkie lewe" lub "wszystkie prawe"
                    # (jeśli nie chcemy ich duplikować - na razie zostawiamy duplikaty dla prostoty)
                    is_all_left = set(
                        p["original_index"] for p in pole_combination
                    ) == set(p["original_index"] for p in left_poles)
                    is_all_right = set(
                        p["original_index"] for p in pole_combination
                    ) == set(p["original_index"] for p in right_poles)
                    if is_all_left and k == len(left_poles) and k < num_poles:
                        continue  # Skip if it's identical to left_only
                    if is_all_right and k == len(right_poles) and k < num_poles:
                        continue  # Skip if it's identical to right_only

                    img_subset = img_no_pole.copy()
                    # Zbierz indeksy palików w tej kombinacji do nazwy pliku
                    indices_in_combination = sorted(
                        [p["original_index"] for p in pole_combination]
                    )
                    indices_str = "_".join(map(str, indices_in_combination))
                    subset_filename = f"{base_filename}_poles_{indices_str}.png"
                    subset_path = OUTPUT_DIRPATH / subset_filename

                    # Wklej paliki z tej kombinacji
                    for pole_data in pole_combination:
                        paste_object(img_subset, pole_data["pixels"], pole_data["mask"])

                    output_images_to_save.append((img_subset, str(subset_path)))
                    combination_count += 1
            print(f"    - Generated {combination_count} combination images.")

            # 5. GENERUJ AUGMENTACJE Z ROTACJĄ (tak jak poprzednio)
            # Przypadek 1: Jeden palik (już obsłużony przez kombinacje k=1, ale dodajemy rotację)
            if num_poles == 1:
                # print("  - Scenario: 1 pole detected. Generating single rotated image.") # Logika rotacji
                img_rot = img_no_pole.copy()
                data = poles_data[0]
                angle_range = (
                    LEFT_POLE_ANGLE_RANGE if data["is_left"] else RIGHT_POLE_ANGLE_RANGE
                )
                angle = random.uniform(*angle_range)
                rotated_pixels, rotated_mask = rotate_object(
                    data["pixels"], data["mask"], data["base"], angle, (w, h)
                )
                paste_object(img_rot, rotated_pixels, rotated_mask)
                output_path = (
                    OUTPUT_DIRPATH
                    / f"{base_filename}_rotated_pole_{data['original_index']}.png"
                )  # Zmieniona nazwa
                output_images_to_save.append((img_rot, str(output_path)))

            # Przypadek 2: Dwa paliki
            elif num_poles == 2:
                # print("  - Scenario: 2 poles detected. Generating 3 standard rotation augmentations.")
                # Rotacja 1: Prawe obrócone w lewo, lewe oryginalne
                img1 = img_no_pole.copy()
                for data in left_poles:
                    paste_object(img1, data["pixels"], data["mask"])
                for data in right_poles:
                    angle = random.uniform(*RIGHT_POLE_ANGLE_RANGE)
                    rotated_pixels, rotated_mask = rotate_object(
                        data["pixels"], data["mask"], data["base"], angle, (w, h)
                    )
                    paste_object(img1, rotated_pixels, rotated_mask)
                path1 = OUTPUT_DIRPATH / f"{base_filename}_rot_rightL_orig_left.png"
                output_images_to_save.append((img1, str(path1)))

                # Rotacja 2: Lewe obrócone w prawo, prawe oryginalne
                img2 = img_no_pole.copy()
                for data in right_poles:
                    paste_object(img2, data["pixels"], data["mask"])
                for data in left_poles:
                    angle = random.uniform(*LEFT_POLE_ANGLE_RANGE)
                    rotated_pixels, rotated_mask = rotate_object(
                        data["pixels"], data["mask"], data["base"], angle, (w, h)
                    )
                    paste_object(img2, rotated_pixels, rotated_mask)
                path2 = OUTPUT_DIRPATH / f"{base_filename}_rot_leftR_orig_right.png"
                output_images_to_save.append((img2, str(path2)))

                # Rotacja 3: Oba obrócone
                img3 = img_no_pole.copy()
                for data in left_poles:
                    angle = random.uniform(*LEFT_POLE_ANGLE_RANGE)
                    rotated_pixels, rotated_mask = rotate_object(
                        data["pixels"], data["mask"], data["base"], angle, (w, h)
                    )
                    paste_object(img3, rotated_pixels, rotated_mask)
                for data in right_poles:
                    angle = random.uniform(*RIGHT_POLE_ANGLE_RANGE)
                    rotated_pixels, rotated_mask = rotate_object(
                        data["pixels"], data["mask"], data["base"], angle, (w, h)
                    )
                    paste_object(img3, rotated_pixels, rotated_mask)
                path3 = OUTPUT_DIRPATH / f"{base_filename}_rot_both_outwards.png"
                output_images_to_save.append((img3, str(path3)))

            # Przypadek 3: Trzy paliki
            elif num_poles == 3:
                # print("  - Scenario: 3 poles detected. Rotating outer poles outwards, middle randomly.")
                img_rot = img_no_pole.copy()
                sorted_poles = sorted(poles_data, key=lambda p: p["base"][0])
                leftmost, middle, rightmost = sorted_poles

                angle_l = random.uniform(*LEFT_POLE_ANGLE_RANGE)
                rot_pix_l, rot_mask_l = rotate_object(
                    leftmost["pixels"],
                    leftmost["mask"],
                    leftmost["base"],
                    angle_l,
                    (w, h),
                )
                angle_r = random.uniform(*RIGHT_POLE_ANGLE_RANGE)
                rot_pix_r, rot_mask_r = rotate_object(
                    rightmost["pixels"],
                    rightmost["mask"],
                    rightmost["base"],
                    angle_r,
                    (w, h),
                )
                middle_angle_range = random.choice(
                    [LEFT_POLE_ANGLE_RANGE, RIGHT_POLE_ANGLE_RANGE]
                )
                angle_m = random.uniform(*middle_angle_range)
                rot_pix_m, rot_mask_m = rotate_object(
                    middle["pixels"], middle["mask"], middle["base"], angle_m, (w, h)
                )

                paste_object(img_rot, rot_pix_l, rot_mask_l)
                paste_object(img_rot, rot_pix_m, rot_mask_m)
                paste_object(img_rot, rot_pix_r, rot_mask_r)
                output_path = OUTPUT_DIRPATH / f"{base_filename}_rot_three_poles.png"
                output_images_to_save.append((img_rot, str(output_path)))

            # Przypadek 4: Więcej niż 3 paliki
            else:  # num_poles > 3
                # print(f"  - Scenario: {num_poles} poles detected. Rotating all left right, all right left.")
                img_rot = img_no_pole.copy()
                for data in poles_data:
                    angle_range = (
                        LEFT_POLE_ANGLE_RANGE
                        if data["is_left"]
                        else RIGHT_POLE_ANGLE_RANGE
                    )
                    angle = random.uniform(*angle_range)
                    rotated_pixels, rotated_mask = rotate_object(
                        data["pixels"], data["mask"], data["base"], angle, (w, h)
                    )
                    paste_object(img_rot, rotated_pixels, rotated_mask)
                output_path = (
                    OUTPUT_DIRPATH / f"{base_filename}_rot_many_poles_outwards.png"
                )
                output_images_to_save.append((img_rot, str(output_path)))

        # --- Koniec generowania dla num_poles > 0 ---
        elif num_poles == 0:
            print("  - Scenario: No poles detected. Only background image is saved.")

        # --- Zapis wszystkich zgromadzonych obrazów ---
        saved_count_for_this_bg = 0
        if output_images_to_save:
            print(
                f"  - Attempting to save {len(output_images_to_save)} images for {base_filename}..."
            )
            for img_data, save_path in output_images_to_save:
                try:
                    # Sprawdzenie czy ścieżka istnieje - na wszelki wypadek
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    success = cv.imwrite(save_path, img_data)
                    if success:
                        saved_count_for_this_bg += 1
                    else:
                        print(
                            f"  - Warning: cv.imwrite failed for {save_path} (returned False)"
                        )
                except Exception as e:
                    print(f"  - Error saving image {save_path}: {e}")
                    # Można zdecydować czy kontynuować zapisywanie reszty, czy pominąć ten obraz tła
            print(
                f"    - Successfully saved {saved_count_for_this_bg} / {len(output_images_to_save)} images."
            )
            total_files_saved += saved_count_for_this_bg
            if (
                saved_count_for_this_bg > 0 and num_poles > 0
            ):  # Liczymy jako przetworzony, jeśli zapisano cokolwiek *poza* samym tłem
                processed_count += 1  # Można argumentować, czy liczyć to per tło czy per zapisany plik augmentacji
        else:
            print(
                f"  - No images generated or collected for saving for {base_filename}."
            )
            skipped_count += (
                1  # Jeśli nic nie zostało wygenerowane, liczymy jako pominięty
            )

    print(f"\n--- Processing finished ---")
    print(
        f"Background images processed (generating at least one output file): {len(image_files) - skipped_count}"
    )
    print(
        f"Augmentation sets generated (for backgrounds with poles): {processed_count}"
    )
    print(f"Skipped/failed background images: {skipped_count}")
    print(f"Total individual image files saved: {total_files_saved}")


if __name__ == "__main__":
    main()
