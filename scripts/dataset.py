import cv2
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def is_video_file(filename: str) -> bool:
    """Checks if a filename has a video extension."""
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def create_output_directory(output_folder: Path):
    """Creates the output directory if it doesn't exist."""
    output_folder.mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(
    video_path: Path, output_folder: Path, frame_interval: int
):
    """Extracts frames from a single video file at a given interval."""
    print(f"Processing video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))  # cv2.VideoCapture expects string path

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path.name}")
        return 0  # Return saved_count = 0 to indicate failure

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames: {total_frames}, FPS: {frame_rate:.2f}")  # Formatted FPS

    saved_count = 0
    for frame_count in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if frame reading fails

        output_filename = (
            output_folder / f"{video_path.stem}_frame_{saved_count:04d}.jpg"
        )  # More descriptive filename, 4 digit frame counter
        cv2.imwrite(str(output_filename), frame)  # cv2.imwrite expects string path
        saved_count += 1

    cap.release()
    print(f"Saved {saved_count} frames from {video_path.name}.")
    return saved_count


def fast_extract_frames(input_folder: str, output_folder: str, frame_interval: int):
    """
    Extracts frames from video files in an input folder and saves them to an output folder.

    Args:
        input_folder (str): Path to the folder containing video files.
        output_folder (str): Path to the folder where frames will be saved.
        frame_interval (int): Interval (in frames) at which to extract frames.
                              e.g., frame_interval=30 means extract every 30th frame.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    create_output_directory(output_path)

    if not input_path.is_dir():
        print(f"Error: Input folder '{input_folder}' is not a valid directory.")
        return

    total_videos_processed = 0
    total_frames_saved = 0

    for video_file in input_path.iterdir():  # Use pathlib's iterdir
        if is_video_file(video_file.name):
            frames_saved = extract_frames_from_video(
                video_file, output_path, frame_interval
            )
            total_frames_saved += frames_saved
            total_videos_processed += 1

    print(f"\nFrame extraction process completed.")
    print(f"Processed {total_videos_processed} video files.")
    print(f"Total frames saved: {total_frames_saved}")


if __name__ == "__main__":
    input_folder = "dataset"  # Replace with your input folder path
    output_folder = "dataset/images"  # Replace with your output folder path
    frame_interval = 150

    fast_extract_frames(input_folder, output_folder, frame_interval)
