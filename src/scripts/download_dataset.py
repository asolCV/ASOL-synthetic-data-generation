from pathlib import Path
from roboflow import Roboflow

# Konfiguracja Roboflow API
ROBOFLOW_API_KEY = "cRNpwpG5hUaBUszly98K"  # Wstaw tutaj swój klucz API Roboflow
ROBOFLOW_WORKSPACE = "asol-mc75w"
ROBOFLOW_PROJECT = "asol-keypoint"
ROBOFLOW_VERSION = 4
ROBOFLOW_FORMAT = "coco-segmentation"


def download_roboflow_dataset(
    api_key, workspace, project, version, format, output_path: Path
):
    """Pobiera dataset z Roboflow."""
    rf = Roboflow(api_key=api_key)
    workspace_obj = rf.workspace(workspace)
    project_obj = workspace_obj.project(project)
    version_obj = project_obj.version(version)
    dataset = version_obj.download(
        model_format=format,
        location=str(
            output_path / f"{version_obj.name.replace(' ', '-')}-{version_obj.version}"
        ),
    )
    print(f"Dataset Roboflow pobrany do: {output_path}")
    return Path(dataset.location)


if __name__ == "__main__":
    # Upewnij się, że katalogi istnieją
    BASE_DIR = Path(".")
    DATASET_PATH = BASE_DIR / "roboflow_dataset"
    DATASET_PATH.mkdir(parents=True, exist_ok=True)

    # Pobierz dataset z Roboflow
    dataset_path = download_roboflow_dataset(
        ROBOFLOW_API_KEY,
        ROBOFLOW_WORKSPACE,
        ROBOFLOW_PROJECT,
        ROBOFLOW_VERSION,
        ROBOFLOW_FORMAT,
        DATASET_PATH,
    )
