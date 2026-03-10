from pathlib import Path
import subprocess


DATASET_NAME = "paultimothymooney/recipenlg"
RAW_DATA_FOLDER = Path("data/raw")


def make_raw_data_folder() -> None:
    """Create the raw data folder if it does not already exist."""
    RAW_DATA_FOLDER.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset() -> None:
    """
    Download the RecipeNLG dataset from Kaggle into data/raw.
    """
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_NAME,
        "-p",
        str(RAW_DATA_FOLDER),
        "--unzip",
    ]

    print("Starting Kaggle download...")
    print("Dataset:", DATASET_NAME)
    print("Saving files to:", RAW_DATA_FOLDER)

    subprocess.run(command, check=True)

    print("Download finished.")


def main() -> None:
    make_raw_data_folder()
    download_kaggle_dataset()


if __name__ == "__main__":
    main()