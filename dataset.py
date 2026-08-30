"""PyTorch dataset for the internal train and validation partitions."""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from split_data import CAMERAS, CLASS_IDS, MANIFEST_COLUMNS, file_sha256


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class DriverDataset(Dataset):
    def __init__(self, data_dir, manifest_path, split, transform=None):
        if split not in ["train", "validation"]:
            raise ValueError("split must be 'train' or 'validation'")

        self.data_dir = Path(data_dir).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser()
        self.transform = transform if transform is not None else build_transform()

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.samples = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != MANIFEST_COLUMNS:
                raise ValueError(f"Manifest columns must be {MANIFEST_COLUMNS}")

            for row in reader:
                if row["split"] not in ["train", "validation"]:
                    raise ValueError("Manifest can only contain train and validation")
                if row["split"] != split:
                    continue
                self.samples.append(self._prepare_sample(row))

        if not self.samples:
            raise ValueError(f"No images found for split: {split}")

    def _prepare_sample(self, row):
        if row["camera"] not in CAMERAS:
            raise ValueError(f"Unknown camera: {row['camera']}")
        if row["class_id"] not in CLASS_IDS:
            raise ValueError(f"Unknown class: {row['class_id']}")

        relative_path = Path(row["relative_path"])
        expected_folder_parts = (row["camera"], "train", row["class_id"])
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 4
            or relative_path.parts[:3] != expected_folder_parts
            or relative_path.suffix.lower() not in [".jpg", ".jpeg"]
        ):
            raise ValueError(f"Invalid relative path: {row['relative_path']}")

        image_path = (self.data_dir / relative_path).resolve()
        try:
            resolved_relative_path = image_path.relative_to(self.data_dir)
        except ValueError as error:
            raise ValueError("Image path is outside the dataset directory") from error

        # Reject links inside image paths.
        paths_to_check = [
            self.data_dir / relative_path.parts[0],
            self.data_dir.joinpath(*relative_path.parts[:2]),
            self.data_dir.joinpath(*relative_path.parts[:3]),
            self.data_dir / relative_path,
        ]
        if any(path.is_symlink() for path in paths_to_check):
            raise ValueError("Symbolic links are not allowed in image paths")
        if resolved_relative_path.parts[:3] != expected_folder_parts:
            raise ValueError("Resolved image path is not in an official train folder")

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {row['relative_path']}")

        return {
            "image_path": image_path,
            "target": CLASS_IDS.index(row["class_id"]),
            "class_id": row["class_id"],
            "camera": row["camera"],
            "sha256": row["sha256"],
        }

    def verify_files(self):
        for sample in self.samples:
            if file_sha256(sample["image_path"]) != sample["sha256"]:
                raise ValueError(
                    f"Image content has changed: {sample['image_path'].name}"
                )
        return len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with Image.open(sample["image_path"]) as image_file:
            image = image_file.convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "target": sample["target"],
            "class_id": sample["class_id"],
            "camera": sample["camera"],
        }
