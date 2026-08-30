"""Create a development-only train/validation manifest."""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from config import (
    EXPECTED_INTERNAL_TRAIN_IMAGES,
    EXPECTED_INTERNAL_VALIDATION_IMAGES,
    EXPECTED_OFFICIAL_TRAIN_IMAGES,
    INTERNAL_SPLIT_SEED,
    INTERNAL_VALIDATION_FRACTION,
)


CAMERAS = ["Camera 1", "Camera 2"]
CLASS_IDS = [f"c{index}" for index in range(10)]
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
MANIFEST_COLUMNS = ["relative_path", "camera", "class_id", "split", "sha256"]
SPLIT_METHOD_NAME = "auc-v2-camera-class-sha256-v1"


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def find_training_images(dataset_root):
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    rows = []
    for camera in CAMERAS:
        for class_id in CLASS_IDS:
            camera_dir = dataset_root / camera
            train_dir = camera_dir / "train"
            class_dir = train_dir / class_id

            for folder in [camera_dir, train_dir, class_dir]:
                if folder.is_symlink():
                    raise ValueError(f"Symbolic links are not allowed: {folder}")
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing folder: {camera}/train/{class_id}")

            image_paths = []
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if path.is_symlink():
                    raise ValueError(f"Symbolic links are not allowed: {path}")
                if path.is_file():
                    image_paths.append(path)
            if not image_paths:
                raise ValueError(f"No images found in {camera}/train/{class_id}")

            for path in image_paths:
                rows.append(
                    {
                        "relative_path": path.relative_to(dataset_root).as_posix(),
                        "camera": camera,
                        "class_id": class_id,
                        "sha256": file_sha256(path),
                    }
                )

    rows.sort(key=lambda row: row["relative_path"])
    return rows


def calculate_validation_quotas(rows, validation_fraction):
    """Allocate the exact validation total across camera/class groups."""

    counts = Counter((row["camera"], row["class_id"]) for row in rows)
    fraction = Decimal(str(validation_fraction))
    exact_quotas = {
        group: Decimal(count) * fraction for group, count in counts.items()
    }
    quotas = {group: int(value) for group, value in exact_quotas.items()}

    target_validation_count = int(
        (Decimal(len(rows)) * fraction).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    remaining_places = target_validation_count - sum(quotas.values())

    ranked_groups = sorted(
        counts,
        key=lambda group: (
            -(exact_quotas[group] - quotas[group]),
            CAMERAS.index(group[0]),
            CLASS_IDS.index(group[1]),
        ),
    )
    for group in ranked_groups[:remaining_places]:
        quotas[group] += 1

    return quotas


def make_group_sort_key(seed, camera, class_id, content_sha256):
    """Create a repeatable sorting key for one image group."""

    text = (
        f"{SPLIT_METHOD_NAME}\0{seed}\0{camera}\0"
        f"{class_id}\0{content_sha256}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_validation_hashes(groups, target, seed, camera, class_id):
    """Choose duplicate-image groups that exactly fill the quota."""

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            make_group_sort_key(seed, camera, class_id, item[0]),
            item[0],
        ),
    )

    possible_totals = {0: []}
    for content_sha256, group_rows in ordered_groups:
        group_size = len(group_rows)
        for current_total in sorted(list(possible_totals), reverse=True):
            new_total = current_total + group_size
            if new_total <= target and new_total not in possible_totals:
                possible_totals[new_total] = (
                    possible_totals[current_total] + [content_sha256]
                )
        if target in possible_totals:
            break

    if target not in possible_totals:
        raise ValueError(
            f"Cannot create the requested split for {camera}/{class_id} "
            "without separating duplicate images"
        )
    return set(possible_totals[target])


def build_manifest(
    dataset_root,
    validation_fraction=INTERNAL_VALIDATION_FRACTION,
    seed=INTERNAL_SPLIT_SEED,
    expected_image_count=EXPECTED_OFFICIAL_TRAIN_IMAGES,
):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    rows = find_training_images(dataset_root)
    if expected_image_count is not None and len(rows) != expected_image_count:
        raise ValueError(
            f"Expected {expected_image_count} official-train images, "
            f"found {len(rows)}"
        )

    # Reject duplicate content with conflicting labels or cameras.
    locations_by_hash = defaultdict(set)
    for row in rows:
        locations_by_hash[row["sha256"]].add(
            (row["camera"], row["class_id"])
        )
    for content_sha256, locations in locations_by_hash.items():
        if len(locations) > 1:
            raise ValueError(
                "Identical image content appears under different labels or cameras: "
                f"{content_sha256}"
            )

    quotas = calculate_validation_quotas(rows, validation_fraction)
    validation_hashes = {}

    for camera in CAMERAS:
        for class_id in CLASS_IDS:
            group_rows = [
                row
                for row in rows
                if row["camera"] == camera and row["class_id"] == class_id
            ]
            groups = defaultdict(list)
            for row in group_rows:
                groups[row["sha256"]].append(row)

            validation_hashes[(camera, class_id)] = choose_validation_hashes(
                groups,
                quotas[(camera, class_id)],
                seed,
                camera,
                class_id,
            )

    manifest_rows = []
    for row in rows:
        selected_hashes = validation_hashes[(row["camera"], row["class_id"])]
        split = "validation" if row["sha256"] in selected_hashes else "train"
        manifest_rows.append({**row, "split": split})

    manifest_rows.sort(key=lambda row: row["relative_path"])
    split_counts = Counter(row["split"] for row in manifest_rows)
    expected_validation_count = int(
        (Decimal(len(rows)) * Decimal(str(validation_fraction))).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    if split_counts["validation"] != expected_validation_count:
        raise RuntimeError("Validation split has the wrong number of images")

    # Keep duplicate content in one split.
    splits_by_hash = defaultdict(set)
    for row in manifest_rows:
        splits_by_hash[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in splits_by_hash.values()):
        raise RuntimeError("Duplicate images were separated across splits")

    return manifest_rows


def save_manifest(rows, output_path):
    output_path = Path(output_path).expanduser()
    if output_path.exists():
        raise FileExistsError(
            f"Manifest already exists: {output_path}. "
            "Use a new output path to avoid replacing a fixed split."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in MANIFEST_COLUMNS})

    return file_sha256(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create the development split")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="local/manifest.csv")
    args = parser.parse_args()

    rows = build_manifest(args.data_dir)
    manifest_hash = save_manifest(rows, args.output)
    counts = Counter(row["split"] for row in rows)

    print(f"train: {counts['train']}")
    print(f"validation: {counts['validation']}")
    print(f"development manifest SHA-256: {manifest_hash}")

    if len(rows) == EXPECTED_OFFICIAL_TRAIN_IMAGES:
        assert counts["train"] == EXPECTED_INTERNAL_TRAIN_IMAGES
        assert counts["validation"] == EXPECTED_INTERNAL_VALIDATION_IMAGES


if __name__ == "__main__":
    main()
