"""Train one model on the internal train and validation splits."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EXPECTED_INTERNAL_TRAIN_IMAGES,
    EXPECTED_INTERNAL_VALIDATION_IMAGES,
    MAX_EPOCHS,
    MODEL_SETTINGS,
    NUM_CLASSES,
    SEEDS,
    WEIGHT_DECAY,
)
from dataset import DriverDataset
from metrics import calculate_metrics, make_confusion_matrix
from models import count_parameters, create_model
from split_data import file_sha256


CHECKPOINT_SCORE_TOLERANCE = 1e-12
RELOADED_VALIDATION_TOLERANCE = 1e-8


def set_seed(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    torch.manual_seed(seed)


def select_device(device_name="auto"):
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this computer")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this computer")
    if device_name not in ["mps", "cuda", "cpu"]:
        raise ValueError("device must be auto, mps, cuda or cpu")
    return torch.device(device_name)


def make_data_loaders(data_dir, manifest_path, seed, batch_size=BATCH_SIZE):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    train_data = DriverDataset(data_dir, manifest_path, "train")
    validation_data = DriverDataset(data_dir, manifest_path, "validation")
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, validation_loader


def run_epoch(model, data_loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    for batch in data_loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device, dtype=torch.int64)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            if logits.shape != (targets.shape[0], NUM_CLASSES):
                raise ValueError(
                    f"Model output must have shape [batch_size, {NUM_CLASSES}]"
                )
            loss = loss_function(logits, targets)

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Loss is not finite")
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        predictions = logits.detach().argmax(dim=1)
        confusion += make_confusion_matrix(predictions, targets)

    if total_samples == 0:
        raise ValueError("The DataLoader did not contain any samples")

    return {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "metrics": calculate_metrics(confusion),
    }


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def save_checkpoint(path, model, epoch, best_macro_f1, model_name, seed):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_validation_macro_f1": best_macro_f1,
        "model_name": model_name,
        "seed": seed,
    }
    torch.save(checkpoint, path)


def train_model(
    model,
    train_loader,
    validation_loader,
    *,
    model_name,
    seed,
    learning_rate,
    output_dir,
    device,
    pretrained,
    manifest_sha256=None,
    weight_decay=WEIGHT_DECAY,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=EARLY_STOPPING_MIN_DELTA,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if max_epochs < 1 or patience < 1 or min_delta < 0:
        raise ValueError("Invalid epoch or early-stopping setting")
    output_dir.mkdir(parents=True)

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.json"
    history = []
    best_validation_macro_f1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "max_epochs"

    record = {
        "stage": "development",
        "official_test_accessed": False,
        "model_name": model_name,
        "seed": seed,
        "device": str(device),
        "pytorch_version": str(torch.__version__),
        "pretrained": pretrained,
        "pretrained_weights": "IMAGENET1K_V1" if pretrained else None,
        "loss_function": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": train_loader.batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "parameter_count": count_parameters(model),
        "train_images": len(train_loader.dataset),
        "validation_images": len(validation_loader.dataset),
        "development_manifest_sha256": manifest_sha256,
        "history": history,
    }

    for epoch in range(1, max_epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        validation_macro_f1 = validation_result["metrics"]["macro_f1"]
        is_new_best = (
            validation_macro_f1 > best_validation_macro_f1 + min_delta
        )
        if is_new_best:
            best_validation_macro_f1 = validation_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                epoch,
                best_validation_macro_f1,
                model_name,
                seed,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train": train_result,
                "validation": validation_result,
                "best_checkpoint_updated": is_new_best,
            }
        )
        record["completed_epochs"] = epoch
        record["best_epoch"] = best_epoch
        record["best_validation_macro_f1"] = best_validation_macro_f1
        write_json(history_path, record)

        print(
            f"epoch {epoch}: train loss={train_result['loss']:.4f}, "
            f"validation loss={validation_result['loss']:.4f}, "
            f"validation macro-F1={validation_macro_f1:.4f}"
        )

        if epochs_without_improvement >= patience:
            stop_reason = "early_stopping"
            break

    # Reload the selected epoch instead of the final epoch.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if checkpoint["model_name"] != model_name or checkpoint["seed"] != seed:
        raise ValueError("Checkpoint does not match this training run")
    if checkpoint["epoch"] != best_epoch:
        raise ValueError("Checkpoint epoch does not match the run record")
    checkpoint_score_difference = abs(
        checkpoint["best_validation_macro_f1"] - best_validation_macro_f1
    )
    if checkpoint_score_difference > CHECKPOINT_SCORE_TOLERANCE:
        raise ValueError("Checkpoint score does not match the run record")
    model.load_state_dict(checkpoint["model_state_dict"])

    selected_validation = run_epoch(
        model,
        validation_loader,
        loss_function,
        device,
    )
    selected_macro_f1 = selected_validation["metrics"]["macro_f1"]
    selected_score_difference = abs(
        selected_macro_f1 - best_validation_macro_f1
    )
    if selected_score_difference > RELOADED_VALIDATION_TOLERANCE:
        raise RuntimeError("Reloaded checkpoint does not match the recorded score")

    record["status"] = "completed"
    record["stop_reason"] = stop_reason
    record["best_checkpoint"] = "best_model.pt"
    record["best_checkpoint_sha256"] = file_sha256(checkpoint_path)
    record["selected_validation"] = selected_validation
    write_json(history_path, record)
    return record


def run_experiment(args):
    set_seed(args.seed)
    manifest_hash = file_sha256(args.manifest)
    settings = MODEL_SETTINGS[args.model]
    model = create_model(args.model, pretrained=settings["pretrained"])
    if count_parameters(model) != settings["expected_parameters"]:
        raise RuntimeError("Model parameter count does not match config.py")

    train_loader, validation_loader = make_data_loaders(
        args.data_dir,
        args.manifest,
        args.seed,
    )
    if len(train_loader.dataset) != EXPECTED_INTERNAL_TRAIN_IMAGES:
        raise ValueError(
            "The internal train split must contain "
            f"{EXPECTED_INTERNAL_TRAIN_IMAGES:,} images"
        )
    if len(validation_loader.dataset) != EXPECTED_INTERNAL_VALIDATION_IMAGES:
        raise ValueError(
            "The internal validation split must contain "
            f"{EXPECTED_INTERNAL_VALIDATION_IMAGES:,} images"
        )
    verified_train_file_count = train_loader.dataset.verify_files()
    verified_validation_file_count = validation_loader.dataset.verify_files()

    device = select_device(args.device)
    record = train_model(
        model,
        train_loader,
        validation_loader,
        model_name=args.model,
        seed=args.seed,
        learning_rate=settings["learning_rate"],
        output_dir=args.output_dir,
        device=device,
        manifest_sha256=manifest_hash,
        pretrained=settings["pretrained"],
    )
    if file_sha256(args.manifest) != manifest_hash:
        record["status"] = "failed"
        record["stop_reason"] = "manifest_changed"
        record["manifest_unchanged"] = False
        write_json(Path(args.output_dir) / "history.json", record)
        raise RuntimeError("The development manifest changed during training")
    record["manifest_unchanged"] = True
    record["verified_train_file_count"] = verified_train_file_count
    record["verified_validation_file_count"] = verified_validation_file_count
    write_json(Path(args.output_dir) / "history.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description="Train one development model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_SETTINGS),
    )
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    args = parser.parse_args()

    result = run_experiment(args)
    print(
        "training completed; best validation macro-F1: "
        f"{result['best_validation_macro_f1']:.4f}"
    )
    print("official test was not accessed")


if __name__ == "__main__":
    main()
