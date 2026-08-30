"""Metrics used to evaluate the ten-class classifier."""

import torch

from config import NUM_CLASSES


INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def make_confusion_matrix(predictions, targets, num_classes=NUM_CLASSES):
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("predictions and targets must be one-dimensional")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.numel() == 0:
        raise ValueError("predictions and targets must not be empty")
    if predictions.dtype not in INTEGER_DTYPES or targets.dtype not in INTEGER_DTYPES:
        raise ValueError("predictions and targets must contain integer labels")

    prediction_values = predictions.detach().cpu().tolist()
    target_values = targets.detach().cpu().tolist()

    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for prediction, target in zip(prediction_values, target_values):
        if prediction < 0 or prediction >= num_classes:
            raise ValueError("prediction is outside the class range")
        if target < 0 or target >= num_classes:
            raise ValueError("target is outside the class range")
        matrix[target, prediction] += 1

    return matrix


def calculate_metrics(confusion_matrix):
    if confusion_matrix.ndim != 2:
        raise ValueError("confusion_matrix must be two-dimensional")
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("confusion_matrix must be square")
    if confusion_matrix.dtype not in INTEGER_DTYPES:
        raise ValueError("confusion_matrix must contain integer counts")

    matrix = confusion_matrix.detach().cpu().to(torch.int64)
    total = int(matrix.sum().item())
    if total == 0:
        raise ValueError("confusion_matrix must contain observations")

    matrix_values = matrix.tolist()
    num_classes = len(matrix_values)
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for class_index in range(num_classes):
        true_positive = matrix_values[class_index][class_index]
        actual_total = sum(matrix_values[class_index])
        predicted_total = sum(
            matrix_values[row][class_index] for row in range(num_classes)
        )

        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    correct = sum(matrix_values[index][index] for index in range(num_classes))
    macro_precision = sum(per_class_precision) / num_classes
    macro_recall = sum(per_class_recall) / num_classes

    return {
        "accuracy": correct / total,
        # Balanced accuracy equals macro recall here.
        "balanced_accuracy": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": sum(per_class_f1) / num_classes,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": matrix_values,
    }
