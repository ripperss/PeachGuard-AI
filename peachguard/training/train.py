"""Запуск обучения модели детекции."""

from pathlib import Path

import torch
from ultralytics import YOLO

from peachguard.training.config import TrainingConfig


def train_model(config: TrainingConfig) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(config.model)
    output_dir = config.output_dir.resolve()

    model.train(
        data=str(config.dataset),
        epochs=config.epochs,
        batch=config.batch,
        imgsz=config.image_size,
        device=device,
        workers=config.workers,
        patience=config.patience,
        lr0=config.learning_rate,
        project=str(output_dir),
        name=config.run_name,
        fraction=config.fraction,
        save_period=config.save_period,
        exist_ok=config.exist_ok,
    )

    return output_dir / config.run_name
