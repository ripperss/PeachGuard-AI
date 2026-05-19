"""Запуск обучения модели детекции."""

from pathlib import Path

import torch
from ultralytics import YOLO

from peachguard.training.config import TrainingConfig


def train_model(config: TrainingConfig) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(config.model)

    model.train(
        data=str(config.dataset),
        epochs=config.epochs,
        batch=config.batch,
        imgsz=config.image_size,
        device=device,
        workers=config.workers,
        patience=config.patience,
        lr0=config.learning_rate,
        project=str(config.output_dir),
        name=config.run_name,
    )

    return config.output_dir
