"""Загрузка параметров обучения из YAML-конфига."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    dataset: Path
    epochs: int
    batch: int
    image_size: int
    workers: int
    patience: int
    learning_rate: float
    output_dir: Path
    run_name: str
    fraction: float
    save_period: int
    exist_ok: bool


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    raw_config = _read_yaml(config_path)

    return TrainingConfig(
        model=str(raw_config["model"]),
        dataset=Path(raw_config["dataset"]),
        epochs=int(raw_config["epochs"]),
        batch=int(raw_config["batch"]),
        image_size=int(raw_config["image_size"]),
        workers=int(raw_config["workers"]),
        patience=int(raw_config["patience"]),
        learning_rate=float(raw_config["learning_rate"]),
        output_dir=Path(raw_config["output_dir"]),
        run_name=str(raw_config["run_name"]),
        fraction=float(raw_config.get("fraction", 1.0)),
        save_period=int(raw_config.get("save_period", -1)),
        exist_ok=bool(raw_config.get("exist_ok", False)),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Конфиг обучения должен быть YAML-словарем: {path}")

    return data
