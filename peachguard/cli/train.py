"""Запуск обучения модели из YAML-конфига."""

from __future__ import annotations

import argparse
from pathlib import Path

from peachguard.training import load_training_config
from peachguard.training.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучить модель детекции болезней персика.")
    parser.add_argument(
        "--config",
        default=Path("configs/train.yaml"),
        type=Path,
        help="Путь к YAML-конфигу обучения",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    output_dir = train_model(config)
    print(f"Результаты обучения сохранены в {output_dir}")


if __name__ == "__main__":
    main()
