"""Запуск обучения модели из YAML-конфига."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from peachguard.training import TrainingConfig
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
    parser.add_argument("--epochs", type=int, help="Переопределить количество эпох")
    parser.add_argument("--batch", type=int, help="Переопределить batch size")
    parser.add_argument("--workers", type=int, help="Переопределить количество dataloader workers")
    parser.add_argument("--patience", type=int, help="Переопределить early stopping patience")
    parser.add_argument("--image-size", type=int, help="Переопределить размер изображения")
    parser.add_argument("--output-dir", type=Path, help="Переопределить директорию результатов")
    parser.add_argument("--run-name", help="Переопределить имя запуска")
    parser.add_argument("--fraction", type=float, help="Доля train-датасета для отладочного запуска")
    parser.add_argument("--save-period", type=int, help="Сохранять checkpoint каждые N эпох")
    parser.add_argument("--exist-ok", action="store_true", help="Разрешить перезапись директории запуска")
    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "image_size": args.image_size,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "fraction": args.fraction,
        "save_period": args.save_period,
    }
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}

    if args.exist_ok:
        clean_overrides["exist_ok"] = True

    return replace(config, **clean_overrides)


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    config = apply_overrides(config, args)
    output_dir = train_model(config)
    print(f"Результаты обучения сохранены в {output_dir}")


if __name__ == "__main__":
    main()
