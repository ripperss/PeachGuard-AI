"""Сборка отчёта по истории обучения из results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from peachguard.training.colab_import import import_colab_notebook
from peachguard.training.report import generate_training_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить графики и сводку по истории обучения YOLO (results.csv).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Папка запуска обучения (Colab export или artifacts/runs/yolo11n-100e)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Куда сохранить отчёт (по умолчанию: <run-dir>/report)",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=None,
        help="PeachGuard.ipynb: извлечь results.csv из stdout Colab перед отчётом",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.notebook is not None:
        csv_path = import_colab_notebook(args.notebook, run_dir=args.run_dir)
        print(f"История обучения извлечена в {csv_path}")

    report_dir = generate_training_report(args.run_dir, output_dir=args.output_dir)
    print(f"Отчёт сохранён в {report_dir}")


if __name__ == "__main__":
    main()
