"""Импорт истории обучения из stdout ноутбука Colab (PeachGuard.ipynb)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

TRAIN_LINE = re.compile(
    r"(?P<epoch>\d+)/\d+\s+[\d.]+G\s+"
    r"(?P<box>[\d.]+)\s+(?P<cls>[\d.]+)\s+(?P<dfl>[\d.]+)\s+\d+",
)
METRICS_LINE = re.compile(
    r"^\s+all\s+\d+\s+\d+\s+"
    r"(?P<precision>[\d.]+)\s+(?P<recall>[\d.]+)\s+"
    r"(?P<map50>[\d.]+)\s+(?P<map50_95>[\d.]+)\s*$",
)
CLASS_METRICS_LINE = re.compile(
    r"^\s+(?P<class_name>\S+)\s+\d+\s+\d+\s+"
    r"(?P<precision>[\d.]+)\s+(?P<recall>[\d.]+)\s+"
    r"(?P<map50>[\d.]+)\s+(?P<map50_95>[\d.]+)\s*$",
)
EARLY_STOPPING = re.compile(r"Best results observed at epoch (\d+)", re.IGNORECASE)


def _notebook_stdout_lines(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cell in data.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            text = output.get("text", "")
            if isinstance(text, list):
                lines.extend(text)
            elif text:
                lines.append(text)
    return lines


def parse_training_log(lines: list[str]) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Собрать строки эпох и метаданные из лога обучения в ноутбуке."""
    rows: list[dict[str, float | int]] = []
    meta: dict[str, Any] = {}
    pending_train: dict[str, float] | None = None

    for raw_line in lines:
        line = raw_line.replace("\u001b[K", "").replace("\r", "")

        stop_match = EARLY_STOPPING.search(line)
        if stop_match:
            meta["best_epoch_early_stopping"] = int(stop_match.group(1))

        train_match = TRAIN_LINE.search(line)
        if train_match:
            pending_train = {
                "train/box_loss": float(train_match.group("box")),
                "train/cls_loss": float(train_match.group("cls")),
                "train/dfl_loss": float(train_match.group("dfl")),
            }
            epoch = int(train_match.group("epoch"))
            pending_train["_epoch"] = epoch
            continue

        metrics_match = METRICS_LINE.match(line)
        if metrics_match and pending_train is not None:
            row: dict[str, float | int] = {
                "epoch": int(pending_train["_epoch"]),
                "train/box_loss": pending_train["train/box_loss"],
                "train/cls_loss": pending_train["train/cls_loss"],
                "train/dfl_loss": pending_train["train/dfl_loss"],
                "metrics/precision(B)": float(metrics_match.group("precision")),
                "metrics/recall(B)": float(metrics_match.group("recall")),
                "metrics/mAP50(B)": float(metrics_match.group("map50")),
                "metrics/mAP50-95(B)": float(metrics_match.group("map50_95")),
            }
            rows.append(row)
            pending_train = None

    if not rows:
        msg = "В ноутбуке не найден лог эпох обучения (train + metrics)."
        raise ValueError(msg)

    meta["epochs_parsed"] = len(rows)
    meta["epochs_total"] = int(rows[-1]["epoch"])
    return rows, meta


def parse_final_class_metrics(lines: list[str]) -> list[dict[str, Any]]:
    """Метрики по классам из финальной валидации best.pt."""
    classes: list[dict[str, Any]] = []
    after_early_stop = False

    for raw_line in lines:
        line = raw_line.replace("\r", "")
        if "EarlyStopping" in line or "Validating" in line:
            after_early_stop = True
        if not after_early_stop:
            continue

        match = CLASS_METRICS_LINE.match(line)
        if not match or match.group("class_name") == "all":
            continue
        classes.append(
            {
                "class_name": match.group("class_name"),
                "precision": float(match.group("precision")),
                "recall": float(match.group("recall")),
                "mAP50": float(match.group("map50")),
                "mAP50-95": float(match.group("map50_95")),
            }
        )

    return classes


def write_results_csv(rows: list[dict[str, float | int]], csv_path: Path) -> None:
    fieldnames = [
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def import_colab_notebook(
    notebook_path: str | Path,
    *,
    run_dir: str | Path,
) -> Path:
    """
    Извлечь results.csv из PeachGuard.ipynb в ``run_dir``.

    Возвращает путь к созданному CSV.
    """
    notebook = Path(notebook_path).resolve()
    output = Path(run_dir).resolve()
    lines = _notebook_stdout_lines(notebook)
    rows, meta = parse_training_log(lines)
    class_metrics = parse_final_class_metrics(lines)

    csv_path = output / "results.csv"
    write_results_csv(rows, csv_path)

    meta_path = output / "colab_meta.json"
    meta["class_metrics_final"] = class_metrics
    meta["notebook"] = str(notebook)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return csv_path
