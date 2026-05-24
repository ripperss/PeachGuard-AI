"""Отчёт по истории обучения YOLO: сводка метрик и графики."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_FILENAME = "results.csv"
SUMMARY_FILENAME = "summary.json"
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
TRAIN_LOSS_KEYS = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
VAL_LOSS_KEYS = ("val/box_loss", "val/cls_loss", "val/dfl_loss")
LR_KEYS = ("lr/pg0", "lr/pg1", "lr/pg2")


def find_results_csv(run_dir: Path) -> Path:
    """Найти results.csv в директории запуска обучения."""
    run_dir = run_dir.resolve()
    direct = run_dir / RESULTS_FILENAME
    if direct.is_file():
        return direct

    matches = sorted(run_dir.rglob(RESULTS_FILENAME))
    if not matches:
        msg = f"Не найден {RESULTS_FILENAME} в {run_dir}. Скопируйте папку запуска из Colab/Ultralytics."
        raise FileNotFoundError(msg)
    return matches[0]


def load_results_csv(csv_path: Path) -> list[dict[str, float | int]]:
    """Загрузить results.csv; пустые ячейки пропускаются."""
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Пустой CSV: {csv_path}")

        rows: list[dict[str, float | int]] = []

        for raw in reader:
            row: dict[str, float | int] = {}
            for key, raw_value in raw.items():
                if key is None:
                    continue
                column = key.strip()
                value = (raw_value or "").strip()
                if not value:
                    continue
                if column == "epoch":
                    row[column] = int(float(value))
                else:
                    row[column] = float(value)
            if "epoch" in row:
                rows.append(row)

    if not rows:
        raise ValueError(f"В {csv_path} нет строк с данными.")
    return rows


def _series(rows: list[dict[str, float | int]], key: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        if key not in row:
            continue
        epochs.append(int(row["epoch"]))
        values.append(float(row[key]))
    return epochs, values


def _best_epoch(rows: list[dict[str, float | int]], metric_key: str) -> dict[str, Any] | None:
    epochs, values = _series(rows, metric_key)
    if not values:
        return None
    best_index = max(range(len(values)), key=values.__getitem__)
    return {"epoch": epochs[best_index], "value": values[best_index]}


def build_summary(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    """Сводка по последней эпохе и лучшим значениям метрик."""
    last = rows[-1]
    last_epoch = int(last["epoch"])

    summary: dict[str, Any] = {
        "epochs_total": last_epoch,
        "last_epoch": {key: last[key] for key in last if key != "epoch"},
    }

    best_metrics: dict[str, Any] = {}
    for key in METRIC_KEYS:
        best = _best_epoch(rows, key)
        if best is not None:
            best_metrics[key] = best
    if best_metrics:
        summary["best_metrics"] = best_metrics

    map50_best = best_metrics.get("metrics/mAP50(B)")
    if map50_best is not None:
        summary["recommended_epoch"] = map50_best["epoch"]

    return summary


def _plot_series(
    ax: plt.Axes,
    rows: list[dict[str, float | int]],
    keys: tuple[str, ...],
    *,
    title: str,
    ylabel: str,
) -> None:
    plotted = False
    for key in keys:
        epochs, values = _series(rows, key)
        if not values:
            continue
        label = key.split("/", 1)[-1].replace("(B)", "")
        ax.plot(epochs, values, label=label, linewidth=1.8)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=8)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(rows: list[dict[str, float | int]], output_dir: Path) -> list[Path]:
    """Построить графики loss, метрик и learning rate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_series(axes[0], rows, TRAIN_LOSS_KEYS, title="Train loss", ylabel="Loss")
    _plot_series(axes[1], rows, VAL_LOSS_KEYS, title="Validation loss", ylabel="Loss")
    fig.tight_layout()
    loss_path = output_dir / "loss.png"
    _save_figure(fig, loss_path)
    saved.append(loss_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_series(ax, rows, METRIC_KEYS, title="Метрики на validation", ylabel="Значение")
    fig.tight_layout()
    metrics_path = output_dir / "metrics.png"
    _save_figure(fig, metrics_path)
    saved.append(metrics_path)

    if any(key in rows[0] for key in LR_KEYS):
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_series(ax, rows, LR_KEYS, title="Learning rate", ylabel="LR")
        fig.tight_layout()
        lr_path = output_dir / "learning_rate.png"
        _save_figure(fig, lr_path)
        saved.append(lr_path)

    return saved


CLASS_LABELS_RU: dict[str, str] = {
    "bacterial_spot": "Бактериальная пятнистость",
    "brown_rot": "Бурый гниль",
    "healthy_peach": "Здоровый персик",
    "shot_hole": "Дырчатость (плод)",
    "shot_hole_leaf": "Дырчатость (лист)",
}


def plot_class_metrics(class_metrics: list[dict[str, Any]], output_dir: Path) -> Path | None:
    """Столбчатая диаграмма mAP50 / mAP50-95 по классам (финальная валидация)."""
    if not class_metrics:
        return None

    labels = [
        CLASS_LABELS_RU.get(item["class_name"], str(item["class_name"]))
        for item in class_metrics
    ]
    map50 = [float(item["mAP50"]) for item in class_metrics]
    map50_95 = [float(item["mAP50-95"]) for item in class_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = range(len(labels))
    width = 0.35
    ax.bar([x - width / 2 for x in x_positions], map50, width, label="mAP50", color="#3498db")
    ax.bar([x + width / 2 for x in x_positions], map50_95, width, label="mAP50-95", color="#2ecc71")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Значение")
    ax.set_title("Метрики по классам (best.pt, validation)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / "class_metrics.png"
    _save_figure(fig, output_path)
    return output_path


def _copy_artifacts(run_dir: Path, output_dir: Path) -> list[str]:
    """Скопировать готовые артефакты YOLO (confusion matrix и т.д.) в отчёт."""
    copied: list[str] = []
    for name in ("confusion_matrix.png", "confusion_matrix_normalized.png", "results.png"):
        source = run_dir / name
        if source.is_file():
            target = output_dir / name
            shutil.copy2(source, target)
            copied.append(name)
    return copied


def generate_training_report(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Собрать отчёт: summary.json, графики и копии артефактов YOLO.

    ``run_dir`` — папка запуска (где лежит results.csv из Colab или ``artifacts/runs/train``).
    ``output_dir`` — куда сохранить отчёт; по умолчанию ``<run_dir>/report``.
    """
    run_path = Path(run_dir).resolve()
    report_dir = Path(output_dir).resolve() if output_dir else run_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_results_csv(run_path)
    rows = load_results_csv(csv_path)

    summary = build_summary(rows)
    summary["source_csv"] = str(csv_path)
    summary["plots"] = [path.name for path in plot_training_curves(rows, report_dir)]
    summary["copied_artifacts"] = _copy_artifacts(run_path, report_dir)

    meta_path = run_path / "colab_meta.json"
    if meta_path.is_file():
        colab_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        summary["colab"] = {
            key: colab_meta[key]
            for key in (
                "best_epoch_early_stopping",
                "epochs_parsed",
                "epochs_total",
                "class_metrics_final",
            )
            if key in colab_meta
        }
        class_metrics = colab_meta.get("class_metrics_final", [])
        class_plot = plot_class_metrics(class_metrics, report_dir)
        if class_plot is not None:
            summary["plots"].append(class_plot.name)

    summary_path = report_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return report_dir
