"""Запуск инференса модели на одном изображении."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peachguard.detection import Detection, PeachDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Распознать болезни персика на изображении.")
    parser.add_argument("--model", required=True, type=Path, help="Путь к весам модели .pt")
    parser.add_argument("--image", required=True, type=Path, help="Путь к изображению")
    parser.add_argument("--image-size", default=1024, type=int, help="Размер изображения для модели")
    parser.add_argument("--confidence", default=0.5, type=float, help="Минимальная уверенность детекции")
    return parser.parse_args()


def detection_to_dict(detection: Detection) -> dict[str, object]:
    return {
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "box": {
            "x1": detection.box.x1,
            "y1": detection.box.y1,
            "x2": detection.box.x2,
            "y2": detection.box.y2,
        },
    }


def main() -> None:
    args = parse_args()
    detector = PeachDetector(args.model)
    detections = detector.predict(
        args.image,
        image_size=args.image_size,
        confidence=args.confidence,
    )

    print(json.dumps([detection_to_dict(item) for item in detections], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
