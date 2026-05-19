"""Интерфейс инференса для модели детекции болезней персика."""

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from peachguard.detection.results import BoundingBox, Detection


class PeachDetector:
    """Обертка над YOLO, возвращающая проектные структуры результатов."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model = YOLO(self.model_path)

    def predict(
        self,
        source: str | Path | Any,
        *,
        image_size: int = 640,
        confidence: float = 0.5,
    ) -> list[Detection]:
        results = self.model.predict(
            source=source,
            imgsz=image_size,
            conf=confidence,
            verbose=False,
        )
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for box, class_id, score in zip(boxes, class_ids, confidences, strict=True):
                label_id = int(class_id)
                x1, y1, x2, y2 = (float(value) for value in box)
                detections.append(
                    Detection(
                        class_id=label_id,
                        confidence=float(score),
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        return detections
