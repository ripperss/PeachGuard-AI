"""Структуры данных для результатов инференса."""

from dataclasses import dataclass

from peachguard.detection import CLASS_ID_TO_NAME


@dataclass(frozen=True)
class BoundingBox:
    """Рамка объекта в формате xyxy: левый верхний и правый нижний углы."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: BoundingBox

    @property
    def class_name(self) -> str:
        return CLASS_ID_TO_NAME.get(self.class_id, str(self.class_id))
