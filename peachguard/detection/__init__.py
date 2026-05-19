from peachguard.detection.classes import CLASS_ID_TO_NAME, CLASS_NAMES
from peachguard.detection.detector import PeachDetector
from peachguard.detection.results import BoundingBox, Detection

__all__ = [
    "BoundingBox",
    "CLASS_ID_TO_NAME",
    "CLASS_NAMES",
    "Detection",
    "PeachDetector",
]
