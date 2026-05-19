"""Классы, которые возвращает модель детекции болезней персика."""

CLASS_NAMES: tuple[str, ...] = (
    "bacterial_spot",
    "brown_rot",
    "healthy_peach",
    "shot_hole",
    "shot_hole_leaf",
)

CLASS_ID_TO_NAME: dict[int, str] = dict(enumerate(CLASS_NAMES))
