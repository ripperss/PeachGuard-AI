"""Классы, которые возвращает модель детекции болезней персика."""

CLASS_NAMES: tuple[str, ...] = (
    "chloroz",
    "citopsoroz",
    "gomoz",
    "kurchavost",
    "monilioz",
    "muchn_rosa",
    "persik",
    "sgnivshie_persiki",
)

CLASS_ID_TO_NAME: dict[int, str] = dict(enumerate(CLASS_NAMES))
