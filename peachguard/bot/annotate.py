"""Отрисовка рамок детекции на изображении."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from peachguard.detection import Detection
from peachguard.bot.labels import label_ru

BOX_COLORS = (
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
)

# Шрифты с поддержкой кириллицы (Windows / Linux / macOS)
_CYRILLIC_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
)


def _load_cyrillic_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _CYRILLIC_FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _supports_cyrillic(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> bool:
    return isinstance(font, ImageFont.FreeTypeFont)


def _image_label(class_name: str, confidence: float, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> str:
    """Подпись на снимке: русский текст, если шрифт умеет кириллицу."""
    percent = f"{confidence * 100:.0f}%"
    if _supports_cyrillic(font):
        return f"{label_ru(class_name)} {percent}"
    readable = class_name.replace("_", " ")
    return f"{readable} {percent}"


def annotate_image(image_path: Path, detections: list[Detection]) -> bytes:
    """Вернуть JPEG с нарисованными рамками и подписями."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_cyrillic_font(size=20)
    line_height = 24 if _supports_cyrillic(font) else 14

    for index, item in enumerate(detections):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        box = item.box
        draw.rectangle(
            (box.x1, box.y1, box.x2, box.y2),
            outline=color,
            width=3,
        )
        label = _image_label(item.class_name, item.confidence, font)
        text_x = box.x1
        text_y = max(0, box.y1 - line_height)

        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((text_x, text_y), label, font=font)
            draw.rectangle(bbox, fill="#000000")
            draw.text((text_x, text_y), label, fill=color, font=font)
        else:
            draw.text((text_x, text_y), label, fill=color, font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
