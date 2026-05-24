"""Форматирование результатов детекции для Telegram."""

from peachguard.detection import Detection
from peachguard.bot.labels import CLASS_HINTS_RU, label_ru


def format_detections(detections: list[Detection]) -> str:
    if not detections:
        return (
            "На фото ничего не найдено с заданным порогом уверенности.\n"
            "Попробуйте другое изображение или снизьте PEACHGUARD_CONFIDENCE."
        )

    lines = [f"Найдено объектов: {len(detections)}", ""]
    for index, item in enumerate(detections, start=1):
        name = item.class_name
        title = label_ru(name)
        hint = CLASS_HINTS_RU.get(name, "")
        percent = item.confidence * 100
        box = item.box
        lines.append(f"{index}. {title} ({name})")
        lines.append(f"   Уверенность: {percent:.1f}%")
        lines.append(f"   Рамка: ({box.x1:.0f}, {box.y1:.0f}) — ({box.x2:.0f}, {box.y2:.0f})")
        if hint:
            lines.append(f"   {hint}")
        lines.append("")

    return "\n".join(lines).rstrip()
