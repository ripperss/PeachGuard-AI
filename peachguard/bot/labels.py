"""Человекочитаемые подписи классов для ответов бота."""

CLASS_LABELS_RU: dict[str, str] = {
    "bacterial_spot": "Бактериальная пятнистость",
    "brown_rot": "Бурый гниль",
    "healthy_peach": "Здоровый персик",
    "shot_hole": "Дырчатость (плод)",
    "shot_hole_leaf": "Дырчатость (лист)",
}

CLASS_HINTS_RU: dict[str, str] = {
    "bacterial_spot": "Тёмные маслянистые пятна на плодах и листьях.",
    "brown_rot": "Бурые вдавленные участки на плодах, быстро распространяется.",
    "healthy_peach": "Признаков типичных болезней на снимке не обнаружено.",
    "shot_hole": "Круглые «дырки» и вдавленные пятна на плоде.",
    "shot_hole_leaf": "Отверстия и некроз на листьях.",
}


def label_ru(class_name: str) -> str:
    return CLASS_LABELS_RU.get(class_name, class_name)
