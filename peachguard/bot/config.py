"""Настройки бота из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BotConfig:
    token: str
    model_path: Path
    image_size: int
    confidence: float

    @classmethod
    def from_env(cls) -> BotConfig:
        token = os.environ.get("PEACHGUARD_BOT_TOKEN", "").strip()
        if not token:
            msg = "Задайте переменную окружения PEACHGUARD_BOT_TOKEN."
            raise ValueError(msg)

        model_raw = os.environ.get("PEACHGUARD_MODEL_PATH", "artifacts/models/best.pt").strip()
        model_path = Path(model_raw)
        if not model_path.is_file():
            msg = f"Веса модели не найдены: {model_path.resolve()}"
            raise FileNotFoundError(msg)

        image_size = int(os.environ.get("PEACHGUARD_IMAGE_SIZE", "640"))
        confidence = float(os.environ.get("PEACHGUARD_CONFIDENCE", "0.5"))
        return cls(
            token=token,
            model_path=model_path,
            image_size=image_size,
            confidence=confidence,
        )
