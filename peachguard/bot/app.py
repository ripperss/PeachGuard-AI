"""Запуск Telegram-бота PeachGuard."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from peachguard.bot.annotate import annotate_image
from peachguard.bot.config import BotConfig
from peachguard.bot.formatting import format_detections
from peachguard.detection import PeachDetector

logger = logging.getLogger(__name__)

WELCOME_TEXT = (
    "PeachGuard AI — детекция болезней персика по фото.\n\n"
    "Отправьте изображение персика или листа, и я верну найденные "
    "заболевания с уверенностью модели и разметкой на снимке.\n\n"
    "Классы: бактериальная пятнистость, бурый гниль, дырчатость, здоровый персик."
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(WELCOME_TEXT)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.photo:
        return

    detector: PeachDetector = context.application.bot_data["detector"]
    config: BotConfig = context.application.bot_data["config"]

    status = await message.reply_text("Анализирую изображение…")

    photo = message.photo[-1]
    telegram_file = await photo.get_file()

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "input.jpg"
        await telegram_file.download_to_drive(custom_path=str(image_path))

        detections = detector.predict(
            image_path,
            image_size=config.image_size,
            confidence=config.confidence,
        )
        text = format_detections(detections)
        await status.edit_text(text)

        if detections:
            annotated = annotate_image(image_path, detections)
            await message.reply_photo(photo=annotated, caption="Разметка модели")


def build_application(config: BotConfig) -> Application:
    detector = PeachDetector(config.model_path)
    app = Application.builder().token(config.token).build()
    app.bot_data["config"] = config
    app.bot_data["detector"] = detector

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    return app


def run_bot() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    config = BotConfig.from_env()
    app = build_application(config)
    logger.info("Бот запущен, модель: %s", config.model_path)
    app.run_polling(allowed_updates=Update.ALL_TYPES)
