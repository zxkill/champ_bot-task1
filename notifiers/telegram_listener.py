"""Обработчик Telegram-уведомлений с интеграцией в журнал диалогов."""

from __future__ import annotations

import logging
from typing import Dict, Optional
from uuid import uuid4

from app.command_processing import process_command
from memory.dialog_log import record_message

logger = logging.getLogger("telegram_listener")


def _extract_user_id(update: Dict) -> str:
    """Выделяет идентификатор пользователя из Telegram-обновления."""

    return str(update.get("message", {}).get("from", {}).get("id", "unknown"))


def _extract_text(update: Dict) -> Optional[str]:
    """Безопасно извлекает текст сообщения."""

    message = update.get("message", {})
    return message.get("text")


def handle_update(update: Dict) -> Dict[str, str]:
    """Основная точка входа для обработки входящих обновлений Telegram."""

    trace_id = str(update.get("trace_id") or update.get("message", {}).get("message_id") or uuid4())
    channel = "telegram"
    user_id = _extract_user_id(update)
    logger.info(
        "Получено обновление из Telegram: trace_id=%s, пользователь=%s", trace_id, user_id
    )
    text = _extract_text(update)
    if text is None:
        logger.warning(
            "В обновлении отсутствует текст: trace_id=%s, пользователь=%s", trace_id, user_id
        )
        record_message(
            trace_id=trace_id,
            channel=channel,
            user_id=user_id,
            role="system",
            text="Сообщение без текста",
            status="ignored",
            stage="telegram_listener",
        )
        return {"trace_id": trace_id, "status": "ignored"}

    record_message(
        trace_id=trace_id,
        channel=channel,
        user_id=user_id,
        role="user",
        text=text,
        status="received",
        stage="telegram_listener",
    )
    logger.debug(
        "Текст успешно извлечён, запускаем обработку команды: trace_id=%s", trace_id
    )
    result = process_command(trace_id=trace_id, channel=channel, user_id=user_id, text=text)
    logger.info(
        "Ответ сформирован и готов к отправке: trace_id=%s, статус=%s", trace_id, result["status"]
    )
    record_message(
        trace_id=trace_id,
        channel=channel,
        user_id=user_id,
        role="system",
        text="Ответ Telegram готов к доставке",
        status=result["status"],
        stage="telegram_listener",
    )
    enriched = {"trace_id": trace_id}
    enriched.update(result)
    return enriched
