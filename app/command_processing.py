"""Логика обработки команд пользователя с использованием журнала диалога."""

from __future__ import annotations

import logging
from typing import Dict

from memory.dialog_log import record_message

# Создаём именованный логгер для более удобного анализа событий.
logger = logging.getLogger("command_processing")


def process_command(
    trace_id: str,
    channel: str,
    user_id: str,
    text: str,
) -> Dict[str, str]:
    """Обрабатывает пользовательскую команду и возвращает ответ системы."""

    logger.info(
        "Начата обработка команды: trace_id=%s, канал=%s, пользователь=%s", trace_id, channel, user_id
    )
    # Фиксируем входящее сообщение в журнале.
    record_message(
        trace_id=trace_id,
        channel=channel,
        user_id=user_id,
        role="user",
        text=text,
        status="processing",
        stage="command_processing",
    )

    # Простейшая бизнес-логика: поддержка нескольких команд.
    normalized = text.strip().lower()
    if not normalized:
        response_text = "Пожалуйста, введите команду."
        status = "empty"
    elif normalized in {"/start", "start"}:
        response_text = "Добро пожаловать! Отправьте запрос, и я постараюсь помочь."
        status = "started"
    elif normalized in {"/help", "help"}:
        response_text = "Доступные команды: /start, /help, echo <текст>."
        status = "help"
    elif normalized.startswith("echo "):
        response_text = text.partition(" ")[2]
        status = "echo"
    else:
        response_text = f"Неизвестная команда: {text}"
        status = "unknown"

    logger.info(
        "Команда обработана: trace_id=%s, пользователь=%s, статус=%s", trace_id, user_id, status
    )
    # Сохраняем ответ системы в журнал.
    record_message(
        trace_id=trace_id,
        channel=channel,
        user_id=user_id,
        role="bot",
        text=response_text,
        status=status,
        stage="command_processing",
    )
    return {"status": status, "response": response_text}
