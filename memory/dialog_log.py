"""Модуль хранения диалоговых сообщений в памяти процесса."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List, Optional

# Инициализируем логгер уровня модуля, чтобы все операции были хорошо видны в логах.
logger = logging.getLogger("dialog_log")


@dataclass
class DialogMessage:
    """Структура, описывающая одно сообщение диалога."""

    trace_id: str
    channel: str
    user_id: str
    role: str
    text: str
    status: str
    created_at: datetime
    metadata: Dict[str, str]

    def to_dict(self) -> Dict[str, str]:
        """Возвращает сообщение в виде словаря для дальнейшей сериализации."""

        # Используем asdict, но конвертируем datetime в ISO-строку для удобства.
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload


# Используем дек с ограничением длины, чтобы хранение не разрасталось бесконечно.
_STORAGE_LIMIT = 1000
_messages: Deque[DialogMessage] = deque(maxlen=_STORAGE_LIMIT)
_lock = threading.Lock()


def _now() -> datetime:
    """Возвращает текущее UTC-время для метки времени сообщения."""

    return datetime.now(timezone.utc)


def record_message(
    trace_id: str,
    channel: str,
    user_id: str,
    role: str,
    text: str,
    status: str,
    **metadata: str,
) -> DialogMessage:
    """Записывает сообщение в память и возвращает созданную структуру."""

    # Собираем метаданные, убеждаемся в строковом типе значений.
    enriched_metadata = {key: str(value) for key, value in metadata.items()}
    message = DialogMessage(
        trace_id=trace_id,
        channel=channel,
        user_id=user_id,
        role=role,
        text=text,
        status=status,
        created_at=_now(),
        metadata=enriched_metadata,
    )
    logger.debug(
        "Сохраняем сообщение в оперативной памяти: trace_id=%s, канал=%s, пользователь=%s, роль=%s, статус=%s",
        trace_id,
        channel,
        user_id,
        role,
        status,
    )
    with _lock:
        _messages.append(message)
        logger.debug(
            "Размер хранилища после добавления: %s сообщений (лимит=%s)",
            len(_messages),
            _STORAGE_LIMIT,
        )
    return message


def iter_messages(
    *,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterable[DialogMessage]:
    """Позволяет итерироваться по сохранённым сообщениям с фильтрацией."""

    logger.debug(
        "Запрошены сообщения: trace_id=%s, user_id=%s, limit=%s", trace_id, user_id, limit
    )
    with _lock:
        # Создаём список, чтобы избежать проблем при дальнейшей итерации.
        snapshot: List[DialogMessage] = list(_messages)
    filtered: List[DialogMessage] = []
    for message in snapshot:
        if trace_id is not None and message.trace_id != trace_id:
            continue
        if user_id is not None and message.user_id != user_id:
            continue
        filtered.append(message)
        if limit is not None and len(filtered) >= limit:
            break
    logger.debug("Отфильтровано %s сообщений", len(filtered))
    return filtered


def clear_storage() -> None:
    """Очищает хранилище сообщений (полезно для юнит-тестов)."""

    with _lock:
        _messages.clear()
    logger.debug("Хранилище диалогов очищено")
