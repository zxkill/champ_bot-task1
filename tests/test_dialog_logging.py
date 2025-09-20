"""Проверки сервисов диалогового журнала и их интеграции."""

from __future__ import annotations

import logging
from typing import List

import pytest

from app.command_processing import process_command
from memory import dialog_log
from memory.dialog_log import DialogMessage, iter_messages, record_message
from notifiers.telegram_listener import handle_update


@pytest.fixture(autouse=True)
def _clear_storage() -> None:
    """Очищает хранилище перед и после каждого теста."""

    dialog_log.clear_storage()
    yield
    dialog_log.clear_storage()


def test_record_and_iter_messages_preserve_metadata() -> None:
    """Проверяет, что запись сообщений сохраняет trace_id и метаданные."""

    record_message(
        trace_id="trace-1",
        channel="telegram",
        user_id="42",
        role="user",
        text="Привет",
        status="received",
        stage="listener",
    )
    record_message(
        trace_id="trace-2",
        channel="web",
        user_id="42",
        role="bot",
        text="Ответ",
        status="done",
        stage="processor",
    )
    found: List[DialogMessage] = list(iter_messages(trace_id="trace-1"))
    assert len(found) == 1
    message = found[0]
    assert message.trace_id == "trace-1"
    assert message.metadata["stage"] == "listener"
    assert message.to_dict()["created_at"].endswith("+00:00")


def test_process_command_logs_user_and_bot_messages() -> None:
    """Команда должна создавать записи для пользователя и бота."""

    result = process_command(trace_id="abc", channel="telegram", user_id="55", text="echo Привет")
    assert result["status"] == "echo"
    assert result["response"] == "Привет"
    stored = list(iter_messages(trace_id="abc"))
    assert len(stored) == 2
    roles = {msg.role for msg in stored}
    assert roles == {"user", "bot"}
    statuses = {msg.status for msg in stored}
    assert statuses == {"processing", "echo"}


def test_handle_update_generates_trace_and_enriches_log() -> None:
    """Проверяет полную цепочку обработки обновления из Telegram."""

    logging.getLogger("command_processing").setLevel(logging.DEBUG)
    update = {
        "message": {
            "message_id": 10,
            "from": {"id": 77},
            "text": "echo тест",
        }
    }
    response = handle_update(update)
    assert response["status"] == "echo"
    assert response["response"] == "тест"
    trace_id = response["trace_id"]
    messages = list(iter_messages(trace_id=trace_id))
    assert len(messages) == 4
    assert {msg.status for msg in messages} >= {"received", "processing", "echo"}
    assert {msg.metadata.get("stage") for msg in messages} == {"telegram_listener", "command_processing"}


def test_handle_update_without_text_is_ignored() -> None:
    """Сообщение без текста должно быть проигнорировано, но зафиксировано."""

    response = handle_update({"message": {"message_id": 11, "from": {"id": 90}}})
    assert response["status"] == "ignored"
    trace_id = response["trace_id"]
    messages = list(iter_messages(trace_id=trace_id))
    assert len(messages) == 1
    assert messages[0].status == "ignored"
    assert messages[0].metadata["stage"] == "telegram_listener"


def test_iter_messages_limit_and_user_filter() -> None:
    """Проверяет ограничения выборки и фильтрацию по пользователю."""

    for index in range(5):
        record_message(
            trace_id=f"trace-{index}",
            channel="telegram",
            user_id="alpha" if index % 2 == 0 else "beta",
            role="user",
            text=f"msg {index}",
            status="received",
            stage="listener",
        )
    messages = list(iter_messages(user_id="alpha", limit=2))
    assert len(messages) == 2
    assert all(msg.user_id == "alpha" for msg in messages)


@pytest.mark.parametrize(
    "text,expected_status",
    [
        ("", "empty"),
        ("/start", "started"),
        ("/help", "help"),
        ("/unknown", "unknown"),
    ],
)
def test_process_command_covers_all_branches(text: str, expected_status: str) -> None:
    """Убеждаемся, что каждая ветка бизнес-логики возвращает ожидаемый статус."""

    result = process_command(
        trace_id=f"branch-{expected_status}",
        channel="telegram",
        user_id="55",
        text=text,
    )
    assert result["status"] == expected_status
