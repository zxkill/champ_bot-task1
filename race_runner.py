"""Скрипт для быстрого прохождения соревновательного маршрута.

В отличие от ``demo.py`` с обнаружением коридоров, здесь маршрут задан
явно набором точек. Такой подход отлично подходит для гоночного сценария
на рисунке: робот начинает в правой комнате, огибает препятствия, проходит
по перемычке и точно заезжает на белый прямоугольник в левой комнате.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Sequence

from navigator import NavigatorConfig, Waypoint, WaypointNavigator


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("race")

CMD_HOST = str(os.getenv("CMD_HOST", "127.0.0.1"))
CMD_PORT = int(os.getenv("CMD_PORT", "5555"))
TEL_HOST = str(os.getenv("TEL_HOST", "0.0.0.0"))
TEL_PORT = int(os.getenv("TEL_PORT", "5600"))
PROTO = str(os.getenv("PROTO", "tcp"))

LIDAR_FOV = float(os.getenv("LIDAR_FOV", "45.0"))


@dataclass(frozen=True)
class RaceRoute:
    """Заготовленный маршрут, удобный для подмены в тестах."""

    waypoints: Sequence[Waypoint]

    @staticmethod
    def default() -> "RaceRoute":
        """Стандартная траектория для мира с двумя комнатами."""

        return RaceRoute(
            waypoints=(
                # 1) Поддерживаем небольшой разгон вперёд, чтобы робот стабилизировал курс.
                Waypoint(18.0, -0.7, 0.75),
                Waypoint(17.6, -0.8, 0.78),
                Waypoint(17.2, -0.9, 0.8),
                # 2) Поворачиваемся вокруг колонны уточек в правой комнате.
                Waypoint(16.2, -1.1, 0.85),
                Waypoint(15.0, -1.15, 0.85),
                Waypoint(13.8, -1.05, 0.85),
                Waypoint(13.2, -0.95, 0.85),
                Waypoint(12.6, -0.85, 0.85),
                # 3) Готовим выход к дверному проёму, встаём в центр перешейка.
                Waypoint(11.4, -0.6, 0.8),
                Waypoint(10.2, -0.4, 0.75),
                Waypoint(9.0, -0.25, 0.7),
                Waypoint(8.4, -0.2, 0.68),
                Waypoint(7.8, -0.15, 0.65),
                # 4) Проходим коридор между комнатами мелкими шагами, чтобы контролировать траекторию.
                Waypoint(6.8, -0.08, 0.6),
                Waypoint(5.8, -0.02, 0.55),
                Waypoint(5.3, 0.01, 0.52),
                Waypoint(4.8, 0.04, 0.5),
                Waypoint(4.3, 0.08, 0.47),
                Waypoint(3.8, 0.12, 0.45),
                # 5) В левой комнате забираем чуть выше, чтобы белая зона оказалась слева от робота.
                Waypoint(2.9, 0.25, 0.4),
                Waypoint(2.0, 0.38, 0.35),
                Waypoint(1.7, 0.42, 0.33),
                Waypoint(1.4, 0.45, 0.32),
                # 6) Финальный заезд на белый прямоугольник с замедлением.
                Waypoint(1.05, 0.38, 0.25),
                Waypoint(0.9, 0.32, 0.2),
            ),
        )


def open_telemetry_socket():
    """Создаёт сокет для чтения телеметрии от контроллера Webots."""

    if PROTO.lower() == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((TEL_HOST, TEL_PORT))
        LOGGER.info("Слушаем телеметрию UDP на %s:%d", TEL_HOST, TEL_PORT)
        return sock

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TEL_HOST, TEL_PORT))
    sock.listen(1)
    LOGGER.info("Ждём TCP-подключение на %s:%d", TEL_HOST, TEL_PORT)
    conn, addr = sock.accept()
    LOGGER.info("TCP-клиент телеметрии подключён: %s", addr)
    return conn


def send_cmd(sock_cmd: socket.socket, v: float, w: float) -> None:
    """Отправляет линейную и угловую скорость контроллеру ``udp_diff``."""

    packet = struct.pack("<2f", v, w)
    sock_cmd.sendto(packet, (CMD_HOST, CMD_PORT))


def recv_all(sock: socket.socket, size: int) -> bytes | None:
    """Гарантированно дочитывает нужное число байт из TCP-сокета."""

    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_tel(sock_tel: socket.socket):
    """Читает очередной пакет телеметрии и раскладывает его на части."""

    if PROTO.lower() == "udp":
        data, _ = sock_tel.recvfrom(65535)
    else:
        size_bytes = sock_tel.recv(4)
        if not size_bytes:
            return None
        size = struct.unpack("<I", size_bytes)[0]
        data = recv_all(sock_tel, size)
        if data is None:
            return None

    if not data.startswith(b"WBTG"):
        LOGGER.warning("Неизвестный формат пакета: %r", data[:4])
        return None

    header_size = 4 + 9 * 4
    odom = struct.unpack("<9f", data[4:header_size])
    odom_x, odom_y, odom_th = odom[0], odom[1], odom[2]
    n = struct.unpack("<I", data[header_size:header_size + 4])[0]
    ranges = []
    if n:
        ranges = struct.unpack(f"<{n}f", data[header_size + 4:header_size + 4 + 4 * n])

    return odom_x, odom_y, odom_th, ranges


def main(route: RaceRoute | None = None) -> None:
    """Точка входа: запускает цикл чтения телеметрии и расчёта команд."""

    route = route or RaceRoute.default()
    navigator = WaypointNavigator(
        waypoints=route.waypoints,
        config=NavigatorConfig(
            lidar_fov_deg=LIDAR_FOV,
            forward_clearance_distance=0.4,
            blocked_turn_speed=1.5,
            scan_fixed_direction=None,  # включаем динамический выбор стороны разворота по yaw_error и лидара
        ),
        logger=logging.getLogger("navigator"),
    )

    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_tel = open_telemetry_socket()

    LOGGER.info("Маршрут содержит %d точек", len(route.waypoints))
    last_time = time.time()

    try:
        while True:
            tel = recv_tel(sock_tel)
            if tel is None:
                continue
            x, y, yaw, ranges = tel
            navigator.update_pose(x, y, yaw)
            navigator.update_scan(ranges)

            now = time.time()
            dt = max(1e-3, now - last_time)
            last_time = now

            command = navigator.step(dt)
            send_cmd(sock_cmd, command["v"], command["w"])

            LOGGER.info(
                "Цель=%s, v=%.2f м/с, w=%.2f рад/с, запас=%.2f м, ошибка=%.2f рад",
                command["target"],
                command["v"],
                command["w"],
                command["range_min"],
                command["yaw_error"],
            )

            if command["target"] is None:
                LOGGER.info("Финиш достигнут, отправляем стоп-сигнал")
                break
    except KeyboardInterrupt:
        LOGGER.info("Получен Ctrl+C — останавливаемся")
    finally:
        send_cmd(sock_cmd, 0.0, 0.0)
        sock_cmd.close()
        sock_tel.close()


if __name__ == "__main__":  # pragma: no cover - CLI обёртка
    main()
