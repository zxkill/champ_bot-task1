"""Клиентский скрипт для автономного прохода через выбранный коридор."""

import logging
import math
import os
import socket
import struct
import time

from tools import (
    compute_required_gap_width,
    corridor_fits,
    corridor_width_margin,
    find_corridors,
)
from corridor_follower import CorridorFollower

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("corridor_demo")

CMD_HOST  = str(os.getenv("CMD_HOST", "127.0.0.1"))
CMD_PORT  = int(os.getenv("CMD_PORT", "5555"))
TEL_HOST  = str(os.getenv("TEL_HOST", "0.0.0.0"))
TEL_PORT  = int(os.getenv("TEL_PORT", "5600"))
PROTO     = str(os.getenv("PROTO", "tcp"))

# === Параметры ===
SAFE_DIST = 0.5
CRASH_DIST = 0.3
RECOVER_TIME = 1               # время отката назад
TURN_TIME = 0.6                # базовое время для поворота
FORWARD_SPEED = 0.6            # м/с
WIDTH_TOLERANCE = float(os.getenv("WIDTH_TOLERANCE", "0.02"))  # м — допустимое сужение коридора
WIDTH_RECOVERY_MARGIN = float(os.getenv("WIDTH_RECOVERY_MARGIN", "0.04"))  # м — доп. поблажка при удержании
CORRIDOR_RECOVERY_HOLD_SEC = float(os.getenv("CORRIDOR_RECOVERY_HOLD_SEC", "0.6"))  # c — как долго держим коридор при просадке

# demo.py (замени строку с from controllers.udp_diff.udp_diff import WHEEL_BASE ...)
WHEEL_BASE   = float(os.getenv("WHEEL_BASE", "0.25"))
WHEEL_RADIUS = float(os.getenv("WHEEL_RADIUS", "0.035"))


# 1) Считаем требуемую ширину проёма
required_width = compute_required_gap_width(
    wheel_base=WHEEL_BASE,
    side_margin=0.04,          # подбери по месту/шуму лидара
    body_extra_each_side=0.02, # свесы корпуса за колёса, если есть (подбери)
)

# 2) Вперёд должен быть запас — возьмём минимум как 0.35 м,
#    но можно завязать на габарит по базе, чтобы не «тыкаться носом»:
min_forward_clearance = max(0.35, WHEEL_BASE * 1.2)


# === Машина состояний ===
state = "FORWARD"
state_until = 0
last_pos = None
start_time = time.time()
last_move_time = start_time
last_control_time = start_time

# Текущее «прилипание» к выбранному коридору. Храним его геометрию/план и дедлайн,
# до которого разрешаем удерживать коридор даже при временных просадках по ширине.
active_corridor = None
active_plan = None
active_corridor_id = None
active_hold_until = 0.0

sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if PROTO == "udp":
    sock_tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_tel.bind((TEL_HOST, TEL_PORT))
else:
    sock_tel = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_tel.bind((TEL_HOST, TEL_PORT))
    sock_tel.listen(1)
    print(f"[client] waiting for telemetry TCP on {TEL_HOST}:{TEL_PORT}...")
    conn, _ = sock_tel.accept()
    sock_tel = conn
    print("[client] connected to udp_diff telemetry")


def send_cmd(v: float, w: float):
    packet = struct.pack("<2f", v, w)
    sock_cmd.sendto(packet, (CMD_HOST, CMD_PORT))


def recv_all(sock, size):
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_tel():
    if PROTO == "udp":
        data, _ = sock_tel.recvfrom(65535)
    else:
        size_bytes = sock_tel.recv(4)
        if not size_bytes:
            return None
        size = struct.unpack("<I", size_bytes)[0]
        data = recv_all(sock_tel, size)

    if not data.startswith(b"WBTG"):  # ВНИМАНИЕ! Было: WBT2
        print('WBT2 -> WBTG')
        return None

    # теперь в пакете: 9 float после "WBTG" (36 байт)
    # "<6f" -> "<9f"
    header_size = 4 + 9 * 4
    odom_x, odom_y, odom_th, vx, vy, vth, wx, wy, wz = struct.unpack("<9f", data[4:header_size])
    n = struct.unpack("<I", data[header_size:header_size + 4])[0]
    ranges = []
    if n > 0:
        ranges = struct.unpack(f"<{n}f", data[header_size + 4:header_size + 4 + 4 * n])

    return odom_x, odom_y, odom_th, (vx, vy, vth), (wx, wy, wz), ranges


try:
    logger.info(
        "Запускаем цикл управления: ожидаем телеметрию от контроллера на %s:%d",
        TEL_HOST, TEL_PORT,
    )
    follower = CorridorFollower(
        wheel_base=WHEEL_BASE,
        wheel_radius=WHEEL_RADIUS,
        logger=logging.getLogger("corridor_follower"),
    )
    while True:
        tel = recv_tel()
        if not tel:
            continue
        x, y, th, vel, gyro, ranges = tel
        vx, vy, vth = vel
        wx, wy, wz = gyro
        if not ranges:
            continue

        corrs = find_corridors(ranges, fov_deg=90.0, max_lookahead=3.0,
                               min_points=3, min_depth_for_corridor=0.5)

        good = []
        near_miss = {}
        for c in corrs:
            ok, plan = corridor_fits(
                c,
                required_width=required_width,
                min_forward_clearance=min_forward_clearance,
                width_tolerance=WIDTH_TOLERANCE,
            )
            if ok:
                c["plan"] = plan
                good.append(c)
            else:
                margin = corridor_width_margin(
                    c,
                    required_width=required_width,
                    width_tolerance=WIDTH_TOLERANCE,
                )
                near_miss[(c["i0"], c["i1"])] = margin
                logger.debug(
                    "Коридор %s отклонён: ширина %.2f м при запросе %.2f м, глубина %.2f м, запас %.3f м",
                    (c["i0"], c["i1"]),
                    c.get("width_at_best", float("nan")),
                    required_width,
                    c.get("depth_min", float("nan")),
                    margin,
                )

        # Фиксируем реальное время шага, чтобы CorridorFollower мог правильно дозировать ускорения.
        current_time = time.time()
        dt = max(1e-3, current_time - last_control_time)
        last_control_time = current_time

        selected = None

        if good:
            best = max(good, key=lambda c: c["plan"]["expected_width"])
            selected = best
            active_hold_until = current_time + CORRIDOR_RECOVERY_HOLD_SEC
        else:
            # Пробуем удержать предыдущий коридор, если просадка по ширине не критичная.
            if active_corridor_id and active_corridor_id in near_miss:
                margin = near_miss[active_corridor_id]
                if margin >= -WIDTH_RECOVERY_MARGIN and current_time < active_hold_until:
                    # Дополнительно увеличиваем допуск, чтобы corridor_fits выдал план
                    # с учётом временной просадки.
                    recovery_tol = WIDTH_TOLERANCE + WIDTH_RECOVERY_MARGIN
                    for c in corrs:
                        if (c["i0"], c["i1"]) == active_corridor_id:
                            ok_rec, plan_rec = corridor_fits(
                                c,
                                required_width=required_width,
                                min_forward_clearance=min_forward_clearance,
                                width_tolerance=recovery_tol,
                            )
                            if ok_rec:
                                c["plan"] = plan_rec
                                selected = c
                                active_hold_until = current_time + CORRIDOR_RECOVERY_HOLD_SEC
                                logger.info(
                                    "Удерживаем коридор %s при временном сужении: запас %.3f м, допуск %.3f м",
                                    active_corridor_id,
                                    margin,
                                    recovery_tol,
                                )
                            else:
                                logger.debug(
                                    "Не удалось удержать коридор %s даже с расширенным допуском %.3f м",
                                    active_corridor_id,
                                    recovery_tol,
                                )
                            break
                else:
                    if margin < -WIDTH_RECOVERY_MARGIN:
                        logger.debug(
                            "Коридор %s слишком сузился: запас %.3f м < допустимого %.3f м",
                            active_corridor_id,
                            margin,
                            -WIDTH_RECOVERY_MARGIN,
                        )
                    if current_time >= active_hold_until:
                        logger.debug(
                            "Время удержания коридора %s истекло (deadline %.2f, сейчас %.2f)",
                            active_corridor_id,
                            active_hold_until,
                            current_time,
                        )

        if selected:
            active_corridor = selected
            active_plan = selected["plan"]
            active_corridor_id = (selected["i0"], selected["i1"])

            ctrl = follower.step(
                dt,
                ranges,
                active_corridor,
                active_plan,
                active_plan["requested_width"],
            )

            send_cmd(ctrl["v"], ctrl["w"])
            logger.info(
                "Коридор %s, ширина %.2f м, глубина %.2f м -> команды: v=%.2f м/с, w=%.2f рад/с",
                active_corridor_id,
                active_plan["expected_width"],
                active_plan["target_depth"],
                ctrl["v"],
                ctrl["w"],
            )
        else:
            # Без безопасного коридора лучше остановиться и пересчитать план.
            send_cmd(0.0, 0.0)
            logger.warning("Подходящих коридоров не найдено — стоим на месте")
            active_corridor = None
            active_plan = None
            active_corridor_id = None

        now = current_time

        # проверка движения (анти-застревание по одометрии)
        # if last_pos is not None:
        #     dx = x - last_pos[0]
        #     dy = y - last_pos[1]
        #     dist_moved = (dx**2 + dy**2)**0.5
        #     if dist_moved > 0.02:
        #         last_move_time = now
        # last_pos = (x, y)
        #
        # stuck = (now - last_move_time > 2.0)
        #
        # n = len(ranges)
        # mid_idx = n // 2
        # right_idx = int(n * 3 / 4)
        # left_idx = int(n * 1 / 4)
        #
        # front_dist = ranges[mid_idx]
        # right_dist = ranges[right_idx]
        # left_dist = ranges[left_idx]
        # print(f"state {state} dist {front_dist} dist {right_dist} dist {left_dist}")
        # # === Управление по состояниям ===
        # if now < state_until:
        #     action = state
        # else:
        #     # аварийная проверка столкновения
        #     if front_dist < CRASH_DIST or right_dist < CRASH_DIST or left_dist < CRASH_DIST:
        #         state = "RECOVER"
        #         state_until = now + RECOVER_TIME
        #         turn_dir = 1.0 if right_dist > left_dist else -1.0
        #         send_cmd(-0.15, 1.0 * turn_dir)  # назад + поворот
        #         action = f"RECOVER(dir={turn_dir:+})"
        #         time.sleep(1)
        #
        #     elif right_dist > SAFE_DIST:
        #         state = "TURN_RIGHT"
        #         state_until = now + TURN_TIME
        #         send_cmd(0.15, -1.0)
        #         action = "TURN_RIGHT"
        #
        #     elif front_dist > SAFE_DIST:
        #         # скорость пропорциональна расстоянию до препятствия
        #         v = min(FORWARD_SPEED, 0.1 + 0.5 * front_dist)
        #         state = "FORWARD"
        #         state_until = now + 0.3
        #         send_cmd(v, 0.0)
        #         action = f"FORWARD(v={v:.2f})"
        #
        #     else:
        #         state = "TURN_LEFT"
        #         state_until = now + TURN_TIME
        #         send_cmd(0.15, 1.0)
        #         action = "TURN_LEFT"


except KeyboardInterrupt:
    logger.info("Получен Ctrl+C — завершаем работу")
    send_cmd(0.0, 0.0)
finally:
    # Гарантируем, что после выхода из скрипта робот не продолжит движение.
    send_cmd(0.0, 0.0)
