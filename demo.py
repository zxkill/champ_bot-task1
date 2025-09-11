import math
import socket
import struct
import time

CMD_HOST = "127.0.0.1"
CMD_PORT = 5555
TEL_HOST = "127.0.0.1"
TEL_PORT = 5600
PROTO = "tcp"

# === Параметры ===
SAFE_DIST = 0.5
CRASH_DIST = 0.3
RECOVER_TIME = 1               # время отката назад
TURN_TIME = 0.6                # базовое время для поворота
FORWARD_SPEED = 0.6            # м/с

# === Машина состояний ===
state = "FORWARD"
state_until = 0
last_pos = None
start_time = time.time()
last_move_time = start_time

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


def recv_tel():
    if PROTO == "udp":
        data, _ = sock_tel.recvfrom(65535)
    else:
        size_bytes = sock_tel.recv(4)
        if not size_bytes:
            return None
        size = struct.unpack("<I", size_bytes)[0]
        data = sock_tel.recv(size)

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
    while True:
        tel = recv_tel()
        if not tel:
            continue
        x, y, th, vel, gyro, ranges = tel
        vx, vy, vth = vel
        wx, wy, wz = gyro
        if not ranges:
            continue

        now = time.time()

        # проверка движения (анти-застревание по одометрии)
        if last_pos is not None:
            dx = x - last_pos[0]
            dy = y - last_pos[1]
            dist_moved = (dx**2 + dy**2)**0.5
            if dist_moved > 0.02:
                last_move_time = now
        last_pos = (x, y)

        stuck = (now - last_move_time > 2.0)

        n = len(ranges)
        mid_idx = n // 2
        right_idx = int(n * 3 / 4)
        left_idx = int(n * 1 / 4)

        front_dist = ranges[mid_idx]
        right_dist = ranges[right_idx]
        left_dist = ranges[left_idx]

        # === Управление по состояниям ===
        if now < state_until:
            action = state
        else:
            # аварийная проверка столкновения
            if front_dist < CRASH_DIST or right_dist < CRASH_DIST or left_dist < CRASH_DIST:
                state = "RECOVER"
                state_until = now + RECOVER_TIME
                turn_dir = 1.0 if right_dist > left_dist else -1.0
                send_cmd(-0.15, 1.0 * turn_dir)  # назад + поворот
                action = f"RECOVER(dir={turn_dir:+})"
                time.sleep(1)

            elif right_dist > SAFE_DIST:
                state = "TURN_RIGHT"
                state_until = now + TURN_TIME
                send_cmd(0.15, -1.0)
                action = "TURN_RIGHT"

            elif front_dist > SAFE_DIST:
                # скорость пропорциональна расстоянию до препятствия
                v = min(FORWARD_SPEED, 0.1 + 0.5 * front_dist)
                state = "FORWARD"
                state_until = now + 0.3
                send_cmd(v, 0.0)
                action = f"FORWARD(v={v:.2f})"

            else:
                state = "TURN_LEFT"
                state_until = now + TURN_TIME
                send_cmd(0.15, 1.0)
                action = "TURN_LEFT"

        print(f"[client] state={state} pos=({x:.2f},{y:.2f},θ={math.degrees(th):.1f}°) "
              f"vel=({vx:.2f},{vy:.2f},{vth:.2f}) gyro=({wx:.2f},{wy:.2f},{wz:.2f}) "
              f"front={front_dist:.2f} right={right_dist:.2f} left={left_dist:.2f} → {action}")

        # time.sleep(0.1)

except KeyboardInterrupt:
    print("[client] stop")
    send_cmd(0.0, 0.0)
