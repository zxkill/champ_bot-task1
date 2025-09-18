"""Навигация по заранее известным опорным точкам с учётом препятствий.

Модуль содержит простой, но надёжный алгоритм для гонки в замкнутом
пространстве: робот следует по списку контрольных точек, при этом каждую
итерацию отслеживает расстояние до препятствий по данным лидара и
корректирует угловую скорость. Благодаря этому можно заранее разметить
маршрут (например, выход из правой комнаты, прохождение дверного проёма и
заезд на белый прямоугольник) и гарантированно удерживать высокую скорость
без касаний стен.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


@dataclass
class Waypoint:
    """Структура для хранения одной целевой точки."""

    x: float
    y: float
    speed: float


@dataclass
class NavigatorConfig:
    """Настройки навигатора, сгруппированные в dataclass для удобства."""

    angle_gain: float = 3.0  # усиление поворотного регулятора
    max_angular_speed: float = 2.4  # ограничение |ω|
    max_linear_speed: float = 0.9  # верхний предел скорости
    distance_threshold: float = 0.18  # расстояние до точки для «зачёта» прохождения
    angle_threshold: float = math.radians(8.0)  # остаточная ошибка ориентации
    in_place_turn_threshold: float = math.radians(35.0)  # когда только поворачиваемся
    avoidance_distance: float = 0.8  # зона, в которой препятствия начинают «толкать» робота
    avoidance_gain: float = 1.6  # коэффициент влияния избегания
    avoidance_max_correction: float = 1.1  # максимально допустимое добавление к ω из-за избегания
    turn_priority_threshold: float = 0.5  # |ω| от регулятора, выше которого избегание не может сменить знак
    slowdown_distance: float = 0.55  # линейная скорость начинает снижаться раньше столкновения
    min_speed: float = 0.05  # нижний предел положительной скорости при движении вперёд
    hard_stop_distance: float = 0.25  # абсолютный порог остановки при опасном сближении
    forward_clearance_distance: float = 0.35  # минимальный зазор перед роботом для разрешения движения вперёд
    clearance_release_margin: float = 0.08  # дополнительный запас для выхода из режима блокировки (гистерезис)
    blocked_turn_speed: float = 1.2  # минимальная |ω|, когда стоим из-за недостаточного зазора
    blocked_direction_bias_threshold: float = 0.07  # насколько должен отличаться свободный простор слева/справа
    blocked_yaw_bias_threshold: float = math.radians(12.0)  # ошибка курса, при которой ей доверяем больше лидара
    blocked_release_rotation: float = math.radians(210.0)  # сколько нужно провернуть корпус, прежде чем снова пытаться ехать вперёд
    lidar_fov_deg: float = 45.0  # сектор обзора используемого лидара
    log_level: int = logging.INFO  # отдельный уровень логов навигатора


@dataclass
class NavigatorState:
    """Вспомогательный объект с накопленной информацией о движении."""

    index: int = 0
    finished: bool = False
    last_range_min: float = math.inf
    last_command: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    blocked_steps: int = 0  # количество последовательных шагов с заблокированным движением вперёд
    blocked_turn_direction: Optional[float] = None  # выбранное направление разворота при блокировке
    prev_yaw: Optional[float] = None  # предыдущий измеренный курс, чтобы разворачивать угол без скачков
    yaw_unwrapped: float = 0.0  # накопленный курс без обрезки в диапазоне [-pi, pi]
    blocked_yaw_last: Optional[float] = None  # курс на предыдущем шаге блокировки для подсчёта интегрального поворота
    blocked_rotation_accum: float = 0.0  # суммарный абсолютный поворот корпуса за время блокировки


class WaypointNavigator:
    """Высокоуровневый модуль управления для соревнования.

    Логика работы на каждом шаге:
    1. Определить активную точку маршрута.
    2. Рассчитать направление на неё и ошибку по курсу.
    3. Если ошибка большая — поворачиваемся на месте, иначе движемся вперёд.
    4. Вносим поправку по препятствиям, используя вектор "отталкивания" от
       ближайших лучей лидара.
    5. Сохраняем диагностические величины и отдаём (v, w).

    Для удобства отладки всё сопровождается подробными русскоязычными логами.
    """

    def __init__(
        self,
        waypoints: Sequence[Waypoint],
        config: NavigatorConfig | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or NavigatorConfig()
        self._waypoints: List[Waypoint] = list(waypoints)
        if not self._waypoints:
            raise ValueError("Список опорных точек не может быть пустым")

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.config.log_level)

        self.pose = (0.0, 0.0, 0.0)
        self._ranges: List[float] = []
        self.state = NavigatorState()

    # ------------------------------------------------------------------
    # Служебные методы
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Приводим угол к диапазону [-pi, pi] для корректных расчётов."""

        wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
        return wrapped

    def _update_unwrapped_yaw(self, yaw: float) -> None:
        """Обновляет накопленный курс без обрезки, чтобы отслеживать полный поворот."""

        if self.state.prev_yaw is None:
            # Первый вызов: просто запоминаем угол и используем его как базу.
            self.state.prev_yaw = yaw
            self.state.yaw_unwrapped = yaw
            self.logger.debug("Инициализация непрерывного угла: %.3f рад", yaw)
            return

        delta = yaw - self.state.prev_yaw
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi

        self.state.yaw_unwrapped += delta
        self.state.prev_yaw = yaw
        self.logger.debug(
            "Обновили непрерывный угол: прибавка %.3f рад, итог %.3f рад",
            delta,
            self.state.yaw_unwrapped,
        )

    def _update_blocked_rotation_tracker(self, entering: bool) -> None:
        """Ведёт учёт накопленного поворота во время фронтальной блокировки."""

        if entering:
            # Сбрасываем счётчики при первом попадании в режим блокировки.
            self.state.blocked_rotation_accum = 0.0
            self.state.blocked_yaw_last = self.state.yaw_unwrapped
            self.logger.debug(
                "Вошли в режим блокировки: стартовый курс %.3f рад",
                self.state.yaw_unwrapped,
            )
            return

        if self.state.blocked_yaw_last is None:
            # Попадаем сюда только в случае восстановления после сброса состояния.
            self.state.blocked_yaw_last = self.state.yaw_unwrapped
            return

        delta = self.state.yaw_unwrapped - self.state.blocked_yaw_last
        self.state.blocked_rotation_accum += abs(delta)
        self.state.blocked_yaw_last = self.state.yaw_unwrapped
        self.logger.debug(
            "Блокировка: добавили %.3f рад, суммарно %.3f рад",
            delta,
            self.state.blocked_rotation_accum,
        )

    def _current_waypoint(self) -> Optional[Waypoint]:
        """Возвращает текущую целевую точку или ``None``, если маршрут пройден."""

        if self.state.finished or self.state.index >= len(self._waypoints):
            return None
        return self._waypoints[self.state.index]

    def _advance_waypoint(self) -> None:
        """Переключаемся на следующую точку и пишем в лог подробности."""

        self.state.index += 1
        if self.state.index >= len(self._waypoints):
            self.state.finished = True
            self.logger.info("Маршрут полностью завершён — робот на целевом прямоугольнике")
        else:
            next_wp = self._waypoints[self.state.index]
            self.logger.info(
                "Переходим к точке %d: (%.2f, %.2f) со скоростью %.2f м/с",
                self.state.index,
                next_wp.x,
                next_wp.y,
                next_wp.speed,
            )

    # ------------------------------------------------------------------
    # Публичные методы для обновления состояния
    # ------------------------------------------------------------------
    def set_waypoints(self, waypoints: Iterable[Waypoint]) -> None:
        """Полностью заменяет маршрут на новый список точек."""

        self._waypoints = list(waypoints)
        if not self._waypoints:
            raise ValueError("Новый маршрут не содержит ни одной точки")
        self.state = NavigatorState()
        self.logger.info("Загружен новый маршрут из %d точек", len(self._waypoints))

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        """Сохраняем текущую позицию и ориентацию, полученные по одометрии."""

        self.pose = (x, y, yaw)
        # Параллельно ведём непрерывный курс, чтобы понимать, на сколько градусов
        # робот уже провернулся относительно начала блокировки.
        self._update_unwrapped_yaw(yaw)

    def update_scan(self, ranges: Sequence[float]) -> None:
        """Принимаем свежий набор расстояний лидара."""

        self._ranges = [float(r) for r in ranges if r == r and r > 0.0]
        if not self._ranges:
            # Если массив пуст, фиксируем это в логах: навигатор перейдёт в
            # максимально консервативный режим (только разворот).
            self.logger.warning("Лидар не предоставил валидных данных — держим тормоза")

    # ------------------------------------------------------------------
    # Основной шаг
    # ------------------------------------------------------------------
    def step(self, dt: float) -> dict:
        """Выполняет расчёт команд на текущей итерации управления."""

        if dt <= 0.0:
            raise ValueError("Временной шаг должен быть положительным")

        waypoint = self._current_waypoint()
        if waypoint is None:
            self.state.last_command = (0.0, 0.0)
            return {"v": 0.0, "w": 0.0, "target": None, "range_min": self.state.last_range_min}

        x, y, yaw = self.pose
        dx = waypoint.x - x
        dy = waypoint.y - y
        distance = math.hypot(dx, dy)

        if distance <= self.config.distance_threshold:
            self.logger.info(
                "Точка %d достигнута: расстояние %.3f м <= порога %.3f м",
                self.state.index,
                distance,
                self.config.distance_threshold,
            )
            self._advance_waypoint()
            waypoint = self._current_waypoint()
            if waypoint is None:
                self.state.last_command = (0.0, 0.0)
                return {"v": 0.0, "w": 0.0, "target": None, "range_min": self.state.last_range_min}
            dx = waypoint.x - x
            dy = waypoint.y - y
            distance = math.hypot(dx, dy)

        desired_yaw = math.atan2(dy, dx)
        yaw_error = self._wrap_angle(desired_yaw - yaw)

        # Базовая угловая скорость — пропорциональный регулятор по углу.
        w_base = self.config.angle_gain * yaw_error
        w_base = max(-self.config.max_angular_speed, min(self.config.max_angular_speed, w_base))
        # Текущая рабочая угловая скорость, которая будет корректироваться избежанием препятствий.
        w_cmd = w_base

        # Линейную скорость включаем только если робот почти смотрит на цель.
        if abs(yaw_error) > self.config.in_place_turn_threshold or not self._ranges:
            v_cmd = 0.0
        else:
            # Чем точнее наведён курс, тем быстрее едем.
            heading_scale = max(0.0, math.cos(yaw_error))
            base_speed = min(waypoint.speed, self.config.max_linear_speed)
            v_cmd = base_speed * heading_scale

        # Обработка данных лидара: минимальное расстояние и «отталкивание» от препятствий.
        range_min = math.inf
        avoidance = 0.0
        avoidance_raw = 0.0
        forward_blocked = False  # флаг: нельзя ехать вперёд из-за тесного прохода
        was_blocked = self.state.blocked_steps > 0  # находились ли мы в режиме блокировки на предыдущем шаге
        if was_blocked:
            # Каждая итерация блокировки обновляет интегральный угол разворота.
            self._update_blocked_rotation_tracker(entering=False)
        left_min = math.inf  # минимальное расстояние в левой половине сектора
        right_min = math.inf  # минимальное расстояние в правой половине сектора
        if self._ranges:
            n = len(self._ranges)
            if n == 1:
                angles = [0.0]
            else:
                start = -self.config.lidar_fov_deg / 2.0
                step = self.config.lidar_fov_deg / (n - 1)
                angles = [math.radians(start + i * step) for i in range(n)]

            for ang, r in zip(angles, self._ranges):
                range_min = min(range_min, r)
                if r < self.config.avoidance_distance:
                    weight = (self.config.avoidance_distance - r) / self.config.avoidance_distance
                    if ang > 0.0:
                        side = -1.0
                    elif ang < 0.0:
                        side = 1.0
                    else:
                        side = 0.0  # центральный луч не даёт подсказку по направлению
                    avoidance_raw += side * weight
                if ang > 0.0:
                    left_min = min(left_min, r)
                elif ang < 0.0:
                    right_min = min(right_min, r)
                else:
                    # центральный луч обновляет оба значения, чтобы отразить фронтальную опасность
                    left_min = min(left_min, r)
                    right_min = min(right_min, r)

            if range_min < self.config.hard_stop_distance:
                self.logger.warning(
                    "Препятствие слишком близко (%.2f м) — экстренно тормозим",
                    range_min,
                )
                v_cmd = 0.0
                forward_blocked = True
            elif range_min < self.config.forward_clearance_distance:
                # Перед роботом недостаточно места, безопаснее остаться на месте и повернуться.
                if v_cmd > 0.0:
                    v_cmd = 0.0
                forward_blocked = True
                if not was_blocked:
                    # Логируем только момент входа в режим блокировки, чтобы не спамить сообщениями.
                    self.logger.info(
                        "Недостаточный зазор спереди: %.2f м < %.2f м — выполняем разворот на месте",
                        range_min,
                        self.config.forward_clearance_distance,
                    )
                else:
                    # В режиме гистерезиса оставляем подробный отладочный след.
                    self.logger.debug(
                        "Продолжаем разворот: фронтальный зазор %.2f м по-прежнему ниже порога %.2f м",
                        range_min,
                        self.config.forward_clearance_distance,
                    )
            elif (
                was_blocked
                and range_min
                < self.config.forward_clearance_distance + self.config.clearance_release_margin
            ):
                # Добавляем гистерезис: даже если зазор слегка вырос, продолжаем разворот,
                # пока не появится устойчивый запас пространства.
                if v_cmd > 0.0:
                    v_cmd = 0.0
                forward_blocked = True
                self.logger.debug(
                    "Удерживаем блокировку: зазор %.2f м меньше порога освобождения %.2f м",
                    range_min,
                    self.config.forward_clearance_distance + self.config.clearance_release_margin,
                )
            elif range_min < self.config.slowdown_distance and v_cmd > 0.0:
                slowdown = max(0.0, min(1.0, range_min / self.config.slowdown_distance))
                v_cmd = max(self.config.min_speed, v_cmd * slowdown)

            if not forward_blocked and was_blocked:
                required_rotation = self.config.blocked_release_rotation
                if self.state.blocked_rotation_accum < required_rotation:
                    forward_blocked = True
                    self.logger.debug(
                        "Продолжаем разворот после освобождения: накоплено %.2f рад < %.2f рад",
                        self.state.blocked_rotation_accum,
                        required_rotation,
                    )

            avoidance = self.config.avoidance_gain * avoidance_raw
            avoidance = max(
                -self.config.avoidance_max_correction,
                min(self.config.avoidance_max_correction, avoidance),
            )
            tentative_w = w_cmd + avoidance
            # Если регулятор ориентации уверенно требует поворота (|ω| выше порога),
            # избегание может лишь ослабить его, но не менять направление.
            if (
                abs(w_base) >= self.config.turn_priority_threshold
                and w_base * tentative_w < 0.0
            ):
                w_cmd = w_base
                self.logger.debug(
                    "Избегание (%.3f рад/с) не меняет знак поворота, сохраняем w=%.3f",
                    avoidance,
                    w_cmd,
                )
            else:
                w_cmd = tentative_w
            w_cmd = max(-self.config.max_angular_speed, min(self.config.max_angular_speed, w_cmd))
        else:
            range_min = math.inf

        if forward_blocked:
            # При блокировке выбираем сторону разворота, опираясь на данные лидара и историю.
            if not was_blocked:
                self._update_blocked_rotation_tracker(entering=True)
            self.state.blocked_steps += 1
            w_cmd = self._resolve_blocked_turn_direction(w_cmd, yaw_error, left_min, right_min)
            # Если вперёд идти нельзя, ускоряем поворот, чтобы быстрее освободить себе путь.
            w_cmd = self._apply_blocked_turn_boost(w_cmd, yaw_error)
        else:
            if was_blocked:
                release_limit = (
                    self.config.forward_clearance_distance
                    + self.config.clearance_release_margin
                )
                if math.isfinite(range_min):
                    self.logger.info(
                        "Фронтальный проход очищен: минимальный зазор %.2f м превышает %.2f м, поворот %.2f рад (порог %.2f рад)",
                        range_min,
                        release_limit,
                        self.state.blocked_rotation_accum,
                        self.config.blocked_release_rotation,
                    )
                else:
                    self.logger.info(
                        "Снимаем блокировку без данных лидара: поворот %.2f рад (порог %.2f рад)",
                        self.state.blocked_rotation_accum,
                        self.config.blocked_release_rotation,
                    )
            # Как только проход устойчиво освобождён, забываем выбранное направление и счётчик.
            self.state.blocked_steps = 0
            self.state.blocked_turn_direction = None
            self.state.blocked_yaw_last = None
            self.state.blocked_rotation_accum = 0.0

        self.state.last_range_min = range_min
        self.state.last_command = (v_cmd, w_cmd)

        self.logger.debug(
            "Цель #%d: (%.2f, %.2f), дистанция %.2f м, ошибка курса %.2f рад -> v=%.3f м/с, w=%.3f рад/с, r_min=%.2f",
            self.state.index,
            waypoint.x,
            waypoint.y,
            distance,
            yaw_error,
            v_cmd,
            w_cmd,
            range_min,
        )

        return {
            "v": v_cmd,
            "w": w_cmd,
            "target": (waypoint.x, waypoint.y),
            "range_min": range_min,
            "yaw_error": yaw_error,
        }

    def _apply_blocked_turn_boost(self, w_cmd: float, yaw_error: float) -> float:
        """Увеличивает скорость разворота, когда фронтальный проход перекрыт.

        Метод вынесен отдельно, чтобы его можно было точечно протестировать.
        Если текущая угловая скорость мала, навигатор подбирает направление
        разворота: сначала пытается следовать знаку ошибки курса, а когда
        она равна нулю, выбирает левый поворот как безопасный вариант.
        После расчёта выполняется жёсткое ограничение по `max_angular_speed`.
        """

        turn_speed = self.config.blocked_turn_speed
        if abs(w_cmd) < turn_speed:
            if w_cmd == 0.0:
                # Отдаём предпочтение знаку ошибки курса, иначе выбираем разворот влево.
                if yaw_error != 0.0:
                    direction_seed = yaw_error
                else:
                    direction_seed = 1.0
            else:
                direction_seed = w_cmd
            direction = math.copysign(1.0, direction_seed)
            boosted = direction * turn_speed
            self.logger.debug(
                "Ускоряем разворот до %.2f рад/с из-за блокировки по фронту",
                boosted,
            )
            w_cmd = boosted

        return max(-self.config.max_angular_speed, min(self.config.max_angular_speed, w_cmd))

    def _resolve_blocked_turn_direction(
        self,
        w_cmd: float,
        yaw_error: float,
        left_min: float,
        right_min: float,
    ) -> float:
        """Выбирает приоритетное направление разворота, когда движение вперёд запрещено.

        Алгоритм действует по ступеням, чтобы минимизировать время простоя:
        1. Если ошибка курса заметная, доверяем ей — значит, целевая точка
           явно сбоку.
        2. Если ошибка мала, но лидар показывает, что слева или справа
           заметно больше пространства, используем это.
        3. Если и это не помогает, берём знак текущей угловой скорости,
           чтобы не менять решение в каждой итерации.
        4. В последнюю очередь чередуем направления при симметрии, чтобы
           «прощупать» оба варианта и не застревать у стены.
        """

        # Подготовим безопасные значения для анализа: если лучей нет, считаем что пространство свободно.
        left_space = left_min if math.isfinite(left_min) else self.config.avoidance_distance
        right_space = right_min if math.isfinite(right_min) else self.config.avoidance_distance

        if self.state.blocked_turn_direction is not None:
            direction = math.copysign(1.0, self.state.blocked_turn_direction)
            decision_reason = "sticky"
        else:
            direction_seed = 0.0
            decision_reason = ""

            if abs(yaw_error) >= self.config.blocked_yaw_bias_threshold:
                direction_seed = yaw_error
                decision_reason = "yaw_error"
            else:
                # Положительная разница означает, что слева свободнее, отрицательная — справа.
                space_delta = left_space - right_space
                if abs(space_delta) >= self.config.blocked_direction_bias_threshold:
                    direction_seed = space_delta
                    decision_reason = "lidar_delta"
                elif w_cmd != 0.0:
                    direction_seed = w_cmd
                    decision_reason = "existing_w"
                else:
                    # Чередуем направления: нечётные шаги — влево, чётные — вправо.
                    direction_seed = 1.0 if (self.state.blocked_steps % 2 == 1) else -1.0
                    decision_reason = "alternating"

            direction = math.copysign(1.0, direction_seed if direction_seed != 0.0 else 1.0)
            self.state.blocked_turn_direction = direction
        # Небольшая ненулевая величина нужна, чтобы `_apply_blocked_turn_boost`
        # сохранил выбранный знак и не вернул дефолтное вращение влево.
        hint_magnitude = max(abs(w_cmd), 1e-3)
        resolved = direction * hint_magnitude

        self.logger.debug(
            "Блокировка спереди: причина выбора=%s, yaw_error=%.3f рад, left=%.2f м, right=%.2f м, шаг=%d -> w=%.3f рад/с",
            decision_reason,
            yaw_error,
            left_space,
            right_space,
            self.state.blocked_steps,
            resolved,
        )

        return resolved
