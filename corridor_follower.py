"""Модуль с логикой безопасного следования по найденному коридору."""

import logging
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class FollowCfg:
    v_max: float = 0.35  # м/с — верхняя скорость в коридоре
    w_max: float = 1.5  # рад/с — верхняя угловая скорость
    a_lin: float = 0.6  # м/с² — ограничение на рост/падение v
    a_ang: float = 4.0  # рад/с² — ограничение на рост/падение w
    k_yaw: float = 2.0  # рад/с на рад — коэффициент поворота на центр коридора
    hold_sec: float = 0.4  # «прилипание» к выбранному коридору, чтобы не дёргаться между двумя


class CorridorFollower:
    """Простая модель ПИД-подобного следования за центром коридора.

    Класс аккуратно ускоряет робота, регулирует угловую скорость и
    автоматически притормаживает при опасном сближении спереди.
    Все основные шаги снабжены подробным логированием для удобной отладки.
    """

    def __init__(self, wheel_base: float, wheel_radius: float, cfg: FollowCfg = FollowCfg(),
                 logger: Optional[logging.Logger] = None):
        # Габариты робота и ограничения движения задаются при создании экземпляра.
        self.cfg = cfg
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.v = 0.0
        self.w = 0.0
        self._hold_t = 0.0
        self._last_segment = None
        # Используем переданный логгер или создаём свой с именем класса.
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def step(self, dt: float, ranges, corridor: dict, plan: dict, required_width: float):
        """Выполняет один шаг управления для прохода в выбранный коридор.

        :param dt: длительность шага дискретизации в секундах (должна быть > 0).
        :param ranges: массив дистанций лидара.
        :param corridor: описание выбранного сегмента из ``find_corridors``.
        :param plan: результат ``corridor_fits`` с целевой глубиной, углом и полями
            ``requested_width``/``width_margin``.
        :param required_width: минимальная требуемая ширина проёма (исходный запрос).
        :return: словарь с командами ``v``, ``w`` и угловыми скоростями колёс.
        """

        if dt <= 0.0:
            raise ValueError("dt должно быть положительным, иначе управление невозможно")

        # 1) оценим свободное расстояние по центру (минимум в небольшом окне)
        n = len(ranges)
        c = n // 2
        window = 4
        front_clear = min(ranges[max(0, c - window): min(n, c + window + 1)] or [plan["target_depth"]])
        front_clear = front_clear if front_clear == front_clear and front_clear > 0 else plan["target_depth"]
        self.logger.debug(
            "Расстояние по центру: %.3f м (цель %.3f м, требуется ширина %.3f м)",
            front_clear, plan["target_depth"], required_width,
        )

        # 2) прилипание к сегменту, чтобы не «прыгало»
        seg_id = (corridor["i0"], corridor["i1"])
        if self._last_segment != seg_id:
            if self._hold_t <= 0.0:
                self._last_segment = seg_id
                self._hold_t = self.cfg.hold_sec
                self.logger.debug("Зафиксирован новый коридор %s на %.2f c", seg_id, self._hold_t)
        else:
            self._hold_t = max(0.0, self._hold_t - dt)
            self.logger.debug("Удерживаем сегмент %s ещё %.2f c", seg_id, self._hold_t)

        # 3) P-регулятор по углу на центр коридора
        yaw_err = math.radians(corridor["ang_center"])  # (+) поворот на право, если центр справа
        w_cmd = self.cfg.k_yaw * yaw_err
        w_cmd = self._clamp(w_cmd, -self.cfg.w_max, self.cfg.w_max)
        self.logger.debug(
            "Целевой поворот: %.3f рад -> ω командное %.3f рад/с (ограничено %.3f)",
            yaw_err, w_cmd, self.cfg.w_max,
        )

        # 4) целевая линейная скорость: тише при повороте и при маленьком зазоре впереди
        #    масштабируем по свободе впереди (например, 0.8 м — уже без ограничений)
        clear_scale = self._clamp(front_clear / 0.8, 0.0, 1.0)
        turn_scale = 1.0 - min(1.0, abs(w_cmd) / self.cfg.w_max)
        v_target = self.cfg.v_max * clear_scale * (0.6 + 0.4 * turn_scale)  # минимум 0.6 при повороте
        self.logger.debug(
            "Масштабы скорости: по расстоянию %.2f, по повороту %.2f -> цель v %.3f м/с",
            clear_scale, turn_scale, v_target,
        )

        # 5) плавные рампы ускорений
        dv = self._clamp(v_target - self.v, -self.cfg.a_lin * dt, self.cfg.a_lin * dt)
        dw = self._clamp(w_cmd - self.w, -self.cfg.a_ang * dt, self.cfg.a_ang * dt)
        self.v += dv
        self.w += dw
        self.logger.debug("Применённые изменения: Δv=%.3f м/с, Δw=%.3f рад/с", dv, dw)

        # 6) страховки: экстренная остановка если совсем близко
        if front_clear < 0.25:
            self.v = max(0.0, self.v - 2.0 * self.cfg.a_lin * dt)
            self.logger.warning(
                "Опасное сближение: %.3f м спереди. Дополнительно тормозим до %.3f м/с",
                front_clear, self.v,
            )

        # 7) перевод в угловые скорости колёс (рад/с)
        # v — линейная, w — угловая (yaw). Для дифф-робота:
        # ω_L = (v - w*B/2) / R;  ω_R = (v + w*B/2) / R
        wL = (self.v - self.w * (self.wheel_base / 2.0)) / self.wheel_radius
        wR = (self.v + self.w * (self.wheel_base / 2.0)) / self.wheel_radius
        self.logger.debug(
            "Команда: v=%.3f м/с, w=%.3f рад/с, ωL=%.3f рад/с, ωR=%.3f рад/с",
            self.v, self.w, wL, wR,
        )

        return {"v": self.v, "w": self.w, "w_left": wL, "w_right": wR,
                "front_clear": front_clear, "required_width": required_width}
