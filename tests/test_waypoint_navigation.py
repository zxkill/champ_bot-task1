"""Проверки для модуля навигации по опорным точкам."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from navigator import NavigatorConfig, Waypoint, WaypointNavigator  # noqa: E402


@pytest.fixture()
def navigator() -> WaypointNavigator:
    """Создаём навигатор с простым маршрутом из двух точек."""

    cfg = NavigatorConfig(
        angle_gain=2.5,
        max_angular_speed=1.5,
        max_linear_speed=0.8,
        lidar_fov_deg=45.0,
        avoidance_distance=0.6,
        avoidance_gain=1.2,
        slowdown_distance=0.5,
        hard_stop_distance=0.22,
        forward_clearance_distance=0.4,
        blocked_turn_speed=0.9,
        in_place_turn_threshold=math.radians(25.0),
    )
    return WaypointNavigator(
        waypoints=[Waypoint(1.0, 0.0, 0.6), Waypoint(2.0, 0.0, 0.5)],
        config=cfg,
    )


def test_constructor_requires_non_empty_route():
    """Создание навигатора без точек должно вызывать ошибку."""

    with pytest.raises(ValueError):
        WaypointNavigator([], NavigatorConfig())


def test_set_waypoints_validates_input(navigator):
    """Пустой список точек при перенастройке также недопустим."""

    with pytest.raises(ValueError):
        navigator.set_waypoints([])

    # После корректной подстановки маршрут должен обновиться и сбросить индекс.
    navigator.set_waypoints([Waypoint(0.5, 0.0, 0.3)])
    assert navigator.state.index == 0


def test_rotates_in_place_when_target_behind(navigator):
    """Если цель находится позади, линейная скорость должна быть нулевой."""

    navigator.update_pose(0.0, 0.0, math.pi)
    navigator.update_scan([1.0] * 5)
    command = navigator.step(0.1)
    assert command["v"] == pytest.approx(0.0, abs=1e-6)
    assert command["w"] < 0.0  # нужно повернуться вправо, чтобы увидеть цель


def test_switches_waypoint_after_reaching_threshold(navigator):
    """При малом расстоянии до точки навигатор должен перейти к следующей."""

    navigator.update_pose(0.95, 0.0, 0.0)
    navigator.update_scan([1.0] * 5)
    first_command = navigator.step(0.1)
    assert navigator.state.index == 1
    assert first_command["target"] == pytest.approx((2.0, 0.0))


def test_obstacle_forces_slowdown(navigator):
    """Близкое препятствие должно значительно снижать скорость и менять курс."""

    navigator.update_pose(0.0, 0.0, 0.0)
    # Спереди препятствие на 0.2 м — гораздо ближе порога замедления.
    ranges = [0.8, 0.6, 0.2, 0.6, 0.8]
    navigator.update_scan(ranges)
    command = navigator.step(0.1)
    assert command["v"] == pytest.approx(0.0, abs=1e-6)
    assert abs(command["w"]) > 0.0


def test_forward_motion_blocked_when_clearance_small(navigator, caplog):
    """При малом зазоре спереди движение должно остановиться и включиться активный разворот."""

    navigator.update_pose(0.0, 0.0, 0.0)
    # Симметричное препятствие прямо перед роботом: избегание не даёт подсказки по направлению.
    navigator.update_scan([0.5, 0.42, 0.33, 0.42, 0.5])
    with caplog.at_level("INFO"):
        command = navigator.step(0.1)

    assert command["v"] == pytest.approx(0.0, abs=1e-6)
    assert abs(command["w"]) >= navigator.config.blocked_turn_speed
    assert "Недостаточный зазор спереди" in caplog.text


def test_blocked_turn_uses_course_sign_when_avoidance_cancels(navigator):
    """Если избегание полностью компенсировало базовый поворот, разворот должен следовать знаку yaw_error."""

    navigator.update_pose(0.0, 0.0, -0.2)
    navigator.config.avoidance_max_correction = 0.5
    navigator.update_scan([0.7, 0.7, 0.7, 0.25, 0.25])

    command = navigator.step(0.1)

    assert command["v"] == pytest.approx(0.0, abs=1e-6)
    assert command["w"] < 0.0


def test_blocked_turn_boosts_existing_rotation(navigator):
    """При малом ненулевом повороте boosting должен сохранить знак и поднять скорость до порога."""

    navigator.update_pose(0.0, 0.0, -0.1)
    navigator.update_scan([0.5, 0.42, 0.33, 0.42, 0.5])

    command = navigator.step(0.1)

    assert command["v"] == pytest.approx(0.0, abs=1e-6)
    assert command["w"] == pytest.approx(navigator.config.blocked_turn_speed)


def test_apply_blocked_turn_boost_prefers_yaw_error(navigator):
    """Отдельный метод должен корректно выбирать направление разворота."""

    result = navigator._apply_blocked_turn_boost(0.0, -0.4)
    assert result == pytest.approx(-navigator.config.blocked_turn_speed)

    result_left = navigator._apply_blocked_turn_boost(0.0, 0.0)
    assert result_left == pytest.approx(navigator.config.blocked_turn_speed)


def test_blocked_direction_prefers_wider_side(navigator):
    """Если слева свободнее, разворот должен происходить влево при малой угловой ошибке."""

    navigator.update_pose(0.0, 0.0, 0.0)
    # Правая половина лидара показывает стену, левая — свободна.
    navigator.update_scan([0.32, 0.31, 0.3, 0.75, 0.9])

    command = navigator.step(0.1)

    assert command["v"] == pytest.approx(0.0, abs=1e-6)
    assert command["w"] > 0.0


def test_blocked_direction_uses_alternating_seed_once(navigator):
    """Первое решение при симметрии должно зависеть от счётчика блокировок."""

    navigator.update_pose(0.0, 0.0, 0.0)
    navigator.update_scan([0.33, 0.33, 0.32, 0.33, 0.33])

    first = navigator.step(0.1)
    assert first["w"] == pytest.approx(navigator.config.blocked_turn_speed)

    # Сбрасываем скан, но оставляем блокировку — направление не должно меняться.
    navigator.update_scan([0.33, 0.33, 0.32, 0.33, 0.33])
    second = navigator.step(0.1)

    assert second["w"] == pytest.approx(navigator.config.blocked_turn_speed)
    assert navigator.state.blocked_turn_direction == pytest.approx(1.0)


def test_blocked_direction_stays_sticky_until_clear(navigator):
    """Как только выбран знак разворота, он должен сохраняться до снятия блокировки."""

    navigator.update_pose(0.0, 0.0, 0.0)
    navigator.update_scan([0.33, 0.33, 0.32, 0.33, 0.33])

    navigator.step(0.1)
    navigator.update_scan([0.33, 0.33, 0.32, 0.33, 0.33])
    navigator.step(0.1)

    # Имитация освобождения прохода: большой зазор спереди.
    navigator.update_scan([1.0, 1.0, 1.0, 1.0, 1.0])
    navigator.step(0.1)

    assert navigator.state.blocked_steps == 0
    assert navigator.state.blocked_turn_direction is None


def test_resolve_blocked_turn_uses_yaw_priority(navigator):
    """Вспомогательный метод должен отдавать приоритет курсовой ошибке при достаточном значении."""

    navigator.state.blocked_steps = 3
    navigator.state.blocked_turn_direction = None
    decision = navigator._resolve_blocked_turn_direction(0.0, math.radians(20.0), 0.4, 0.4)
    assert decision > 0.0

    navigator.state.blocked_steps = 2
    navigator.state.blocked_turn_direction = None
    decision_right = navigator._resolve_blocked_turn_direction(0.0, -math.radians(20.0), 0.4, 0.4)
    assert decision_right < 0.0


def test_apply_blocked_turn_boost_respects_max_speed(navigator):
    """Когда угловая скорость уже велика, boosting должен ограничить её максимумом."""

    boosted = navigator._apply_blocked_turn_boost(2.0, 0.0)
    assert boosted == pytest.approx(navigator.config.max_angular_speed)


def test_slowdown_region_between_stop_and_clear(navigator):
    """Если препятствие чуть дальше порога остановки, включаем плавное замедление."""

    navigator.update_pose(0.0, 0.0, 0.0)
    navigator.update_scan([0.55, 0.48, 0.45, 0.48, 0.55])
    command = navigator.step(0.1)
    assert command["v"] > 0.0
    assert command["v"] < navigator.config.max_linear_speed


def test_avoidance_cannot_flip_turn_direction(navigator):
    """Когда нужно резко развернуться, добавка избегания не должна разворачивать робот."""

    # Немного разворачиваем робота так, чтобы yaw_error был около -0.3 рад (цель справа позади).
    navigator.update_pose(0.0, 0.0, 0.3)
    # Уменьшаем требуемый зазор, чтобы навигатор не блокировал движение и применил ограничение знака.
    navigator.config.forward_clearance_distance = 0.2
    # Справа (углы < 0) располагаются препятствия на расстоянии 0.25 м,
    # что пытается сместить поворот влево. Проверяем, что ограничение сохраняет знак ω.
    navigator.update_scan([0.25, 0.25, 0.7, 0.7, 0.7])
    command = navigator.step(0.1)
    assert command["w"] < 0.0


def test_refuses_non_positive_dt(navigator):
    """Нулевой шаг интегрирования должен приводить к ошибке."""

    navigator.update_pose(0.0, 0.0, 0.0)
    navigator.update_scan([1.0] * 5)
    with pytest.raises(ValueError):
        navigator.step(0.0)


def test_handles_empty_scan_and_finishes_route(navigator):
    """Навигатор должен останавливаться, если лидар не дал данных или маршрут завершён."""

    navigator.update_scan([float("nan"), -1.0])
    navigator.update_pose(0.0, 0.0, 0.0)
    cmd_turn = navigator.step(0.1)
    assert cmd_turn["v"] == 0.0

    # Последовательно закрываем обе точки маршрута.
    navigator.update_scan([1.0])  # одиночный луч покрывает ветку n == 1
    navigator.update_pose(1.0, 0.0, 0.0)
    navigator.step(0.1)

    navigator.update_pose(2.0, 0.0, 0.0)
    command = navigator.step(0.1)
    assert command["target"] is None
    assert navigator.state.finished is True

    repeat = navigator.step(0.1)
    assert repeat["target"] is None

