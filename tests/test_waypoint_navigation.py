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


def test_slowdown_region_between_stop_and_clear(navigator):
    """Если препятствие чуть дальше порога остановки, включаем плавное замедление."""

    navigator.update_pose(0.0, 0.0, 0.0)
    navigator.update_scan([0.5, 0.4, 0.35, 0.4, 0.5])
    command = navigator.step(0.1)
    assert command["v"] > 0.0
    assert command["v"] < navigator.config.max_linear_speed


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

