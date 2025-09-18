"""Тесты для логики поиска и прохождения коридоров."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from corridor_follower import CorridorFollower, FollowCfg
from tools import (
    _angles_deg,
    compute_required_gap_width,
    corridor_fits,
    corridor_width_margin,
    find_corridors,
)


def test_angles_and_find_corridors_identify_full_opening():
    """Проверяем, что равномерные лучи формируют один широкий коридор."""
    ranges = [1.5] * 21  # имитируем равномерно свободное пространство
    corridors = find_corridors(ranges, fov_deg=90.0, max_lookahead=2.0,
                               min_points=3, min_depth_for_corridor=0.5)
    assert len(corridors) == 1
    corridor = corridors[0]
    assert corridor["i0"] == 0 and corridor["i1"] == len(ranges) - 1
    # центр сегмента должен совпадать с прямым направлением
    assert math.isclose(corridor["ang_center"], 0.0, abs_tol=1e-6)
    # ширина на лучшей глубине должна быть положительной
    assert corridor["width_at_best"] > 0.0


def test_corridor_fits_filters_by_width():
    """Алгоритм должен отбрасывать слишком узкие коридоры."""
    ranges = [0.2, 0.4, 1.0, 1.0, 1.0, 0.4, 0.2]
    corridors = find_corridors(ranges, fov_deg=90.0, max_lookahead=2.0,
                               min_points=3, min_depth_for_corridor=0.3)
    assert len(corridors) == 1
    corridor = corridors[0]

    required_width = compute_required_gap_width(wheel_base=0.25, side_margin=0.05,
                                                body_extra_each_side=0.02)
    ok, plan = corridor_fits(
        corridor,
        required_width=required_width,
        min_forward_clearance=0.4,
        width_tolerance=0.02,
    )
    assert ok is True
    assert plan["expected_width"] >= plan["requested_width"]
    assert plan["width_tolerance"] == pytest.approx(0.02)
    assert plan["width_margin"] >= 0.0

    # Если запросить заведомо большую ширину, проход должен отвергаться
    too_wide, _ = corridor_fits(corridor, required_width=5.0, min_forward_clearance=0.4)
    assert too_wide is False


def test_corridor_fits_tolerance_prevents_dropout():
    """Небольшой запас по ширине должен спасать коридор от исчезновения."""
    corridor = {
        "ang_left": -30.0,
        "ang_right": 30.0,
        "ang_center": 0.0,
        "depth_min": 0.31,
    }
    required_width = 0.37

    ok_no_tol, plan_no_tol = corridor_fits(
        corridor,
        required_width=required_width,
        min_forward_clearance=0.3,
        width_tolerance=0.0,
    )
    assert ok_no_tol is False and plan_no_tol is None

    ok_with_tol, plan_with_tol = corridor_fits(
        corridor,
        required_width=required_width,
        min_forward_clearance=0.3,
        width_tolerance=0.02,
    )
    assert ok_with_tol is True
    assert plan_with_tol["width_margin"] == pytest.approx(-0.02, abs=1e-6)
    assert plan_with_tol["requested_width"] == pytest.approx(required_width)


def test_corridor_width_margin_identifies_near_miss_and_recovery():
    """Функция должна показывать отрицательный, но ограниченный запас при узком проёме."""
    corridor = {
        "ang_left": -25.0,
        "ang_right": 25.0,
        "ang_center": 0.0,
        "depth_min": 0.3,
    }
    required_width = 0.33
    base_tol = 0.02

    margin = corridor_width_margin(corridor, required_width, base_tol)
    assert margin < 0.0
    assert margin > -0.05

    ok_recover, plan_recover = corridor_fits(
        corridor,
        required_width=required_width,
        min_forward_clearance=0.25,
        width_tolerance=base_tol + 0.04,
    )
    assert ok_recover is True
    assert plan_recover["width_margin"] < 0.0
    assert plan_recover["width_margin"] > -0.08


@pytest.fixture()
def follower() -> CorridorFollower:
    """Создаём последователя с немного увеличенными пределами для быстрого отклика."""
    cfg = FollowCfg(v_max=0.4, w_max=1.2, a_lin=1.0, a_ang=5.0, k_yaw=2.0, hold_sec=0.1)
    return CorridorFollower(wheel_base=0.25, wheel_radius=0.035, cfg=cfg)


def _make_corridor_plan(ranges):
    corridors = find_corridors(ranges, fov_deg=90.0, max_lookahead=2.0,
                               min_points=3, min_depth_for_corridor=0.3)
    assert corridors, "должен быть найден хотя бы один коридор"
    corridor = corridors[0]
    ok, plan = corridor_fits(
        corridor,
        required_width=0.35,
        min_forward_clearance=0.4,
        width_tolerance=0.02,
    )
    assert ok and plan
    corridor["plan"] = plan
    return corridor, plan


def test_corridor_follower_moves_forward_and_stays_straight(follower):
    """При широком проёме робот должен ехать прямо с равными угловыми скоростями колёс."""
    ranges = [1.5] * 21
    corridor, plan = _make_corridor_plan(ranges)
    result = follower.step(0.1, ranges, corridor, plan, plan["requested_width"])
    assert result["v"] > 0.0
    assert abs(result["w"]) < 1e-6
    assert math.isclose(result["w_left"], result["w_right"], rel_tol=1e-6)


def test_corridor_follower_emergency_brake_on_close_obstacle(follower):
    """При резком появлении препятствия спереди скорость должна быстро падать к нулю."""
    ranges_clear = [1.5] * 21
    corridor, plan = _make_corridor_plan(ranges_clear)
    # Разгоняемся на чистом участке
    follower.step(0.1, ranges_clear, corridor, plan, plan["requested_width"])
    assert follower.v > 0.0

    # Теперь создаём препятствие непосредственно перед центром
    ranges_blocked = ranges_clear.copy()
    center = len(ranges_blocked) // 2
    for offset in range(-2, 3):
        ranges_blocked[center + offset] = 0.15
    corridor_close, plan_close = _make_corridor_plan(ranges_clear)
    result = follower.step(
        0.1,
        ranges_blocked,
        corridor_close,
        plan_close,
        plan_close["requested_width"],
    )
    assert result["front_clear"] < 0.25
    assert result["v"] == pytest.approx(0.0, abs=1e-3)


def test_corridor_follower_rejects_non_positive_dt(follower):
    """Отрицательное или нулевое время шага должно приводить к ошибке."""
    ranges = [1.5] * 21
    corridor, plan = _make_corridor_plan(ranges)
    with pytest.raises(ValueError):
        follower.step(0.0, ranges, corridor, plan, plan["requested_width"])


def test_angles_and_sanitizing_edge_cases():
    """Отдельно проверяем вспомогательные ветки: короткие массивы, NaN и inf."""
    assert _angles_deg(1, 90.0) == [0.0]
    assert find_corridors([], fov_deg=90.0) == []

    ranges = [float("nan"), 0.0, float("inf"), 10.0, 0.6]
    corridors = find_corridors(ranges, fov_deg=90.0, max_lookahead=1.0,
                               min_points=1, min_depth_for_corridor=0.3)
    assert corridors  # очищенные значения должны позволить найти сегмент

    # Коридор с некорректной глубиной и геометрией должен быть отвергнут
    bad_depth = {"ang_left": -10.0, "ang_right": 10.0, "ang_center": 0.0, "depth_min": 0.0}
    ok, plan = corridor_fits(bad_depth, required_width=0.3, min_forward_clearance=0.2)
    assert not ok and plan is None

    bad_geometry = {"ang_left": 0.0, "ang_right": 0.0, "ang_center": 0.0, "depth_min": 1.0}
    ok, plan = corridor_fits(bad_geometry, required_width=0.3, min_forward_clearance=0.2)
    assert not ok and plan is None

    # corridor_width_margin должен консервативно реагировать на неполные данные.
    assert corridor_width_margin({}, 0.3, 0.02) == float("-inf")
