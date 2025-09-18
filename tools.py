import math
from typing import List, Dict, Tuple, Optional

def _angles_deg(n: int, fov_deg: float) -> List[float]:
    """Равномерные углы от -FOV/2 до +FOV/2 включительно."""
    if n <= 1:
        return [0.0]
    start = -fov_deg / 2.0
    step = fov_deg / (n - 1)
    return [start + i * step for i in range(n)]

def find_corridors(
    ranges: List[float],
    fov_deg: float = 90.0,
    max_lookahead: float = 3.0,
    min_points: int = 3,
    min_depth_for_corridor: float = 0.5,
) -> List[Dict]:
    """
    1) Находит «коридоры» — непрерывные угловые сегменты, где вперёд есть свободное пространство.
       Сегмент — это подряд идущие лучи, у которых дистанция >= min_depth_for_corridor.
       Для каждого сегмента считаем геометрию (углы/глубину/эффективную ширину).
    Возврат: список dict со свойствами коридора:
      {
        'i0','i1',                  # индексы первого/последнего луча
        'ang_left','ang_right',     # границы (градусы, левая<0, правая>0)
        'ang_center',               # центр (градусы)
        'depth_min',                # минимальная глубина внутри сегмента (ограничивает проезд)
        'depth_best',               # глубина, на которой ширина максимальна (ограничена max_lookahead и depth_min)
        'width_at_best',            # поперечная ширина на depth_best
      }
    """
    n = len(ranges)
    if n == 0:
        return []
    ang = _angles_deg(n, fov_deg)

    # «чистим» бесконечности и NaN
    clamped = []
    for r in ranges:
        if not (r == r) or r <= 0.0:          # NaN или некорректно
            clamped.append(0.0)
        elif math.isinf(r) or r > max_lookahead:
            clamped.append(max_lookahead)
        else:
            clamped.append(float(r))

    # бинарная маска «достаточно свободно прямо»
    free = [ri >= min_depth_for_corridor for ri in clamped]

    corridors: List[Dict] = []
    i = 0
    while i < n:
        if not free[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and free[j + 1]:
            j += 1

        # сегмент i..j найден
        if (j - i + 1) >= min_points:
            depth_min = min(clamped[i : j + 1])
            # прикидываем лучшую «рабочую» глубину: не дальше, чем можем видеть
            depth_best = min(depth_min, max_lookahead)

            ang_left = ang[i]
            ang_right = ang[j]
            ang_center = 0.5 * (ang_left + ang_right)

            # ширина поперёк на линии x = depth_best:
            # y = x * tan(theta); ширина = x * (tan(right) - tan(left))
            t_left = math.tan(math.radians(ang_left))
            t_right = math.tan(math.radians(ang_right))
            width_at_best = max(0.0, depth_best * (t_right - t_left))

            corridors.append({
                "i0": i, "i1": j,
                "ang_left": ang_left, "ang_right": ang_right, "ang_center": ang_center,
                "depth_min": depth_min, "depth_best": depth_best,
                "width_at_best": width_at_best,
            })

        i = j + 1

    return corridors

def corridor_fits(
    corridor: dict,
    required_width: float,
    min_forward_clearance: float = 0.35,
) -> tuple[bool, dict | None]:
    need = required_width
    left = corridor["ang_left"]
    right = corridor["ang_right"]
    center = corridor["ang_center"]
    dmax = corridor["depth_min"]
    if dmax <= 0.0:
        return False, None
    gap_tan = math.tan(math.radians(right)) - math.tan(math.radians(left))
    if gap_tan <= 0.0:
        return False, None
    d_need = need / gap_tan
    if d_need <= dmax:
        d_best = max(d_need, min_forward_clearance)
        width_best = d_best * gap_tan
        plan = {
            "target_depth": d_best,
            "target_yaw_deg": center,
            "expected_width": width_best,
            "required_width": need,
        }
        return True, plan
    return False, None


def compute_required_gap_width(
    wheel_base: float,
    side_margin: float = 0.04,
    body_extra_each_side: float = 0.02,
    min_pass_width: float | None = None,
) -> float:
    """
    Возвращает требуемую ширину коридора (проёма) с учётом:
      - wheel_base: колея (расстояние между колёсами), т.е. нижняя оценка ширины робота
      - body_extra_each_side: выступы корпуса/бампера за колёса с каждой стороны
      - side_margin: технологический зазор с каждой стороны (страховка/ошибка лидара/люфт)
      - min_pass_width: если задана минимальная проходная ширина из твоей конфигурации,
        берём максимум из расчётной и этой
    Итог: required_width = max(wheel_base + 2*(body_extra_each_side + side_margin), min_pass_width?)
    """
    raw = wheel_base + 2.0 * (body_extra_each_side + side_margin)
    return max(raw, min_pass_width) if min_pass_width else raw
