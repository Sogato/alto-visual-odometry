"""
Прогон одометрии: от пар кадров к траектории.

Модуль связывает все слои. По каждой паре кадров он вызывает детектор, матчер и
обе геометрии, замеряет время каждого этапа, накапливает траекторию и
записывает подробности в таблицу. Анализы берут данные из этой таблицы и
детекторы повторно не гоняют.

Расход памяти здесь не меряется: прогон идёт в общем процессе вместе со всем,
что уже загружено. Память меряет vo_module/analysis/performance.py, запуская
каждую связку в отдельном процессе.

Масштаб шага зависит от геометрии. Разложение гомографии даёт сдвиг в долях
расстояния до плоскости, а расстояние это высота полёта из телеметрии, поэтому
шаг сразу получается в метрах. Essential Matrix даёт только направление, и её
шаги берутся единичными. Подставлять длину из эталона нельзя: это подсказка
алгоритму, обессмысливающая анализ накопления ошибки. Для таких методов
масштаб подбирается один раз на всю траекторию при сравнении с эталоном.

Отказ геометрии на паре означает нулевое перемещение на этом шаге, а сам шаг
помечается в таблице. Повтор предыдущего смещения маскировал бы отказ и делал
траекторию глаже, чем она есть.
"""

# Стандартные библиотеки
import sys
import time
from pathlib import Path
from typing import Any

# Сторонние библиотеки
import cv2
import numpy as np
import pandas as pd

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from console import PROGRESS_FORMS, clear_progress, print_progress
from vo_module import features, pose
from vo_module.dataset import Dataset, rotation_angle


# === ЕДИНИЦЫ ИЗМЕРЕНИЯ ===
MS_PER_SECOND = 1000.0     # Перевод секунд в миллисекунды для замеров времени


# ────────────────────────────────────────────────────────────────────────────
# Замеры
# ────────────────────────────────────────────────────────────────────────────

def synchronize() -> None:
    """
    Дожидается завершения расчётов на видеокарте.

    Вычисления на GPU выполняются асинхронно: управление возвращается до того,
    как работа закончена, и без ожидания замер показал бы только длительность
    постановки задачи в очередь.

    Та же функция повторена в vo_module/analysis/performance.py: импорт
    пайплайна притащил бы в замеряемый процесс pandas.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────────
# Шаг траектории
# ────────────────────────────────────────────────────────────────────────────

def metric_step(outcome: dict[str, Any], height: float) -> np.ndarray:
    """
    Переводит результат геометрии в перемещение камеры за шаг.

    Сдвиг в долях расстояния до плоскости умножается на высоту полёта и
    становится метрами. Если геометрия дала только направление, шаг берётся
    единичным, и длина восстанавливается общим множителем при сравнении с
    эталоном.

    Args:
        outcome: результат функции восстановления позы.
        height: высота над землёй, м.

    Returns:
        Вектор перемещения в осях камеры на начало шага. Нулевой, если
        геометрия не отработала.
    """
    if not outcome["ok"]:
        return np.zeros(3)

    if outcome.get("translation_over_distance") is not None:
        return np.asarray(outcome["translation_over_distance"],
                          dtype=float) * height

    direction = np.asarray(outcome["translation"], dtype=float)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else direction


def accumulate(position: np.ndarray, orientation: np.ndarray,
               step_rotation: np.ndarray, step_translation: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Добавляет к накопленной позе перемещение за один шаг.

    Шаг задан в осях камеры на начало шага, поэтому перед сложением переводится
    в исходные оси накопленной ориентацией.

    Args:
        position: накопленная позиция.
        orientation: накопленная ориентация.
        step_rotation: поворот камеры за шаг.
        step_translation: перемещение камеры за шаг в осях начала шага.

    Returns:
        Пару из новой позиции и новой ориентации.
    """
    return position + orientation @ step_translation, orientation @ step_rotation


# ────────────────────────────────────────────────────────────────────────────
# Ошибки относительно эталона
# ────────────────────────────────────────────────────────────────────────────

def rotation_error(outcome: dict[str, Any], motion: dict[str, Any],
                   to_reference: np.ndarray) -> float:
    """
    Считает угол между оценённым поворотом камеры за шаг и эталонным.

    Сравнивать углы самих поворотов нельзя: у двух поворотов на одинаковый угол
    вокруг разных осей разность углов равна нулю, и оценка, крутящая камеру
    вокруг посторонней оси, получила бы нулевую ошибку. Поэтому считается
    поворот, переводящий одну оценку в другую, и берётся его угол.

    Оценка приводится к осям эталона сопряжением, а не умножением с одной
    стороны: оси меняются и у того, что поворачивают, и у результата.

    Args:
        outcome: результат восстановления позы.
        motion: эталонное движение из dataset.relative_motion.
        to_reference: поворот из системы камеры в систему эталона.

    Returns:
        Угол ошибки в градусах. NaN при отказе геометрии.
    """
    if not outcome["ok"]:
        return float("nan")

    estimated = (to_reference @ np.asarray(outcome["rotation"], dtype=float)
                 @ to_reference.T)
    reference = np.asarray(motion["rotation"], dtype=float)
    return rotation_angle(reference.T @ estimated)


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """
    Считает угол между двумя векторами.

    Args:
        first: первый вектор.
        second: второй вектор.

    Returns:
        Угол в градусах, от нуля до ста восьмидесяти. NaN, если хотя бы один из
        векторов нулевой.
    """
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        return float("nan")

    cosine = float(np.clip(np.dot(first / first_norm, second / second_norm),
                           -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def direction_error(outcome: dict[str, Any], motion: dict[str, Any],
                    to_reference: np.ndarray) -> tuple[float, float, float]:
    """
    Считает угол между оценённым направлением сдвига и эталонным.

    Оценка выражена в осях камеры, эталон в осях носителя, поэтому перед
    сравнением направление переводится измеренным поворотом между системами.

    Возвращаются три величины. Полная учитывает все три составляющие, включая
    вертикальную. Горизонтальная считается по проекции на плоскость движения и
    ближе связана с точностью траектории: полёт почти горизонтальный, и ошибка
    в вертикальной составляющей на форму маршрута влияет слабо. Третья величина
    это горизонтальная ошибка со знаком, по которой систематический увод в одну
    сторону отличается от случайного разброса.

    Знак берётся по векторному произведению: положительный означает поворот
    оценки против часовой стрелки относительно эталона.

    Args:
        outcome: результат восстановления позы.
        motion: эталонное движение из dataset.relative_motion.
        to_reference: поворот из системы камеры в систему эталона.

    Returns:
        Тройку из полной, горизонтальной и знаковой горизонтальной ошибки в
        градусах. Составляющая равна NaN, если считать её не по чему.
    """
    if not outcome["ok"]:
        return float("nan"), float("nan"), float("nan")

    estimated = to_reference @ np.asarray(outcome["translation"], dtype=float)
    reference = np.asarray(motion["delta_body"], dtype=float)

    plane_estimated, plane_reference = estimated[:2], reference[:2]
    plane_error = angle_between(plane_estimated, plane_reference)

    if np.isnan(plane_error):
        signed = float("nan")
    else:
        cross = (plane_reference[0] * plane_estimated[1]
                 - plane_reference[1] * plane_estimated[0])
        dot = float(np.dot(plane_reference, plane_estimated))
        signed = float(np.degrees(np.arctan2(cross, dot)))

    return angle_between(estimated, reference), plane_error, signed


# ────────────────────────────────────────────────────────────────────────────
# Прогон
# ────────────────────────────────────────────────────────────────────────────

def run(data: Dataset, frontend: dict[str, str], step: int,
        start: int = 0, stop: int | None = None, limit: int | None = None,
        progress_label: str = "") -> dict[str, Any]:
    """
    Прогоняет одометрию по последовательности кадров одной связкой.

    Обе геометрии считаются поверх одних и тех же сопоставленных точек, поэтому
    разница между ними относится к модели, а дорогая часть работы выполняется
    один раз.

    Перед основным циклом идёт прогревочная пара: первое обращение к детектору
    и матчеру подтягивает библиотеки и веса, и без прогрева их загрузка попала
    бы в замер времени первой пары.

    Высота над землёй берётся по первому кадру пары.

    Args:
        data: загруженный датасет.
        frontend: описание связки из config.FRONTENDS.
        step: шаг прореживания кадров.
        start: индекс первого кадра участка.
        stop: индекс, до которого идти, не включая. По умолчанию до конца.
        limit: предельное число пар. По умолчанию сколько получится.
        progress_label: пояснение для индикатора выполнения. Пустая строка
            отключает индикатор, что нужно при вложенных прогонах.

    Returns:
        Словарь с полями dataset, frontend, label, step, pairs с таблицей по
        парам, trajectories и orientations по каждой геометрии, reference с
        эталонной траекторией, приведённой к началу координат первой точки,
        reference_rotations и elapsed_s.

    Raises:
        KeyError: не заполнены калибровочные константы датасета либо в
            описании связки указан неизвестный детектор или матчер.
        ValueError: шаг прореживания меньше единицы.
    """
    calibration = config.CALIBRATION[data.name]
    missing = [field for field in config.CALIBRATION_REQUIRED
               if calibration.get(field) is None]
    if missing:
        raise KeyError(f"для датасета {data.name} не заполнено: "
                       f"{', '.join(missing)}")

    intrinsics = pose.camera_matrix(calibration["focal_px"],
                                    calibration["principal_point"])
    to_reference = np.asarray(calibration["rotation_cam_to_gt"], dtype=float)
    elevation = calibration.get("ground_elevation_m") or 0.0

    detector = features.DETECTORS[frontend["detector"]]
    matcher = features.MATCHERS[frontend["matcher"]]
    needs_second = features.MATCHER_NEEDS_SECOND[frontend["matcher"]]

    # Робастная оценка перебирает случайные подмножества точек. Без фиксации
    # генератора повторный прогон на тех же данных даёт слегка другие числа
    cv2.setRNGSeed(config.RANDOM_SEED)

    pairs = data.pairs(step, start=start, stop=stop, limit=limit)
    width, height = data.image_size
    geometry_keys = [item["key"] for item in config.GEOMETRIES]

    # Накопленная поза по каждой геометрии, начинается в начале координат
    positions = {key: [np.zeros(3)] for key in geometry_keys}
    orientations = {key: [np.eye(3)] for key in geometry_keys}

    if pairs:
        warm_first_gray = data.gray(pairs[0][0])
        warm_second_gray = data.gray(pairs[0][1])
        warm_first = detector(warm_first_gray)
        warm_second = detector(warm_second_gray) if needs_second else warm_first
        matcher(warm_first, warm_second, warm_first_gray, warm_second_gray)
        synchronize()

    records: list[dict[str, Any]] = []

    # Отдельное имя: внутри цикла started переиспользуется для замера этапов
    run_started = time.perf_counter()

    for index, (first_index, second_index) in enumerate(pairs):
        if progress_label:
            print_progress(index, len(pairs), progress_label)

        first_gray = data.gray(first_index)
        second_gray = data.gray(second_index)

        started = time.perf_counter()
        first = detector(first_gray)
        # Оптическому потоку детекция второго кадра не нужна: он ищет точки
        # первого кадра прямо по пикселям второго
        second = detector(second_gray) if needs_second else first
        synchronize()
        detect_ms = (time.perf_counter() - started) * MS_PER_SECOND

        started = time.perf_counter()
        matches = matcher(first, second, first_gray, second_gray)
        synchronize()
        match_ms = (time.perf_counter() - started) * MS_PER_SECOND

        motion = data.relative_motion(first_index, second_index)
        above_ground = float(data.altitudes[first_index] - elevation)

        record: dict[str, Any] = {
            "pair": index,
            "first": first_index,
            "second": second_index,
            "keypoints_first": first["count"],
            "keypoints_second": second["count"],
            "matches": matches["count"],
            "candidates": matches["candidates"],
            "spread": features.spatial_spread(matches["points_first"],
                                              width, height),
            "detect_ms": detect_ms,
            "match_ms": match_ms,
            "height_m": above_ground,
            "gt_distance": motion["distance"],
            "gt_rotation": motion["rotation_angle"],
        }

        for key in geometry_keys:
            started = time.perf_counter()
            outcome = pose.POSE_BY_KEY[key](matches["points_first"],
                                           matches["points_second"], intrinsics)
            geometry_ms = (time.perf_counter() - started) * MS_PER_SECOND

            full_error, plane_error, signed_error = direction_error(
                outcome, motion, to_reference)
            step_translation = metric_step(outcome, above_ground)
            step_rotation = (np.asarray(outcome["rotation"], dtype=float)
                             if outcome["ok"] else np.eye(3))

            new_position, new_orientation = accumulate(
                positions[key][-1], orientations[key][-1],
                step_rotation, step_translation)
            positions[key].append(new_position)
            orientations[key].append(new_orientation)

            record.update({
                f"{key}_ok": outcome["ok"],
                f"{key}_ms": geometry_ms,
                f"{key}_inliers": outcome["inliers"],
                f"{key}_inlier_ratio": outcome["inlier_ratio"],
                f"{key}_residual": outcome["residual"],
                f"{key}_rotation_error": rotation_error(outcome, motion,
                                                        to_reference),
                f"{key}_rotation": (rotation_angle(outcome["rotation"])
                                    if outcome["ok"] else np.nan),
                f"{key}_direction_error": full_error,
                f"{key}_direction_error_2d": plane_error,
                f"{key}_direction_signed": signed_error,
                f"{key}_metric": (outcome.get("translation_over_distance")
                                  is not None),
            })

        records.append(record)

    elapsed = time.perf_counter() - run_started
    if progress_label:
        clear_progress(progress_label, len(pairs), elapsed, PROGRESS_FORMS)

    # Эталон берётся в тех же кадрах, что и оценка: первый кадр первой пары и
    # вторые кадры всех пар
    used = ([pairs[0][0]] + [second for _, second in pairs]) if pairs else []

    if used:
        reference = np.column_stack([data.positions[used],
                                     data.altitudes[used]])
        reference = reference - reference[0]
        reference_rotations = data.rotations[used]
    else:
        reference = np.empty((0, 3))
        reference_rotations = np.empty((0, 3, 3))

    return {
        "dataset": data.name,
        "frontend": frontend["key"],
        "label": frontend["label"],
        "step": step,
        "pairs": pd.DataFrame(records),
        "trajectories": {key: np.array(positions[key]) for key in geometry_keys},
        "orientations": {key: np.array(orientations[key])
                         for key in geometry_keys},
        "reference": reference,
        "reference_rotations": reference_rotations,
        "elapsed_s": elapsed,
    }


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """
    Сводит таблицу по парам к нескольким числам.

    Показатели берутся медианой: отдельные неудачные пары не должны определять
    итоговую оценку.

    Args:
        result: результат run.

    Returns:
        Словарь с полями pairs, keypoints, matches, spread, detect_ms,
        match_ms и по вложенному словарю на каждую геометрию с полями failures,
        inlier_ratio, residual, direction_error, direction_error_2d,
        geometry_ms и metric. Поле ok равно False при пустой таблице, и тогда
        остальных полей нет.
    """
    table = result["pairs"]
    if table.empty:
        return {"ok": False}

    summary: dict[str, Any] = {
        "ok": True,
        "pairs": len(table),
        "keypoints": float(table["keypoints_first"].median()),
        "matches": float(table["matches"].median()),
        "spread": float(table["spread"].median()),
        "detect_ms": float(table["detect_ms"].median()),
        "match_ms": float(table["match_ms"].median()),
    }

    for item in config.GEOMETRIES:
        key = item["key"]
        summary[key] = {
            "failures": int((~table[f"{key}_ok"]).sum()),
            "inlier_ratio": float(table[f"{key}_inlier_ratio"].median()),
            "residual": float(table[f"{key}_residual"].median()),
            "direction_error": float(table[f"{key}_direction_error"].median()),
            "direction_error_2d": float(
                table[f"{key}_direction_error_2d"].median()),
            "geometry_ms": float(table[f"{key}_ms"].median()),
            # Отказавшая пара помечена как неметрическая, поэтому геометрия
            # считается метрической, если её дала хотя бы одна пара
            "metric": bool(table[f"{key}_metric"].any()),
        }

    return summary
