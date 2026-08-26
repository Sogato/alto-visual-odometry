"""
Метрики точности восстановленной траектории.

Модуль сравнивает оценённую траекторию с эталонной. Прямое сравнение координат
невозможно: оценка живёт в системе координат первой камеры, эталон в проекции
UTM, а у части методов ещё и неизвестен масштаб. Поэтому траектории сначала
совмещают функцией align, а ошибку считают после.

Совмещение ищет поворот, сдвиг и при необходимости общий множитель, наилучшим
образом накладывающие одну траекторию на другую. Множитель один на всю
траекторию: это приведение результата к сопоставимому виду, а не подсказка
алгоритму на каждом шаге.

Метрик три. Абсолютная ошибка показывает, насколько разошлись траектории в
целом, и растёт вместе с накоплением. Относительная ошибка считается на
отрезках заданной длины и от накопления не зависит, поэтому показывает качество
самих шагов. Ошибка масштаба применима только к методам, которые
восстанавливают длину перемещения.

Ориентация здесь не сравнивается: накопленные повороты записаны в разных осях,
и без известного преобразования между системами сопоставить их нельзя. Ошибку
поворота считает vo_module/analysis/geometry.py, где для этого есть измеренный
поворот между системами.
"""

# Стандартные библиотеки
from typing import Any

# Сторонние библиотеки
import numpy as np


# === ДЛИНЫ ОТРЕЗКОВ ДЛЯ ОТНОСИТЕЛЬНОЙ ОШИБКИ ===
# Значения в метрах. Короткие отрезки показывают качество отдельных шагов,
# длинные показывают, как быстро копится ошибка
SEGMENT_LENGTHS: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0, 800.0)

# Наименьшее число точек в отрезке, при котором его совмещение осмысленно
MIN_SEGMENT_POINTS = 3


# ────────────────────────────────────────────────────────────────────────────
# Совмещение траекторий
# ────────────────────────────────────────────────────────────────────────────

def align(source: np.ndarray, target: np.ndarray,
          with_scale: bool = True) -> dict[str, Any]:
    """
    Находит преобразование, накладывающее одну траекторию на другую.

    Решается задача Умеямы: ищутся поворот, сдвиг и общий множитель,
    минимизирующие сумму квадратов расстояний между соответствующими точками.
    Решение получается через разложение по сингулярным числам взаимной
    ковариации. Отражения запрещаются, иначе траектория могла бы совпасть со
    своим зеркальным отражением.

    Args:
        source: оценённая траектория, массив (N, 3).
        target: эталонная траектория, массив (N, 3).
        with_scale: искать ли общий множитель. Нужен там, где метод не
            восстанавливает длину перемещения.

    Returns:
        Словарь с полями rotation в виде матрицы 3x3, translation, scale и
        aligned с приведённой траекторией.

    Raises:
        ValueError: траектории пусты либо их длины не совпадают.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if len(source) != len(target):
        raise ValueError(f"длины траекторий не совпадают: {len(source)} и "
                         f"{len(target)}")
    if len(source) == 0:
        raise ValueError("нечего совмещать: траектории пусты")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_shifted = source - source_center
    target_shifted = target - target_center

    covariance = target_shifted.T @ source_shifted / len(source)
    left, singular, right = np.linalg.svd(covariance)

    # Запрет отражения: определитель произведения должен быть положительным
    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        correction[2, 2] = -1.0

    rotation = left @ correction @ right

    scale = 1.0
    if with_scale:
        variance = float((source_shifted ** 2).sum() / len(source))
        if variance > 0:
            scale = float(np.trace(np.diag(singular) @ correction) / variance)

    translation = target_center - scale * rotation @ source_center
    aligned = (scale * rotation @ source.T).T + translation

    return {
        "rotation": rotation,
        "translation": translation,
        "scale": scale,
        "aligned": aligned,
    }


def path_length(trajectory: np.ndarray) -> float:
    """
    Считает длину ломаной, проходящей через точки траектории.

    Args:
        trajectory: массив (N, 3).

    Returns:
        Длину пути в тех же единицах, что и координаты. Ноль, если точек
        меньше двух.
    """
    if len(trajectory) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())


# ────────────────────────────────────────────────────────────────────────────
# Абсолютная ошибка
# ────────────────────────────────────────────────────────────────────────────

def absolute_error(estimated: np.ndarray, reference: np.ndarray,
                   with_scale: bool = True) -> dict[str, Any]:
    """
    Считает расхождение траекторий после их совмещения.

    Величина накапливается вдоль маршрута, поэтому отражает суммарный дрейф, а
    не качество отдельного шага. Отдельно возвращается ошибка конечной точки:
    она нагляднее среднего, когда траектория плавно уходит в сторону.

    Args:
        estimated: оценённая траектория, массив (N, 3).
        reference: эталонная траектория, массив (N, 3).
        with_scale: подбирать ли общий множитель при совмещении.

    Returns:
        Словарь с полями rmse, mean, median, max и final в метрах, final_ratio
        в долях эталонного пути, scale, path_length, distances с расхождением в
        каждой точке и aligned с приведённой траекторией. Поле final_ratio
        равно NaN при нулевой длине эталонного пути.

    Raises:
        ValueError: траектории пусты либо их длины не совпадают.
    """
    alignment = align(estimated, reference, with_scale)
    distances = np.linalg.norm(alignment["aligned"] - reference, axis=1)
    path = path_length(reference)

    return {
        "rmse": float(np.sqrt(np.mean(distances ** 2))),
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "max": float(distances.max()),
        "final": float(distances[-1]),
        # Доля от пройденного пути позволяет сравнивать маршруты разной длины
        "final_ratio": float(distances[-1] / path) if path > 0 else float("nan"),
        "scale": alignment["scale"],
        "path_length": path,
        "distances": distances,
        "aligned": alignment["aligned"],
    }


# ────────────────────────────────────────────────────────────────────────────
# Относительная ошибка
# ────────────────────────────────────────────────────────────────────────────

def relative_error(estimated: np.ndarray, reference: np.ndarray,
                   lengths: tuple[float, ...] = SEGMENT_LENGTHS,
                   with_scale: bool = True) -> dict[float, dict[str, float]]:
    """
    Считает ошибку на отрезках маршрута заданной длины.

    Каждый отрезок совмещается со своим эталоном отдельно, поэтому величина не
    накапливается и характеризует качество самих шагов. Она позволяет отличить
    метод, который плохо считает каждый шаг, от метода, который считает шаги
    неплохо, но систематически уводит траекторию.

    Зависимость ошибки от длины отрезка отвечает на вопрос задания о влиянии
    длины траектории: рост ошибки с длиной означает систематический увод,
    ровный ход означает накопление одного лишь случайного шума.

    Отрезки берутся от каждой точки эталона, поэтому стоимость расчёта растёт
    как произведение числа точек на длину отрезка.

    Args:
        estimated: оценённая траектория, массив (N, 3).
        reference: эталонная траектория, массив (N, 3).
        lengths: длины отрезков в метрах.
        with_scale: подбирать ли множитель при совмещении каждого отрезка.

    Returns:
        Словарь, сопоставляющий длине отрезка поля median и mean в процентах от
        длины отрезка и count с числом отрезков. Длины, для которых не нашлось
        ни одного отрезка, в словарь не попадают.
    """
    reference = np.asarray(reference, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    # Накопленная длина пути до каждой точки эталона: по ней ищутся концы
    # отрезков нужной длины
    steps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(steps)])

    result: dict[float, dict[str, float]] = {}

    for length in lengths:
        errors: list[float] = []

        for start in range(len(reference)):
            finish = int(np.searchsorted(travelled, travelled[start] + length))
            if finish >= len(reference):
                break

            piece_estimated = estimated[start:finish + 1]
            piece_reference = reference[start:finish + 1]
            if len(piece_reference) < MIN_SEGMENT_POINTS:
                continue

            alignment = align(piece_estimated, piece_reference, with_scale)
            drift = float(np.linalg.norm(alignment["aligned"][-1]
                                         - piece_reference[-1]))
            errors.append(drift / length * 100.0)

        if errors:
            values = np.array(errors)
            result[length] = {
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "count": len(values),
            }

    return result


# ────────────────────────────────────────────────────────────────────────────
# Ошибка масштаба
# ────────────────────────────────────────────────────────────────────────────

def scale_error(estimated: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """
    Оценивает, насколько верно метод восстановил длину перемещения.

    Применимо только к методам, дающим метрический сдвиг. У методов, которые
    восстанавливают лишь направление, длины нет вовсе.

    Отношение длин путей возвращается рядом с процентом, поэтому видно, откуда
    процент взялся.

    Args:
        estimated: оценённая траектория в метрах, массив (N, 3).
        reference: эталонная траектория, массив (N, 3).

    Returns:
        Словарь с полями total_ratio и error_percent со знаком, показывающим
        направление ошибки. Поле ok равно False, если эталонный путь нулевой, и
        тогда остальных полей нет.
    """
    estimated_steps = np.linalg.norm(np.diff(estimated, axis=0), axis=1)
    reference_steps = np.linalg.norm(np.diff(reference, axis=0), axis=1)

    if not np.any(reference_steps > 0):
        return {"ok": False}

    total = float(estimated_steps.sum() / reference_steps.sum())

    return {
        "ok": True,
        "total_ratio": total,
        "error_percent": (total - 1.0) * 100.0,
    }
