"""
Измерение физических констант датасета.

Кадры имеют размер 500 на 500, тогда как камера ALTO снимает 1600 на 1200, и
способ их получения нигде не записан. Вырезка из центра сохраняет фокусное
расстояние, сжатие уменьшило бы его втрое, поэтому оно проверяется по кадрам.
Камера смотрит вниз, так что сдвиг картинки s связан со сдвигом камеры d и
высотой над землёй h как s = f * d / h. Перемещение и высота известны из
эталона, сдвиг измеряется сопоставлением кадров, и каждая пара даёт собственную
оценку фокусного расстояния.

В config идёт паспортное значение, а не измеренное: измерение уверенно различает
гипотезы, отличающиеся втрое, но для уточнения самой константы слишком шумное.
Смысл колонки altitude тоже берётся из описания датасета.

Собственно измеряется здесь поворот между системой камеры и системой эталона.
Рядом собираются две справочные величины: шум эталонных позиций и доля точек,
согласных с гомографией.

Сопоставление кадров зашито жёстко: SIFT, перебор дескрипторов, гомография.
Настройки этой связки заданы в модуле и не берутся из config.
"""

# Стандартные библиотеки
import sys
import time
from pathlib import Path
from typing import Any

# Сторонние библиотеки
import cv2
import numpy as np
from scipy import signal

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from console import (MISSING, PROGRESS_FORMS, STATUS_FAIL, STATUS_NONE, STATUS_OK,
                     block_line, block_note, block_wrapped, clear_progress,
                     close_block, describe_exception, format_number, open_block,
                     print_banner, print_legend, print_progress, print_section,
                     print_table)
from vo_module.dataset import Dataset


# === ПАСПОРТНОЕ ФОКУСНОЕ РАССТОЯНИЕ ===
# Фокусное расстояние в пикселях по паспорту камеры. Именно с ним сверяется
# измеренное по кадрам значение
DATASHEET_FOCAL_PX = config.LENS_FOCAL_MM / config.PIXEL_PITCH_MM

# Расхождение измеренного фокусного с паспортным, до которого паспортное значение
# считается подтверждённым. Порог широкий: соперничающие гипотезы о происхождении
# кадров отличаются втрое
MAX_FOCAL_DEVIATION = 0.15

# === ПАРАМЕТРЫ ВЫБОРКИ ПАР ===
CALIB_STEP = 30       # Шаг между кадрами пары
CALIB_PAIRS = 150     # Сколько пар оставить, распределяются равномерно по маршруту

# Доля пар с наименьшим наклоном камеры, остающихся в выборке. Наклон сдвигает
# картинку так же, как перемещение, но к перемещению отношения не имеет
TILT_KEEP_FRACTION = 0.6

# Доля наблюдений с наименьшим полным поворотом, по которым ищется поворот осей.
# Здесь отбор идёт по полному углу, а не по наклону: разворот вокруг оптической
# оси поворачивает направление сдвига в кадре, а поворот осей определяется как
# раз по направлениям
ROTATION_KEEP_FRACTION = 0.6

# === ПАРАМЕТРЫ ИЗМЕРИТЕЛЬНОЙ СВЯЗКИ ===
# Заданы здесь, а не в config: настройка сравниваемых методов не должна влиять
# на измеренные константы
CALIB_SIFT_FEATURES = 2000        # Предел числа точек SIFT
CALIB_RATIO = 0.75                # Порог теста отношения расстояний Лоу
CALIB_RANSAC_THRESHOLD = 3.0      # Порог невязки перепроекции, px
CALIB_RANSAC_ITERS = 5000         # Предел числа итераций RANSAC
CALIB_RANSAC_CONFIDENCE = 0.999   # Требуемая вероятность найти верную модель

MIN_INLIERS = 30                  # Нижняя граница числа согласных точек в паре
MIN_PIXEL_SHIFT = 5.0             # Нижняя граница сдвига центра кадра, px

# === СМЫСЛ КОЛОНКИ ALTITUDE ===
# Взято из описания датасета: Cisneros et al., ALTO, arXiv:2207.12317, где высота
# заявлена над землёй. Модуль эту константу подставляет, а не выводит: проверка
# опиралась бы на постоянство фокусного расстояния, а оценка растёт с высотой
# полёта по неустановленной причине
ALTITUDE_MODE = "agl"
GROUND_ELEVATION_M = 0.0

# Наименьший размах высот, при котором зависимость оценки от высоты измерима,
# в долях от высоты полёта
MIN_ALTITUDE_SPAN_FRACTION = 0.05

# === ПОРОГИ РЕШЕНИЙ ===
# Поворот осей определяется, только если направления движения достаточно
# разнообразны: для параллельных векторов поворот и поворот с отражением дают
# одинаковый результат. Разброс сравнивается с собственной невязкой направлений,
# потому что сам по себе он включает шум и на прямом участке не обнуляется
MIN_DIRECTION_SPREAD = 2.0        # Нижняя граница разброса направлений, град
DIRECTION_SPREAD_FACTOR = 3.0     # Во сколько раз разброс должен превышать невязку

MIN_OBSERVATIONS = 10             # Нижняя граница числа наблюдений для измерения

# === ОКНО СГЛАЖИВАНИЯ ДЛЯ ШУМОВОГО ПОЛА ===
NOISE_WINDOW = 21                 # Длина окна фильтра Савицкого и Голея, кадров
NOISE_POLYORDER = 3               # Степень многочлена в окне


# ────────────────────────────────────────────────────────────────────────────
# Измерительная связка
# ────────────────────────────────────────────────────────────────────────────

def transform_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Применяет гомографию к набору точек.

    Args:
        homography: матрица 3x3.
        points: массив формы (N, 2).

    Returns:
        Массив формы (N, 2) с преобразованными точками.
    """
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ np.asarray(homography).T
    return projected[:, :2] / projected[:, 2:3]


def center_shift(homography: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Возвращает сдвиг центра кадра под действием гомографии.

    Args:
        homography: матрица 3x3.
        width: ширина кадра, px.
        height: высота кадра, px.

    Returns:
        Вектор сдвига из двух компонент, px.
    """
    center = np.array([[(width - 1) / 2.0, (height - 1) / 2.0]])
    return (transform_points(homography, center) - center)[0]


def tilt_angle(relative_rotation: np.ndarray) -> float:
    """
    Возвращает наклон камеры между кадрами, без вращения вокруг её оси.

    Поворот раскладывается по Родригу на вектор, направление которого задаёт ось
    вращения, а длина угол. Составляющая вдоль оптической оси отбрасывается: она
    разворачивает кадр вокруг центра, но сам центр не смещает.

    Args:
        relative_rotation: матрица относительного поворота 3x3 в осях носителя.

    Returns:
        Угол наклона в градусах.
    """
    vector = cv2.Rodrigues(np.asarray(relative_rotation, dtype=float))[0].ravel()
    axis = config.CAMERA_AXIS_IN_BODY
    return float(np.degrees(np.linalg.norm(np.delete(vector, axis))))


def match_pair(detector: cv2.Feature2D, matcher: cv2.DescriptorMatcher,
               first_gray: np.ndarray,
               second_gray: np.ndarray) -> dict[str, Any] | None:
    """
    Сопоставляет два кадра и строит между ними гомографию.

    Args:
        detector: детектор SIFT.
        matcher: матчер перебором дескрипторов.
        first_gray: первый кадр в градациях серого.
        second_gray: второй кадр в градациях серого.

    Returns:
        Словарь с полями homography, keypoints_first, keypoints_second, matches,
        inlier_ratio и residual. None, если точек или согласных с гомографией
        сопоставлений оказалось меньше MIN_INLIERS.
    """
    first_kp, first_desc = detector.detectAndCompute(first_gray, None)
    second_kp, second_desc = detector.detectAndCompute(second_gray, None)

    if first_desc is None or second_desc is None:
        return None
    if len(first_kp) < MIN_INLIERS or len(second_kp) < MIN_INLIERS:
        return None

    # Тест отношения расстояний: пара принимается, если ближайший дескриптор
    # заметно ближе второго по близости
    candidates = matcher.knnMatch(first_desc, second_desc, k=2)
    good = [pair[0] for pair in candidates
            if len(pair) == 2 and pair[0].distance < CALIB_RATIO * pair[1].distance]

    if len(good) < MIN_INLIERS:
        return None

    src = np.float32([first_kp[match.queryIdx].pt for match in good])
    dst = np.float32([second_kp[match.trainIdx].pt for match in good])

    homography, mask = cv2.findHomography(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=CALIB_RANSAC_THRESHOLD,
        maxIters=CALIB_RANSAC_ITERS, confidence=CALIB_RANSAC_CONFIDENCE)

    if homography is None:
        return None

    # Маска приводится к плоскому виду: её форма зависит от версии OpenCV
    inliers = np.asarray(mask).ravel().astype(bool)
    if int(inliers.sum()) < MIN_INLIERS:
        return None

    projected = transform_points(homography, src[inliers])
    residuals = np.linalg.norm(projected - dst[inliers], axis=1)

    return {
        "homography": homography,
        "keypoints_first": len(first_kp),
        "keypoints_second": len(second_kp),
        "matches": len(good),
        "inlier_ratio": float(inliers.mean()),
        "residual": float(np.median(residuals)),
    }


def select_pairs(data: Dataset) -> list[tuple[int, int]]:
    """
    Отбирает пары кадров с наименьшим наклоном камеры по эталону.

    Пары берутся скользящим окном с шагом CALIB_STEP, поэтому перекрываются. Из
    них остаётся доля TILT_KEEP_FRACTION с наименьшим наклоном, а затем не более
    CALIB_PAIRS штук, распределённых равномерно по маршруту.

    Вращение вокруг оптической оси при отборе не учитывается: оно разворачивает
    кадр, но центр не сдвигает.

    Args:
        data: загруженный датасет.

    Returns:
        Список отобранных пар кадров.
    """
    all_pairs = [(index, index + CALIB_STEP)
                 for index in range(len(data) - CALIB_STEP)]
    if not all_pairs:
        return []

    tilts = np.array([tilt_angle(data.relative_motion(first, second)["rotation"])
                      for first, second in all_pairs])
    threshold = float(np.quantile(tilts, TILT_KEEP_FRACTION))
    kept = [pair for pair, tilt in zip(all_pairs, tilts) if tilt <= threshold]

    if len(kept) > CALIB_PAIRS:
        indices = np.linspace(0, len(kept) - 1, CALIB_PAIRS).astype(int)
        kept = [kept[index] for index in indices]

    return kept


def collect_observations(data: Dataset, chosen: list[tuple[int, int]],
                         label: str) -> list[dict[str, Any]]:
    """
    Собирает измерения по отобранным парам кадров.

    Пара пропускается, если её не удалось сопоставить, если сдвиг центра меньше
    MIN_PIXEL_SHIFT либо если эталонное перемещение нулевое.

    Args:
        data: загруженный датасет.
        chosen: пары кадров из select_pairs.
        label: пояснение для индикатора выполнения.

    Returns:
        Список наблюдений по одному на удавшуюся пару. Поля наблюдения:
        shift_norm, motion_px, altitude, distance, delta_world, delta_body,
        rotation_angle, inlier_ratio, residual, matches и keypoints.
    """
    if not chosen:
        return []

    detector = cv2.SIFT_create(nfeatures=CALIB_SIFT_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    width, height = data.image_size
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()

    for position, (first, second) in enumerate(chosen):
        print_progress(position, len(chosen), label)

        match = match_pair(detector, matcher, data.gray(first), data.gray(second))
        if match is None:
            continue

        shift = center_shift(match["homography"], width, height)
        shift_norm = float(np.linalg.norm(shift))
        if shift_norm < MIN_PIXEL_SHIFT:
            continue

        motion = data.relative_motion(first, second)
        if motion["distance"] <= 0:
            continue

        observations.append({
            "shift_norm": shift_norm,
            # Знак обращён: картинка едет навстречу камере, местность уплывает
            # назад, когда носитель летит вперёд. Поворот осей ищется по движению
            # камеры, чтобы измеренная константа применялась к выходу pose
            "motion_px": -shift,
            "altitude": float((data.altitudes[first] + data.altitudes[second]) / 2),
            "distance": motion["distance"],
            "delta_world": motion["delta_world"],
            "delta_body": motion["delta_body"],
            "rotation_angle": motion["rotation_angle"],
            "inlier_ratio": match["inlier_ratio"],
            "residual": match["residual"],
            "matches": match["matches"],
            "keypoints": (match["keypoints_first"] + match["keypoints_second"]) / 2,
        })

    clear_progress(label, len(chosen), time.perf_counter() - started, PROGRESS_FORMS)
    return observations


# ────────────────────────────────────────────────────────────────────────────
# Оценка фокусного расстояния
# ────────────────────────────────────────────────────────────────────────────

def project_shifts(observations: list[dict[str, Any]],
                   rotation: dict[str, Any]) -> None:
    """
    Добавляет к наблюдениям сдвиг, спроецированный на ожидаемое направление.

    Боковое покачивание камеры добавляет к истинному сдвигу перпендикулярную
    составляющую, отчего длина вектора сдвига смещена вверх. Направление, в
    котором камера должна была сместиться, известно из поворота осей, и проекция
    на него делает вклад покачивания симметричным вокруг нуля.

    Если поворот осей не определён, в поле shift_projected записывается длина
    вектора сдвига.

    Args:
        observations: результат collect_observations, изменяется на месте.
        rotation: результат fit_rotation.
    """
    if not rotation.get("ok"):
        for item in observations:
            item["shift_projected"] = item["shift_norm"]
        return

    # Матрица переводит направления кадра в направления эталона, обратное
    # преобразование для ортогональной матрицы это транспонирование
    plane = np.asarray(rotation["matrix"])[:2, :2]
    key = "delta_body" if rotation["reference"] == "body" else "delta_world"

    for item in observations:
        reference = np.asarray(item[key])[:2]
        norm = np.linalg.norm(reference)
        if norm == 0:
            item["shift_projected"] = item["shift_norm"]
            continue

        expected = plane.T @ (reference / norm)
        item["shift_projected"] = float(np.dot(item["motion_px"], expected))


def check_focal(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Сверяет измеренное фокусное расстояние с паспортным.

    Каждая пара кадров даёт собственную оценку: сдвиг картинки, умноженный на
    высоту и делённый на перемещение. Итоговой берётся медиана. Оценка ведётся
    по полю shift_projected, если оно посчитано, иначе по длине вектора сдвига.

    Args:
        observations: результат collect_observations.

    Returns:
        Словарь с полями measured_px, datasheet_px, deviation, agrees и
        ground_sample_m. Поле ok равно False, если наблюдений меньше
        MIN_OBSERVATIONS.
    """
    if len(observations) < MIN_OBSERVATIONS:
        return {"ok": False}

    distance = np.array([item["distance"] for item in observations])
    altitude = np.array([item["altitude"] for item in observations])
    shift = np.array([item.get("shift_projected", item["shift_norm"])
                      for item in observations])

    focals = shift * altitude / distance
    measured = float(np.median(focals))
    deviation = abs(measured - DATASHEET_FOCAL_PX) / DATASHEET_FOCAL_PX

    return {
        "ok": True,
        "measured_px": measured,
        "datasheet_px": DATASHEET_FOCAL_PX,
        "deviation": deviation,
        "agrees": deviation <= MAX_FOCAL_DEVIATION,
        # Размер участка земли, приходящийся на один пиксель кадра
        "ground_sample_m": float(np.median(altitude / focals)),
    }


def measure_altitude_drift(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Измеряет, насколько оценка фокусного расстояния зависит от высоты полёта.

    Фокусное расстояние камеры постоянно, поэтому оценки, полученные на разных
    высотах, обязаны совпадать. Расхождение служит мерой доверия к измерению.
    Величиной дрейфа берётся изменение оценки поперёк всего диапазона высот по
    линейной подгонке, в долях от самой оценки. Знак отбрасывается.

    Измерение выполняется, только если размах высот составляет не менее
    MIN_ALTITUDE_SPAN_FRACTION от высоты полёта.

    Args:
        observations: результат collect_observations.

    Returns:
        Словарь с полями measurable, altitude_span_m, altitude_span_fraction и
        drift. Поле drift равно None, если размаха высот не хватило. Поле ok
        равно False, если наблюдений меньше MIN_OBSERVATIONS.
    """
    if len(observations) < MIN_OBSERVATIONS:
        return {"ok": False}

    distance = np.array([item["distance"] for item in observations])
    altitude = np.array([item["altitude"] for item in observations])
    shift = np.array([item.get("shift_projected", item["shift_norm"])
                      for item in observations])
    focal = shift * altitude / distance

    span = float(altitude.max() - altitude.min())
    span_fraction = span / float(np.median(altitude))
    measurable = span_fraction >= MIN_ALTITUDE_SPAN_FRACTION

    result: dict[str, Any] = {
        "ok": True,
        "measurable": measurable,
        "altitude_span_m": span,
        "altitude_span_fraction": span_fraction,
        "drift": None,
    }

    if measurable:
        slope = float(np.polyfit(altitude, focal, 1)[0])
        result["drift"] = abs(slope) * span / float(np.median(focal))

    return result


# ────────────────────────────────────────────────────────────────────────────
# Поворот между системой камеры и системой эталона
# ────────────────────────────────────────────────────────────────────────────

def fit_plane_rotation(image_vectors: np.ndarray,
                       reference_vectors: np.ndarray) -> dict[str, Any]:
    """
    Находит плоский поворот, согласующий направления в кадре с эталонными.

    Задача решается как ортогональная задача Прокруста: ищется матрица 2x2,
    переводящая единичные векторы движения камеры в осях кадра в единичные
    векторы перемещения по эталону. Отражение не запрещается, поскольку ось y
    изображения направлена вниз.

    Args:
        image_vectors: массив формы (N, 2) со сдвигами, px.
        reference_vectors: массив формы (N, 2) с эталонными перемещениями.

    Returns:
        Словарь с полями matrix, determinant, angle в градусах, error_median,
        error_max и direction_spread.
    """
    image_unit = image_vectors / np.linalg.norm(image_vectors, axis=1, keepdims=True)
    reference_unit = (reference_vectors
                      / np.linalg.norm(reference_vectors, axis=1, keepdims=True))

    correlation = reference_unit.T @ image_unit
    left, _, right = np.linalg.svd(correlation)
    matrix = left @ right

    rotated = image_unit @ matrix.T
    cosines = np.clip(np.sum(rotated * reference_unit, axis=1), -1.0, 1.0)
    errors = np.degrees(np.arccos(cosines))

    # Угловой разброс эталонных направлений вокруг их среднего: он определяет,
    # различимы ли поворот и поворот с отражением
    headings = np.arctan2(reference_unit[:, 1], reference_unit[:, 0])
    mean_direction = np.arctan2(np.sin(headings).mean(), np.cos(headings).mean())
    relative = np.degrees(np.angle(np.exp(1j * (headings - mean_direction))))

    return {
        "matrix": matrix,
        "determinant": float(np.linalg.det(matrix)),
        "angle": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        "error_median": float(np.median(errors)),
        "error_max": float(np.max(errors)),
        "direction_spread": float(relative.max() - relative.min()),
    }


def fit_rotation(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Определяет поворот из системы камеры в систему эталона.

    Проверяются две гипотезы о том, с чем согласован кадр: с мировыми осями или
    с осями носителя. Верной считается та, что даёт меньшую невязку. Итоговая
    матрица 3x3 получается достройкой плоского поворота третьей осью, знак
    которой берётся равным определителю плоской части.

    В расчёт идёт доля ROTATION_KEEP_FRACTION наблюдений с наименьшим полным
    поворотом. Если после отбора их осталось меньше MIN_OBSERVATIONS, берутся
    все наблюдения.

    Args:
        observations: результат collect_observations.

    Returns:
        Словарь с полями reference, matrix, angle, determinant, error_median,
        error_max, error_world, error_body, direction_spread, spread_required,
        determined, rotation_threshold и used. Поле determined показывает,
        хватило ли разнообразия направлений. Поле ok равно False, если
        наблюдений меньше MIN_OBSERVATIONS.
    """
    result: dict[str, Any] = {"ok": False}
    if len(observations) < MIN_OBSERVATIONS:
        return result

    angles = np.array([item["rotation_angle"] for item in observations])
    threshold = float(np.quantile(angles, ROTATION_KEEP_FRACTION))
    steady = [item for item, angle in zip(observations, angles) if angle <= threshold]

    # Оценка на неполных данных надёжнее, чем её отсутствие
    if len(steady) < MIN_OBSERVATIONS:
        steady = observations
    observations = steady

    result["rotation_threshold"] = threshold
    result["used"] = len(observations)

    image_vectors = np.array([item["motion_px"] for item in observations], dtype=float)
    world_vectors = np.array([item["delta_world"][:2] for item in observations],
                             dtype=float)
    body_vectors = np.array([item["delta_body"][:2] for item in observations],
                            dtype=float)

    world_fit = fit_plane_rotation(image_vectors, world_vectors)
    body_fit = fit_plane_rotation(image_vectors, body_vectors)

    chosen_name = ("world" if world_fit["error_median"] <= body_fit["error_median"]
                   else "body")
    chosen = world_fit if chosen_name == "world" else body_fit

    matrix = np.eye(3)
    matrix[:2, :2] = chosen["matrix"]
    matrix[2, 2] = chosen["determinant"]

    spread_required = max(MIN_DIRECTION_SPREAD,
                          DIRECTION_SPREAD_FACTOR * chosen["error_median"])

    result.update({
        "ok": True,
        "reference": chosen_name,
        "matrix": matrix,
        "angle": chosen["angle"],
        "determinant": chosen["determinant"],
        "error_median": chosen["error_median"],
        "error_max": chosen["error_max"],
        "error_world": world_fit["error_median"],
        "error_body": body_fit["error_median"],
        "direction_spread": chosen["direction_spread"],
        "spread_required": spread_required,
        "determined": chosen["direction_spread"] >= spread_required,
    })
    return result


# ────────────────────────────────────────────────────────────────────────────
# Справочные величины
# ────────────────────────────────────────────────────────────────────────────

def estimate_noise_floor(data: Dataset) -> dict[str, Any]:
    """
    Оценивает шум эталонных позиций по отклонению от гладкой траектории.

    Вертолёт на коротком интервале не может двигаться рывками, поэтому истинная
    траектория близка к гладкой кривой. За шум эталона принимаются отклонения
    записанных позиций от локально подогнанного многочлена. Величина задаёт
    предел осмысленности сравнений между методами.

    Args:
        data: загруженный датасет.

    Returns:
        Словарь с полями rms в метрах и relative_to_step. Поле ok равно False,
        если кадров меньше NOISE_WINDOW плюс два.
    """
    if len(data) < NOISE_WINDOW + 2:
        return {"ok": False}

    easting = data.positions[:, 0]
    northing = data.positions[:, 1]

    smooth_e = signal.savgol_filter(easting, NOISE_WINDOW, NOISE_POLYORDER)
    smooth_n = signal.savgol_filter(northing, NOISE_WINDOW, NOISE_POLYORDER)
    residuals = np.hypot(easting - smooth_e, northing - smooth_n)

    # Края окна сглаживания менее надёжны и отбрасываются
    margin = NOISE_WINDOW
    if len(residuals) > 2 * margin:
        residuals = residuals[margin:-margin]

    step_median = float(np.median(np.hypot(np.diff(easting), np.diff(northing))))
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "ok": True,
        "rms": rms,
        "relative_to_step": rms / step_median,
    }


def summarize_planarity(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Обобщает, насколько хорошо гомография описывает сопоставленные точки.

    Гомография описывает движение точек, лежащих в одной плоскости. Чем выше
    доля согласных с ней точек, тем ближе сцена к плоской и тем хуже обусловлена
    задача восстановления позы через Essential Matrix.

    Args:
        observations: результат collect_observations.

    Returns:
        Словарь с полями inlier_ratio_median, inlier_ratio_min и
        residual_median. Поле ok равно False при пустом списке наблюдений.
    """
    if not observations:
        return {"ok": False}

    ratios = np.array([item["inlier_ratio"] for item in observations])
    residuals = np.array([item["residual"] for item in observations])

    return {
        "ok": True,
        "inlier_ratio_median": float(np.median(ratios)),
        "inlier_ratio_min": float(np.min(ratios)),
        "residual_median": float(np.median(residuals)),
    }


# ────────────────────────────────────────────────────────────────────────────
# Калибровка одного датасета
# ────────────────────────────────────────────────────────────────────────────

def calibrate(name: str) -> dict[str, Any]:
    """
    Выполняет все измерения по одному датасету.

    Args:
        name: имя датасета, ключ в config.DATASETS.

    Returns:
        Словарь с полями name, ok, error, observations, image_size,
        principal_point, rotation, focal, altitude, noise и planarity. При
        отказе заполняется error, а результаты измерений отсутствуют.
    """
    result: dict[str, Any] = {"name": name, "ok": False, "error": None}

    try:
        data = Dataset(name)
    except Exception as error:
        result["error"] = describe_exception(error)
        return result

    pairs = select_pairs(data)
    observations = collect_observations(data, pairs, f"{name}: сопоставление пар")

    if len(observations) < MIN_OBSERVATIONS:
        result["error"] = f"удалось сопоставить только {len(observations)} пар"
        return result

    # Поворот осей считается первым: он нужен, чтобы спроецировать сдвиги на
    # ожидаемое направление и убрать вклад бокового покачивания
    rotation = fit_rotation(observations)
    project_shifts(observations, rotation)

    width, height = data.image_size

    result.update({
        "ok": True,
        "observations": len(observations),
        "image_size": (width, height),
        # Главная точка принимается в центре кадра: её смещение определяется по
        # заметному изменению направления движения, которого на маршруте нет
        "principal_point": ((width - 1) / 2.0, (height - 1) / 2.0),
        "rotation": rotation,
        "focal": check_focal(observations),
        "altitude": measure_altitude_drift(observations),
        "noise": estimate_noise_floor(data),
        "planarity": summarize_planarity(observations),
    })
    return result


# ────────────────────────────────────────────────────────────────────────────
# Сведение оценок к единым константам
# ────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Сводит оценки по датасетам к одному набору констант.

    Камера и её крепление одни и те же во всех датасетах, поэтому раздельные
    оценки служат повторными измерениями одной величины.

    Фокусным расстоянием принимается паспортное значение: измерения расходятся
    между маршрутами сильнее, чем их медиана расходится с паспортом. По той же
    причине расхождение с паспортом отдаётся диапазоном по маршрутам, а не одним
    числом по медиане.

    Поворот берётся только по тем датасетам, где направления движения достаточно
    разнообразны. На прямолинейном участке задача вырождена, и решение выпадает
    произвольно, вплоть до зеркального отражения.

    Args:
        results: результаты calibrate по каждому датасету.

    Returns:
        Словарь с принятыми константами, разбросом оценок и списками датасетов,
        по которым они получены. Поле ok равно False, если ни один датасет не
        дал оценки фокусного расстояния.
    """
    usable = [item for item in results if item["ok"] and item["focal"].get("ok")]
    aggregated: dict[str, Any] = {"ok": False}
    if not usable:
        return aggregated

    measured = np.array([item["focal"]["measured_px"] for item in usable])

    # Дрейф оценки по высоте: мера доверия к измерению, а не к колонке altitude.
    # Собирается по тем маршрутам, где высота менялась достаточно
    drifts = [item["altitude"]["drift"] for item in usable
              if item["altitude"].get("measurable")]

    aggregated.update({
        "ok": True,
        "focal_px": DATASHEET_FOCAL_PX,
        "measured_median": float(np.median(measured)),
        "measured_min": float(measured.min()),
        "measured_max": float(measured.max()),
        "measured_spread": float((measured.max() - measured.min())
                                 / np.median(measured)),
        "deviation_min": float(min(item["focal"]["deviation"] for item in usable)),
        "deviation_max": float(max(item["focal"]["deviation"] for item in usable)),
        "agrees": all(item["focal"]["agrees"] for item in usable),
        "focal_sources": [item["name"] for item in usable],
        "altitude_mode": ALTITUDE_MODE,
        "ground_elevation_m": GROUND_ELEVATION_M,
        "drift_min": min(drifts) if drifts else None,
        "drift_max": max(drifts) if drifts else None,
    })

    determined = [item for item in usable
                  if item["rotation"].get("ok") and item["rotation"].get("determined")]
    # Отбор по имени, а не по самому словарю: словари содержат массивы numpy,
    # и сравнение таких словарей между собой неоднозначно
    determined_names = {item["name"] for item in determined}

    aggregated["rotation_sources"] = [item["name"] for item in determined]
    aggregated["rotation_rejected"] = [item["name"] for item in usable
                                       if item["name"] not in determined_names]

    if determined:
        angles = np.array([item["rotation"]["angle"] for item in determined])
        determinants = np.array([item["rotation"]["determinant"]
                                 for item in determined])

        angle = float(np.median(angles))
        determinant = float(np.sign(np.median(determinants)))
        radians = np.deg2rad(angle)

        matrix = np.eye(3)
        matrix[:2, :2] = [[np.cos(radians), -np.sin(radians)],
                          [np.sin(radians), np.cos(radians)]]
        # Отражение вносится сменой знака второго столбца, чтобы определитель
        # плоской части совпал с измеренным
        if determinant < 0:
            matrix[:2, 1] *= -1
        matrix[2, 2] = determinant

        aggregated.update({
            "rotation_angle": angle,
            "rotation_spread": float(angles.max() - angles.min()),
            "rotation_matrix": matrix,
        })
    else:
        aggregated.update({"rotation_angle": None, "rotation_matrix": None})

    return aggregated


# ────────────────────────────────────────────────────────────────────────────
# Отчёт об измерениях
# ────────────────────────────────────────────────────────────────────────────

def report_summary(results: list[dict[str, Any]]) -> None:
    """
    Печатает таблицу измерений по датасетам.

    Args:
        results: результаты calibrate по каждому датасету.
    """
    headers = ["Датасет", "Пар", "f, px", "Δ, %", "м/px", "Плоск.%",
               "Поворот", "Невязка", "Шум, м"]
    rows: list[list[Any]] = []

    for result in results:
        if not result["ok"] or not result["focal"].get("ok"):
            rows.append([result["name"], MISSING, "", "", "", "", "", "", ""])
            continue

        focal = result["focal"]
        rotation = result["rotation"]
        noise = result["noise"]

        rows.append([
            result["name"],
            result["observations"],
            format_number(focal["measured_px"], 0),
            format_number(focal["deviation"] * 100, 1),
            format_number(focal["ground_sample_m"], 3),
            format_number(result["planarity"]["inlier_ratio_median"] * 100, 1),
            format_number(rotation.get("angle"), 1) if rotation.get("ok") else "",
            format_number(rotation.get("error_median"), 1) if rotation.get("ok") else "",
            format_number(noise.get("rms"), 3) if noise.get("ok") else "",
        ])

    print_section("ИЗМЕРЕНИЯ ПО ДАТАСЕТАМ")
    print()
    print_table(headers, rows)
    print()
    print_legend([
        ("f", "фокусное расстояние, измеренное по кадрам"),
        ("Δ", "его отличие от паспортного"),
        ("м/px", "участок земли, приходящийся на один пиксель кадра"),
        ("Плоск.%", "доля точек, согласных с гомографией: выше значит ровнее сцена"),
        ("Поворот", "угол между осями камеры и эталона, град"),
        ("Невязка", "ошибка направления при поиске этого угла, град"),
        ("Шум", "разброс эталонных позиций вокруг гладкой траектории"),
    ])


def report_adopted(aggregated: dict[str, Any]) -> None:
    """
    Печатает блок с принятыми константами и их неопределённостью.

    Args:
        aggregated: результат aggregate.
    """
    open_block("ПРИНЯТЫЕ КОНСТАНТЫ", "общие для всех датасетов")

    if not aggregated.get("ok"):
        block_line("Состояние", "измерений не хватило", STATUS_FAIL)
        close_block()
        return

    block_line("Фокусное расстояние",
               f"{format_number(aggregated['focal_px'], 0)} px, паспортное")
    block_line("Измерено по кадрам",
               f"{format_number(aggregated['measured_min'], 0)} ... "
               f"{format_number(aggregated['measured_max'], 0)} px, медиана "
               f"{format_number(aggregated['measured_median'], 0)}, "
               f"разброс {format_number(aggregated['measured_spread'] * 100, 0)} %")
    block_line("Расхождение с паспортом",
               f"{format_number(aggregated['deviation_min'] * 100, 1)} ... "
               f"{format_number(aggregated['deviation_max'] * 100, 1)} % "
               f"по маршрутам",
               STATUS_OK if aggregated["agrees"] else STATUS_FAIL)

    if aggregated.get("drift_max") is not None:
        block_line("Дрейф оценки по высоте",
                   f"{format_number(aggregated['drift_min'] * 100, 0)} ... "
                   f"{format_number(aggregated['drift_max'] * 100, 0)} %, "
                   f"причина не установлена", STATUS_NONE)

    block_line("Смысл колонки altitude", "над землёй, из описания датасета")

    if aggregated.get("rotation_angle") is not None:
        block_line("Поворот камеры к эталону",
                   f"{format_number(aggregated['rotation_angle'], 2)} град, "
                   f"расхождение оценок "
                   f"{format_number(aggregated['rotation_spread'], 2)} град")
        if aggregated["rotation_rejected"]:
            block_wrapped("Исключены из оценки",
                          ", ".join(aggregated["rotation_rejected"])
                          + ": направления движения слишком однородны")
    else:
        block_line("Поворот камеры к эталону", "не определён", STATUS_FAIL)

    block_note("Кадры имеют размер 500 на 500, тогда как камера снимает 1600 на "
               "1200. Вырезка из центра сохраняет фокусное расстояние, сжатие "
               "уменьшило бы его втрое: измерение согласуется с паспортным, "
               "значит кадры вырезаны без сжатия.")

    block_note("Принимается паспортное значение, а не медиана измерений: "
               "разброс между маршрутами превышает расхождение с паспортом. "
               "Множитель масштаба при этом общий для всех сравниваемых "
               "методов и при сравнении между собой сокращается.")

    close_block()


def report_config_values(results: list[dict[str, Any]],
                         aggregated: dict[str, Any]) -> None:
    """
    Печатает измеренные значения в виде, готовом к переносу в config.

    Фокусное расстояние, поворот осей и режим отсчёта высоты одинаковы у всех
    датасетов, главная точка печатается по размеру кадра каждого из них.

    Фокусное печатается ссылкой на константы камеры из config, а не числом:
    частное бесконечно в десятичной записи, и округление оторвало бы константу
    от её источника.

    Args:
        results: результаты calibrate по каждому датасету.
        aggregated: результат aggregate.
    """
    print_section("ЗНАЧЕНИЯ ДЛЯ CONFIG.CALIBRATION")
    print()

    if not aggregated.get("ok"):
        print("    измерение не выполнено")
        print()
        return

    matrix = aggregated.get("rotation_matrix")

    for result in results:
        if not result["ok"] or not result["focal"].get("ok"):
            print(f'    "{result["name"]}": измерение не выполнено')
            print()
            continue

        point = result["principal_point"]

        print(f'    "{result["name"]}": {{')
        print(f'        "focal_px": LENS_FOCAL_MM / PIXEL_PITCH_MM')
        print(f'        "principal_point": ({point[0]:.1f}, {point[1]:.1f}),')
        print(f'        "altitude_mode": "{aggregated["altitude_mode"]}",')
        print(f'        "ground_elevation_m": '
              f'{aggregated["ground_elevation_m"]:.1f},')

        # Матрица печатается построчно, ровно как лежит в конфиге: перенос
        # сводится к копированию, и строки не выходят за ширину файла
        if matrix is None:
            print('        "rotation_cam_to_gt": None,')
        else:
            print('        "rotation_cam_to_gt": [')
            for row in matrix:
                print("            ["
                      + ", ".join(f"{value:.6f}" for value in row) + "],")
            print("        ],")

        print("    },")
        print()


# ────────────────────────────────────────────────────────────────────────────
# Точка входа
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Калибрует все датасеты из конфига и печатает отчёт.
    """
    print_banner([
        "КАЛИБРОВКА ДАТАСЕТОВ",
        "Проверка фокусного расстояния и измерение поворота осей",
    ])

    results: list[dict[str, Any]] = []
    for name in config.DATASETS:
        try:
            result = calibrate(name)
        except Exception as error:
            result = {"name": name, "ok": False, "error": describe_exception(error)}
        results.append(result)

    aggregated = aggregate(results)
    report_summary(results)
    report_adopted(aggregated)
    report_config_values(results, aggregated)

    failed = [result["name"] for result in results if not result["ok"]]
    lines = [f"ИТОГ: измерено датасетов {len(results) - len(failed)} из {len(results)}"]
    if failed:
        lines.append("Не измерено: " + ", ".join(failed))
    print_banner(lines)

    print()


if __name__ == "__main__":
    main()
