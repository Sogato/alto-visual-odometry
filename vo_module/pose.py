"""
Восстановление движения камеры по сопоставленным точкам.

Модуль составляет третий слой пайплайна: от пар соответствующих точек к
повороту и сдвигу камеры между кадрами.

Essential Matrix описывает движение камеры в произвольной трёхмерной сцене и
опирается на параллакс, то есть на разное смещение близких и далёких точек. На
плоской сцене параллакса нет, и задача вырождается, что для съёмки с высоты
вниз существенно. Длину сдвига эта модель не даёт: одна и та же картина
получается при вдвое большем сдвиге и вдвое дальше расположенной сцене.

Гомография описывает движение точек одной плоскости, поэтому на плоской сцене
задача не вырождена. Её разложение даёт до четырёх решений, из которых нужное
выбирается по наклону нормали, и сдвиг получается делённым на расстояние до
плоскости, так что известная высота полёта сразу превращает его в метры.

Функции восстановления позы возвращают словарь одной формы и не бросают
исключений при неудаче: не сработавшая оценка описывается результатом
empty_pose с полем ok, равным False.

Движение отдаётся в едином соглашении: перемещение самой камеры от первого
кадра ко второму, выраженное в осях первого кадра. OpenCV отдаёт обратное
преобразование, перевод делает to_camera_motion.

Готовые модели перечислены в реестрах POSE_BY_KEY и EXTRA_POSE_BY_KEY.
"""

# Стандартные библиотеки
import sys
from pathlib import Path
from typing import Any, Callable

# Сторонние библиотеки
import cv2
import numpy as np

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


# === СООТВЕТСТВИЕ ИМЁН МЕТОДОВ РОБАСТНОЙ ОЦЕНКИ ===
# Метод хранится в конфиге строкой, а не константой OpenCV: так его видно в
# отчёте и не приходится импортировать cv2 ради чтения настроек
ROBUST_METHOD_CODES: dict[str, int] = {
    "RANSAC": cv2.RANSAC,
    "LMEDS": cv2.LMEDS,
    "RHO": cv2.RHO,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_PROSAC": cv2.USAC_PROSAC,
}


# ────────────────────────────────────────────────────────────────────────────
# Вспомогательное
# ────────────────────────────────────────────────────────────────────────────

def camera_matrix(focal: float,
                  principal_point: tuple[float, float]) -> np.ndarray:
    """
    Собирает матрицу внутренней калибровки камеры.

    Матрица переводит направления в трёхмерном пространстве в координаты на
    снимке. Масштаб по обеим осям считается одинаковым: разделить их можно
    только при движении в разных направлениях относительно кадра, а маршрут
    почти прямолинеен.

    Args:
        focal: фокусное расстояние, px.
        principal_point: координаты главной точки, px.

    Returns:
        Матрицу 3x3.
    """
    return np.array([
        [focal, 0.0, principal_point[0]],
        [0.0, focal, principal_point[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def robust_method(name: str) -> int:
    """
    Переводит название метода робастной оценки в код OpenCV.

    Args:
        name: название из ROBUST_METHOD_CODES.

    Returns:
        Код метода для передачи в OpenCV.

    Raises:
        KeyError: метод с таким названием не поддерживается.
    """
    if name not in ROBUST_METHOD_CODES:
        raise KeyError(f"неизвестный метод робастной оценки: {name}")
    return ROBUST_METHOD_CODES[name]


def flat_mask(mask: Any, size: int) -> np.ndarray:
    """
    Приводит маску согласных точек к плоскому логическому массиву.

    Форма маски зависит от версии OpenCV, поэтому опираться на неё нельзя.

    Args:
        mask: маска, возвращённая OpenCV, либо None.
        size: сколько точек подавалось на вход.

    Returns:
        Логический массив длиной size. При отсутствии маски все точки
        считаются согласными.
    """
    if mask is None:
        return np.ones(size, dtype=bool)
    return np.asarray(mask).ravel().astype(bool)


def to_camera_motion(rotation: np.ndarray,
                     translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Переводит результат OpenCV в движение камеры в осях первого кадра.

    OpenCV возвращает преобразование координат из первого кадра во второй:
    точка сцены, записанная в осях первой камеры, переходит в оси второй. Для
    накопления траектории нужно обратное, то есть куда сместилась и как
    повернулась сама камера. Поворот заменяется транспонированным, сдвиг
    переносится в оси первого кадра со сменой знака.

    Args:
        rotation: матрица поворота из OpenCV.
        translation: вектор сдвига из OpenCV.

    Returns:
        Пару из матрицы поворота камеры и вектора её сдвига.
    """
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float).ravel()
    return rotation.T, -(rotation.T @ translation)


def empty_pose(reason: str) -> dict[str, Any]:
    """
    Возвращает результат неудавшейся оценки позы.

    Набор полей урезан по сравнению с успешным результатом: matrix, mask,
    normal и normal_alignment отсутствуют. Общие поля есть всегда, поэтому
    вызывающий код читает их без проверок, а остальные через get.

    Args:
        reason: почему оценка не выполнена.

    Returns:
        Словарь с полем ok, равным False, полем reason и обнулёнными
        показателями.
    """
    return {
        "ok": False,
        "reason": reason,
        "rotation": None,
        "translation": None,
        "translation_over_distance": None,
        "inliers": 0,
        "total": 0,
        "inlier_ratio": 0.0,
        "residual": float("nan"),
    }


def find_homography(points_first: np.ndarray, points_second: np.ndarray,
                    method: str | None, threshold: float | None) -> tuple:
    """
    Ищет гомографию робастной оценкой с настройками из конфига.

    Args:
        points_first: координаты точек первого кадра, массив (N, 2).
        points_second: координаты соответствующих точек второго кадра.
        method: название метода робастной оценки либо None для значения из
            конфига.
        threshold: порог невязки перепроекции в пикселях либо None для
            значения из конфига.

    Returns:
        Пару из матрицы 3x3 и маски согласных точек. Матрица равна None, если
        решение не найдено.

    Raises:
        KeyError: метод с таким названием не поддерживается.
    """
    return cv2.findHomography(
        points_first, points_second,
        method=robust_method(method or config.RANSAC_METHOD),
        ransacReprojThreshold=(threshold if threshold is not None
                               else config.RANSAC_THRESHOLD_HOMOGRAPHY),
        maxIters=config.RANSAC_MAX_ITERS,
        confidence=config.RANSAC_CONFIDENCE)


def reprojection_residual(homography: np.ndarray, points_first: np.ndarray,
                          points_second: np.ndarray) -> float:
    """
    Считает медианную ошибку перепроекции точек через гомографию.

    Args:
        homography: матрица 3x3.
        points_first: точки первого кадра.
        points_second: точки второго кадра.

    Returns:
        Медианную невязку, px. NaN, если точек нет.
    """
    if len(points_first) == 0:
        return float("nan")
    projected = apply_homography(homography, points_first)
    return float(np.median(np.linalg.norm(projected - points_second, axis=1)))


def apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
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


# ────────────────────────────────────────────────────────────────────────────
# Essential Matrix
# ────────────────────────────────────────────────────────────────────────────

def pose_from_essential(points_first: np.ndarray, points_second: np.ndarray,
                        intrinsics: np.ndarray, method: str | None = None,
                        threshold: float | None = None) -> dict[str, Any]:
    """
    Восстанавливает поворот и направление сдвига через Essential Matrix.

    Матрица связывает соответствующие точки двух кадров условием, что точка, её
    образ на втором кадре и линия между центрами камер лежат в одной плоскости.
    Разложение даёт четыре сочетания поворота и сдвига, из которых OpenCV
    выбирает то, при котором точки оказываются перед обеими камерами.

    Args:
        points_first: координаты точек первого кадра, массив (N, 2).
        points_second: координаты соответствующих точек второго кадра.
        intrinsics: матрица внутренней калибровки.
        method: название метода робастной оценки. По умолчанию из конфига.
        threshold: порог расстояния до эпиполярной линии, px. По умолчанию из
            конфига.

    Returns:
        Словарь с полями ok, reason, rotation, translation в виде единичного
        вектора направления, translation_over_distance со значением None,
        inliers, total, inlier_ratio, residual, matrix и mask. При неудаче
        результат empty_pose.

    Raises:
        KeyError: метод с таким названием не поддерживается.
    """
    total = len(points_first)
    if total < config.MIN_MATCHES_FOR_GEOMETRY:
        return empty_pose(f"мало сопоставлений: {total}")

    essential, mask = cv2.findEssentialMat(
        points_first, points_second, cameraMatrix=intrinsics,
        method=robust_method(method or config.RANSAC_METHOD),
        prob=config.RANSAC_CONFIDENCE,
        threshold=(threshold if threshold is not None
                   else config.RANSAC_THRESHOLD_ESSENTIAL),
        maxIters=config.RANSAC_MAX_ITERS)

    if essential is None or essential.shape != (3, 3):
        return empty_pose("матрица не найдена")

    inliers = flat_mask(mask, total)
    if int(inliers.sum()) < config.MIN_MATCHES_FOR_GEOMETRY:
        return empty_pose(f"мало согласных точек: {int(inliers.sum())}")

    # Маска передаётся дальше: recoverPose уточняет её, оставляя точки, которые
    # оказались перед обеими камерами
    passed, rotation, translation, pose_mask = cv2.recoverPose(
        essential, points_first, points_second, cameraMatrix=intrinsics,
        mask=None if mask is None else np.asarray(mask).copy())

    if passed < 1:
        return empty_pose("поза не восстановлена")

    accepted = flat_mask(pose_mask, total)
    residual = epipolar_residual(points_first[accepted], points_second[accepted],
                                 essential, intrinsics)

    motion_rotation, motion_translation = to_camera_motion(rotation, translation)

    return {
        "ok": True,
        "reason": "",
        "rotation": motion_rotation,
        "translation": motion_translation,
        # Масштаб через Essential Matrix недоступен принципиально
        "translation_over_distance": None,
        "inliers": int(accepted.sum()),
        "total": total,
        "inlier_ratio": float(accepted.mean()),
        "residual": residual,
        "matrix": essential,
        "mask": accepted,
    }


def epipolar_residual(points_first: np.ndarray, points_second: np.ndarray,
                      essential: np.ndarray, intrinsics: np.ndarray) -> float:
    """
    Считает медианное расстояние от точек до их эпиполярных линий.

    Каждой точке первого кадра соответствует на втором не точка, а прямая, на
    которой она обязана лежать. Отклонение от этой прямой и есть невязка
    модели.

    Args:
        points_first: точки первого кадра.
        points_second: точки второго кадра.
        essential: матрица Essential.
        intrinsics: матрица внутренней калибровки.

    Returns:
        Медианную невязку, px. NaN, если точек нет.
    """
    if len(points_first) == 0:
        return float("nan")

    # Переход к фундаментальной матрице: она работает в пикселях, а Essential
    # в нормированных координатах камеры
    inverse = np.linalg.inv(intrinsics)
    fundamental = inverse.T @ essential @ inverse

    first = np.hstack([points_first, np.ones((len(points_first), 1))])
    second = np.hstack([points_second, np.ones((len(points_second), 1))])

    lines = first @ fundamental.T
    norms = np.hypot(lines[:, 0], lines[:, 1])
    norms[norms == 0] = np.finfo(float).eps

    distances = np.abs(np.sum(lines * second, axis=1)) / norms
    return float(np.median(distances))


# ────────────────────────────────────────────────────────────────────────────
# Гомография
# ────────────────────────────────────────────────────────────────────────────

def pose_from_homography(points_first: np.ndarray, points_second: np.ndarray,
                         intrinsics: np.ndarray, method: str | None = None,
                         threshold: float | None = None) -> dict[str, Any]:
    """
    Восстанавливает поворот и сдвиг через гомографию плоской сцены.

    Гомография связывает точки одной плоскости на двух кадрах. Её разложение
    даёт до четырёх сочетаний поворота, сдвига и нормали плоскости, выбор среди
    которых делает select_homography_solution.

    Сдвиг возвращается делённым на расстояние до плоскости, поэтому умножение
    на высоту полёта даёт перемещение в метрах.

    Args:
        points_first: координаты точек первого кадра, массив (N, 2).
        points_second: координаты соответствующих точек второго кадра.
        intrinsics: матрица внутренней калибровки.
        method: название метода робастной оценки. По умолчанию из конфига.
        threshold: порог невязки перепроекции, px. По умолчанию из конфига.

    Returns:
        Словарь с полями ok, reason, rotation, translation в виде единичного
        вектора направления, translation_over_distance, normal,
        normal_alignment, inliers, total, inlier_ratio, residual, matrix и
        mask. При неудаче результат empty_pose.

    Raises:
        KeyError: метод с таким названием не поддерживается.
    """
    total = len(points_first)
    if total < config.MIN_MATCHES_FOR_GEOMETRY:
        return empty_pose(f"мало сопоставлений: {total}")

    homography, mask = find_homography(points_first, points_second,
                                       method, threshold)
    if homography is None:
        return empty_pose("матрица не найдена")

    inliers = flat_mask(mask, total)
    if int(inliers.sum()) < config.MIN_MATCHES_FOR_GEOMETRY:
        return empty_pose(f"мало согласных точек: {int(inliers.sum())}")

    solution = select_homography_solution(homography, intrinsics)
    if solution is None:
        return empty_pose("разложение не дало подходящего решения")

    residual = reprojection_residual(homography, points_first[inliers],
                                     points_second[inliers])

    motion_rotation, motion_translation = to_camera_motion(
        solution["rotation"], solution["translation"])
    norm = np.linalg.norm(motion_translation)

    return {
        "ok": True,
        "reason": "",
        "rotation": motion_rotation,
        "translation": (motion_translation / norm if norm > 0
                        else motion_translation),
        # Разложение даёт сдвиг в долях расстояния до плоскости, то есть высоты
        "translation_over_distance": motion_translation,
        "normal": solution["normal"],
        "normal_alignment": solution["alignment"],
        "inliers": int(inliers.sum()),
        "total": total,
        "inlier_ratio": float(inliers.mean()),
        "residual": residual,
        "matrix": homography,
        "mask": inliers,
    }


def select_homography_solution(homography: np.ndarray,
                               intrinsics: np.ndarray) -> dict[str, Any] | None:
    """
    Выбирает одно из решений разложения гомографии.

    Разложение даёт до четырёх математически равноправных вариантов. Берётся
    тот, у которого нормаль плоскости сильнее прижата к оптической оси: камера
    снимает землю сверху, а варианты с почти горизонтальной нормалью описывают
    вертикально стоящую плоскость. Отбор OpenCV по видимости точек не
    применяется, поэтому сами точки функции не нужны.

    Пара из нормали и сдвига определена с точностью до общего знака, поэтому
    при отрицательной третьей составляющей нормали разворачиваются обе.

    Args:
        homography: матрица гомографии.
        intrinsics: матрица внутренней калибровки.

    Returns:
        Словарь с полями rotation, translation, normal и alignment, где
        alignment это третья составляющая нормали. None, если разложение не
        дало ни одного варианта.
    """
    count, rotations, translations, normals = cv2.decomposeHomographyMat(
        homography, intrinsics)
    if count == 0:
        return None

    best: dict[str, Any] | None = None
    best_alignment = -np.inf

    for index in range(count):
        normal = np.asarray(normals[index], dtype=float).ravel()

        sign = -1.0 if normal[2] < 0 else 1.0
        aligned_normal = normal * sign
        alignment = aligned_normal[2]

        if alignment <= best_alignment:
            continue

        best_alignment = alignment
        best = {
            "rotation": np.asarray(rotations[index], dtype=float),
            "translation": (np.asarray(translations[index], dtype=float).ravel()
                            * sign),
            "normal": aligned_normal,
            "alignment": alignment,
        }

    return best


def pose_from_homography_fixed_normal(
        points_first: np.ndarray, points_second: np.ndarray,
        intrinsics: np.ndarray, method: str | None = None,
        threshold: float | None = None) -> dict[str, Any]:
    """
    Восстанавливает движение из гомографии при заданной нормали плоскости.

    Общее разложение ищет одновременно поворот, сдвиг и наклон наблюдаемой
    плоскости. На съёмке земли сверху эти величины плохо разделяются: сдвиг
    картинки одинаково объясняется и перемещением камеры, и её наклоном, и
    уклоном местности.

    Здесь наклон не ищется, а задаётся: камера смотрит вниз, значит нормаль
    земли совпадает с оптической осью. Тогда матрица в координатах камеры
    распадается на сумму поворота и сдвига, размазанного по третьему столбцу.
    Первые два столбца оказываются столбцами матрицы поворота, третий столбец
    поворота достраивается векторным произведением, а остаток третьего столбца
    и есть сдвиг, делённый на высоту. Неизвестных остаётся две вместо трёх, и
    вариантов решения не возникает.

    Args:
        points_first: координаты точек первого кадра, массив (N, 2).
        points_second: координаты соответствующих точек второго кадра.
        intrinsics: матрица внутренней калибровки.
        method: название метода робастной оценки. По умолчанию из конфига.
        threshold: порог невязки перепроекции, px. По умолчанию из конфига.

    Returns:
        Словарь того же состава, что и у pose_from_homography. Поле normal
        всегда равно оптической оси, normal_alignment равно единице.

    Raises:
        KeyError: метод с таким названием не поддерживается.
    """
    total = len(points_first)
    if total < config.MIN_MATCHES_FOR_GEOMETRY:
        return empty_pose(f"мало сопоставлений: {total}")

    homography, mask = find_homography(points_first, points_second,
                                       method, threshold)
    if homography is None:
        return empty_pose("матрица не найдена")

    inliers = flat_mask(mask, total)
    if int(inliers.sum()) < config.MIN_MATCHES_FOR_GEOMETRY:
        return empty_pose(f"мало согласных точек: {int(inliers.sum())}")

    # Переход к координатам камеры: там гомография равна сумме поворота и
    # сдвига, умноженного на нормаль
    normalized = np.linalg.inv(intrinsics) @ np.asarray(homography) @ intrinsics

    # Матрица определена с точностью до множителя. Первые два столбца обязаны
    # быть единичными, поскольку это столбцы поворота, отсюда и находится
    # множитель
    scale = np.sqrt(np.linalg.norm(normalized[:, 0])
                    * np.linalg.norm(normalized[:, 1]))
    if scale == 0:
        return empty_pose("вырожденная матрица")
    normalized = normalized / scale

    # Знак тоже неоднозначен. Камера смотрит на землю почти прямо, значит
    # третий диагональный элемент поворота близок к единице
    if normalized[2, 2] < 0:
        normalized = -normalized

    # Ближайший к первым двум столбцам поворот: достраивается третий столбец,
    # затем тройка приводится к строго ортогональному виду
    first_axis = normalized[:, 0]
    second_axis = normalized[:, 1]
    third_axis = np.cross(first_axis, second_axis)
    approximate = np.column_stack([first_axis, second_axis, third_axis])

    left, _, right = np.linalg.svd(approximate)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(left @ right)))])
    rotation = left @ correction @ right

    # Остаток третьего столбца сверх поворота и есть сдвиг, выраженный в долях
    # расстояния до плоскости
    translation_over_distance = normalized[:, 2] - rotation[:, 2]

    residual = reprojection_residual(homography, points_first[inliers],
                                     points_second[inliers])

    motion_rotation, motion_translation = to_camera_motion(
        rotation, translation_over_distance)
    norm = np.linalg.norm(motion_translation)

    return {
        "ok": True,
        "reason": "",
        "rotation": motion_rotation,
        "translation": (motion_translation / norm if norm > 0
                        else motion_translation),
        "translation_over_distance": motion_translation,
        "normal": np.array([0.0, 0.0, 1.0]),
        "normal_alignment": 1.0,
        "inliers": int(inliers.sum()),
        "total": total,
        "inlier_ratio": float(inliers.mean()),
        "residual": residual,
        "matrix": homography,
        "mask": inliers,
    }


# ────────────────────────────────────────────────────────────────────────────
# Реестры
# ────────────────────────────────────────────────────────────────────────────

# Модели по ключу поля key в config.GEOMETRIES
POSE_BY_KEY: dict[str, Callable[..., dict[str, Any]]] = {
    "essential": pose_from_essential,
    "homography": pose_from_homography,
}

# Вариант с заданной нормалью в основной реестр не входит: он не относится к
# трём подходам из задания и разбирается отдельным экспериментом в анализе
# геометрии, где сравниваются способы извлечь позу из одной и той же матрицы
EXTRA_POSE_BY_KEY: dict[str, Callable[..., dict[str, Any]]] = {
    "homography_fixed": pose_from_homography_fixed_normal,
}
