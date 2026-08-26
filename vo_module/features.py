"""
Детекторы ключевых точек и матчеры с единым интерфейсом.

Модуль собирает первые два слоя пайплайна: поиск точек на кадре и связывание
точек двух кадров. Реализации разные по природе, от классических дескрипторов
до нейросетей, но наружу все выглядят одинаково.

Детектор принимает кадр и возвращает словарь с координатами точек, их
описаниями и метрикой сравнения описаний. Матчер принимает результаты детектора
по обоим кадрам, сами кадры и возвращает пары сопоставленных координат. Кадры
нужны только оптическому потоку, остальные матчеры их не читают.

Оптический поток Лукаса и Канаде дескрипторов не считает вовсе: точки первого
кадра он ищет на втором, обращаясь к пикселям. Детекция на втором кадре ему не
нужна, и это отмечено в реестре MATCHER_NEEDS_SECOND, по которому пайплайн
решает, запускать ли детектор второй раз.

Готовые детекторы и матчеры перечислены в реестрах DETECTORS и MATCHERS, откуда
их берёт пайплайн по ключам из config.FRONTENDS.
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


# === КЕШ ТЯЖЁЛЫХ ОБЪЕКТОВ ===
# Детекторы OpenCV и нейросетевые модели создаются заметно дольше, чем работают
# на одном кадре. Создание внутри цикла исказило бы замеры времени. Ключ это имя
# детектора, матчера либо метрика сравнения дескрипторов
_CACHE: dict[str, Any] = {}


# ────────────────────────────────────────────────────────────────────────────
# Детекторы
# ────────────────────────────────────────────────────────────────────────────

def detect_sift(gray: np.ndarray) -> dict[str, Any]:
    """
    Находит точки детектором SIFT и считает их дескрипторы.

    Args:
        gray: кадр в градациях серого.

    Returns:
        Результат детектора: points, descriptors, norm, raw и count.
    """
    if "sift" not in _CACHE:
        _CACHE["sift"] = cv2.SIFT_create(**config.SIFT_PARAMS)

    keypoints, descriptors = _CACHE["sift"].detectAndCompute(gray, None)
    return pack_opencv_result(keypoints, descriptors, cv2.NORM_L2)


def detect_orb(gray: np.ndarray) -> dict[str, Any]:
    """
    Находит точки детектором ORB и считает их дескрипторы.

    Args:
        gray: кадр в градациях серого.

    Returns:
        Результат детектора: points, descriptors, norm, raw и count.
    """
    if "orb" not in _CACHE:
        _CACHE["orb"] = cv2.ORB_create(**config.ORB_PARAMS)

    keypoints, descriptors = _CACHE["orb"].detectAndCompute(gray, None)
    # Дескрипторы ORB двоичные, поэтому сравниваются расстоянием Хэмминга,
    # то есть числом несовпавших битов
    return pack_opencv_result(keypoints, descriptors, cv2.NORM_HAMMING)


def detect_superpoint(gray: np.ndarray) -> dict[str, Any]:
    """
    Находит точки нейросетью SuperPoint.

    Точки и их описания выдаёт одна сеть, обученная так, чтобы находимые точки
    были устойчивы к смене ракурса и освещения.

    Поле descriptors остаётся пустым, а описания лежат внутри raw, откуда их
    берёт LightGlue. В реестре связок SuperPoint идёт только с ним, и
    выкладывание означало бы перенос массива из памяти видеокарты в оперативную
    на каждом кадре внутри замеряемого участка.

    Args:
        gray: кадр в градациях серого.

    Returns:
        Результат детектора: points, descriptors со значением None, norm со
        значением None, raw с выходом сети и count.
    """
    import torch
    from lightglue import SuperPoint

    if "superpoint" not in _CACHE:
        _CACHE["superpoint"] = (SuperPoint(**config.SUPERPOINT_PARAMS)
                                .eval().to(torch_device()))

    tensor = torch.from_numpy(gray).float()[None] / 255.0
    with torch.inference_mode():
        raw = _CACHE["superpoint"].extract(tensor.to(torch_device()))

    points = raw["keypoints"][0].cpu().numpy().astype(np.float32)
    return {
        "points": points,
        "descriptors": None,
        "norm": None,
        "raw": raw,
        "count": len(points),
    }


def pack_opencv_result(keypoints: tuple, descriptors: np.ndarray | None,
                       norm: int) -> dict[str, Any]:
    """
    Приводит выход детектора OpenCV к общему виду.

    Args:
        keypoints: точки в формате OpenCV.
        descriptors: массив дескрипторов либо None, если точек не нашлось.
        norm: метрика сравнения дескрипторов, константа cv2.NORM_*.

    Returns:
        Результат детектора: points, descriptors, norm, raw со значением None
        и count.
    """
    points = (np.array([point.pt for point in keypoints], dtype=np.float32)
              if keypoints else np.empty((0, 2), dtype=np.float32))
    return {
        "points": points,
        "descriptors": descriptors,
        "norm": norm,
        "raw": None,
        "count": len(points),
    }


def torch_device() -> str:
    """
    Возвращает устройство для нейросетевых моделей.

    Returns:
        Строку cuda при доступной видеокарте, иначе cpu.
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────────────────────────────────────────────────────────
# Матчеры
# ────────────────────────────────────────────────────────────────────────────

def match_brute_force(first: dict[str, Any], second: dict[str, Any],
                      first_gray: np.ndarray, second_gray: np.ndarray
                      ) -> dict[str, Any]:
    """
    Сопоставляет точки перебором дескрипторов с тестом отношения расстояний.

    Для каждой точки первого кадра ищутся два ближайших дескриптора во втором.
    Пара принимается, если ближайший ближе второго в config.BF_RATIO_THRESHOLD
    раз: при близких расстояниях точка похожа сразу на несколько мест, и
    сопоставление неоднозначно.

    Args:
        first: результат детектора на первом кадре.
        second: результат детектора на втором кадре.
        first_gray: первый кадр, не используется.
        second_gray: второй кадр, не используется.

    Returns:
        Результат сопоставления: points_first, points_second, count и
        candidates с числом точек, для которых искалась пара.
    """
    if first["descriptors"] is None or second["descriptors"] is None:
        return empty_matches()
    if len(first["points"]) < 2 or len(second["points"]) < 2:
        return empty_matches()

    key = f"bf_{first['norm']}"
    if key not in _CACHE:
        _CACHE[key] = cv2.BFMatcher(first["norm"])

    candidates = _CACHE[key].knnMatch(first["descriptors"],
                                      second["descriptors"], k=2)
    good = [pair[0] for pair in candidates
            if len(pair) == 2
            and pair[0].distance < config.BF_RATIO_THRESHOLD * pair[1].distance]

    if not good:
        return empty_matches()

    first_index = np.array([match.queryIdx for match in good])
    second_index = np.array([match.trainIdx for match in good])

    return {
        "points_first": first["points"][first_index],
        "points_second": second["points"][second_index],
        "count": len(good),
        "candidates": len(candidates),
    }


def match_lucas_kanade(first: dict[str, Any], second: dict[str, Any],
                       first_gray: np.ndarray, second_gray: np.ndarray
                       ) -> dict[str, Any]:
    """
    Прослеживает точки первого кадра на втором методом Лукаса и Канаде.

    Метод подбирает смещение небольшого окна вокруг каждой точки так, чтобы
    яркость в нём совпала. Он предполагает, что смещение невелико и яркость
    между кадрами не изменилась. Пирамида изображений частично снимает первое
    ограничение: на уменьшенных копиях большое смещение выглядит малым.

    Найденные точки проверяются обратным прослеживанием: точка прогоняется
    назад на первый кадр, и при расхождении больше config.LK_BACKWARD_THRESHOLD
    пара отбрасывается.

    Args:
        first: результат детектора на первом кадре.
        second: не используется, детекция второго кадра не нужна.
        first_gray: первый кадр.
        second_gray: второй кадр.

    Returns:
        Результат сопоставления: points_first, points_second, count и
        candidates с числом точек, для которых искалась пара.
    """
    points = first["points"]
    if len(points) < 2:
        return empty_matches()

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.LK_PARAMS["criteria_max_iter"],
                config.LK_PARAMS["criteria_epsilon"])
    options = {
        "winSize": config.LK_PARAMS["winSize"],
        "maxLevel": config.LK_PARAMS["maxLevel"],
        "criteria": criteria,
    }

    source = points.reshape(-1, 1, 2)
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        first_gray, second_gray, source, None, **options)
    if forward is None:
        return empty_matches()

    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        second_gray, first_gray, forward, None, **options)
    if backward is None:
        return empty_matches()

    # Точка принимается, если оба прохода удались и обратный вернул её
    # достаточно близко к исходному положению
    drift = np.linalg.norm(source - backward, axis=2).ravel()
    accepted = ((status_forward.ravel() == 1) & (status_backward.ravel() == 1)
                & (drift < config.LK_BACKWARD_THRESHOLD))

    if not np.any(accepted):
        return empty_matches()

    return {
        "points_first": points[accepted],
        "points_second": forward.reshape(-1, 2)[accepted],
        "count": int(accepted.sum()),
        "candidates": len(points),
    }


def match_lightglue(first: dict[str, Any], second: dict[str, Any],
                    first_gray: np.ndarray, second_gray: np.ndarray
                    ) -> dict[str, Any]:
    """
    Сопоставляет точки нейросетью LightGlue.

    Сеть смотрит на оба набора точек целиком и учитывает их взаимное
    расположение, а не только близость описаний, поэтому отсеивает
    сопоставления, противоречащие общей картине смещения, ещё до геометрической
    проверки.

    Args:
        first: результат детектора SuperPoint на первом кадре.
        second: результат детектора SuperPoint на втором кадре.
        first_gray: первый кадр, не используется.
        second_gray: второй кадр, не используется.

    Returns:
        Результат сопоставления: points_first, points_second, count и
        candidates с числом точек на менее богатом из двух кадров.
    """
    import torch
    from lightglue import LightGlue

    if first["raw"] is None or second["raw"] is None:
        return empty_matches()

    if "lightglue" not in _CACHE:
        _CACHE["lightglue"] = (LightGlue(**config.LIGHTGLUE_PARAMS)
                               .eval().to(torch_device()))

    with torch.inference_mode():
        result = _CACHE["lightglue"]({"image0": first["raw"],
                                      "image1": second["raw"]})

    pairs = result["matches"][0].cpu().numpy()
    if len(pairs) == 0:
        return empty_matches()

    return {
        "points_first": first["points"][pairs[:, 0]],
        "points_second": second["points"][pairs[:, 1]],
        "count": len(pairs),
        "candidates": min(first["count"], second["count"]),
    }


def empty_matches() -> dict[str, Any]:
    """
    Возвращает пустой результат сопоставления.

    Форма совпадает с обычным результатом, поэтому вызывающий код не разбирает
    особый случай отдельно.

    Returns:
        Результат сопоставления с пустыми массивами координат и нулевыми
        счётчиками.
    """
    return {
        "points_first": np.empty((0, 2), dtype=np.float32),
        "points_second": np.empty((0, 2), dtype=np.float32),
        "count": 0,
        "candidates": 0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Свойства набора точек
# ────────────────────────────────────────────────────────────────────────────

def spatial_spread(points: np.ndarray, width: int, height: int) -> float:
    """
    Оценивает, насколько широко точки разбросаны по кадру.

    Считается площадь выпуклой оболочки точек в долях площади кадра. Величина
    связывает неоднородность текстуры с качеством геометрии: если контрастны
    лишь отдельные участки кадра, сопоставления скапливаются в них, и модель
    оказывается закреплена только там.

    Args:
        points: координаты точек, массив (N, 2).
        width: ширина кадра, px.
        height: высота кадра, px.

    Returns:
        Долю площади кадра, занятую оболочкой точек. Ноль, если точек меньше
        трёх. Значение превышает единицу, когда точки разбросаны шире кадра.
    """
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(cv2.contourArea(hull) / (width * height))


# ────────────────────────────────────────────────────────────────────────────
# Реестры
# ────────────────────────────────────────────────────────────────────────────

# Детекторы по ключу поля detector в config.FRONTENDS
DETECTORS: dict[str, Callable[[np.ndarray], dict[str, Any]]] = {
    "sift": detect_sift,
    "orb": detect_orb,
    "superpoint": detect_superpoint,
}

# Матчеры по ключу поля matcher в config.FRONTENDS
MATCHERS: dict[str, Callable[..., dict[str, Any]]] = {
    "bf": match_brute_force,
    "lk": match_lucas_kanade,
    "lightglue": match_lightglue,
}

# Нужна ли матчеру детекция на втором кадре. Оптический поток обходится без
# неё, и это его главное преимущество по стоимости: детектор запускается вдвое
# реже
MATCHER_NEEDS_SECOND: dict[str, bool] = {
    "bf": True,
    "lk": False,
    "lightglue": True,
}
