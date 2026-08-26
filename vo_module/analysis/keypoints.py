"""
Устойчивость ключевых точек: повторяемость и влияние текстуры.

Модуль отвечает на пункт задания о ключевых точках. Результатами прогона он не
пользуется, поскольку речь только о детекторе, и делает собственный проход по
выборке пар кадров.

Главная измеряемая величина это повторяемость. Детектор полезен не тем, сколько
точек он нашёл, а тем, находит ли он одни и те же места на разных снимках.
Точки первого кадра переносятся на второй известным преобразованием, и
проверяется, есть ли рядом точка, найденная там независимо. Преобразование
берётся из отдельной измерительной связки, одной и той же для всех детекторов.

Число найденных точек показателем не служит: оно ограничено одинаково для всех
детекторов, поэтому различия ищутся в повторяемости и в точности постановки
точки.
"""

# Стандартные библиотеки
import sys
import time
from pathlib import Path
from typing import Any

# Сторонние библиотеки
import cv2
import numpy as np
from scipy import spatial, stats

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from console import (PROGRESS_FORMS, STATUS_NONE, block_line, block_note,
                     clear_progress, close_block, format_number, open_block,
                     print_progress, print_section, print_table)
from vo_module import features
from vo_module.dataset import Dataset
from visualization import analysis as figures


# === ПАРАМЕТРЫ ПРОХОДА ===
# Пары распределяются равномерно по маршруту: местность вдоль полёта меняется,
# и выборка подряд с начала описывала бы только его начало
SAMPLE_PAIRS = 60      # Сколько пар кадров взять

# Точка считается найденной повторно, если после переноса на второй кадр рядом
# оказалась точка, найденная там независимо. Порог в пикселях.
#
# Перенос идёт гомографией, то есть по плоской модели сцены, а сцена плоская не
# везде. Объект, поднятый над землёй, переносится с промахом, и точка на дереве,
# честно найденная на обоих кадрах, засчитывалась бы как потерянная. Расширять
# порог дальше нельзя: чем он шире, тем чаще рядом случайно оказывается
# посторонняя точка
MATCH_RADIUS = 4.0

# Отступ от края кадра: точки у самой границы могли выйти за пределы второго
# кадра не по вине детектора
BORDER_MARGIN = 8.0

# === ИЗМЕРИТЕЛЬНАЯ СВЯЗКА ===
# Преобразование между кадрами берётся одно и то же для всех детекторов, иначе
# повторяемость зависела бы от того, чьими точками оно построено
REFERENCE_DETECTOR = "sift"
REFERENCE_MATCHER = "bf"

# === НАЗВАНИЯ ДЕТЕКТОРОВ ===
# Порядок задаёт и порядок строк таблицы, и порядок рядов на графике
DETECTOR_LABELS: dict[str, str] = {
    "sift": "SIFT",
    "orb": "ORB",
    "superpoint": "SuperPoint",
}

# === ПОРОГИ ВЫВОДОВ ===
# Относительный разброс текстуры вдоль маршрута, ниже которого связь с ней
# измеряется по шуму
MIN_TEXTURE_RANGE = 0.25

# === НИЖНИЕ ГРАНИЦЫ ОБЪЁМА ДАННЫХ ===
MIN_POINTS = 20        # Точек на кадре для расчёта повторяемости
MIN_PAIRS_FOR_LINK = 5  # Пар для оценки связи повторяемости с текстурой


# ────────────────────────────────────────────────────────────────────────────
# Свойства кадра
# ────────────────────────────────────────────────────────────────────────────

def texture(gray: np.ndarray) -> float:
    """
    Считает насыщенность кадра деталями как среднюю величину градиента яркости.

    Величина отражает наличие контуров и перепадов, за которые цепляются
    детекторы.

    Args:
        gray: кадр в градациях серого.

    Returns:
        Средний градиент, нормированный на полную шкалу яркости.
    """
    normalized = gray.astype(np.float32) / 255.0
    dx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.hypot(dx, dy).mean())


# ────────────────────────────────────────────────────────────────────────────
# Повторяемость
# ────────────────────────────────────────────────────────────────────────────

def reference_transform(first: dict[str, Any], second: dict[str, Any],
                        first_gray: np.ndarray,
                        second_gray: np.ndarray) -> np.ndarray | None:
    """
    Находит преобразование между кадрами по точкам измерительной связки.

    Связка одна и та же для всех детекторов: если бы преобразование строилось
    точками самого проверяемого детектора, повторяемость измерялась бы
    относительно его же ошибок.

    Точки приходят готовыми, поскольку измерительный детектор входит в число
    проверяемых, и повторный запуск считал бы их второй раз.

    Args:
        first: результат измерительного детектора на первом кадре.
        second: результат измерительного детектора на втором кадре.
        first_gray: первый кадр.
        second_gray: второй кадр.

    Returns:
        Матрицу гомографии либо None, если сопоставлений не хватило или решение
        не найдено.
    """
    matcher = features.MATCHERS[REFERENCE_MATCHER]
    matches = matcher(first, second, first_gray, second_gray)

    if matches["count"] < config.MIN_MATCHES_FOR_GEOMETRY:
        return None

    homography, _ = cv2.findHomography(
        matches["points_first"], matches["points_second"], method=cv2.RANSAC,
        ransacReprojThreshold=config.RANSAC_THRESHOLD_HOMOGRAPHY,
        maxIters=config.RANSAC_MAX_ITERS, confidence=config.RANSAC_CONFIDENCE)

    return homography


def repeatability(first_points: np.ndarray, second_points: np.ndarray,
                  homography: np.ndarray, width: int,
                  height: int) -> dict[str, Any]:
    """
    Считает, какая доля точек первого кадра снова найдена на втором.

    Точки первого кадра переносятся на второй известным преобразованием. Точка
    считается найденной повторно, если рядом с перенесённым положением есть
    точка второго кадра, найденная там независимо.

    В счёт идут только точки, попавшие в пределы второго кадра с отступом от
    края: вышедшие за границу не найдены не по вине детектора, а из-за
    смещения камеры.

    Args:
        first_points: точки первого кадра, массив (N, 2).
        second_points: точки второго кадра, массив (M, 2).
        homography: преобразование из первого кадра во второй.
        width: ширина кадра, px.
        height: высота кадра, px.

    Returns:
        Словарь с полем rate, долей повторно найденных точек от нуля до
        единицы. Поле ok равно False, если точек меньше MIN_POINTS либо после
        отсечения края их осталось меньше того же числа, и тогда поля rate нет.
    """
    if len(first_points) < MIN_POINTS or len(second_points) < MIN_POINTS:
        return {"ok": False}

    homogeneous = np.hstack([first_points, np.ones((len(first_points), 1))])
    projected = homogeneous @ np.asarray(homography).T
    mapped = projected[:, :2] / projected[:, 2:3]

    inside = ((mapped[:, 0] > BORDER_MARGIN)
              & (mapped[:, 0] < width - BORDER_MARGIN)
              & (mapped[:, 1] > BORDER_MARGIN)
              & (mapped[:, 1] < height - BORDER_MARGIN))
    visible = mapped[inside]

    if len(visible) < MIN_POINTS:
        return {"ok": False}

    tree = spatial.cKDTree(second_points)
    distances, _ = tree.query(visible)

    return {"ok": True, "rate": float((distances < MATCH_RADIUS).mean())}


def chance_rate(points: float, width: int, height: int) -> float:
    """
    Оценивает долю совпадений, возникающих по случайности.

    Совпадением считается наличие любой точки второго кадра ближе MATCH_RADIUS.
    При достаточной плотности такая точка находится рядом и без всякой связи с
    исходной, поэтому у повторяемости есть пол, ниже которого она не опускается
    ни при каком детекторе.

    Считается по фактическому числу точек детектора, а не по общему пределу:
    чем гуще точки, тем чаще ближайшая попадает в радиус случайно, и общий для
    всех пол это различие скрыл бы.

    Точки второго кадра считаются случайно рассыпанными по кадру, тогда доля
    равна вероятности того, что ближайшая из них окажется внутри радиуса.
    Настоящие точки садятся на контуры и распределены неравномерно, поэтому
    величина годится как порядок: вычитать её из повторяемости нельзя,
    показывать рядом нужно.

    Args:
        points: сколько точек детектор находит на кадре.
        width: ширина кадра, px.
        height: высота кадра, px.

    Returns:
        Долю случайных совпадений в процентах.
    """
    density = points / (width * height)
    return float(1.0 - np.exp(-density * np.pi * MATCH_RADIUS ** 2)) * 100


# ────────────────────────────────────────────────────────────────────────────
# Проход по кадрам
# ────────────────────────────────────────────────────────────────────────────

def measure(name: str) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    """
    Прогоняет все детекторы по выборке пар и собирает наблюдения.

    Размер кадра возвращается вместе с наблюдениями: он нужен для оценки
    случайных совпадений, а создание датасета второй раз ради двух чисел
    означало бы заново прочесть телеметрию и декодировать кадр.

    Args:
        name: имя датасета.

    Returns:
        Пару из списка наблюдений и размера кадра в пикселях. Поля наблюдения:
        detector, points, rate в процентах и gradient. Список пуст, если пар
        кадров не набралось.

    Raises:
        KeyError: не заполнен config.FRAME_STEP либо датасета нет в конфиге.
        FileNotFoundError: нет файла телеметрии или каталога с кадрами.
    """
    # То же предусловие, что у pipeline.run: без проверки проход упал бы глубже,
    # на составлении пар
    if config.FRAME_STEP is None:
        raise KeyError("не заполнен config.FRAME_STEP, "
                       "запусти service/frame_step.py")

    data = Dataset(name)
    width, height = data.image_size

    all_pairs = data.pairs(config.FRAME_STEP)
    if not all_pairs:
        return [], (width, height)

    count = min(SAMPLE_PAIRS, len(all_pairs))
    chosen = [all_pairs[index] for index in
              np.linspace(0, len(all_pairs) - 1, count).astype(int)]

    detectors = {key: features.DETECTORS[key] for key in DETECTOR_LABELS
                 if key in features.DETECTORS}

    label = f"{name}: повторяемость точек"
    started = time.perf_counter()
    records: list[dict[str, Any]] = []

    for position, (first_index, second_index) in enumerate(chosen):
        print_progress(position, len(chosen), label)

        first_gray = data.gray(first_index)
        second_gray = data.gray(second_index)

        detected = {key: (detector(first_gray), detector(second_gray))
                    for key, detector in detectors.items()}

        # Измерительный детектор входит в число проверяемых, поэтому его точки
        # берутся из общего набора. Отдельный запуск нужен только в случае,
        # когда он оттуда исключён
        if REFERENCE_DETECTOR in detected:
            reference_points = detected[REFERENCE_DETECTOR]
        else:
            reference_detector = features.DETECTORS[REFERENCE_DETECTOR]
            reference_points = (reference_detector(first_gray),
                                reference_detector(second_gray))

        homography = reference_transform(*reference_points,
                                         first_gray, second_gray)
        if homography is None:
            continue

        gradient = texture(first_gray)

        for key, (first, second) in detected.items():
            outcome = repeatability(first["points"], second["points"],
                                    homography, width, height)
            if not outcome["ok"]:
                continue

            records.append({
                "detector": key,
                "points": first["count"],
                "rate": outcome["rate"] * 100,
                "gradient": gradient,
            })

    clear_progress(label, len(chosen), time.perf_counter() - started,
                   PROGRESS_FORMS)
    return records, (width, height)


def summarize(records: list[dict[str, Any]], width: int,
              height: int) -> list[dict[str, Any]]:
    """
    Сводит наблюдения к показателям по каждому детектору.

    Связь повторяемости с текстурой оценивается ранговой корреляцией Спирмена:
    зависимость не обязана быть линейной, важно лишь, растёт ли повторяемость
    вместе с насыщенностью кадра деталями.

    Args:
        records: результат measure.
        width: ширина кадра, px, нужна для оценки случайных совпадений.
        height: высота кадра, px.

    Returns:
        Список записей по одной на детектор, в порядке DETECTOR_LABELS. Поля
        записи: detector, label, pairs, rate, chance, rate_worst, texture_link,
        texture_range, texture_median и at_limit с долей кадров у предела числа
        точек. Поля texture_link и texture_range равны NaN, если пар меньше
        MIN_PAIRS_FOR_LINK.
    """
    summary: list[dict[str, Any]] = []

    for key, label in DETECTOR_LABELS.items():
        own = [record for record in records if record["detector"] == key]
        if not own:
            continue

        rates = np.array([record["rate"] for record in own])
        gradients = np.array([record["gradient"] for record in own])
        points = np.array([record["points"] for record in own], dtype=float)

        if len(own) >= MIN_PAIRS_FOR_LINK:
            texture_link = float(stats.spearmanr(gradients, rates).statistic)
            # Насколько широко меняется текстура вдоль маршрута. При узком
            # диапазоне связь измеряется по шуму
            texture_range = float(np.percentile(gradients, 90)
                                  - np.percentile(gradients, 10))
        else:
            texture_link = float("nan")
            texture_range = float("nan")

        summary.append({
            "detector": key,
            "label": label,
            "pairs": len(own),
            "rate": float(np.median(rates)),
            "chance": chance_rate(float(np.median(points)), width, height),
            "rate_worst": float(np.percentile(rates, 10)),
            "texture_link": texture_link,
            "texture_range": texture_range,
            "texture_median": float(np.median(gradients)),
            "at_limit": float(np.mean(points >= config.MAX_KEYPOINTS)) * 100,
        })

    return summary


# ────────────────────────────────────────────────────────────────────────────
# Отчёт
# ────────────────────────────────────────────────────────────────────────────

def report_table(summary: list[dict[str, Any]]) -> None:
    """
    Печатает таблицу показателей по детекторам.

    Доля случайных совпадений стоит рядом с повторяемостью, потому что задаёт
    отсчёт: повторяемость начинается не от нуля. Доля кадров у предела числа
    точек относится к влиянию текстуры: если детектор не всегда набирает даже
    разрешённое количество, значит на однородных участках маршрута ему не за
    что зацепиться.

    Охват кадра точками сюда не входит: он относится не к устойчивости точек, а
    к тому, насколько широко закреплена геометрическая модель, и разбирается в
    vo_module/analysis/geometry.py.

    Args:
        summary: результат summarize.
    """
    headers = ["Детектор", "Пар", "У предела %", "Повтор. %", "Случайно %",
               "Худшие 10 %"]
    rows = [[item["label"], item["pairs"],
             format_number(item["at_limit"], 0),
             format_number(item["rate"], 1),
             format_number(item["chance"], 1),
             format_number(item["rate_worst"], 1)]
            for item in summary]

    print_section("ПОВТОРЯЕМОСТЬ ПО ДЕТЕКТОРАМ")
    print()
    print_table(headers, rows)


def report_texture(summary: list[dict[str, Any]]) -> None:
    """
    Печатает связь повторяемости с насыщенностью кадра деталями.

    Печатается само значение связи, без вердикта по порогу: порог отсекал бы
    слабую, но согласованную между детекторами зависимость.

    Args:
        summary: результат summarize.
    """
    open_block("ВЛИЯНИЕ ТЕКСТУРЫ")

    for item in summary:
        span_relative = (item["texture_range"] / item["texture_median"]
                         if item["texture_median"] else 0.0)
        narrow = span_relative < MIN_TEXTURE_RANGE
        note = ", разброс текстуры мал" if narrow else ""

        block_line(item["label"],
                   f"{format_number(item['texture_link'], 2)} "
                   f"по {item['pairs']} парам{note}",
                   STATUS_NONE if narrow else "")

    block_note("Связь ранговая: положительная означает, что на насыщенных "
               "деталями кадрах точки ставятся устойчивее.")

    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Точка входа анализа
# ────────────────────────────────────────────────────────────────────────────

def run(results: dict[str, dict[str, Any]], name: str) -> None:
    """
    Выполняет анализ устойчивости ключевых точек.

    Args:
        results: результаты pipeline.run по ключу связки, не используются:
            повторяемость это свойство детектора, а сопоставление и геометрия
            к ней отношения не имеют. Аргумент есть ради общего вида точек
            входа всех анализов.
        name: имя датасета.
    """
    records, (width, height) = measure(name)
    summary = summarize(records, width, height)

    if not summary:
        open_block("УСТОЙЧИВОСТЬ КЛЮЧЕВЫХ ТОЧЕК")
        block_line("Состояние", "не удалось собрать наблюдения", STATUS_NONE)
        close_block()
        return

    report_table(summary)
    report_texture(summary)

    path = figures.keypoints(records, summary, name, list(DETECTOR_LABELS))

    open_block("ГРАФИК КЛЮЧЕВЫХ ТОЧЕК")
    block_line("Файл", path.name)
    block_line("Содержание", "повторяемость против насыщенности кадра")
    close_block()
