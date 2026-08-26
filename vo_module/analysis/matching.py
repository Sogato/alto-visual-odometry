"""
Качество сопоставления: доля согласных точек и влияние отбраковки.

Модуль отвечает на пункт задания о сопоставлении и разбирает два вопроса.

Первый: какая доля сопоставлений оказывается согласована с геометрической
моделью. Величина зависит не только от матчера, но и от модели, поэтому
считается для обеих геометрий отдельно. Сама по себе она мера внутренней
согласованности: точки способны дружно согласоваться с неверной моделью, и
тогда доля высока при неправильно восстановленном движении. Поэтому рядом
проверяется её связь с ошибкой направления.

Второй: что даёт робастная оценка. Перебираются методы, которых в OpenCV
несколько, и пороги невязки. При слишком строгом пороге отбрасывается почти
всё, при слишком свободном не отбрасывается ничего, и модель строится в том
числе по ошибочным сопоставлениям. Заведомо большой порог служит опорой,
показывающей, что было бы без отбраковки вовсе.

Для перебора нужен собственный проход по кадрам: сопоставления в основном
прогоне не сохраняются. Сопоставление при этом делается один раз на пару, а все
варианты отбраковки считаются поверх одних и тех же точек, поэтому разница
относится к отбраковке, а не к случайностям детектора.
"""

# Стандартные библиотеки
import sys
import time
from pathlib import Path
from typing import Any

# Сторонние библиотеки
import numpy as np
from scipy import stats

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from console import (PROGRESS_FORMS, STATUS_NONE, STATUS_OK, TABLE_INDENT,
                     block_line, block_note, clear_progress, close_block,
                     format_number, open_block, print_legend, print_progress,
                     print_section, print_table)
from vo_module import features, pose
from vo_module.dataset import Dataset
from visualization import analysis as figures


# === ПАРАМЕТРЫ ПЕРЕБОРА ===
# Пары распределяются равномерно по маршруту, чтобы охватить разную местность
SWEEP_PAIRS = 40      # Сколько пар кадров взять

# Пороги невязки в пикселях. Последнее значение заведомо велико и отбраковку
# фактически отключает, поэтому служит опорой: столько было бы без неё
THRESHOLD_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0)
NO_REJECTION_THRESHOLD = 50.0

# === ЕДИНИЦЫ ИЗМЕРЕНИЯ ===
MS_PER_SECOND = 1000.0   # Перевод секунд в миллисекунды для замеров времени

# === НИЖНИЕ ГРАНИЦЫ ОБЪЁМА ДАННЫХ ===
MIN_PAIRS_FOR_LINK = 10  # Пар для оценки связи доли согласных точек с ошибкой


# ────────────────────────────────────────────────────────────────────────────
# Показатели из основного прогона
# ────────────────────────────────────────────────────────────────────────────

def collect(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Собирает показатели сопоставления по каждой связке и геометрии.

    Связь доли согласных точек с ошибкой направления оценивается ранговой
    корреляцией Спирмена: зависимость не обязана быть линейной, важно лишь,
    падает ли ошибка с ростом доли.

    Args:
        results: результаты pipeline.run по ключу связки.

    Returns:
        Список записей по одной на сочетание связки и геометрии. Поля записи:
        frontend, label, geometry, geometry_label, ratio в процентах,
        ratio_worst, residual, error, link, а также ratio_series и
        error_series с рядами по парам для графика. Поле link равно NaN, если
        пар меньше MIN_PAIRS_FOR_LINK.
    """
    records: list[dict[str, Any]] = []

    for frontend in config.FRONTENDS:
        result = results.get(frontend["key"])
        if result is None or result["pairs"].empty:
            continue

        table = result["pairs"]

        for item in config.GEOMETRIES:
            key = item["key"]
            ratio = table[f"{key}_inlier_ratio"]
            errors = table[f"{key}_direction_error_2d"]

            usable = ratio.notna() & errors.notna()
            link = (float(stats.spearmanr(ratio[usable], errors[usable]).statistic)
                    if usable.sum() >= MIN_PAIRS_FOR_LINK else float("nan"))

            records.append({
                "frontend": frontend["key"],
                "label": frontend["label"],
                "geometry": key,
                "geometry_label": item["label"],
                "ratio": float(ratio.median()) * 100,
                "ratio_worst": float(np.percentile(ratio.dropna(), 10)) * 100,
                "residual": float(table[f"{key}_residual"].median()),
                "error": float(errors.median()),
                "link": link,
                "ratio_series": ratio[usable].to_numpy() * 100,
                "error_series": errors[usable].to_numpy(),
            })

    return records


# ────────────────────────────────────────────────────────────────────────────
# Перебор вариантов отбраковки
# ────────────────────────────────────────────────────────────────────────────

def gather_matches(name: str) -> list[dict[str, Any]]:
    """
    Собирает сопоставления по выборке пар для каждой связки.

    Сопоставление делается один раз на пару, а все варианты отбраковки потом
    считаются поверх одних и тех же точек, иначе разница между вариантами
    смешалась бы со случайностями детектора.

    Детекции переиспользуются между связками: SIFT и ORB встречаются в конфиге
    дважды, с перебором дескрипторов и с оптическим потоком, и без общего кеша
    один и тот же детектор гонялся бы по кадру несколько раз.

    Args:
        name: имя датасета.

    Returns:
        Список наборов сопоставленных точек. Поля набора: frontend, label,
        points_first, points_second, reference с единичным вектором эталонного
        направления, intrinsics и to_reference. Список пуст, если пар кадров не
        набралось.

    Raises:
        KeyError: не заполнен config.FRAME_STEP, не заполнены калибровочные
            константы датасета либо датасета нет в конфиге.
        FileNotFoundError: нет файла телеметрии или каталога с кадрами.
    """
    # Те же предусловия, что у pipeline.run: без проверки проход упал бы глубже,
    # на составлении пар или на построении матрицы камеры
    if config.FRAME_STEP is None:
        raise KeyError("не заполнен config.FRAME_STEP, "
                       "запусти service/frame_step.py")

    calibration = config.CALIBRATION[name]
    missing = [field for field in config.CALIBRATION_REQUIRED
               if calibration.get(field) is None]
    if missing:
        raise KeyError(f"для датасета {name} не заполнено: "
                       f"{', '.join(missing)}")

    data = Dataset(name)
    intrinsics = pose.camera_matrix(calibration["focal_px"],
                                    calibration["principal_point"])
    to_reference = np.asarray(calibration["rotation_cam_to_gt"], dtype=float)

    all_pairs = data.pairs(config.FRAME_STEP)
    if not all_pairs:
        return []

    count = min(SWEEP_PAIRS, len(all_pairs))
    chosen = [all_pairs[index] for index in
              np.linspace(0, len(all_pairs) - 1, count).astype(int)]

    label = f"{name}: сопоставления для перебора"
    started = time.perf_counter()
    collected: list[dict[str, Any]] = []

    for position, (first_index, second_index) in enumerate(chosen):
        print_progress(position, len(chosen), label)

        first_gray = data.gray(first_index)
        second_gray = data.gray(second_index)

        motion = data.relative_motion(first_index, second_index)
        reference = np.asarray(motion["delta_body"], dtype=float)[:2]
        if np.linalg.norm(reference) == 0:
            continue
        reference = reference / np.linalg.norm(reference)

        # Кеш на пару кадров: ключ это имя детектора и признак кадра
        detected: dict[tuple[str, bool], dict[str, Any]] = {}

        def detect(detector_key: str, is_first: bool) -> dict[str, Any]:
            if (detector_key, is_first) not in detected:
                gray = first_gray if is_first else second_gray
                detected[(detector_key, is_first)] = (
                    features.DETECTORS[detector_key](gray))
            return detected[(detector_key, is_first)]

        for frontend in config.FRONTENDS:
            detector_key = frontend["detector"]
            matcher = features.MATCHERS[frontend["matcher"]]
            needs_second = features.MATCHER_NEEDS_SECOND[frontend["matcher"]]

            first = detect(detector_key, True)
            second = detect(detector_key, False) if needs_second else first
            matches = matcher(first, second, first_gray, second_gray)

            if matches["count"] < config.MIN_MATCHES_FOR_GEOMETRY:
                continue

            collected.append({
                "frontend": frontend["key"],
                "label": frontend["label"],
                "points_first": matches["points_first"],
                "points_second": matches["points_second"],
                "reference": reference,
                "intrinsics": intrinsics,
                "to_reference": to_reference,
            })

    clear_progress(label, len(chosen), time.perf_counter() - started,
                   PROGRESS_FORMS)
    return collected


def evaluate(sample: dict[str, Any], geometry: str, method: str | None,
             threshold: float | None) -> dict[str, float] | None:
    """
    Применяет к готовым сопоставлениям заданный вариант отбраковки.

    Args:
        sample: набор сопоставленных точек из gather_matches.
        geometry: ключ геометрии.
        method: название метода робастной оценки либо None для значения из
            конфига.
        threshold: порог невязки в пикселях либо None для значения из конфига.

    Returns:
        Словарь с полями ratio в процентах, error в градусах и time_ms. None,
        если геометрия не отработала либо направление оказалось вырожденным.
    """
    started = time.perf_counter()
    outcome = pose.POSE_BY_KEY[geometry](sample["points_first"],
                                         sample["points_second"],
                                         sample["intrinsics"],
                                         method=method, threshold=threshold)
    elapsed = (time.perf_counter() - started) * MS_PER_SECOND

    if not outcome["ok"]:
        return None

    estimated = (sample["to_reference"]
                 @ np.asarray(outcome["translation"], dtype=float))[:2]
    norm = np.linalg.norm(estimated)
    if norm == 0:
        return None

    cosine = float(np.clip(np.dot(estimated / norm, sample["reference"]),
                           -1.0, 1.0))

    return {
        "ratio": outcome["inlier_ratio"] * 100,
        "error": float(np.degrees(np.arccos(cosine))),
        "time_ms": elapsed,
    }


def sweep(samples: list[dict[str, Any]], geometry: str,
          variants: dict[str, tuple[str | None, float | None]]
          ) -> list[dict[str, Any]]:
    """
    Прогоняет все варианты отбраковки по собранным сопоставлениям.

    Args:
        samples: результат gather_matches.
        geometry: ключ геометрии.
        variants: название варианта и пара из метода и порога.

    Returns:
        Список записей по одной на вариант, в порядке variants. Поля записи:
        variant, ratio, error и time_ms, все медианные по парам. Поле ok равно
        False, если ни одна пара не отработала, и тогда остальных полей нет.
    """
    outcome: list[dict[str, Any]] = []

    for name, (method, threshold) in variants.items():
        collected = [evaluate(sample, geometry, method, threshold)
                     for sample in samples]
        usable = [item for item in collected if item is not None]

        if not usable:
            outcome.append({"variant": name, "ok": False})
            continue

        outcome.append({
            "variant": name,
            "ok": True,
            "ratio": float(np.median([item["ratio"] for item in usable])),
            "error": float(np.median([item["error"] for item in usable])),
            "time_ms": float(np.median([item["time_ms"] for item in usable])),
        })

    return outcome


# ────────────────────────────────────────────────────────────────────────────
# Отчёт
# ────────────────────────────────────────────────────────────────────────────

def report_table(records: list[dict[str, Any]]) -> None:
    """
    Печатает долю согласных точек по связкам и геометриям.

    Число кандидатов и число сопоставлений сюда не входят: первое упирается в
    общий предел числа точек и одинаково у всех связок, второе уже напечатано в
    сводке прогона.

    Args:
        records: результат collect.
    """
    headers = ["Связка", "Геометрия", "Доля %", "Худшие 10 %", "Невязка",
               "Ошибка"]
    rows: list[list[Any]] = []
    previous = ""

    for record in records:
        # Название связки печатается один раз на пару строк: повтор в соседней
        # строке читался бы как отдельное измерение
        label = record["label"] if record["label"] != previous else ""
        previous = record["label"]
        rows.append([label, record["geometry_label"],
                     format_number(record["ratio"], 1),
                     format_number(record["ratio_worst"], 1),
                     format_number(record["residual"], 2),
                     format_number(record["error"], 1)])

    print_section("ДОЛЯ СОГЛАСНЫХ ТОЧЕК")
    print()
    print_table(headers, rows)
    print()
    print_legend([
        ("Доля %", "сопоставлений, согласных с моделью, от числа найденных"),
        ("Худшие 10 %", "доля согласных на десятой части худших пар"),
        ("Невязка", "расстояние от точки до модели у согласных, px"),
        ("Ошибка", "ошибка направления движения на паре, град"),
    ])


def report_link(records: list[dict[str, Any]]) -> None:
    """
    Печатает, означает ли высокая доля согласных точек точный результат.

    Печатается само значение связи, без вердикта по порогу: порог давал бы
    одинаковый ответ и там, где связи нет, и там, где она слаба, но устойчива.

    Args:
        records: результат collect.
    """
    open_block("СОГЛАСИЕ И ТОЧНОСТЬ")

    for item in config.GEOMETRIES:
        own = [record["link"] for record in records
               if record["geometry"] == item["key"]
               and not np.isnan(record["link"])]
        if not own:
            block_line(item["label"], "данных не хватило", STATUS_NONE)
            continue
        block_line(item["label"],
                   f"{format_number(float(np.median(own)), 2)} "
                   f"по {len(own)} связкам")

    block_note("Связь ранговая, между долей согласных точек и ошибкой "
               "направления. Отрицательная означает, что доля служит признаком "
               "качества. Сама по себе доля это мера внутренней "
               "согласованности: точки способны дружно согласоваться с неверной "
               "моделью, и тогда она высока при неверном движении.")

    close_block()


def report_variants(title: str, outcome: dict[str, list[dict[str, Any]]],
                    note: str, legend: bool = False) -> None:
    """
    Печатает таблицу с результатами перебора вариантов отбраковки.

    Геометрии идут колонками, а не строками, как в остальных таблицах: строк
    тут до семи, и разнесение по геометриям удвоило бы их число ради сравнения,
    которое делается взглядом поперёк строки. Полные названия в заголовки не
    помещаются, поэтому колонки помечены первой буквой, а расшифровка вынесена
    под таблицу.

    Args:
        title: заголовок таблицы.
        outcome: результат sweep по каждой геометрии, ключ это её подпись.
        note: пояснение под таблицей, строки разделяются переводом строки.
        legend: печатать ли расшифровку колонок. У второй таблицы она та же, и
            повтор занял бы больше места, чем сами данные.
    """
    headers = ["Вариант"]
    for label in outcome:
        letter = label[0]
        headers += [f"{letter} согл. %", f"{letter} ошибка", f"{letter} мс"]

    variants = [item["variant"] for item in next(iter(outcome.values()))]
    rows: list[list[Any]] = []

    for variant in variants:
        row: list[Any] = [variant]
        for records in outcome.values():
            found = next((item for item in records
                          if item["variant"] == variant), None)
            if found is None or not found["ok"]:
                row += ["", "", ""]
            else:
                row += [format_number(found["ratio"], 1),
                        format_number(found["error"], 1),
                        format_number(found["time_ms"], 2)]
        rows.append(row)

    print_section(title)
    print()
    print_table(headers, rows)
    print()

    if legend:
        print_legend([(label[0], label) for label in outcome]
                     + [("согл. %", "доля согласных с моделью точек"),
                        ("ошибка", "ошибка направления движения, град"),
                        ("мс", "время оценки геометрии на паре")])
        print()

    for line in note.split("\n"):
        print(f"{' ' * TABLE_INDENT}{line}")


def report_rejection(thresholds: dict[str, list[dict[str, Any]]]) -> None:
    """
    Печатает, что даёт отбраковка по сравнению с её отсутствием.

    Лучшим порогом иногда оказывается тот, что отбраковку отключает: так бывает,
    когда среди сопоставлений почти нет грубых промахов, и отсев только сужает
    набор точек, по которому строится модель. Выигрыш для такого случая не
    считается: вычитание дало бы ноль от сравнения строки с самой собой.

    Args:
        thresholds: результат sweep по порогам, по каждой геометрии. Названия
            вариантов должны читаться как числа: по ним отыскивается порог,
            отключающий отбраковку.
    """
    open_block("ЧТО ДАЁТ ОТБРАКОВКА")

    for geometry, records in thresholds.items():
        usable = [item for item in records if item["ok"]]
        if not usable:
            block_line(geometry, "данных не хватило", STATUS_NONE)
            continue

        without = next((item for item in usable
                        if float(item["variant"]) == NO_REJECTION_THRESHOLD),
                       None)
        best = min(usable, key=lambda item: item["error"])

        if without is None:
            block_line(geometry,
                       f"лучший порог {best['variant']} px, ошибка "
                       f"{format_number(best['error'], 1)} град")
            continue

        if best is without:
            block_line(geometry,
                       f"{format_number(without['error'], 1)} град, отбраковка "
                       f"не помогает", STATUS_NONE)
            continue

        gain = without["error"] - best["error"]
        block_line(geometry,
                   f"{format_number(without['error'], 1)} град без отбраковки, "
                   f"{format_number(best['error'], 1)} при пороге "
                   f"{best['variant']} px")
        block_line(f"{geometry}, выигрыш", f"{format_number(gain, 1)} град",
                   STATUS_OK if gain > 0 else STATUS_NONE)

    block_note(f"Порог в {NO_REJECTION_THRESHOLD:g} пикселей отбраковку "
               f"фактически отключает: согласными признаются почти все "
               f"сопоставления. Разница с лучшим порогом и показывает, что даёт "
               f"робастная оценка на этих данных.")

    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Точка входа анализа
# ────────────────────────────────────────────────────────────────────────────

def run(results: dict[str, dict[str, Any]], name: str) -> None:
    """
    Выполняет анализ качества сопоставления.

    Args:
        results: результаты pipeline.run по ключу связки.
        name: имя датасета.
    """
    records = collect(results)
    if not records:
        open_block("СОСТОЯНИЕ")
        block_line("Данные", "нет данных для анализа", STATUS_NONE)
        close_block()
        return

    report_table(records)
    report_link(records)

    samples = gather_matches(name)
    if not samples:
        return

    method_variants = {method: (method, None)
                       for method in config.RANSAC_METHODS_COMPARED}
    threshold_variants = {f"{value:g}": (None, value)
                          for value in THRESHOLD_GRID}

    methods = {item["label"]: sweep(samples, item["key"], method_variants)
               for item in config.GEOMETRIES}
    thresholds = {item["label"]: sweep(samples, item["key"], threshold_variants)
                  for item in config.GEOMETRIES}

    report_variants("МЕТОДЫ РОБАСТНОЙ ОЦЕНКИ", methods,
                    "Порог невязки у всех вариантов взят из конфига,\n"
                    "различается только способ поиска согласного набора точек.",
                    legend=True)
    report_variants("ПОРОГ НЕВЯЗКИ, ПИКСЕЛИ", thresholds,
                    "Метод у всех вариантов один и тот же, различается порог.\n"
                    "Последняя строка отбраковку фактически отключает.")
    report_rejection(thresholds)

    path = figures.matching(records, methods, thresholds, name)

    open_block("ГРАФИК СОПОСТАВЛЕНИЯ")
    block_line("Файл", path.name)
    block_line("Содержание", "влияние порога отбраковки")
    close_block()
