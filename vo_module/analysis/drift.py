"""
Накопление ошибки: как расхождение растёт с длиной маршрута.

Модуль отвечает на пункт задания о дрейфе. Одометрия складывает перемещения
одно за другим, поэтому ошибка каждого шага остаётся в траектории навсегда, и
вопрос в том, по какому закону она копится.

Если ошибки шагов независимы и не смещены в одну сторону, они частично гасят
друг друга, и расхождение растёт примерно как корень из длины пути. При
постоянном уводе расхождение растёт прямо пропорционально пути. Показатель
роста, найденный по отрезкам разной длины, различает эти случаи и позволяет
предсказать поведение на маршруте длиннее имеющегося.

Отдельно проверяется роль редких грубых промахов: средняя ошибка на паре может
быть небольшой, но один разворот на полторы сотни градусов уводит траекторию
безвозвратно. Поэтому рядом со средней ошибкой считается доля таких промахов и
сравнивается их связь с итоговым дрейфом.
"""

# Стандартные библиотеки
import sys
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
from console import (MISSING, STATUS_NONE, block_line, block_note, close_block,
                     format_number, open_block, print_legend, print_section,
                     print_table)
from vo_module import metrics
from vo_module.dataset import rotation_angle
from visualization import analysis as figures


# === ПОРОГИ ТРАКТОВКИ ===
# Ошибка направления выше этого угла считается грубым промахом: перемещение
# оценено настолько неверно, что шаг уводит траекторию, а не уточняет её
OUTLIER_ANGLE = 45.0

# Границы трактовки показателя роста. Около половины он означает случайное
# накопление, около единицы постоянный увод, границы взяты посередине
GROWTH_RANDOM = 0.7      # Ниже этого значения накопление считается случайным
GROWTH_LINEAR = 0.9      # От этого значения накопление считается уводом

# === НИЖНИЕ ГРАНИЦЫ ОБЪЁМА ДАННЫХ ===
MIN_LENGTHS_FOR_FIT = 3   # Длин отрезков для подгонки показателя роста
MIN_SETS_FOR_LINK = 4     # Наборов для оценки связи величины с дрейфом


# ────────────────────────────────────────────────────────────────────────────
# Закон накопления
# ────────────────────────────────────────────────────────────────────────────

def growth_exponent(relative: dict[float, dict[str, float]]) -> dict[str, Any]:
    """
    Находит, по какому закону ошибка растёт с длиной отрезка.

    Расхождение на отрезке длиной L описывается степенной зависимостью от этой
    длины. Показатель степени около половины отвечает случайному накоплению,
    когда ошибки шагов гасят друг друга, около единицы постоянному уводу, когда
    они складываются.

    Показатель находится подгонкой прямой в двойных логарифмических осях: там
    степенная зависимость превращается в линейную, и наклон прямой равен
    искомому показателю. Подгонка ведётся по расхождению в метрах, а не по его
    отношению к длине отрезка.

    Args:
        relative: результат metrics.relative_error, ошибка по длинам отрезков.

    Returns:
        Словарь с полями exponent, kind со словесной трактовкой, quality с
        качеством подгонки, lengths и errors. Поле ok равно False, если длин
        отрезков меньше MIN_LENGTHS_FOR_FIT либо среди расхождений есть нули, и
        тогда остальных полей нет.
    """
    lengths = sorted(relative)
    if len(lengths) < MIN_LENGTHS_FOR_FIT:
        return {"ok": False}

    errors = [relative[length]["median"] * length / 100.0 for length in lengths]

    horizontal = np.log(np.array(lengths, dtype=float))
    vertical = np.log(np.array(errors, dtype=float))
    if not np.all(np.isfinite(vertical)):
        return {"ok": False}

    fit = stats.linregress(horizontal, vertical)
    exponent = float(fit.slope)

    if exponent < GROWTH_RANDOM:
        kind = "случайное накопление"
    elif exponent < GROWTH_LINEAR:
        kind = "промежуточное"
    else:
        kind = "постоянный увод"

    return {
        "ok": True,
        "exponent": exponent,
        "kind": kind,
        "quality": float(fit.rvalue ** 2),
        "lengths": lengths,
        "errors": errors,
    }


def heading_drift(orientations: np.ndarray, reference: np.ndarray,
                  to_reference: np.ndarray) -> dict[str, Any]:
    """
    Считает, как расходится накопленная ориентация с эталонной.

    Ошибка поворота на отдельном шаге мала, но копится иначе, чем ошибка
    перемещения: неверный поворот разворачивает все последующие шаги, поэтому
    даже небольшой систематический подворот сворачивает траекторию в спираль.

    Сравниваются накопленные ориентации, приведённые к общим осям. Разность
    углов поворота для этого не годится: у двух поворотов на одинаковый угол
    вокруг разных осей она равна нулю, и оценка, уводящая камеру в сторону,
    получила бы нулевое расхождение.

    Расхождение на шаге и итоговое расхождение считаются по-разному. На каждом
    шаге берётся угол поворота, переводящего оценку в эталон. Итог берётся
    между конечными ориентациями, потому что ошибки шагов не складываются
    числами, а перемножаются как повороты и способны частично гасить друг
    друга.

    Args:
        orientations: накопленные повороты оценки, массив (N, 3, 3).
        reference: эталонные повороты, массив (N, 3, 3).
        to_reference: поворот из системы камеры в систему эталона.

    Returns:
        Словарь с полями final в градусах, per_step с медианной ошибкой шага и
        steps с числом шагов. Поле ok равно False, если поворотов меньше двух,
        и тогда остальных полей нет.
    """
    if len(orientations) < 2 or len(reference) < 2:
        return {"ok": False}

    count = min(len(orientations), len(reference))

    def aligned(matrix: np.ndarray) -> np.ndarray:
        # При смене осей поворот преобразуется сопряжением: оси меняются и у
        # того, что поворачивают, и у результата
        return to_reference @ np.asarray(matrix, dtype=float) @ to_reference.T

    steps: list[float] = []
    for index in range(1, count):
        estimated = aligned(orientations[index - 1].T @ orientations[index])
        expected = reference[index - 1].T @ reference[index]
        steps.append(rotation_angle(expected.T @ estimated))

    final = rotation_angle(np.asarray(reference[count - 1], dtype=float).T
                           @ aligned(orientations[count - 1]))

    return {
        "ok": True,
        "final": final,
        "per_step": float(np.median(steps)),
        "steps": len(steps),
    }


# ────────────────────────────────────────────────────────────────────────────
# Сбор показателей
# ────────────────────────────────────────────────────────────────────────────

def collect(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Считает показатели накопления по каждой связке и геометрии.

    Общий множитель при сравнении с эталоном подбирается только тем геометриям,
    которые сами метрического масштаба не дают.

    Args:
        results: результаты pipeline.run по ключу связки.

    Returns:
        Список записей по одной на сочетание связки и геометрии. Поля записи:
        frontend, label, geometry, geometry_label, path_length, final,
        final_ratio, rmse, median_error, outliers, heading, relative, growth,
        distances и travelled с пройденным путём к каждой паре.
    """
    records: list[dict[str, Any]] = []

    for frontend in config.FRONTENDS:
        result = results.get(frontend["key"])
        if result is None or result["pairs"].empty:
            continue

        table = result["pairs"]
        to_reference = np.asarray(
            config.CALIBRATION[result["dataset"]]["rotation_cam_to_gt"],
            dtype=float)

        for item in config.GEOMETRIES:
            key = item["key"]
            # Отказавшая пара помечена как неметрическая, поэтому геометрия
            # считается метрической, если её дала хотя бы одна пара
            metric = bool(table[f"{key}_metric"].any())
            trajectory = result["trajectories"][key]
            reference = result["reference"]

            absolute = metrics.absolute_error(trajectory, reference,
                                              with_scale=not metric)
            relative = metrics.relative_error(trajectory, reference,
                                              with_scale=not metric)

            errors = table[f"{key}_direction_error_2d"].dropna()
            if len(errors):
                median_error = float(errors.median())
                outliers = float((errors > OUTLIER_ANGLE).mean()) * 100.0
            else:
                median_error = float("nan")
                outliers = float("nan")

            heading = heading_drift(result["orientations"][key],
                                    result["reference_rotations"], to_reference)

            # Пройденный путь к каждой паре нужен только как ось графика, но
            # считается здесь: у модуля рисования эталона нет
            travelled = (np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(reference, axis=0), axis=1))])
                if len(reference) else np.empty(0))

            records.append({
                "frontend": frontend["key"],
                "label": frontend["label"],
                "geometry": key,
                "geometry_label": item["label"],
                "path_length": absolute["path_length"],
                "final": absolute["final"],
                "final_ratio": absolute["final_ratio"] * 100,
                "rmse": absolute["rmse"],
                "median_error": median_error,
                "outliers": outliers,
                "heading": heading,
                "relative": relative,
                "growth": growth_exponent(relative),
                "distances": absolute["distances"],
                "travelled": travelled,
            })

    return records


def compare_predictors(records: list[dict[str, Any]]
                       ) -> dict[str, dict[str, Any]]:
    """
    Проверяет, что сильнее связано с итоговым дрейфом.

    Сравниваются две величины: обычная ошибка на паре и доля грубых промахов.
    Связь оценивается ранговой корреляцией Спирмена и считается отдельно по
    каждой геометрии: у моделей разная природа ошибки, и связь по объединённому
    набору отражала бы главным образом различие между моделями.

    Args:
        records: результат collect.

    Returns:
        Словарь по ключу геометрии с полями by_median, by_outliers и count.
        Поле ok равно False, если пригодных наборов меньше MIN_SETS_FOR_LINK, и
        тогда остаётся только count.
    """
    outcome: dict[str, dict[str, Any]] = {}

    for item in config.GEOMETRIES:
        own = [record for record in records if record["geometry"] == item["key"]]

        drift = np.array([record["final_ratio"] for record in own])
        median = np.array([record["median_error"] for record in own])
        outliers = np.array([record["outliers"] for record in own])

        usable = ~(np.isnan(drift) | np.isnan(median) | np.isnan(outliers))
        if usable.sum() < MIN_SETS_FOR_LINK:
            outcome[item["key"]] = {"ok": False, "count": int(usable.sum())}
            continue

        by_median, _ = stats.spearmanr(median[usable], drift[usable])
        by_outliers, _ = stats.spearmanr(outliers[usable], drift[usable])

        outcome[item["key"]] = {
            "ok": True,
            "by_median": float(by_median),
            "by_outliers": float(by_outliers),
            "count": int(usable.sum()),
        }

    return outcome


# ────────────────────────────────────────────────────────────────────────────
# Отчёт
# ────────────────────────────────────────────────────────────────────────────

def report_table(records: list[dict[str, Any]]) -> None:
    """
    Печатает таблицу накопления ошибки.

    Длина пути вынесена из таблицы в расшифровку: эталон один на все связки, и
    колонка повторяла бы одно число. Средняя ошибка вдоль маршрута и ошибка
    направления на паре сюда не входят: первая почти повторяет конечное
    расхождение, вторая напечатана в разборе геометрии.

    Args:
        records: результат collect.
    """
    headers = ["Связка", "Геометрия", "Конец, м", "Дрейф %", "Неточных %",
               "Степень"]
    rows: list[list[Any]] = []
    previous = ""

    for record in records:
        growth = record["growth"]
        # Имя связки печатается один раз на группу её геометрий
        label = record["label"] if record["label"] != previous else ""
        previous = record["label"]

        rows.append([
            label,
            record["geometry_label"],
            format_number(record["final"], 0),
            format_number(record["final_ratio"], 1),
            format_number(record["outliers"], 1),
            format_number(growth["exponent"], 2) if growth["ok"] else MISSING,
        ])

    path_length = float(np.median([record["path_length"] for record in records]))

    print_section("НАКОПЛЕНИЕ ВДОЛЬ МАРШРУТА")
    print()
    print_table(headers, rows)
    print()
    print_legend([
        ("Конец, м", f"расхождение с эталоном в конце пути длиной "
                     f"{format_number(path_length, 0)} м"),
        ("Дрейф %", "то же расхождение в долях пройденного пути"),
        ("Неточных %", f"доля пар с ошибкой направления больше "
                       f"{OUTLIER_ANGLE:.0f} градусов"),
        ("Степень", "показатель роста ошибки с длиной отрезка: около половины "
                    "случайное накопление, около единицы постоянный увод"),
    ])


def report_growth(records: list[dict[str, Any]]) -> None:
    """
    Печатает вывод о законе накопления по каждой геометрии.

    Показатель и качество подгонки берутся медианой по связкам, словесная
    трактовка самой частой из них.

    Args:
        records: результат collect.
    """
    open_block("ЗАКОН НАКОПЛЕНИЯ")

    for item in config.GEOMETRIES:
        own = [record["growth"] for record in records
               if record["geometry"] == item["key"] and record["growth"]["ok"]]
        if not own:
            block_line(item["label"], "данных не хватило", STATUS_NONE)
            continue

        exponent = float(np.median([growth["exponent"] for growth in own]))
        quality = float(np.median([growth["quality"] for growth in own]))
        kinds = [growth["kind"] for growth in own]

        block_line(item["label"],
                   f"показатель {format_number(exponent, 2)}, "
                   f"качество подгонки {format_number(quality, 2)}")
        block_line(f"{item['label']}, вывод", max(set(kinds), key=kinds.count))

    block_note("Показатель около половины означает, что ошибки шагов гасят друг "
               "друга и расхождение растёт как корень из пути. Около единицы "
               "означает постоянный увод, при котором расхождение растёт прямо "
               "пропорционально пути. По показателю оценивается, что будет на "
               "маршруте длиннее имеющегося: при показателе единица удвоение "
               "пути удваивает ошибку, при показателе половина увеличивает её "
               "лишь в полтора раза.")

    close_block()


def report_heading(records: list[dict[str, Any]]) -> None:
    """
    Печатает, насколько разошлась накопленная ориентация.

    Args:
        records: результат collect.
    """
    open_block("НАКОПЛЕНИЕ ОШИБКИ ПОВОРОТА")

    step_counts: list[int] = []

    for item in config.GEOMETRIES:
        own = [record["heading"] for record in records
               if record["geometry"] == item["key"] and record["heading"]["ok"]]
        if not own:
            block_line(item["label"], "данных не хватило", STATUS_NONE)
            continue

        final = float(np.median([heading["final"] for heading in own]))
        per_step = float(np.median([heading["per_step"] for heading in own]))
        step_counts.extend(heading["steps"] for heading in own)

        block_line(item["label"],
                   f"{format_number(final, 1)} град к концу, "
                   f"по {format_number(per_step, 2)} на шаге")

    # Число шагов одинаково у всех геометрий, поэтому печатается одной строкой
    block_line("Шагов в маршруте",
               int(np.median(step_counts)) if step_counts else MISSING)

    block_note("Ошибка поворота копится иначе, чем ошибка перемещения: неверный "
               "поворот разворачивает все последующие шаги. Расхождение к концу "
               "маршрута при этом не равно сумме ошибок шагов: повороты "
               "перемножаются и способны частично гасить друг друга, поэтому "
               "итог берётся сравнением конечных ориентаций, а не накоплением "
               "чисел.")

    close_block()


def report_predictors(comparison: dict[str, dict[str, Any]]) -> None:
    """
    Печатает, что сильнее связано с итоговым дрейфом, по каждой геометрии.

    Args:
        comparison: результат compare_predictors.
    """
    open_block("ЧТО ОПРЕДЕЛЯЕТ ДРЕЙФ")

    for item in config.GEOMETRIES:
        outcome = comparison.get(item["key"], {})

        if not outcome.get("ok"):
            block_line(item["label"],
                       f"наборов мало: {outcome.get('count', 0)}", STATUS_NONE)
            continue

        block_line(item["label"],
                   f"ошибка {format_number(outcome['by_median'], 2)}, "
                   f"промахи {format_number(outcome['by_outliers'], 2)}, "
                   f"по {outcome['count']} связкам")

    block_note("Печатаются оба значения без вывода о сильнейшем: связь "
               "считается по пяти связкам, и на таком числе точек объявлять "
               "победителя нельзя, даже когда одно значение больше другого. "
               "Смешивать геометрии тоже нельзя: у них разная природа ошибки, "
               "и общая связь отражала бы различие между моделями.")

    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Точка входа анализа
# ────────────────────────────────────────────────────────────────────────────

def run(results: dict[str, dict[str, Any]], name: str) -> None:
    """
    Выполняет анализ накопления ошибки по результатам прогона.

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
    report_growth(records)
    report_heading(records)
    report_predictors(compare_predictors(records))

    path = figures.drift(records, name)

    open_block("ГРАФИК НАКОПЛЕНИЯ")
    block_line("Файл", path.name)
    block_line("Содержание", "относительная ошибка по длине отрезка")
    close_block()
