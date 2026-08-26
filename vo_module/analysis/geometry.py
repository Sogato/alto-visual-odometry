"""
Геометрический анализ: сравнение способов восстановить движение.

Модуль отвечает на пункт задания о геометрии. Обе модели считаются по одним и
тем же сопоставленным точкам, поэтому разница между ними относится к самой
модели, а не к случайностям детектора.

Сравнения по средней ошибке недостаточно. Ошибка раскладывается на две части,
которые ведут себя по-разному. Систематическая часть это постоянный увод в одну
сторону, он накапливается вдоль маршрута прямо пропорционально пройденному
пути. Случайная часть это разброс вокруг нуля, отдельные ошибки гасят друг
друга, и накопление идёт как корень из числа шагов. Поэтому модель с большей
средней ошибкой, но случайной по природе, может дать траекторию точнее, чем
модель с меньшей, но систематической. Разделяются эти части только по знаковой
ошибке: у беззнаковой увод и разброс выглядят одинаково.

Отдельно проверяется, связана ли ошибка гомографии с охватом кадра, и
сравниваются два способа извлечь движение из одной и той же матрицы: общее
разложение и вариант с заранее заданной нормалью плоскости.
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
from console import (MISSING, STATUS_NONE, STATUS_OK, block_line, block_note,
                     block_wrapped, close_block, format_number, open_block,
                     plural, print_legend, print_section, print_table)
from vo_module import features, metrics, pose
from vo_module.dataset import Dataset
from visualization import analysis as figures


# === ФОРМЫ СЧЁТНЫХ СЛОВ ===
# Дательный падеж: «по одной паре», «по двум парам». Готовые формы из console
# стоят в именительном и здесь не подходят
PAIR_DATIVE: tuple[str, str, str] = ("паре", "парам", "парам")

# === ПАРАМЕТРЫ РАЗЛОЖЕНИЯ ОШИБКИ ===
# Разброс считается как половина межквартильного размаха, поэтому отдельные
# неудачные пары не определяют оценку
SPREAD_QUANTILES = (25, 75)      # Границы размаха, проценты

# Во сколько раз смещение должно превышать собственную погрешность, чтобы
# считаться отличимым от нуля
BIAS_SIGNIFICANCE = 2.0

# Ошибка направления выше этого угла означает не неточность, а разворот: оценка
# указывает в сторону, противоположную движению. У Essential это происходит при
# неверном выборе знака переноса, когда триангуляция вырождается на малой базе
REVERSAL_ANGLE = 150.0

# Доля разворотов, выше которой ставится отметка. Порог различает редкий отказ и
# систематический: при одном развороте на сотню шагов траектория его переживает,
# при одном на дюжину нет
MAX_REVERSALS = 2.0

# === ПАРАМЕТРЫ ДОПОЛНИТЕЛЬНОГО ОПЫТА ===
# Сравнение способов извлечь позу из гомографии идёт на отдельной небольшой
# выборке: оно не входит в основной прогон, поскольку не относится к трём
# подходам из задания
EXTRA_FRONTEND = "sift_bf"       # Связка, на которой ставится опыт
EXTRA_PAIRS = 40                 # Сколько пар кадров взять

# Разница между способами извлечь позу, ниже которой она незначима, проценты
EXTRACTION_TOLERANCE = 10.0

# === НИЖНИЕ ГРАНИЦЫ ОБЪЁМА ДАННЫХ ===
MIN_PAIRS_FOR_DECOMPOSE = 4      # Пар для разложения ошибки на части
MIN_PAIRS_FOR_CORRELATION = 10   # Пар для оценки связи ошибки с охватом кадра


# ────────────────────────────────────────────────────────────────────────────
# Разложение ошибки
# ────────────────────────────────────────────────────────────────────────────

def decompose(signed: np.ndarray) -> dict[str, Any]:
    """
    Раскладывает знаковую ошибку направления на смещение и разброс.

    Смещением берётся медиана, то есть постоянный увод в одну сторону.
    Разбросом берётся половина межквартильного размаха, то есть случайная
    часть. Медиана и квартили устойчивы к отдельным грубым промахам, которых
    при сопоставлении всегда несколько.

    Отдельно считается доля разворотов, то есть пар, где оценка указывает почти
    противоположно движению. В смещение и разброс они не входят, поскольку
    медиана и квартили к ним нечувствительны, а траекторию каждый такой шаг
    отбрасывает назад.

    Args:
        signed: знаковые ошибки направления в градусах, NaN пропускаются.

    Returns:
        Словарь с полями bias, scatter, noise с погрешностью смещения,
        significant с признаком его отличимости от нуля, reversals в процентах
        и count. При числе пар меньше MIN_PAIRS_FOR_DECOMPOSE числовые поля
        равны NaN, а significant равно False.
    """
    values = np.asarray(signed, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < MIN_PAIRS_FOR_DECOMPOSE:
        return {"bias": float("nan"), "scatter": float("nan"),
                "noise": float("nan"), "significant": False,
                "reversals": float("nan"), "count": len(values)}

    bias = float(np.median(values))
    low, high = np.percentile(values, SPREAD_QUANTILES)
    scatter = float((high - low) / 2)

    # Медиана сама определена неточно: её погрешность падает как корень из
    # числа пар. Смещение меньше этой величины неотличимо от нуля
    noise = float(scatter / np.sqrt(len(values)))

    return {
        "bias": bias,
        "scatter": scatter,
        "noise": noise,
        "significant": bool(abs(bias) > BIAS_SIGNIFICANCE * noise),
        "reversals": float(np.mean(np.abs(values) > REVERSAL_ANGLE)) * 100,
        "count": len(values),
    }


def collect(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Собирает показатели по каждой связке и геометрии.

    Доля согласных точек и невязка сюда не входят: они относятся к пункту о
    сопоставлении и печатаются там же.

    Эталонный поворот за шаг берётся из телеметрии, поэтому одинаков у всех
    связок. В каждую запись он кладётся для того, чтобы таблица не зависела от
    отдельного источника.

    Args:
        results: результаты pipeline.run по ключу связки.

    Returns:
        Список записей по одной на сочетание связки и геометрии. Поля записи:
        gt_rotation, frontend, label, geometry, geometry_label, error, bias,
        scatter, bias_noise, bias_significant, reversals, rotation_error и
        scale_error. Поле scale_error равно None у геометрий без метрического
        масштаба.
    """
    records: list[dict[str, Any]] = []
    reference_rotation: list[float] = []

    for frontend in config.FRONTENDS:
        result = results.get(frontend["key"])
        if result is None or result["pairs"].empty:
            continue

        table = result["pairs"]
        reference_rotation.append(float(table["gt_rotation"].median()))

        for item in config.GEOMETRIES:
            key = item["key"]
            parts = decompose(table[f"{key}_direction_signed"].to_numpy())

            # Отказавшая пара помечена как неметрическая, поэтому геометрия
            # считается метрической, если её дала хотя бы одна пара
            metric = bool(table[f"{key}_metric"].any())
            scale = (metrics.scale_error(result["trajectories"][key],
                                         result["reference"])
                     if metric else {"ok": False})

            records.append({
                "gt_rotation": float("nan"),
                "frontend": frontend["key"],
                "label": frontend["label"],
                "geometry": key,
                "geometry_label": item["label"],
                "error": float(table[f"{key}_direction_error_2d"].median()),
                "bias": parts["bias"],
                "scatter": parts["scatter"],
                "bias_noise": parts["noise"],
                "bias_significant": parts["significant"],
                "reversals": parts["reversals"],
                "rotation_error": float(table[f"{key}_rotation_error"].median()),
                "scale_error": (scale.get("error_percent") if scale.get("ok")
                                else None),
            })

    # Эталонный поворот проставляется после обхода: иначе первые записи
    # получили бы медиану по неполному набору связок
    if reference_rotation:
        median_rotation = float(np.median(reference_rotation))
        for record in records:
            record["gt_rotation"] = median_rotation

    return records


def correlate_spread(results: dict[str, dict[str, Any]],
                     geometry: str) -> dict[str, Any]:
    """
    Проверяет, связана ли ошибка с тем, насколько широко точки покрывают кадр.

    Если сопоставления скапливаются в одной части кадра, модель закреплена
    только там, а за пределами этой области ничем не ограничена. Проверка
    ведётся по всем парам всех связок сразу: чем шире набор условий, тем
    надёжнее вывод о наличии связи.

    Связь оценивается ранговой корреляцией Спирмена: зависимость не обязана
    быть линейной, важно лишь, растёт ли ошибка вместе с кучностью точек.

    Args:
        results: результаты pipeline.run по ключу связки.
        geometry: ключ геометрии.

    Returns:
        Словарь с полями correlation, spread, error, owner с ключом связки для
        каждой точки и count. Поле ok равно False, если пар меньше
        MIN_PAIRS_FOR_CORRELATION, и тогда остальных полей нет.
    """
    spread: list[float] = []
    error: list[float] = []
    owner: list[str] = []

    for key, result in results.items():
        table = result["pairs"]
        if table.empty:
            continue
        values = table[["spread", f"{geometry}_direction_error_2d"]].dropna()
        spread.extend(values["spread"].tolist())
        error.extend(values[f"{geometry}_direction_error_2d"].tolist())
        owner.extend([key] * len(values))

    if len(spread) < MIN_PAIRS_FOR_CORRELATION:
        return {"ok": False}

    correlation, _ = stats.spearmanr(spread, error)
    return {
        "ok": True,
        "correlation": float(correlation),
        "spread": np.array(spread),
        "error": np.array(error),
        "owner": owner,
        "count": len(spread),
    }


# ────────────────────────────────────────────────────────────────────────────
# Дополнительный опыт: способы извлечь позу из гомографии
# ────────────────────────────────────────────────────────────────────────────

def compare_extraction(name: str) -> dict[str, Any]:
    """
    Сравнивает два способа получить движение из одной и той же гомографии.

    Общее разложение ищет одновременно поворот, сдвиг и наклон наблюдаемой
    плоскости. На съёмке земли сверху эти величины плохо разделяются: сдвиг
    картинки одинаково объясняется и перемещением камеры, и её наклоном. Второй
    способ наклон не ищет, а задаёт: камера смотрит вниз, значит нормаль земли
    совпадает с оптической осью, и неизвестных остаётся на одну меньше.

    Опыт отвечает на вопрос, в оценке ли плоскости причина ошибки гомографии.
    Для сравнения рядом считается Essential Matrix на тех же точках.

    Args:
        name: имя датасета.

    Returns:
        Словарь с полями frontend, pairs, errors с медианной ошибкой каждого
        способа в градусах и counts с числом удавшихся пар. Поле ok равно
        False, если связки EXTRA_FRONTEND нет в конфиге, и тогда остальных
        полей нет.

    Raises:
        TypeError: не заполнен config.FRAME_STEP.
        KeyError: датасета нет в конфиге.
        FileNotFoundError: нет файла телеметрии или каталога с кадрами.
    """
    frontend = next((item for item in config.FRONTENDS
                     if item["key"] == EXTRA_FRONTEND), None)
    if frontend is None:
        return {"ok": False}

    calibration = config.CALIBRATION[name]
    data = Dataset(name)
    intrinsics = pose.camera_matrix(calibration["focal_px"],
                                    calibration["principal_point"])
    to_reference = np.asarray(calibration["rotation_cam_to_gt"], dtype=float)

    detector = features.DETECTORS[frontend["detector"]]
    matcher = features.MATCHERS[frontend["matcher"]]
    pairs = data.pairs(config.FRAME_STEP, limit=EXTRA_PAIRS)

    methods = {
        "Общее разложение": pose.pose_from_homography,
        "Заданная нормаль": pose.EXTRA_POSE_BY_KEY["homography_fixed"],
        "Essential": pose.pose_from_essential,
    }
    errors: dict[str, list[float]] = {label: [] for label in methods}
    previous_index: int | None = None
    previous: dict[str, Any] | None = None

    for first_index, second_index in pairs:
        first_gray = data.gray(first_index)
        second_gray = data.gray(second_index)

        # Пары идут встык, поэтому детекция второго кадра пары годится первым
        # кадром следующей и вдвое сокращает работу детектора
        first = (previous if previous_index == first_index
                 else detector(first_gray))
        second = detector(second_gray)
        previous_index, previous = second_index, second

        matches = matcher(first, second, first_gray, second_gray)
        if matches["count"] < config.MIN_MATCHES_FOR_GEOMETRY:
            continue

        motion = data.relative_motion(first_index, second_index)
        reference = np.asarray(motion["delta_body"], dtype=float)[:2]
        if np.linalg.norm(reference) == 0:
            continue
        reference = reference / np.linalg.norm(reference)

        for label, function in methods.items():
            outcome = function(matches["points_first"],
                               matches["points_second"], intrinsics)
            if not outcome["ok"]:
                continue

            estimated = (to_reference @ np.asarray(outcome["translation"],
                                                   dtype=float))[:2]
            norm = np.linalg.norm(estimated)
            if norm == 0:
                continue

            cosine = float(np.clip(np.dot(estimated / norm, reference),
                                   -1.0, 1.0))
            errors[label].append(float(np.degrees(np.arccos(cosine))))

    return {
        "ok": True,
        "frontend": frontend["label"],
        "pairs": len(pairs),
        "errors": {label: (float(np.median(values)) if values else float("nan"))
                   for label, values in errors.items()},
        "counts": {label: len(values) for label, values in errors.items()},
    }


# ────────────────────────────────────────────────────────────────────────────
# Отчёт
# ────────────────────────────────────────────────────────────────────────────

def report_table(records: list[dict[str, Any]]) -> None:
    """
    Печатает таблицу показателей по связкам и геометриям.

    Увод меньше собственной погрешности медианы печатается пометкой вместо
    числа: он неотличим от нуля.

    Args:
        records: результат collect.
    """
    headers = ["Связка", "Геометрия", "Ошибка", "Увод", "Разброс", "Поворот",
               "Масштаб"]
    reference = float(np.median([record["gt_rotation"] for record in records]))
    rows: list[list[Any]] = []
    previous = ""

    for record in records:
        # Название связки печатается один раз на пару строк: повтор в соседней
        # строке читался бы как отдельное измерение
        label = record["label"] if record["label"] != previous else ""
        previous = record["label"]

        rows.append([
            label,
            record["geometry_label"],
            format_number(record["error"], 1),
            (format_number(record["bias"], 1) if record["bias_significant"]
             else "~0"),
            format_number(record["scatter"], 1),
            format_number(record["rotation_error"], 2),
            (format_number(record["scale_error"], 1)
             if record["scale_error"] is not None else MISSING),
        ])

    print_section("РАЗЛОЖЕНИЕ ОШИБКИ ПО ГЕОМЕТРИЯМ")
    print()
    print_table(headers, rows)
    print()
    print_legend([
        ("Ошибка", "ошибка направления движения на паре, град"),
        ("Увод", "постоянная часть ошибки, увод в одну сторону, град"),
        ("Разброс", "случайная часть ошибки, град"),
        ("Поворот", f"угол между оценённым и эталонным поворотом за шаг, град, "
                    f"при повороте за шаг {format_number(reference, 2)} град"),
        ("Масштаб", "отклонение длины пути от эталонной, %, у Essential "
                    "масштаба нет"),
    ])


def report_decomposition(records: list[dict[str, Any]]) -> None:
    """
    Печатает вывод о природе ошибки каждой модели.

    Смещение сравнивается с собственной погрешностью: медиана по конечному
    числу пар сама определена неточно, и значение меньше этой погрешности
    ничего не говорит об уводе.

    Args:
        records: результат collect.
    """
    open_block("ПРИРОДА ОШИБКИ")

    for item in config.GEOMETRIES:
        own = [record for record in records if record["geometry"] == item["key"]]
        if not own:
            continue

        bias = float(np.nanmedian([abs(record["bias"]) for record in own]))
        scatter = float(np.nanmedian([record["scatter"] for record in own]))
        noise = float(np.nanmedian([record["bias_noise"] for record in own]))
        significant = sum(1 for record in own if record["bias_significant"])

        block_line(item["label"],
                   f"смещение {format_number(bias, 1)} +- "
                   f"{format_number(noise, 1)}, разброс "
                   f"{format_number(scatter, 1)} град")

        if significant == 0:
            verdict, status = "увода нет, ошибка случайная", STATUS_OK
        elif bias > scatter:
            verdict, status = "преобладает постоянный увод", STATUS_NONE
        else:
            verdict = f"увод у {significant} из {len(own)}, разброс больше"
            status = STATUS_OK

        block_line(f"{item['label']}, вывод", verdict, status)

        reversals = float(np.nanmedian([record["reversals"] for record in own]))
        block_line(f"{item['label']}, разворотов",
                   f"{format_number(reversals, 1)} % пар",
                   STATUS_OK if reversals <= MAX_REVERSALS else STATUS_NONE)

    block_note("Постоянный увод накапливается вдоль маршрута прямо "
               "пропорционально пройденному пути, случайный разброс как корень "
               "из числа шагов. Если увода нет, накопление идёт медленнее, и "
               "модель с большей ошибкой на паре может дать траекторию точнее.")

    block_note("Разложение относится только к направлению движения. Ошибка "
               "длины шага в него не входит, а копится она линейно, поэтому "
               "вывод об отсутствии увода не означает, что и накопление пойдёт "
               "как у случайной ошибки: показатель роста в разборе накопления "
               "учитывает обе части и выходит выше.")

    block_note(f"Развороты это пары с ошибкой больше "
               f"{REVERSAL_ANGLE:.0f} градусов: оценка указывает назад. Ни в "
               f"смещение, ни в разброс они не попадают, поскольку медиана и "
               f"квартили к ним нечувствительны, но на траекторию влияют "
               f"сильнее любой неточности. Отметка ставится выше "
               f"{MAX_REVERSALS:.0f} процентов.")

    close_block()


def report_correlation(correlations: dict[str, dict[str, Any]]) -> None:
    """
    Печатает результат проверки связи ошибки с охватом кадра.

    Печатается само значение связи, без вердикта по порогу: порог давал бы один
    и тот же ответ и там, где связи нет, и там, где она слаба, но устойчива.

    Args:
        correlations: результат correlate_spread по каждой геометрии.
    """
    open_block("ЗАВИСИМОСТЬ ОТ ОХВАТА КАДРА")

    for item in config.GEOMETRIES:
        correlation = correlations.get(item["key"], {})
        if not correlation.get("ok"):
            block_line(item["label"], "данных не хватило", STATUS_NONE)
            continue

        count = correlation["count"]
        block_line(item["label"],
                   f"{format_number(correlation['correlation'], 2)} "
                   f"по {count} {plural(count, PAIR_DATIVE)}")

    block_note("Положительная связь означает, что ошибка растёт вместе с "
               "охватом, отрицательная что падает. Кучные точки закрепляют "
               "модель лишь в одной части кадра, поэтому у гомографии ожидается "
               "отрицательная связь.")

    close_block()


def report_extraction(extraction: dict[str, Any]) -> None:
    """
    Печатает результат сравнения способов извлечь позу из гомографии.

    Знак разницы говорит о разном: фиксация нормали может как помочь, так и
    помешать, и выводы из этих случаев противоположны.

    Args:
        extraction: результат compare_extraction.
    """
    open_block("СПОСОБЫ ИЗВЛЕЧЬ ПОЗУ ИЗ ГОМОГРАФИИ")

    if not extraction.get("ok"):
        block_line("Состояние", "опыт не выполнен", STATUS_NONE)
        close_block()
        return

    block_line("Связка", extraction["frontend"])
    block_line("Пар в опыте", extraction["pairs"])

    for label, value in extraction["errors"].items():
        block_line(label, f"{format_number(value, 1)} град, "
                          f"удалось на {extraction['counts'][label]} парах")

    general = extraction["errors"].get("Общее разложение", float("nan"))
    fixed = extraction["errors"].get("Заданная нормаль", float("nan"))

    if not np.isnan(general) and not np.isnan(fixed):
        difference = (fixed - general) / general * 100 if general else float("nan")

        if abs(difference) < EXTRACTION_TOLERANCE:
            verdict = "оценка наклона плоскости на результат не влияет"
        elif difference > 0:
            verdict = ("фиксация нормали ухудшает результат: наклон плоскости "
                       "не совпадает с оптической осью, и его оценка полезна")
        else:
            verdict = ("фиксация нормали улучшает результат: оценка наклона "
                       "вносит больше ошибки, чем пользы")

        block_line("Разница", format_number(difference, 1, "%"))
        block_wrapped("Вывод", verdict)

    block_note("Общее разложение ищет поворот, сдвиг и наклон плоскости "
               "одновременно. Вариант с заданной нормалью наклон не ищет: "
               "камера смотрит вниз, значит нормаль земли можно считать "
               "совпадающей с оптической осью. Сравнение показывает, помогает "
               "ли такое упрощение или наблюдаемая поверхность на самом деле "
               "заметно наклонена.")

    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Точка входа анализа
# ────────────────────────────────────────────────────────────────────────────

def run(results: dict[str, dict[str, Any]], name: str) -> None:
    """
    Выполняет геометрический анализ по результатам прогона.

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

    correlations = {item["key"]: correlate_spread(results, item["key"])
                    for item in config.GEOMETRIES}

    report_table(records)
    report_decomposition(records)
    report_correlation(correlations)
    report_extraction(compare_extraction(name))

    # Знаковые ошибки по всем связкам сразу: гистограмма строится по общему
    # ряду, а чтение таблицы прогона это работа анализа, а не рисования
    signed: dict[str, np.ndarray] = {}
    for item in config.GEOMETRIES:
        column = f"{item['key']}_direction_signed"
        values = [result["pairs"][column].dropna().to_numpy()
                  for result in results.values() if not result["pairs"].empty]
        signed[item["key"]] = np.concatenate(values) if values else np.empty(0)

    paths = figures.geometry(records, correlations[config.GEOMETRIES[0]["key"]],
                             signed, name)

    open_block("ГРАФИКИ ГЕОМЕТРИИ")
    block_line("Ошибка направления", paths[0].name)
    block_line("Восстановление поворота", paths[1].name)
    close_block()
