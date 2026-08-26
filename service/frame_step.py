"""
Исследование шага прореживания кадров.

Соседние кадры сняты примерно через два с половиной метра пути, и картинка
между ними смещается на единицы пикселей, что сопоставимо с точностью
позиционирования точки. Поэтому пары составляются не из соседних кадров, а
через FRAME_STEP кадров.

Увеличивать шаг бесконечно нельзя: кадры перестают перекрываться, а оптический
поток теряет применимость, поскольку ищет точку в небольшом окне рядом с
прежним положением. Модуль перебирает сетку config.FRAME_STEP_GRID и находит
промежуток, где шаг уже достаточен для геометрии, но ещё не разрушил
сопоставление.

Все значения шага прогоняются по одному и тому же участку маршрута, а не по
одинаковому числу пар: иначе при большом шаге охват по километрам оказался бы
кратно больше, и накопленную ошибку нельзя было бы сравнивать. Перебор идёт на
нескольких участках разной длины, чтобы видеть, зависит ли выбор от того, какой
кусок маршрута попался.

Результат работы модуля это значение FRAME_STEP для config.
"""

# Стандартные библиотеки
import sys
import time
from pathlib import Path
from typing import Any

# Сторонние библиотеки
import numpy as np

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from console import (FRAME_FORMS, MISSING, STATUS_FAIL, block_line,
                     clear_progress, close_block, describe_exception,
                     format_number, open_block, plural, print_banner,
                     print_legend, print_progress, print_section, print_table)
from vo_module import metrics, pipeline
from vo_module.dataset import Dataset
from visualization import service as figures


# === ПАРАМЕТРЫ ИССЛЕДОВАНИЯ ===
STUDY_DATASET = "val_sample"                # Датасет, на котором ведётся перебор
STUDY_SPANS: tuple[int, ...] = (300, 600, 900)   # Длины участков, кадров

# === ПОРОГИ ПРИГОДНОСТИ ===
MIN_SUCCESS_RATE = 0.9    # Доля удавшихся пар, ниже которой связка не работает
MIN_PAIRS = 8             # Нижняя граница числа пар для осмысленной оценки

# Накопленная ошибка выше этой доли пути означает развалившуюся траекторию.
# Такие значения отбрасываются: они смещают медиану и растягивают масштаб графика
MAX_SANE_DRIFT = 200.0

# === ФОРМЫ СЧЁТНЫХ СЛОВ ===
# Порядок форм: одна конфигурация, две конфигурации, пять конфигураций
CONFIG_FORMS: tuple[str, str, str] = ("конфигурация", "конфигурации",
                                      "конфигураций")


# ────────────────────────────────────────────────────────────────────────────
# Прогон сетки
# ────────────────────────────────────────────────────────────────────────────

def expected_shift(table: Any, focal: float) -> float:
    """
    Считает, на сколько пикселей должна была сместиться картинка.

    Для надирной камеры смещение равно фокусному расстоянию, умноженному на
    пройденный путь и делённому на высоту.

    Args:
        table: таблица по парам из прогона.
        focal: фокусное расстояние, px.

    Returns:
        Медианное ожидаемое смещение, px. NaN при пустой таблице.
    """
    if table.empty:
        return float("nan")
    shifts = focal * table["gt_distance"] / table["height_m"]
    return float(np.median(shifts))


def measure(data: Dataset, frontend: dict[str, str], step: int, span: int,
            focal: float) -> dict[str, Any]:
    """
    Прогоняет одну связку на одном значении шага и сводит результат к числам.

    Накопленная ошибка считается по отношению к пройденному пути и отбрасывается,
    если превышает MAX_SANE_DRIFT. Она измеряется с подгонкой масштаба, если
    геометрия сама метрического масштаба не даёт.

    Args:
        data: загруженный датасет.
        frontend: описание связки из config.FRONTENDS.
        step: шаг прореживания.
        span: длина участка, кадров.
        focal: фокусное расстояние, px.

    Returns:
        Словарь с полями frontend, step, pairs, shift_px, matches и по одному
        полю на каждую геометрию. Поле геометрии содержит success_rate,
        inlier_ratio, direction_error и drift, либо ok со значением False.
    """
    result = pipeline.run(data, frontend, step, stop=span, progress_label="")
    table = result["pairs"]

    record: dict[str, Any] = {
        "frontend": frontend["key"],
        "step": step,
        "pairs": len(table),
        "shift_px": expected_shift(table, focal),
        "matches": float(table["matches"].median()) if not table.empty else 0.0,
    }

    for item in config.GEOMETRIES:
        key = item["key"]
        if table.empty:
            record[key] = {"ok": False}
            continue

        successes = int(table[f"{key}_ok"].sum())
        metric = bool(table[f"{key}_metric"].iloc[0])

        drift = float("nan")
        if len(table) >= MIN_PAIRS:
            absolute = metrics.absolute_error(result["trajectories"][key],
                                              result["reference"],
                                              with_scale=not metric)
            value = absolute["final_ratio"] * 100.0
            drift = value if value <= MAX_SANE_DRIFT else float("nan")

        record[key] = {
            "ok": True,
            "success_rate": successes / len(table),
            "inlier_ratio": float(table[f"{key}_inlier_ratio"].median()),
            "direction_error": float(table[f"{key}_direction_error_2d"].median()),
            "drift": drift,
        }

    return record


def run_grid(data: Dataset, span: int) -> list[dict[str, Any]]:
    """
    Прогоняет все связки на всех значениях шага из config.FRAME_STEP_GRID.

    Args:
        data: загруженный датасет.
        span: длина участка, кадров.

    Returns:
        Список записей measure по одной на комбинацию связки и шага. Отказавшая
        комбинация даёт запись только с полями frontend, step и pairs, поэтому
        отсеивается функцией usable.
    """
    focal = config.CALIBRATION[data.name]["focal_px"]
    total = len(config.FRONTENDS) * len(config.FRAME_STEP_GRID)
    label = f"{data.name}: участок {span} кадров"

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    done = 0

    for frontend in config.FRONTENDS:
        for step in config.FRAME_STEP_GRID:
            print_progress(done, total, label)
            try:
                records.append(measure(data, frontend, step, span, focal))
            except Exception:
                records.append({"frontend": frontend["key"],
                                "step": step, "pairs": 0})
            done += 1

    clear_progress(label, total, time.perf_counter() - started, CONFIG_FORMS)
    return records


# ────────────────────────────────────────────────────────────────────────────
# Выбор рабочего значения
# ────────────────────────────────────────────────────────────────────────────

def usable(record: dict[str, Any], geometry: str) -> bool:
    """
    Проверяет, работает ли связка на этом шаге при заданной геометрии.

    Args:
        record: запись из run_grid.
        geometry: ключ геометрии.

    Returns:
        True, если пар не меньше MIN_PAIRS и доля удавшихся не меньше
        MIN_SUCCESS_RATE.
    """
    info = record.get(geometry)
    if not info or not info.get("ok"):
        return False
    return (record["pairs"] >= MIN_PAIRS
            and info["success_rate"] >= MIN_SUCCESS_RATE)


def usable_everywhere(record: dict[str, Any]) -> bool:
    """
    Проверяет, что связка отработала при обеих геометриях.

    Шаг прореживания один на всё сравнение, поэтому пригодным считается только
    тот, при котором работают все связки при всех геометриях.

    Args:
        record: запись из run_grid.

    Returns:
        True, если связка отработала при каждой геометрии.
    """
    return all(usable(record, item["key"]) for item in config.GEOMETRIES)


def choose_step(records: list[dict[str, Any]], geometry: str) -> dict[str, Any]:
    """
    Выбирает шаг по наихудшей из связок.

    Шаг общий для всего дальнейшего сравнения, поэтому берётся тот, при котором
    худшая связка ошибается меньше всего. Выбор по среднему подстроил бы условия
    сравнения под ту часть методов, которых большинство.

    Показатели считаются по заданной геометрии, а пригодность шага по всем
    сразу.

    Args:
        records: результат run_grid.
        geometry: ключ геометрии, по которой ведётся выбор.

    Returns:
        Словарь с полями chosen, usable_steps и summary. Поле chosen равно None,
        если ни на одном шаге не отработали все связки. Поле usable_steps
        перечисляет все шаги, где отработали все связки. Элемент summary
        содержит step, working, working_everywhere, total, shift_px,
        worst_error, median_error и all_working.
    """
    by_step: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_step.setdefault(record["step"], []).append(record)

    summary: list[dict[str, Any]] = []
    for step, group in sorted(by_step.items()):
        working = [item for item in group if usable(item, geometry)]
        working_everywhere = [item for item in group if usable_everywhere(item)]
        errors = [item[geometry]["direction_error"] for item in working]
        shifts = [item["shift_px"] for item in group
                  if not np.isnan(item.get("shift_px", np.nan))]

        summary.append({
            "step": step,
            "working": len(working),
            "working_everywhere": len(working_everywhere),
            "total": len(group),
            "shift_px": float(np.median(shifts)) if shifts else float("nan"),
            "worst_error": float(np.max(errors)) if errors else float("nan"),
            "median_error": float(np.median(errors)) if errors else float("nan"),
            "all_working": len(working_everywhere) == len(group),
        })

    complete = [item for item in summary if item["all_working"]
                and not np.isnan(item["worst_error"])]
    chosen = (min(complete, key=lambda item: item["worst_error"])["step"]
              if complete else None)

    return {"chosen": chosen,
            "usable_steps": [item["step"] for item in complete],
            "summary": summary}


# ────────────────────────────────────────────────────────────────────────────
# Подготовка рядов для графика
# ────────────────────────────────────────────────────────────────────────────

def build_series(records: list[dict[str, Any]],
                 geometry: str) -> dict[str, dict[str, list[float]]]:
    """
    Разворачивает записи прогона в плоские ряды по каждой связке.

    Неудавшиеся точки заменяются на NaN, поэтому линия на графике рвётся в месте
    отказа, а не соединяет соседние значения через пропуск. Доля согласных точек
    переводится в проценты.

    Args:
        records: результат run_grid.
        geometry: ключ геометрии.

    Returns:
        Словарь «ключ связки: ряды», где ряды содержат steps, matches,
        inlier_ratio, direction_error и drift.
    """
    series: dict[str, dict[str, list[float]]] = {}

    for frontend in config.FRONTENDS:
        own = sorted([item for item in records
                      if item["frontend"] == frontend["key"]],
                     key=lambda item: item["step"])
        if not own:
            continue

        rows: dict[str, list[float]] = {"steps": [item["step"] for item in own],
                                        "matches": [item.get("matches", float("nan"))
                                                    for item in own]}

        for field in ("inlier_ratio", "direction_error", "drift"):
            values = []
            for item in own:
                outcome = item.get(geometry, {})
                value = (outcome.get(field, float("nan"))
                         if outcome.get("ok") else float("nan"))
                if field == "inlier_ratio" and not np.isnan(value):
                    value *= 100
                values.append(value)
            rows[field] = values

        series[frontend["key"]] = rows

    return series


# ────────────────────────────────────────────────────────────────────────────
# Отчёт по участку
# ────────────────────────────────────────────────────────────────────────────

def number_or_blank(value: float) -> str:
    """
    Приводит показатель к строке, оставляя пустоту вместо отсутствующего числа.

    Пустая ячейка, а не пометка словом, сохраняет колонке числовое выравнивание.

    Args:
        value: измеренное значение либо NaN.

    Returns:
        Число с одним знаком после запятой либо пустую строку.
    """
    return "" if np.isnan(value) else format_number(value, 1)


def report_steps(span: int, choices: dict[str, dict[str, Any]]) -> None:
    """
    Печатает таблицу по шагам сразу для всех геометрий.

    Строки шагов, ожидаемый сдвиг картинки и пригодность шага у геометрий общие,
    различаются только столбцы показателей. Накопленная ошибка в таблицу не
    выводится: у гомографии масштаб задаёт высота, а Essential получает его
    подгонкой под эталон, поэтому соседние столбцы приглашали бы сравнить
    величины, полученные по разным правилам.

    Args:
        span: длина участка, кадров.
        choices: результат choose_step по каждой геометрии.
    """
    print_section(f"ЗАВИСИМОСТЬ ОТ ШАГА, УЧАСТОК {span} КАДРОВ")
    print()

    geometries = [item for item in config.GEOMETRIES if item["key"] in choices]

    headers = ["Шаг", "Сдвиг, px", "Везде"]
    for item in geometries:
        letter = item["label"][0]
        headers += [f"{letter}: худшая", f"{letter}: медиана"]

    # Шаги и ожидаемый сдвиг берутся из любой геометрии: они общие
    base = choices[geometries[0]["key"]]["summary"]
    rows: list[list[Any]] = []

    for position, entry in enumerate(base):
        row: list[Any] = [entry["step"],
                          format_number(entry["shift_px"], 0),
                          f"{entry['working_everywhere']} из {entry['total']}"]
        for item in geometries:
            own = choices[item["key"]]["summary"][position]
            row += [number_or_blank(own["worst_error"]),
                    number_or_blank(own["median_error"])]
        rows.append(row)

    print_table(headers, rows)


def report_step_legend() -> None:
    """
    Печатает расшифровку колонок таблицы зависимости от шага.

    Печатается один раз на весь отчёт: обозначения от участка не зависят.
    """
    print()
    print_legend([(item["label"][0], item["label"]) for item in config.GEOMETRIES]
                 + [("Везде", "сколько связок отработало при всех геометриях"),
                    ("Худшая", "ошибка направления у худшей связки, град"),
                    ("Медиана", "медиана ошибки направления по связкам, град"),
                    ("пусто", "геометрия на этом шаге не отработала")])


def report_frontends(records: list[dict[str, Any]], span: int) -> None:
    """
    Печатает ошибку направления по каждой связке и каждому шагу.

    Это единственное место, где видно, какая именно связка ограничивает шаг
    сверху: сводная таблица показывает лишь худшее значение, не называя
    виновника. Таблица печатается по первой геометрии из config.GEOMETRIES, её
    название выносится в заголовок.

    Args:
        records: результат run_grid.
        span: длина участка, кадров.
    """
    geometry = config.GEOMETRIES[0]["key"]
    geometry_label = config.GEOMETRIES[0]["label"]
    steps = sorted({item["step"] for item in records})

    headers = ["Связка"] + [str(step) for step in steps]
    rows: list[list[Any]] = []

    for frontend in config.FRONTENDS:
        row: list[Any] = [frontend["label"]]
        for step in steps:
            found = next((item for item in records
                          if item["frontend"] == frontend["key"]
                          and item["step"] == step), None)
            if found is None or not found.get(geometry, {}).get("ok"):
                row.append("")
            else:
                row.append(format_number(found[geometry]["direction_error"], 1))
        rows.append(row)

    print_section(f"ОШИБКА НАПРАВЛЕНИЯ ПО СВЯЗКАМ, ГРАДУСЫ, "
                  f"{geometry_label.upper()}, УЧАСТОК {span} КАДРОВ")
    print()
    print_table(headers, rows)


# ────────────────────────────────────────────────────────────────────────────
# Итоговый выбор
# ────────────────────────────────────────────────────────────────────────────

def report_choice(by_span: dict[int, dict[str, int | None]],
                  usable_by_span: list[list[int]]) -> int | None:
    """
    Печатает выбор по каждому участку и итоговое значение для config.

    Совпадение выбора на участках разной длины означает, что он не зависит от
    того, какой кусок маршрута попался.

    Итоговым берётся наименьший из шагов, пригодных сразу на всех участках при
    всех геометриях: при равной пригодности короткий шаг даёт больше пар и
    подробнее траекторию. Лучший на отдельном участке шаг для этого не годится,
    поскольку на другом участке может не работать вовсе.

    Args:
        by_span: выбранный шаг по каждой длине участка и каждой геометрии.
        usable_by_span: списки пригодных шагов, по одному на каждую пару
            участка и геометрии.

    Returns:
        Выбранный шаг либо None, если пригодных везде шагов нет.
    """
    print_section("ВЫБОР ПО УЧАСТКАМ")
    print()

    headers = ["Участок, кадров"] + [item["label"] for item in config.GEOMETRIES]
    rows = [[span] + [choices.get(item["label"]) or MISSING
                      for item in config.GEOMETRIES]
            for span, choices in sorted(by_span.items())]
    print_table(headers, rows)

    # Пригоден тот шаг, что отработал в каждом случае без исключения
    common: set[int] | None = None
    for steps in usable_by_span:
        common = set(steps) if common is None else common & set(steps)
    common = common or set()

    best = [value for choices in by_span.values()
            for value in choices.values() if value is not None]

    print_section("ЗНАЧЕНИЕ ДЛЯ CONFIG")
    print()

    if not common:
        print("    FRAME_STEP: определить не удалось")
        print()
        if best:
            print("    Ни один шаг не отработал на всех участках при обеих "
                  "геометриях сразу.")
            print()
        return None

    chosen = min(common)
    print(f"    FRAME_STEP = {chosen}    # {plural(chosen, FRAME_FORMS)}")
    print()

    print(f"    Пригодны везде: {', '.join(str(step) for step in sorted(common))}.")
    print()

    if best and min(best) < chosen:
        print(f"    При меньшем шаге, {min(best)}, показатель лучше,")
        print("    но там отрабатывают не все связки.")
        print()

    return chosen


# ────────────────────────────────────────────────────────────────────────────
# Точка входа
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Перебирает шаг прореживания на нескольких участках и выбирает значение.
    """
    print_banner([
        "ИССЛЕДОВАНИЕ ШАГА ПРОРЕЖИВАНИЯ",
        f"датасет {STUDY_DATASET}, участки {', '.join(map(str, STUDY_SPANS))} кадров",
    ])

    calibration = config.CALIBRATION.get(STUDY_DATASET, {})
    missing = [field for field in config.CALIBRATION_REQUIRED
               if calibration.get(field) is None]
    if missing:
        open_block("ПОДГОТОВКА")
        block_line("Калибровка", "не заполнено: " + ", ".join(missing), STATUS_FAIL)
        close_block()
        return

    try:
        data = Dataset(STUDY_DATASET)
    except Exception as error:
        open_block("ПОДГОТОВКА")
        block_line("Загрузка", describe_exception(error), STATUS_FAIL)
        close_block()
        return

    spans = [span for span in STUDY_SPANS if span <= len(data)]
    if not spans:
        open_block("ПОДГОТОВКА")
        block_line("Длина датасета", f"{len(data)} кадров, короче самого "
                                     f"короткого участка", STATUS_FAIL)
        close_block()
        return

    by_span: dict[int, dict[str, int | None]] = {}
    usable_by_span: list[list[int]] = []
    last_records: list[dict[str, Any]] = []
    last_span = 0
    last_usable: list[int] = []

    for span in spans:
        # Отбивка ставится перед индикатором, а не перед заголовком таблицы:
        # заголовок отбивает begin_line внутри print_section, а строка
        # индикатора иначе прилипает к последней строке предыдущей таблицы
        if span != spans[0]:
            print()

        records = run_grid(data, span)

        choices: dict[str, dict[str, Any]] = {}
        for item in config.GEOMETRIES:
            choice = choose_step(records, item["key"])
            choices[item["key"]] = choice
            usable_by_span.append(choice["usable_steps"])

        report_steps(span, choices)

        # Расшифровка и разбор по связкам печатаются один раз, после последней
        # таблицы: обозначения от участка не зависят, а виновник ограничения
        # шага от его длины не меняется
        if span == spans[-1]:
            report_step_legend()
            report_frontends(records, span)
            last_records, last_span = records, span
            # Шаги, пригодные на этом участке при обеих геометриях: именно они
            # задают затенение на картинке
            last_usable = sorted(set.intersection(
                *(set(choice["usable_steps"]) for choice in choices.values())))

        by_span[span] = {item["label"]: choices[item["key"]]["chosen"]
                         for item in config.GEOMETRIES}

    chosen = report_choice(by_span, usable_by_span)

    # Картинка строится одна, по самому длинному участку, но по обеим
    # геометриям: затенение слева объясняется именно второй из них, и без её
    # кривых граница выглядела бы взятой с потолка.
    #
    # Затенение считается по этому же участку, а не по всему исследованию:
    # иначе шаг оказался бы затенён из-за короткого участка, которого на
    # картинке нет. Выбор рабочего значения при этом остаётся общим, его
    # показывают таблицы
    if chosen is not None and last_records:
        figures.frame_step(
            {item["key"]: build_series(last_records, item["key"])
             for item in config.GEOMETRIES},
            STUDY_DATASET, last_span, chosen, last_usable)

    print()


if __name__ == "__main__":
    main()
