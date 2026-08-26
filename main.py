"""
Точка входа: прогон одометрии и построение анализа.

Модуль спрашивает, на каком датасете работать и какой анализ строить, прогоняет
все связки и передаёт результаты выбранным анализам.

Каждая связка проходит по последовательности один раз, а обе геометрии
считаются поверх уже сопоставленных точек. Поэтому наборов результатов
получается десять при пяти проходах по кадрам, и разница между геометриями
относится к модели, а не к случайностям детектора.

Отказ одного анализа не мешает остальным.
"""

# Стандартные библиотеки
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Сторонние библиотеки
import numpy as np

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from console import (MISSING, PROGRESS_FORMS, STATUS_FAIL, block_line,
                     block_note, block_wrapped, clear_progress, close_block,
                     describe_exception, format_number, open_block,
                     print_banner, print_legend, print_section, print_table)
from vo_module import metrics, pipeline
from vo_module.dataset import Dataset
from visualization import analysis as figures


# === АНАЛИЗЫ ===
# Соответствуют пунктам задания. Каждый модуль предоставляет функцию run,
# принимающую результаты прогона и имя датасета.
#
# Буква пункта хранится здесь, а не в самом модуле анализа: место в задании это
# свойство отчёта, а не измерения
ANALYSES: tuple[dict[str, str], ...] = (
    {"key": "keypoints", "letter": "A", "label": "Устойчивость ключевых точек",
     "module": "vo_module.analysis.keypoints"},
    {"key": "matching", "letter": "B", "label": "Качество сопоставления",
     "module": "vo_module.analysis.matching"},
    {"key": "geometry", "letter": "C", "label": "Геометрический анализ",
     "module": "vo_module.analysis.geometry"},
    {"key": "drift", "letter": "D", "label": "Накопление ошибки",
     "module": "vo_module.analysis.drift"},
    {"key": "performance", "letter": "E", "label": "Нагрузка на устройство",
     "module": "vo_module.analysis.performance"},
)

# === ОТЛАДОЧНЫЕ ОГРАНИЧЕНИЯ ===
# Предел числа пар на связку. Нужен при отладке на длинных датасетах, где полный
# проход занимает минуты. None означает всю последовательность
MAX_PAIRS: int | None = None


# ────────────────────────────────────────────────────────────────────────────
# Меню
# ────────────────────────────────────────────────────────────────────────────

def ask(title: str, options: list[str], extra: str = "") -> int | None:
    """
    Показывает пронумерованный список и ждёт выбора.

    Args:
        title: заголовок вопроса.
        options: варианты ответа.
        extra: дополнительный вариант, получающий номер ноль.

    Returns:
        Номер выбранного варианта, ноль для дополнительного, либо None, если
        ввод прерван.
    """
    print()
    print(f"  {title}")
    print()
    if extra:
        print(f"    0. {extra}")
    for number, option in enumerate(options, start=1):
        print(f"    {number}. {option}")
    print()

    lowest = 0 if extra else 1
    while True:
        try:
            answer = input(f"  Номер от {lowest} до {len(options)}: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Прерванный ввод оставляет курсор в середине строки
            print()
            return None

        if answer.isdigit() and lowest <= int(answer) <= len(options):
            return int(answer)
        print("  Такого варианта нет, попробуйте ещё раз.")


def choose_dataset() -> str | None:
    """
    Спрашивает, на каком датасете работать.

    Returns:
        Имя датасета либо None, если ввод прерван.
    """
    names = list(config.DATASETS)
    choice = ask("На каком датасете работать?", names)
    return None if choice is None else names[choice - 1]


def choose_analyses() -> list[dict[str, str]] | None:
    """
    Спрашивает, какой анализ строить.

    Returns:
        Описания выбранных анализов либо None, если ввод прерван.
    """
    labels = [item["label"] for item in ANALYSES]
    picked = ask("Какой анализ построить?", labels, extra="все")
    if picked is None:
        return None
    return list(ANALYSES) if picked == 0 else [ANALYSES[picked - 1]]


# ────────────────────────────────────────────────────────────────────────────
# Подготовка
# ────────────────────────────────────────────────────────────────────────────

def check_config(dataset_name: str) -> list[str]:
    """
    Проверяет, что все измеренные константы заполнены.

    Args:
        dataset_name: имя датасета.

    Returns:
        Список незаполненного. Пустой список означает готовность.
    """
    missing: list[str] = []

    if config.FRAME_STEP is None:
        missing.append("FRAME_STEP")

    calibration = config.CALIBRATION.get(dataset_name, {})
    for field in config.CALIBRATION_REQUIRED:
        if calibration.get(field) is None:
            missing.append(f"CALIBRATION[{dataset_name}].{field}")

    return missing


def report_missing(missing: list[str]) -> None:
    """
    Печатает, каких констант не хватает для прогона.

    Args:
        missing: результат check_config.
    """
    open_block("ПОДГОТОВКА")
    block_line("Константы", "не заполнены", STATUS_FAIL)
    block_wrapped("Не хватает", ", ".join(missing))
    block_note("Заполняются по выводу модулей из service. Порядок запуска "
               "описан в докстринге config.py.")
    close_block()


def load_dataset(name: str) -> Dataset | None:
    """
    Загружает датасет, сообщая о причине отказа.

    Args:
        name: имя датасета.

    Returns:
        Загруженный датасет либо None, если загрузить не удалось.
    """
    try:
        return Dataset(name)
    except Exception as error:
        open_block("ПОДГОТОВКА")
        block_line("Загрузка", describe_exception(error), STATUS_FAIL)
        close_block()
        return None


def report_setup(data: Dataset, pairs: int) -> None:
    """
    Печатает объёмы предстоящей работы.

    Args:
        data: загруженный датасет.
        pairs: сколько пар обработает каждая связка.
    """
    open_block("ИСХОДНЫЕ ДАННЫЕ", data.name)
    block_line("Кадров в датасете", len(data))
    block_line("Пар на связку", pairs)
    block_line("Связок", len(config.FRONTENDS))
    block_line("Геометрий", len(config.GEOMETRIES))
    block_line("Наборов результатов",
               len(config.FRONTENDS) * len(config.GEOMETRIES))
    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Прогон
# ────────────────────────────────────────────────────────────────────────────

def run_all(data: Dataset, step: int) -> dict[str, dict[str, Any]]:
    """
    Прогоняет все связки по датасету.

    Отказ одной связки не прерывает остальные: она просто не попадает в
    результаты.

    Args:
        data: загруженный датасет.
        step: шаг прореживания.

    Returns:
        Результаты pipeline.run по ключу связки.
    """
    results: dict[str, dict[str, Any]] = {}

    for frontend in config.FRONTENDS:
        label = f"{data.name}: {frontend['label']}"
        started = time.perf_counter()
        try:
            results[frontend["key"]] = pipeline.run(
                data, frontend, step, limit=MAX_PAIRS, progress_label=label)
        except Exception as error:
            # Индикатор гасится вручную: pipeline.run до своего вызова
            # clear_progress не дошёл
            clear_progress(label, 0, time.perf_counter() - started,
                           PROGRESS_FORMS)
            open_block(f"СВЯЗКА {frontend['label']}")
            block_line("Состояние", describe_exception(error), STATUS_FAIL)
            close_block()

    return results


def report_run(results: dict[str, dict[str, Any]]) -> None:
    """
    Печатает сводку по прогону: объёмы, время и точность каждой связки.

    Геометрии разнесены по строкам, как в таблицах анализов: колонками их
    заголовки не умещаются в ширину отчёта.

    Число пар и число точек сюда не входят: пары одинаковы у всех связок и
    напечатаны в блоке исходных данных, а число точек упирается в общий предел
    и описывает не детектор, а настройку.

    Args:
        results: результат run_all.
    """
    headers = ["Связка", "Геометрия", "Сопост.", "Охват %", "мс", "Ошибка",
               "Дрейф %"]
    rows: list[list[Any]] = []

    for frontend in config.FRONTENDS:
        result = results.get(frontend["key"])
        if result is None:
            rows.append([frontend["label"], MISSING, "", "", "", "", ""])
            continue

        summary = pipeline.summarize(result)
        shared = [format_number(summary["matches"], 0),
                  format_number(summary["spread"] * 100, 0),
                  format_number(summary["detect_ms"] + summary["match_ms"], 1)]

        for position, item in enumerate(config.GEOMETRIES):
            info = summary[item["key"]]
            absolute = metrics.absolute_error(
                result["trajectories"][item["key"]], result["reference"],
                with_scale=not info["metric"])

            # Название связки и общие для геометрий величины печатаются один
            # раз: повтор в соседней строке читался бы как второе измерение
            head = [frontend["label"] if position == 0 else "", item["label"]]
            rows.append(head + (shared if position == 0 else ["", "", ""])
                        + [format_number(info["direction_error_2d"], 1),
                           format_number(absolute["final_ratio"] * 100, 1)])

    print_section("СВОДКА ПО ПРОГОНУ")
    print()
    print_table(headers, rows)
    print()
    print_legend([
        ("Сопост.", "сопоставлений на пару после отбраковки"),
        ("Охват %", "доля площади кадра, покрытая сопоставленными точками"),
        ("мс", "время детектора и матчера на пару, геометрия не входит"),
        ("Ошибка", "ошибка направления движения на паре, град"),
        ("Дрейф %", "расхождение с эталоном в конце пути, доля от его длины"),
    ])


def report_trajectories(results: dict[str, dict[str, Any]], name: str) -> None:
    """
    Рисует траектории всех связок и печатает, где искать файлы.

    Совмещение с эталоном идёт по всей длине: так наилучшим образом
    накладывается форма, и этим же совмещением считаются метрики.

    Геометриям, восстанавливающим длину перемещения, общий множитель не
    подбирается, иначе их преимущество оказалось бы скрыто. Знание об этом есть
    только здесь, поэтому на график уходят уже совмещённые траектории.

    Args:
        results: результат run_all.
        name: имя датасета.
    """
    if not results:
        return

    reference = next(iter(results.values()))["reference"]
    panels: list[dict[str, Any]] = []

    for item in config.GEOMETRIES:
        key = item["key"]
        whole: dict[str, np.ndarray] = {}

        for frontend_key, result in results.items():
            table = result["pairs"]
            if table.empty:
                continue

            # Отказавшая пара помечена как неметрическая, поэтому геометрия
            # считается метрической, если её дала хотя бы одна пара
            metric = bool(table[f"{key}_metric"].any())
            whole[frontend_key] = metrics.absolute_error(
                result["trajectories"][key], result["reference"],
                with_scale=not metric)["aligned"]

        panels.append({"key": key, "label": item["label"], "whole": whole})

    paths = figures.trajectories(reference, panels, name)

    open_block("ТРАЕКТОРИИ")
    for panel, path in zip(panels, paths):
        block_line(panel["label"], path.name)
    block_note("Траектории совмещены с эталоном по всей длине: так наилучшим "
               "образом накладывается форма, и этим же совмещением считаются "
               "метрики. Начало кривой поэтому не обязано совпадать с началом "
               "маршрута.")
    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Анализы
# ────────────────────────────────────────────────────────────────────────────

def load_analysis(item: dict[str, str]) -> tuple[Callable[..., Any] | None, str]:
    """
    Загружает модуль анализа и возвращает его точку входа.

    Args:
        item: описание анализа из ANALYSES.

    Returns:
        Пару из функции run и описания отказа. При успехе описание пусто, при
        отказе функция равна None.
    """
    try:
        module = importlib.import_module(item["module"])
    except Exception as error:
        return None, describe_exception(error)

    function = getattr(module, "run", None)
    if function is None:
        return None, "в модуле нет функции run"
    return function, ""


def run_analyses(chosen: list[dict[str, str]],
                 results: dict[str, dict[str, Any]], name: str) -> None:
    """
    Запускает выбранные анализы по результатам прогона.

    Каждый раздел открывается баннером с буквой пункта задания: заголовок с
    линией внутри анализа выглядит так же, как его начало, и без баннера
    граница между пунктами теряется.

    Args:
        chosen: описания анализов из ANALYSES.
        results: результат run_all.
        name: имя датасета.
    """
    for item in chosen:
        print_banner([f"ПУНКТ {item['letter']}. {item['label'].upper()}"])

        function, problem = load_analysis(item)
        if function is None:
            open_block("СОСТОЯНИЕ")
            block_line("Модуль", "не загружен", STATUS_FAIL)
            block_wrapped("Ожидается", item["module"].replace(".", "/") + ".py")
            block_wrapped("Причина", problem)
            close_block()
            continue

        try:
            function(results, name)
        except Exception as error:
            open_block("СОСТОЯНИЕ")
            block_line("Анализ", describe_exception(error), STATUS_FAIL)
            close_block()


# ────────────────────────────────────────────────────────────────────────────
# Точка входа
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Спрашивает датасет и анализ, прогоняет связки и строит выбранное.
    """
    print_banner(["ОЦЕНКА СОБСТВЕННОГО ДВИЖЕНИЯ ПО ДАННЫМ ALTO"])

    name = choose_dataset()
    if name is None:
        return

    missing = check_config(name)
    if missing:
        report_missing(missing)
        return

    chosen = choose_analyses()
    if chosen is None:
        return

    data = load_dataset(name)
    if data is None:
        return

    step = config.FRAME_STEP
    print_banner(["ПРОГОН", f"датасет {name}, шаг прореживания {step}"])
    report_setup(data, len(data.pairs(step, limit=MAX_PAIRS)))

    results = run_all(data, step)
    if not results:
        open_block("ИТОГ")
        block_line("Результаты", "ни одна связка не отработала", STATUS_FAIL)
        close_block()
        return

    report_run(results)
    report_trajectories(results, name)
    run_analyses(chosen, results, name)

    print_banner([f"ГОТОВО: связок {len(results)} из {len(config.FRONTENDS)}",
                  f"графики сохранены в {config.PLOTS_DIR}"])
    print()


if __name__ == "__main__":
    main()
