"""
Нагрузка на устройство: время работы и потребление памяти.

Модуль отвечает на пункт задания о нагрузке. Замер устроен так, чтобы цифры
можно было сравнивать между собой, а это требует нескольких предосторожностей.

Каждая связка меряется в отдельном процессе. Иначе всё, что выделяется при
первом обращении к библиотекам, приписалось бы той связке, которая запустилась
первой, и она выглядела бы во много раз прожорливее остальных.

Память делится на две части. Разовая нужна, чтобы поднять библиотеки и веса
моделей, и от числа обрабатываемых кадров не зависит. Рабочая тратится во время
обработки и показывает, сколько нужно сверх разовой.

Время делится по этапам: поиск точек, сопоставление, восстановление движения.
Так видно, где узкое место.

Замер повторяется в двух режимах по числу потоков процессора. Классические
методы OpenCV используют все ядра, а нейросетевая связка работает на
видеокарте, и сравнение в режиме по умолчанию отражало бы больше устройство
машины, чем свойства алгоритмов. Замер в один поток показывает вычислительную
стоимость саму по себе.
"""

# Стандартные библиотеки
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

# Сторонние библиотеки
import numpy as np

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from console import (STATUS_FAIL, STATUS_NONE, block_line, block_note,
                     block_wrapped, close_block, format_number, open_block,
                     print_legend, print_note, print_section, print_table)
from visualization import analysis as figures


# === ПАРАМЕТРЫ ЗАМЕРА ===
# Пар достаточно для устойчивой медианы, но мало настолько, чтобы все запуски в
# отдельных процессах уложились в разумное время
PERF_PAIRS = 40

# Сколько пар прогреть перед замером. Первые вызовы включают подбор алгоритмов
# и выделение буферов, и их время работу не характеризует
WARMUP_PAIRS = 3

# Сколько раз повторить замер. Берётся лучший результат: посторонняя нагрузка на
# машине может только замедлить работу, но не ускорить её. Двух повторов мало:
# лучший из двух слабо защищает от нагрузки, попавшей на оба замера, а разброс
# по двум значениям это не разброс, а разница между двумя числами
REPEATS = 3

# Режимы по числу потоков процессора. Первый считается основным: по нему идут
# отчёты о темпе обработки и о цене точности
THREAD_MODES: tuple[tuple[str, int | None], ...] = (
    ("все ядра", None),
    ("один поток", 1),
)

# === ТАЙМАУТЫ ОТДЕЛЬНЫХ ПРОЦЕССОВ, СЕКУНДЫ ===
PROCESS_TIMEOUT = 600    # Сколько ждать результат замера
SHUTDOWN_TIMEOUT = 5     # Сколько ждать завершения после принудительной остановки

# === ПОРОГИ ВЫВОДОВ ===
# Наименьшая разница между режимами по числу потоков, при которой она считается
# выигрышем. Служит нижней границей: настоящий порог берётся по разбросу
# замеров, если тот оказался больше
THREAD_NOISE = 1.15

# Насколько связка должна опережать следующую по точности, чтобы называться
# самой точной, в градусах. Меньший отрыв не воспроизводится: он меняется от
# прогона к прогону вместе с выбором точек при робастной оценке
ACCURACY_MARGIN = 1.0

# Геометрия, по которой берётся ошибка в разборе цены точности. Задана ключом, а
# не позицией в config.GEOMETRIES: перестановка геометрий в конфиге молча
# сменила бы модель, по которой считается ошибка
COST_GEOMETRY = "essential"

# === ЕДИНИЦЫ ИЗМЕРЕНИЯ ===
BYTES_PER_MB = 1024 * 1024   # Перевод байт в мегабайты
MS_PER_SECOND = 1000.0       # Перевод секунд в миллисекунды


# ────────────────────────────────────────────────────────────────────────────
# Замер в отдельном процессе
# ────────────────────────────────────────────────────────────────────────────

def synchronize() -> None:
    """
    Дожидается завершения расчётов на видеокарте.

    Вычисления на GPU выполняются асинхронно: без ожидания замер показал бы
    только длительность постановки задачи в очередь.

    Та же функция есть в vo_module/pipeline.py: импорт пайплайна притащил бы в
    замеряемый процесс pandas.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def video_memory() -> float:
    """
    Возвращает пиковое потребление видеопамяти.

    Returns:
        Объём в мегабайтах. Ноль, если видеокарта не использовалась.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / BYTES_PER_MB)
    except Exception:
        pass
    return 0.0


def worker(dataset_name: str, frontend_key: str, pairs: int,
           threads: int | None, channel: Any) -> None:
    """
    Меряет одну связку в отдельном процессе и отправляет результат.

    Функция выполняется в новом процессе, поэтому все библиотеки загружаются в
    нём заново. Именно это и нужно: разовые затраты на загрузку относятся к той
    связке, которая их вызвала, а не к запущенной первой.

    Отказ не выпускается наружу, а уходит в очередь описанием: процесс,
    упавший с исключением, оставил бы родителя ждать результата до таймаута.

    Args:
        dataset_name: имя датасета.
        frontend_key: ключ связки.
        pairs: сколько пар обработать сверх прогревочных.
        threads: сколько потоков разрешить OpenCV, None означает без
            ограничения.
        channel: очередь для передачи результата родительскому процессу.
    """
    try:
        import cv2
        import psutil

        import config as settings
        from vo_module import features, pose
        from vo_module.dataset import Dataset

        if threads is not None:
            cv2.setNumThreads(threads)

        # Генератор фиксируется и здесь: процесс новый, состояние в нём своё, а
        # замер должен воспроизводиться от запуска к запуску так же, как прогон
        cv2.setRNGSeed(settings.RANDOM_SEED)

        process = psutil.Process()
        started_memory = process.memory_info().rss / BYTES_PER_MB

        frontend = next(item for item in settings.FRONTENDS
                        if item["key"] == frontend_key)
        detector = features.DETECTORS[frontend["detector"]]
        matcher = features.MATCHERS[frontend["matcher"]]
        needs_second = features.MATCHER_NEEDS_SECOND[frontend["matcher"]]

        calibration = settings.CALIBRATION[dataset_name]
        intrinsics = pose.camera_matrix(calibration["focal_px"],
                                        calibration["principal_point"])

        data = Dataset(dataset_name)
        chosen = data.pairs(settings.FRAME_STEP, limit=pairs + WARMUP_PAIRS)

        detect: list[float] = []
        match: list[float] = []
        geometry: dict[str, list[float]] = {item["key"]: []
                                            for item in settings.GEOMETRIES}
        loaded_memory = started_memory
        peak_memory = started_memory

        for index, (first_index, second_index) in enumerate(chosen):
            first_gray = data.gray(first_index)
            second_gray = data.gray(second_index)

            began = time.perf_counter()
            first = detector(first_gray)
            second = detector(second_gray) if needs_second else first
            synchronize()
            detect_ms = (time.perf_counter() - began) * MS_PER_SECOND

            began = time.perf_counter()
            matches = matcher(first, second, first_gray, second_gray)
            synchronize()
            match_ms = (time.perf_counter() - began) * MS_PER_SECOND

            geometry_ms: dict[str, float] = {}
            for item in settings.GEOMETRIES:
                began = time.perf_counter()
                pose.POSE_BY_KEY[item["key"]](matches["points_first"],
                                              matches["points_second"],
                                              intrinsics)
                geometry_ms[item["key"]] = ((time.perf_counter() - began)
                                            * MS_PER_SECOND)

            # Прогревочные пары в замер не идут, но память после них уже
            # включает загруженные модели, и это как раз разовые затраты
            if index < WARMUP_PAIRS:
                loaded_memory = process.memory_info().rss / BYTES_PER_MB
                peak_memory = loaded_memory
                continue

            # Берётся наибольшее значение за время работы, а не последнее:
            # сборщик мусора может освободить память к концу замера, и разница
            # между двумя точками оказалась бы отрицательной
            peak_memory = max(peak_memory,
                              process.memory_info().rss / BYTES_PER_MB)

            detect.append(detect_ms)
            match.append(match_ms)
            for key, value in geometry_ms.items():
                geometry[key].append(value)

        channel.put({
            "ok": True,
            "frontend": frontend_key,
            "threads": threads,
            "pairs": len(detect),
            "detect_ms": float(np.median(detect)) if detect else float("nan"),
            "match_ms": float(np.median(match)) if match else float("nan"),
            "geometry_ms": {key: (float(np.median(values)) if values
                                  else float("nan"))
                            for key, values in geometry.items()},
            "startup_mb": loaded_memory - started_memory,
            "working_mb": max(peak_memory - loaded_memory, 0.0),
            "total_mb": peak_memory,
            "vram_mb": video_memory(),
        })

    except Exception as error:
        channel.put({"ok": False, "frontend": frontend_key, "threads": threads,
                     "error": f"{type(error).__name__}: {error}"})


def measure_once(dataset_name: str, frontend_key: str,
                 threads: int | None) -> dict[str, Any]:
    """
    Выполняет один замер связки в отдельном процессе.

    При невозможности создать процесс замер не выполняется: считать в текущем
    процессе бессмысленно, поскольку память окажется засчитана неверно.

    Процесс, не уложившийся в PROCESS_TIMEOUT, останавливается принудительно.
    Иначе он продолжал бы занимать ядра и память, пока идут остальные замеры, и
    портил бы их результат.

    Args:
        dataset_name: имя датасета.
        frontend_key: ключ связки.
        threads: ограничение числа потоков OpenCV.

    Returns:
        Результат замера из worker либо словарь с полями ok, равным False,
        frontend, threads и error.
    """
    process = None
    try:
        context = multiprocessing.get_context("spawn")
        channel = context.Queue()
        process = context.Process(target=worker,
                                  args=(dataset_name, frontend_key, PERF_PAIRS,
                                        threads, channel))
        process.start()
        outcome = channel.get(timeout=PROCESS_TIMEOUT)
        process.join(timeout=PROCESS_TIMEOUT)
        return outcome
    except Exception as error:
        return {"ok": False, "frontend": frontend_key, "threads": threads,
                "error": f"{type(error).__name__}: {error}"}
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=SHUTDOWN_TIMEOUT)


def measure(dataset_name: str, frontend_key: str,
            threads: int | None) -> dict[str, Any]:
    """
    Меряет связку несколько раз и берёт лучший результат.

    Посторонняя нагрузка на машине может замедлить работу, но не ускорить её,
    поэтому наименьшее из полученных времён ближе к собственной стоимости
    алгоритма, чем среднее.

    Args:
        dataset_name: имя датасета.
        frontend_key: ключ связки.
        threads: ограничение числа потоков OpenCV.

    Returns:
        Лучший из полученных результатов, дополненный полями attempts с числом
        удавшихся замеров и spread с отношением худшего времени к лучшему. Если
        не удался ни один замер, возвращается описание последнего отказа.
    """
    attempts = [measure_once(dataset_name, frontend_key, threads)
                for _ in range(REPEATS)]
    usable = [item for item in attempts if item["ok"]]

    if not usable:
        return attempts[-1]

    best = min(usable, key=total_time)
    best["attempts"] = len(usable)
    best["spread"] = ((max(total_time(item) for item in usable)
                       / total_time(best)) if total_time(best) > 0
                      else float("nan"))
    return best


def run_all(dataset_name: str) -> list[dict[str, Any]]:
    """
    Меряет все связки во всех режимах по числу потоков.

    Args:
        dataset_name: имя датасета.

    Returns:
        Список результатов замера, дополненных полями mode с названием режима и
        label с подписью связки.

    Raises:
        KeyError: не заполнен config.FRAME_STEP, не заполнены калибровочные
            константы датасета либо датасета нет в конфиге.
    """
    # Предусловия проверяются до запуска процессов: без проверки все замеры
    # отработали бы вхолостую и вернули одинаковый отказ
    if config.FRAME_STEP is None:
        raise KeyError("не заполнен config.FRAME_STEP, "
                       "запусти service/frame_step.py")

    calibration = config.CALIBRATION[dataset_name]
    missing = [field for field in config.CALIBRATION_REQUIRED
               if calibration.get(field) is None]
    if missing:
        raise KeyError(f"для датасета {dataset_name} не заполнено: "
                       f"{', '.join(missing)}")

    outcome: list[dict[str, Any]] = []

    for label, threads in THREAD_MODES:
        for frontend in config.FRONTENDS:
            result = measure(dataset_name, frontend["key"], threads)
            result["mode"] = label
            result["label"] = frontend["label"]
            outcome.append(result)

    return outcome


# ────────────────────────────────────────────────────────────────────────────
# Производные величины
# ────────────────────────────────────────────────────────────────────────────

def total_time(record: dict[str, Any]) -> float:
    """
    Считает полное время обработки одной пары кадров.

    В сумму входят поиск точек, сопоставление и восстановление движения по всем
    геометриям.

    Args:
        record: результат замера.

    Returns:
        Время в миллисекундах. NaN, если замер не выполнен.
    """
    if not record["ok"]:
        return float("nan")

    geometry = sum(value for value in record["geometry_ms"].values()
                   if not np.isnan(value))
    return record["detect_ms"] + record["match_ms"] + geometry


def required_rate() -> float:
    """
    Считает, сколько пар в секунду нужно обрабатывать, чтобы поспевать за
    съёмкой.

    Камера снимает с известной частотой, а пары берутся через заданное число
    кадров. Отношение и даёт нужную скорость обработки.

    Returns:
        Число пар в секунду.
    """
    step = config.FRAME_STEP or 1
    return config.CAMERA_FPS / step


# ────────────────────────────────────────────────────────────────────────────
# Отчёт
# ────────────────────────────────────────────────────────────────────────────

def report_table(records: list[dict[str, Any]]) -> None:
    """
    Печатает таблицу времени и памяти.

    Время геометрии сюда не входит, хотя и учтено в полном: она занимает
    единицы миллисекунд у всех связок и считается одним и тем же кодом, поэтому
    узким местом не бывает и связки не различает.

    Скорость в парах в секунду тоже не выводится: это обратная величина полного
    времени из соседней колонки, а там, где она нужна, её печатает блок о темпе
    обработки вместе с запасом.

    Рабочая память слита с разовой в одну колонку: обработка кадров добавляет
    единицы мегабайт при сотнях, потраченных на загрузку библиотек и весов.

    Args:
        records: результат run_all.
    """
    headers = ["Связка", "Потоки", "Точки", "Сопост.", "Всего", "Разброс",
               "Память"]
    rows: list[list[Any]] = []

    for record in records:
        # Название печатается в каждой строке: замеры сгруппированы по режиму
        # потоков, поэтому соседние строки относятся к разным связкам
        label = record["label"]

        if not record["ok"]:
            rows.append([label, record["mode"], "отказ", "", "", "", ""])
            continue

        rows.append([
            label, record["mode"],
            format_number(record["detect_ms"], 1),
            format_number(record["match_ms"], 1),
            format_number(total_time(record), 1),
            format_number(record["spread"], 2),
            format_number(record["startup_mb"] + record["working_mb"], 0),
        ])

    video = [record for record in records
             if record["ok"] and record["vram_mb"] > 1]

    print_section("ВРЕМЯ И ПАМЯТЬ ПО СВЯЗКАМ")
    print()
    print_table(headers, rows)
    print()
    print_legend([
        ("Точки", "поиск ключевых точек на паре кадров, мс"),
        ("Сопост.", "сопоставление найденных точек, мс"),
        ("Всего", "полное время пары, включая восстановление движения"),
        ("Разброс", f"во сколько раз худший из {REPEATS} замеров медленнее "
                    f"лучшего, единица означает повторяемый результат"),
        ("Память", "разовая загрузка библиотек и весов плюс расход на "
                   "обработку, МБ"),
    ])

    if video:
        peak = max(record["vram_mb"] for record in video)
        names = ", ".join(sorted({record["label"] for record in video}))
        print()
        print_note(f"Видеопамять расходует только {names}: "
                   f"{format_number(peak, 0)} МБ. Остальные связки считают на "
                   f"процессоре.")


def report_realtime(records: list[dict[str, Any]]) -> None:
    """
    Печатает скорость обработки рядом с темпом съёмки.

    Отметки о годности здесь нет: работа в темпе съёмки это не требование
    задания, а справка о применимости на борту.

    Args:
        records: результат run_all.
    """
    needed = required_rate()
    open_block("ТЕМП ОБРАБОТКИ", f"съёмка даёт {needed:.1f} пар в секунду")

    default_mode = THREAD_MODES[0][0]
    for record in records:
        if not record["ok"] or record["mode"] != default_mode:
            continue

        full = total_time(record)
        rate = MS_PER_SECOND / full if full > 0 else 0.0
        block_line(record["label"],
                   f"{format_number(rate, 1)} пар в секунду, "
                   f"{format_number(rate / needed, 1)} от темпа съёмки")

    block_note(f"Темп съёмки получен делением частоты кадров на шаг "
               f"прореживания: камера ALTO снимает "
               f"{config.CAMERA_FPS:.0f} кадров в секунду, обрабатывается "
               f"каждый {config.FRAME_STEP}-й. Столько пар в секунду нужно "
               f"успевать, чтобы работать на борту без задержки.")

    block_note("Задание такого требования не ставит: оно просит измерить время "
               "и память, а не проверить пригодность к работе в реальном "
               "времени. Поэтому отношение к темпу съёмки печатается справкой, "
               "без отметки о годности.")

    close_block()


def report_threads(records: list[dict[str, Any]]) -> None:
    """
    Печатает, насколько связки выигрывают от многих ядер процессора.

    Выигрыш признаётся только тогда, когда он превышает разброс самих замеров.
    Постоянного порога мало: у связки с большим разбросом разница между режимами
    в этом разбросе тонет, хотя формально выглядит выигрышем.

    Отношение проверяется в обе стороны. Значение около единицы означает, что
    связка от числа потоков не зависит, а значение заметно меньше единицы не
    означает ничего: на всех ядрах не может быть медленнее, чем на одном, и
    такой исход говорит лишь о посторонней нагрузке во время одного из замеров.

    Args:
        records: результат run_all.
    """
    open_block("ВЛИЯНИЕ ЧИСЛА ПОТОКОВ")

    fast_mode, slow_mode = THREAD_MODES[0][0], THREAD_MODES[-1][0]

    for frontend in config.FRONTENDS:
        fast = next((record for record in records if record["ok"]
                     and record["frontend"] == frontend["key"]
                     and record["mode"] == fast_mode), None)
        slow = next((record for record in records if record["ok"]
                     and record["frontend"] == frontend["key"]
                     and record["mode"] == slow_mode), None)

        if fast is None or slow is None:
            block_line(frontend["label"], "замер не выполнен", STATUS_NONE)
            continue

        many, single = total_time(fast), total_time(slow)
        gain = single / many if many > 0 else float("nan")
        noise = max(THREAD_NOISE, fast["spread"] or 1.0, slow["spread"] or 1.0)

        if gain >= noise:
            verdict, status = f"выигрыш {format_number(gain, 1)} раза", ""
        elif gain >= 1.0 / noise:
            verdict, status = "в пределах разброса", ""
        else:
            verdict, status = "замер ненадёжен", STATUS_NONE

        block_line(frontend["label"],
                   f"{format_number(many, 1)} против "
                   f"{format_number(single, 1)} мс, {verdict}", status)

    block_note("Классические методы OpenCV используют все ядра процессора, а "
               "нейросетевая связка считает на видеокарте и от числа потоков "
               "почти не зависит. Поэтому прямое сравнение отражает не только "
               "свойства алгоритмов, но и устройство машины, на которой они "
               "запущены. Замер в один поток показывает вычислительную "
               "стоимость саму по себе.")

    block_note("Выигрышем считается разница, превышающая разброс самих замеров, "
               "а не постоянный порог: у нейросетевой связки повторы расходятся "
               "сильнее, чем режимы между собой, и разница между ними в этом "
               "разбросе тонет.")

    close_block()


def report_cost(records: list[dict[str, Any]],
                results: dict[str, dict[str, Any]]) -> None:
    """
    Печатает, во что обходится точность.

    Ошибка берётся из основного прогона по геометрии COST_GEOMETRY, её название
    стоит в заголовке блока. Самая точная связка называется лишь тогда, когда
    отрыв от следующей превышает ACCURACY_MARGIN: меньшая разница от прогона к
    прогону объявляла бы победителем то одну связку, то другую.

    Args:
        records: результат run_all.
        results: результаты pipeline.run по ключу связки.
    """
    geometry = next((item for item in config.GEOMETRIES
                     if item["key"] == COST_GEOMETRY), None)
    if geometry is None:
        open_block("ЦЕНА ТОЧНОСТИ")
        block_line("Состояние",
                   f"геометрии {COST_GEOMETRY} нет в конфиге", STATUS_NONE)
        close_block()
        return

    open_block("ЦЕНА ТОЧНОСТИ", geometry["label"])

    default_mode = THREAD_MODES[0][0]
    collected: list[tuple[str, float, float]] = []

    for record in records:
        if not record["ok"] or record["mode"] != default_mode:
            continue
        result = results.get(record["frontend"])
        if result is None or result["pairs"].empty:
            continue
        error = float(result["pairs"][
            f"{geometry['key']}_direction_error_2d"].median())
        collected.append((record["label"], total_time(record), error))

    if not collected:
        block_line("Состояние", "данных не хватило", STATUS_NONE)
        close_block()
        return

    fastest = min(collected, key=lambda item: item[1])
    most_accurate = min(collected, key=lambda item: item[2])

    # Перечень всех связок сюда не входит: время стоит в таблице выше, ошибка в
    # разборе геометрии, и повтор занял бы пять строк ради двух крайних случаев
    block_line("Самая быстрая",
               f"{fastest[0]}, {format_number(fastest[1], 1)} мс, "
               f"{format_number(fastest[2], 1)} град")

    rivals = sorted(item[2] for item in collected)
    margin = rivals[1] - rivals[0] if len(rivals) > 1 else float("inf")

    if margin >= ACCURACY_MARGIN:
        block_line("Самая точная",
                   f"{most_accurate[0]}, "
                   f"{format_number(most_accurate[1], 1)} мс, "
                   f"{format_number(most_accurate[2], 1)} град")
    else:
        block_line("Самая точная",
                   f"не выделяется: {format_number(rivals[0], 1)} и "
                   f"{format_number(rivals[1], 1)} град", STATUS_NONE)

    if fastest[0] != most_accurate[0] and margin >= ACCURACY_MARGIN:
        times = (most_accurate[1] / fastest[1] if fastest[1] > 0
                 else float("nan"))
        better = (fastest[2] / most_accurate[2] if most_accurate[2] > 0
                  else float("nan"))
        block_wrapped("Размен",
                      f"самая точная работает в {format_number(times, 1)} раза "
                      f"дольше самой быстрой и ошибается в "
                      f"{format_number(better, 1)} раза меньше")

    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Точка входа анализа
# ────────────────────────────────────────────────────────────────────────────

def run(results: dict[str, dict[str, Any]], name: str) -> None:
    """
    Выполняет анализ нагрузки на устройство.

    Args:
        results: результаты pipeline.run по ключу связки.
        name: имя датасета.

    Raises:
        KeyError: не заполнены константы конфига, нужные для замера.
    """
    records = run_all(name)
    usable = [record for record in records if record["ok"]]

    if not usable:
        open_block("СОСТОЯНИЕ")
        block_line("Замеры", "ни один не выполнен", STATUS_FAIL)
        failed = next((record for record in records if not record["ok"]), None)
        if failed:
            block_wrapped("Причина", failed["error"])
        close_block()
        return

    report_table(records)
    report_realtime(records)
    report_threads(records)
    report_cost(records, results)

    # Что входит в полное время обработки пары, решает этот модуль, поэтому
    # сумма проставляется здесь, а не выводится при рисовании
    for record in records:
        record["total_ms"] = total_time(record)

    path = figures.performance(records, name,
                               [label for label, _ in THREAD_MODES])

    open_block("ГРАФИК НАГРУЗКИ")
    block_line("Файл", path.name)
    block_line("Содержание", "влияние числа потоков процессора")
    close_block()
