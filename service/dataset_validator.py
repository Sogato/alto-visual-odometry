"""
Проверка целостности датасетов и сбор характеристик маршрутов.

Состав проверок:

1. Соответствие кадров и телеметрии, связка идёт по колонке name.
2. Непрерывность нумерации кадров: разрыв ломает относительное перемещение.
3. Нормировка кватернионов ориентации: отклонение означает иной смысл колонок.
4. Геометрия маршрута: шаг между кадрами, длина пути, диапазон высот,
   извилистость.
5. Свойства изображений: размеры, число каналов, текстурность.
6. Пересечение датасетов между собой по координатам: совпадающие наборы не
   являются независимыми наблюдениями.

Формальные проверки сворачиваются в строку на датасет и расписываются только при
отклонении. Числовые характеристики маршрута и кадров идут в сводную таблицу.
"""

# Стандартные библиотеки
import sys
import time
from pathlib import Path
from typing import Any

# Сторонние библиотеки
import cv2
import numpy as np
import pandas as pd

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from console import (FRAME_FORMS, MISSING, STATUS_FAIL, STATUS_NONE, STATUS_OK,
                     block_line, block_note, block_wrapped, clear_progress,
                     close_block, describe_exception, format_number, open_block,
                     plural, print_banner, print_note, print_progress,
                     print_section, print_table)


# === ПАРАМЕТРЫ АНАЛИЗА ИЗОБРАЖЕНИЙ ===
# Размеры и каналы проверяются у всех файлов, статистика по пикселям на выборке
PIXEL_SAMPLE_SIZE = 30          # Число кадров в выборке для расчёта текстурности
PROGRESS_EVERY = 100            # Через сколько кадров перерисовывать индикатор

# === ДОПУСКИ И ПОРОГИ ===
QUATERNION_NORM_TOLERANCE = 1e-3   # Допуск на отклонение нормы кватерниона от единицы
MAX_GAPS_SHOWN = 5                 # Сколько разрывов нумерации печатать до свёртки

# Точность округления координат при поиске совпадающих датасетов, знаков после
# запятой. Позиции берутся из одного источника и должны совпадать точно
OVERLAP_PRECISION = 2

# База для оценки поворота маршрута, м. Задана в метрах, а не в кадрах, поэтому
# величина сравнима между датасетами с разным шагом съёмки
TURNING_BASELINE_M = 25.0

# === ФОРМЫ СЧЁТНЫХ СЛОВ ===
# Порядок форм: одна позиция, две позиции, пять позиций
POSITION_FORMS: tuple[str, str, str] = ("общая позиция", "общие позиции",
                                        "общих позиций")


# ────────────────────────────────────────────────────────────────────────────
# Чтение данных
# ────────────────────────────────────────────────────────────────────────────

def read_image(path: Path) -> np.ndarray | None:
    """
    Читает изображение с диска.

    Файл читается как байты и декодируется отдельно: cv2.imread не работает с
    путями, содержащими символы вне latin-1.

    Args:
        path: путь к файлу изображения.

    Returns:
        Массив изображения как есть, включая альфа-канал, либо None при отказе.
    """
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def frame_number(filename: str) -> int | None:
    """
    Извлекает порядковый номер кадра из имени файла.

    Args:
        filename: имя файла вида 000446.png.

    Returns:
        Номер кадра либо None, если имя не числовое.
    """
    stem = Path(filename).stem
    return int(stem) if stem.isdigit() else None


def load_telemetry(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Читает файл телеметрии и проверяет наличие обязательных колонок.

    Args:
        path: путь к query.csv.

    Returns:
        Кортеж из таблицы и списка недостающих колонок. При отказе чтения
        таблица равна None, а недостающими считаются все колонки.
    """
    required = [config.COLUMN_NAME, config.COLUMN_EASTING, config.COLUMN_NORTHING,
                config.COLUMN_ALTITUDE, *config.COLUMNS_QUATERNION]

    try:
        frame = pd.read_csv(path)
    except Exception:
        return None, required

    missing = [column for column in required if column not in frame.columns]
    return frame, missing


# ────────────────────────────────────────────────────────────────────────────
# Соответствие кадров и телеметрии
# ────────────────────────────────────────────────────────────────────────────

def match_frames(telemetry: pd.DataFrame, images_dir: Path) -> dict[str, Any]:
    """
    Связывает кадры на диске со строками телеметрии по имени файла.

    Связка идёт по имени файла, а не по позиции строки.

    Args:
        telemetry: таблица телеметрии.
        images_dir: каталог с кадрами.

    Returns:
        Словарь с полями csv_rows, files_total, matched, matched_count,
        disk_only, csv_only и csv_duplicates. Поле matched содержит имена
        совпавших кадров по возрастанию.
    """
    files = sorted(path.name for path in images_dir.glob(f"*{config.IMAGE_EXTENSION}"))
    file_set = set(files)

    csv_names = telemetry[config.COLUMN_NAME].astype(str).tolist()
    csv_set = set(csv_names)

    matched = [name for name in files if name in csv_set]

    return {
        "csv_rows": len(csv_names),
        "files_total": len(files),
        "matched": matched,
        "matched_count": len(matched),
        "disk_only": len(file_set - csv_set),
        "csv_only": len(csv_set - file_set),
        "csv_duplicates": len(csv_names) - len(csv_set),
    }


def check_numbering(names: list[str]) -> dict[str, Any]:
    """
    Проверяет непрерывность нумерации совпавших кадров.

    Обычным шагом считается самый частый, отрезки с иным шагом попадают в
    разрывы.

    Args:
        names: имена совпавших кадров.

    Returns:
        Словарь с полями first, last, gaps и non_numeric. Поле regular_step
        присутствует, только когда числовых имён хотя бы два. Поле non_numeric
        равно True, если числовых имён нет вовсе.
    """
    numbers = sorted(value for value in (frame_number(name) for name in names)
                     if value is not None)

    if not numbers:
        return {"first": None, "last": None, "gaps": [], "non_numeric": True}

    if len(numbers) < 2:
        return {"first": numbers[0], "last": numbers[0], "gaps": [],
                "non_numeric": False}

    diffs = np.diff(numbers)
    unique, counts = np.unique(diffs, return_counts=True)

    regular = int(unique[np.argmax(counts)])
    gaps = [(int(numbers[i]), int(numbers[i + 1]))
            for i, diff in enumerate(diffs) if diff != regular]

    return {
        "first": int(numbers[0]),
        "last": int(numbers[-1]),
        "regular_step": regular,
        "gaps": gaps,
        "non_numeric": False,
    }


def check_quaternions(telemetry: pd.DataFrame, names: list[str]) -> dict[str, Any]:
    """
    Проверяет нормировку кватернионов ориентации.

    Отклонением считается разница нормы с единицей больше
    QUATERNION_NORM_TOLERANCE.

    Args:
        telemetry: таблица телеметрии.
        names: имена совпавших кадров.

    Returns:
        Словарь с полями norm_min, norm_max и bad_count. Пустой словарь, если
        подходящих строк нет.
    """
    subset = telemetry[telemetry[config.COLUMN_NAME].astype(str).isin(names)]
    values = subset[list(config.COLUMNS_QUATERNION)].to_numpy(dtype=float)

    if len(values) == 0:
        return {}

    norms = np.linalg.norm(values, axis=1)
    deviation = np.abs(norms - 1.0)

    return {
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "bad_count": int((deviation > QUATERNION_NORM_TOLERANCE).sum()),
    }


# ────────────────────────────────────────────────────────────────────────────
# Геометрия маршрута
# ────────────────────────────────────────────────────────────────────────────

def turning_on_baseline(easting: np.ndarray, northing: np.ndarray,
                        baseline_m: float) -> float:
    """
    Считает суммарный поворот маршрута по точкам через заданный путь.

    Маршрут прореживается по пройденному расстоянию: опорные точки берутся через
    каждые baseline_m метров пути, курс считается по ним. Разрывы нумерации не
    учитываются, в расчёт входят все переданные позиции.

    Args:
        easting: координаты вдоль оси восток, м.
        northing: координаты вдоль оси север, м.
        baseline_m: расстояние между опорными точками, м.

    Returns:
        Сумму модулей изменений курса в градусах. Ноль, если опорных точек
        меньше трёх.
    """
    distances = np.hypot(np.diff(easting), np.diff(northing))
    travelled = np.concatenate([[0.0], np.cumsum(distances)])

    marks = np.arange(0.0, travelled[-1], baseline_m)
    indices = np.unique(np.searchsorted(travelled, marks))

    # Для одного поворота нужно три точки: два отрезка и угол между ними
    if len(indices) < 3:
        return 0.0

    heading = np.unwrap(np.arctan2(np.diff(easting[indices]),
                                   np.diff(northing[indices])))
    changes = np.degrees(np.abs(np.diff(heading)))
    return float(changes.sum())


def check_trajectory(telemetry: pd.DataFrame, names: list[str],
                     regular_step: int | None = None) -> dict[str, Any]:
    """
    Считает характеристики маршрута по совпавшим кадрам.

    Шаг маршрута и длина пути считаются по отрезкам между соседними кадрами.
    Отрезки через разрыв нумерации в них не входят: между их концами лежит
    непройденный участок. Число исключённых отрезков возвращается в
    dropped_steps.

    Извилистость описывается суммой модулей поворотов на базе
    TURNING_BASELINE_M и считается по всем позициям, включая отрезки через
    разрыв. Курс берётся по направлению перемещения, а не по кватерниону
    ориентации.

    Args:
        telemetry: таблица телеметрии.
        names: имена совпавших кадров.
        regular_step: обычный шаг нумерации из check_numbering. Отрезки с иным
            шагом исключаются из шага маршрута и длины пути.

    Returns:
        Словарь с полями dropped_steps, step_min, step_median, step_max,
        path_length, altitude_min, altitude_median, altitude_max,
        altitude_span и turning_total. Пустой словарь, если позиций меньше двух.
    """
    subset = telemetry[telemetry[config.COLUMN_NAME].astype(str).isin(names)]
    subset = subset.sort_values(config.COLUMN_NAME)

    easting = subset[config.COLUMN_EASTING].to_numpy(dtype=float)
    northing = subset[config.COLUMN_NORTHING].to_numpy(dtype=float)
    altitude = subset[config.COLUMN_ALTITUDE].to_numpy(dtype=float)

    # Пустой словарь, а не словарь с иным набором ключей: потребитель отличает
    # отсутствие данных одной проверкой
    if len(easting) < 2:
        return {}

    steps = np.hypot(np.diff(easting), np.diff(northing))

    # Разрывы исключаются, только когда номера кадров полностью соответствуют
    # позициям. Повторы имён в телеметрии дают больше строк, чем имён, и тогда
    # отрезки не сопоставить с номерами
    numbers = [frame_number(name) for name in sorted(names)]
    if (regular_step is not None and None not in numbers
            and len(numbers) == len(easting)):
        continuous = np.diff(np.array(numbers)) == regular_step
    else:
        continuous = np.ones(len(steps), dtype=bool)

    walked = steps[continuous] if continuous.any() else steps

    return {
        "dropped_steps": int((~continuous).sum()),
        "step_min": float(walked.min()),
        "step_median": float(np.median(walked)),
        "step_max": float(walked.max()),
        "path_length": float(walked.sum()),
        "altitude_min": float(altitude.min()),
        "altitude_median": float(np.median(altitude)),
        "altitude_max": float(altitude.max()),
        "altitude_span": float(altitude.max() - altitude.min()),
        "turning_total": turning_on_baseline(easting, northing, TURNING_BASELINE_M),
    }


def build_signature(telemetry: pd.DataFrame,
                    names: list[str]) -> set[tuple[float, float]]:
    """
    Строит множество позиций датасета для поиска совпадений с другими.

    Координаты округляются до OVERLAP_PRECISION знаков.

    Args:
        telemetry: таблица телеметрии.
        names: имена совпавших кадров.

    Returns:
        Множество пар координат в метрах.
    """
    subset = telemetry[telemetry[config.COLUMN_NAME].astype(str).isin(names)]
    easting = subset[config.COLUMN_EASTING].round(OVERLAP_PRECISION)
    northing = subset[config.COLUMN_NORTHING].round(OVERLAP_PRECISION)
    return set(zip(easting.tolist(), northing.tolist()))


# ────────────────────────────────────────────────────────────────────────────
# Свойства изображений
# ────────────────────────────────────────────────────────────────────────────

def check_image_formats(images_dir: Path, names: list[str],
                        label: str = "") -> dict[str, Any]:
    """
    Проверяет размеры и число каналов у всех кадров датасета.

    Обходятся все кадры, а не выборка.

    Args:
        images_dir: каталог с кадрами.
        names: имена совпавших кадров.
        label: пояснение для индикатора выполнения.

    Returns:
        Словарь с полями sizes, channels и unreadable. В sizes и channels лежат
        распределения «значение: число кадров».
    """
    sizes: dict[tuple[int, int], int] = {}
    channels: dict[int, int] = {}
    unreadable: list[str] = []
    started = time.perf_counter()

    for index, name in enumerate(names):
        if index % PROGRESS_EVERY == 0:
            print_progress(index, len(names), label)

        image = read_image(images_dir / name)
        if image is None:
            unreadable.append(name)
            continue

        height, width = image.shape[:2]
        depth = image.shape[2] if image.ndim == 3 else 1
        sizes[(width, height)] = sizes.get((width, height), 0) + 1
        channels[depth] = channels.get(depth, 0) + 1

    clear_progress(label, len(names) - len(unreadable),
                   time.perf_counter() - started, FRAME_FORMS)
    return {"sizes": sizes, "channels": channels, "unreadable": unreadable}


def frame_gradient(image: np.ndarray) -> float:
    """
    Считает меру текстурности кадра как среднюю величину градиента яркости.

    Кадр переводится в градации серого, альфа-канал отбрасывается. Градиент
    берётся оператором Собеля по обеим осям.

    Args:
        image: кадр как прочитан с диска.

    Returns:
        Среднюю величину градиента, нормированную на полную шкалу яркости.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0

    dx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.hypot(dx, dy).mean())


def check_pixels(images_dir: Path, names: list[str],
                 label: str = "") -> dict[str, Any]:
    """
    Считает текстурность на выборке кадров и проверяет альфа-канал.

    Выборка из PIXEL_SAMPLE_SIZE кадров берётся равномерно по всей
    последовательности. Заодно проверяется постоянство альфа-канала.

    Args:
        images_dir: каталог с кадрами.
        names: имена совпавших кадров.
        label: пояснение для индикатора выполнения.

    Returns:
        Словарь с полями sampled, has_alpha, alpha_constant и gradient, где
        gradient содержит median, min и max. При пустой выборке возвращается
        словарь с одним полем sampled, равным нулю.
    """
    if not names:
        return {"sampled": 0}

    count = min(PIXEL_SAMPLE_SIZE, len(names))
    indices = np.linspace(0, len(names) - 1, count).astype(int)

    started = time.perf_counter()
    gradients: list[float] = []
    alpha_constant = True
    has_alpha = False

    for position, index in enumerate(indices):
        print_progress(position, len(indices), label)

        image = read_image(images_dir / names[index])
        if image is None:
            continue

        if image.ndim == 3 and image.shape[2] == 4:
            has_alpha = True
            alpha = image[:, :, 3]
            if int(alpha.min()) != int(alpha.max()):
                alpha_constant = False

        gradients.append(frame_gradient(image))

    clear_progress(label, len(gradients), time.perf_counter() - started, FRAME_FORMS)

    if not gradients:
        return {"sampled": 0}

    values = np.array(gradients, dtype=float)
    return {
        "sampled": len(gradients),
        "has_alpha": has_alpha,
        "alpha_constant": alpha_constant,
        "gradient": {"median": float(np.median(values)),
                     "min": float(values.min()),
                     "max": float(values.max())},
    }


# ────────────────────────────────────────────────────────────────────────────
# Проверка одного датасета
# ────────────────────────────────────────────────────────────────────────────

def validate_dataset(name: str) -> dict[str, Any]:
    """
    Выполняет все проверки одного датасета.

    Args:
        name: имя датасета, ключ в config.DATASETS.

    Returns:
        Словарь с полями name, ok, error, warnings и результатами проверок:
        matching, numbering, trajectory, quaternions, signature, formats,
        pixels. При недоступных данных заполняется error, а результаты проверок
        отсутствуют.
    """
    paths = config.DATASETS[name]
    result: dict[str, Any] = {"name": name, "ok": False, "error": None, "warnings": []}

    if not paths["telemetry"].is_file():
        result["error"] = "нет файла телеметрии"
        return result
    if not paths["images"].is_dir():
        result["error"] = "нет каталога с кадрами"
        return result

    telemetry, missing_columns = load_telemetry(paths["telemetry"])
    if telemetry is None:
        result["error"] = "не удалось прочитать телеметрию"
        return result
    if missing_columns:
        result["error"] = "нет колонок: " + ", ".join(missing_columns)
        return result

    matching = match_frames(telemetry, paths["images"])
    result["matching"] = matching

    names = matching["matched"]
    if not names:
        result["error"] = "ни один кадр не совпал с телеметрией"
        return result

    result["numbering"] = check_numbering(names)
    result["trajectory"] = check_trajectory(telemetry, names,
                                            result["numbering"].get("regular_step"))
    result["quaternions"] = check_quaternions(telemetry, names)
    result["signature"] = build_signature(telemetry, names)
    result["formats"] = check_image_formats(paths["images"], names,
                                            f"{name}: чтение кадров")
    result["pixels"] = check_pixels(paths["images"], names,
                                    f"{name}: статистика пикселей")

    result["warnings"] = collect_warnings(result)
    result["ok"] = True
    return result


def pixels_alpha_ok(pixels: dict[str, Any]) -> bool:
    """
    Проверяет, что альфа-канал можно отбрасывать.

    Args:
        pixels: результат check_pixels.

    Returns:
        True, если альфа-канала нет либо он постоянен.
    """
    return not pixels.get("has_alpha") or bool(pixels.get("alpha_constant"))


def collect_warnings(result: dict[str, Any]) -> list[str]:
    """
    Собирает замечания, которые не блокируют работу.

    Args:
        result: результат validate_dataset с заполненными проверками.

    Returns:
        Список текстов замечаний.
    """
    warnings: list[str] = []
    matching = result["matching"]
    numbering = result["numbering"]

    # Датасет, собранный как срез длинной последовательности, пользуется полным
    # файлом телеметрии. Лишние строки в нём не дефект: признак среза это все
    # кадры с диска, обеспеченные телеметрией, при непрерывной нумерации
    is_slice = (not matching["disk_only"] and not numbering.get("gaps")
                and not numbering.get("non_numeric"))

    if matching["disk_only"]:
        warnings.append(f"кадров без телеметрии: {matching['disk_only']}")
    if matching["csv_only"] and not is_slice:
        warnings.append(f"строк телеметрии без кадров: {matching['csv_only']}")
    if matching["csv_duplicates"]:
        warnings.append(f"повторяющихся имён в телеметрии: {matching['csv_duplicates']}")

    if numbering.get("non_numeric"):
        warnings.append("имена кадров не числовые, порядок определить нельзя")
    elif numbering.get("gaps"):
        warnings.append(f"разрывов в нумерации: {len(numbering['gaps'])}")

    trajectory = result["trajectory"]
    if trajectory.get("dropped_steps"):
        warnings.append(f"отрезков через разрыв не учтено в длине пути: "
                        f"{trajectory['dropped_steps']}")

    quaternions = result["quaternions"]
    if quaternions.get("bad_count"):
        warnings.append(f"кватернионов вне допуска: {quaternions['bad_count']}")

    formats = result["formats"]
    if len(formats["sizes"]) > 1:
        warnings.append(f"разных размеров кадра: {len(formats['sizes'])}")
    if len(formats["channels"]) > 1:
        warnings.append(f"разного числа каналов: {len(formats['channels'])}")
    if formats["unreadable"]:
        warnings.append(f"не прочитано файлов: {len(formats['unreadable'])}")

    if not pixels_alpha_ok(result["pixels"]):
        warnings.append("альфа-канал меняется между кадрами")

    return warnings


# ────────────────────────────────────────────────────────────────────────────
# Отчёт о целостности
# ────────────────────────────────────────────────────────────────────────────

def describe_integrity(result: dict[str, Any]) -> str:
    """
    Сводит формальные проверки датасета к одной строке.

    Args:
        result: результат validate_dataset.

    Returns:
        Строку вида «10436 кадров, 500x500x4, без разрывов». Разнородный формат
        кадров обозначается пометкой, перечень вариантов печатает
        report_anomalies.
    """
    matching = result["matching"]
    numbering = result["numbering"]
    formats = result["formats"]

    count = matching["matched_count"]
    parts = [f"{count} {plural(count, FRAME_FORMS)}"]

    sizes, channels = formats["sizes"], formats["channels"]
    if len(sizes) == 1 and len(channels) == 1:
        (width, height), = sizes
        depth, = channels
        parts.append(f"{width}x{height}x{depth}")
    else:
        parts.append("формат разный")

    if numbering.get("non_numeric"):
        parts.append("имена не числовые")
    elif numbering.get("gaps"):
        parts.append(f"разрывов {len(numbering['gaps'])}")
    else:
        parts.append("без разрывов")

    return ", ".join(parts)


def report_anomalies(result: dict[str, Any]) -> None:
    """
    Расписывает отклонения, свёрнутые в строке describe_integrity.

    Args:
        result: результат validate_dataset.
    """
    matching = result["matching"]
    numbering = result["numbering"]
    formats = result["formats"]
    quaternions = result["quaternions"]

    # Отрицание цепочки равенств, а не цепочка неравенств: при 1684 строках,
    # 3 файлах и 3 совпадениях условие a != b != c дало бы ложь и скрыло
    # расхождение, ради которого проверка и нужна
    if not (matching["csv_rows"] == matching["files_total"]
            == matching["matched_count"]):
        block_line("  строк телеметрии", matching["csv_rows"])
        block_line("  файлов в каталоге", matching["files_total"])
        block_line("  совпало по имени", matching["matched_count"])

    if numbering.get("gaps"):
        shown = numbering["gaps"][:MAX_GAPS_SHOWN]
        text = ", ".join(f"{start}-{end}" for start, end in shown)
        if len(numbering["gaps"]) > MAX_GAPS_SHOWN:
            text += f" и ещё {len(numbering['gaps']) - MAX_GAPS_SHOWN}"
        block_wrapped("  разрывы нумерации", text)

    if len(formats["sizes"]) > 1:
        block_line("  размеры кадров",
                   ", ".join(f"{w}x{h}: {n}" for (w, h), n in formats["sizes"].items()),
                   STATUS_FAIL)

    if len(formats["channels"]) > 1:
        block_line("  число каналов",
                   ", ".join(f"{d}: {n}" for d, n in formats["channels"].items()),
                   STATUS_FAIL)

    if quaternions.get("bad_count"):
        block_line("  норма кватернионов",
                   f"{format_number(quaternions['norm_min'], 4)} ... "
                   f"{format_number(quaternions['norm_max'], 4)}, "
                   f"вне допуска {quaternions['bad_count']}", STATUS_FAIL)

    if result["warnings"]:
        block_wrapped("  замечания", "; ".join(result["warnings"]))


def report_integrity(results: list[dict[str, Any]]) -> None:
    """
    Печатает блок целостности: строка на датасет, подробности при отклонении.

    Args:
        results: результаты validate_dataset по каждому датасету.
    """
    open_block("ЦЕЛОСТНОСТЬ ДАННЫХ", f"проверено {len(results)}")

    for result in results:
        if result["error"]:
            block_line(result["name"], result["error"], STATUS_FAIL)
            continue

        clean = not result["warnings"]
        block_line(result["name"], describe_integrity(result),
                   STATUS_OK if clean else STATUS_NONE)
        if not clean:
            report_anomalies(result)

    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Сводная таблица
# ────────────────────────────────────────────────────────────────────────────

def format_gradient(gradient: dict[str, float] | None) -> str:
    """
    Собирает ячейку с медианой градиента и его разбросом по выборке.

    Args:
        gradient: словарь с полями min, median и max либо None.

    Returns:
        Строку вида 0.15 (0.06-0.48), где первое число медиана, а в скобках
        наименьшее и наибольшее значения. При отсутствии данных MISSING.
    """
    if not gradient:
        return MISSING
    return (f"{format_number(gradient['median'])} "
            f"({format_number(gradient['min'])}-{format_number(gradient['max'])})")


def report_summary(results: list[dict[str, Any]]) -> None:
    """
    Печатает сводную таблицу характеристик датасетов.

    Шаг, высота и градиент приводятся медианой.

    Args:
        results: результаты validate_dataset по каждому датасету.
    """
    headers = ["Датасет", "Кадров", "Шаг, м", "Путь, м", "Высота, м",
               "Поворот", "Градиент"]
    rows: list[list[Any]] = []

    for result in results:
        if result["error"]:
            rows.append([result["name"], MISSING, "", "", "", "", ""])
            continue

        trajectory = result["trajectory"]
        altitude = MISSING
        if trajectory:
            altitude = (f"{format_number(trajectory['altitude_min'], 0)}-"
                        f"{format_number(trajectory['altitude_max'], 0)}")

        rows.append([
            result["name"],
            result["matching"]["matched_count"],
            format_number(trajectory.get("step_median")),
            format_number(trajectory.get("path_length"), 0),
            altitude,
            format_number(trajectory.get("turning_total"), 1),
            format_gradient(result["pixels"].get("gradient")),
        ])

    print_section("СВОДКА ПО ДАТАСЕТАМ")
    print()
    print_table(headers, rows)
    print()
    print_note(f"Поворот: извилистость маршрута, сумма изменений курса по модулю "
               f"в градусах. Курс берётся через каждые {TURNING_BASELINE_M:.0f} м пути.")
    print_note(f"Градиент: насыщенность кадра деталями, медиана по "
               f"{PIXEL_SAMPLE_SIZE} кадрам, в скобках разброс.")


# ────────────────────────────────────────────────────────────────────────────
# Пересечение датасетов
# ────────────────────────────────────────────────────────────────────────────

def check_overlaps(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ищет датасеты, описывающие один и тот же участок полёта.

    Сравниваются множества координат, поэтому находится и полное совпадение, и
    вложение одного набора в другой. Датасеты с ошибками пропускаются.

    Args:
        results: результаты validate_dataset по каждому датасету.

    Returns:
        Список словарей с полями left, right, shared и relation, где relation
        принимает значения «полное совпадение», «первый вложен во второй»,
        «второй вложен в первый» либо «частичное пересечение».
    """
    usable = [item for item in results if item["ok"]]
    findings: list[dict[str, Any]] = []

    for first in range(len(usable)):
        for second in range(first + 1, len(usable)):
            left, right = usable[first], usable[second]
            left_set, right_set = left["signature"], right["signature"]
            shared = left_set & right_set
            if not shared:
                continue

            if left_set == right_set:
                relation = "полное совпадение"
            elif left_set <= right_set:
                relation = "первый вложен во второй"
            elif right_set <= left_set:
                relation = "второй вложен в первый"
            else:
                relation = "частичное пересечение"

            findings.append({
                "left": left["name"],
                "right": right["name"],
                "shared": len(shared),
                "relation": relation,
            })

    return findings


def report_overlaps(findings: list[dict[str, Any]], total: int) -> None:
    """
    Печатает блок с результатом поиска совпадающих датасетов.

    Args:
        findings: результат check_overlaps.
        total: сколько датасетов проверено.
    """
    open_block("СОВПАДЕНИЕ ДАТАСЕТОВ", f"проверено {total}")

    if not findings:
        block_line("Пересечения по координатам", "не обнаружены", STATUS_OK)
        close_block()
        return

    for finding in findings:
        block_line(f"{finding['left']} и {finding['right']}",
                   f"{finding['shared']} {plural(finding['shared'], POSITION_FORMS)}, "
                   f"{finding['relation']}")

    block_note("Совпадающие датасеты не считаются независимыми наблюдениями: "
               "одинаковые метрики на них отражают повторный прогон одних и тех "
               "же данных.")
    close_block()


# ────────────────────────────────────────────────────────────────────────────
# Точка входа
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Проверяет все датасеты из конфига и печатает отчёт.
    """
    print_banner([
        "ПРОВЕРКА ДАТАСЕТОВ",
        "Пригодность данных к работе и характеристики маршрутов",
    ])

    results: list[dict[str, Any]] = []
    for name in config.DATASETS:
        try:
            result = validate_dataset(name)
        except Exception as error:
            result = {"name": name, "ok": False, "warnings": [],
                      "error": describe_exception(error)}
        results.append(result)

    report_integrity(results)
    report_overlaps(check_overlaps(results), len(results))
    report_summary(results)

    failed = [result["name"] for result in results if not result["ok"]]
    warned = [result["name"] for result in results if result["ok"] and result["warnings"]]

    lines = [f"ИТОГ: пригодно датасетов {len(results) - len(failed)} из {len(results)}"]
    if failed:
        lines.append("Недоступны: " + ", ".join(failed))
    if warned:
        lines.append("С замечаниями: " + ", ".join(warned))
    print_banner(lines)

    print()


if __name__ == "__main__":
    main()
