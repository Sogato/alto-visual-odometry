"""
Сбор сведений о вычислительном окружении.

Модуль определяет характеристики машины, версии установленных библиотек,
доступность CUDA и состав сборки OpenCV. Проверка ограничена наличием
компонентов, корректность их работы не оценивается.

Функции сбора при любом исходе возвращают словарь ожидаемой формы, недоступные
значения в нём равны None.

Замеры производительности выполняет vo_module/analysis/performance.py, оттуда
же вызывается hardware_line.
"""

# Стандартные библиотеки
import platform
import sys
from pathlib import Path
from typing import Any

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from console import (MISSING, STATUS_FAIL, STATUS_NONE, block_line, close_block,
                     describe_exception, fit, format_bytes, open_block, plural,
                     print_banner, print_columns)


# === БИБЛИОТЕКИ, ВЕРСИИ КОТОРЫХ ПОПАДУТ В ОТЧЁТ ===
# Ключ это имя пакета для importlib.metadata, значение это имя модуля для
# импорта: пакет opencv-python импортируется как cv2
TRACKED_PACKAGES: dict[str, str] = {
    "numpy": "numpy",
    "opencv-python": "cv2",
    "torch": "torch",
    "torchvision": "torchvision",
    "kornia": "kornia",
    "lightglue": "lightglue",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "psutil": "psutil",
    "nvidia-ml-py": "pynvml",
}

# === ПАКЕТЫ, БЕЗ КОТОРЫХ НЕ РАБОТАЕТ ОДИН ИЗ МЕТОДОВ ===
# Имя пакета и последствия его отсутствия, читает report_summary
CRITICAL_PACKAGES: dict[str, str] = {
    "opencv-python": "методы на основе SIFT и ORB",
    "torch": "метод на основе SuperPoint и LightGlue",
    "lightglue": "метод на основе SuperPoint и LightGlue",
}

# === ФУНКЦИИ OPENCV, ИСПОЛЬЗУЕМЫЕ ПРОЕКТОМ ===
# Проверяются как атрибуты модуля: в части сборок детекторы вынесены в contrib
REQUIRED_CV_ATTRS: tuple[str, ...] = (
    "SIFT_create",
    "ORB_create",
    "BFMatcher",
    "calcOpticalFlowPyrLK",
    "findHomography",
    "findEssentialMat",
    "recoverPose",
    "decomposeHomographyMat",
)

# === РАСКЛАДКА СПИСКА БИБЛИОТЕК ===
LIB_COLUMNS = 2                 # Число колонок в блоке версий
LIB_NAME_WIDTH = 17             # Ширина имени пакета внутри ячейки
LIB_VERSION_WIDTH = 20          # Ширина версии внутри ячейки, дальше обрезка

# Пометка отсутствующего пакета. По совпадению с ней report_summary определяет
# наличие пакета
NOT_INSTALLED = "не установлен"

# Примечания к версиям, приписываются в скобках. LightGlue ставится из
# репозитория и объявляет версию 0.0
VERSION_NOTES: dict[str, str] = {
    "lightglue": "из репозитория",
}

# === ФОРМЫ СЧЁТНЫХ СЛОВ ===
# Порядок форм: одно ядро, два ядра, пять ядер
CORE_FORMS: tuple[str, str, str] = ("ядро", "ядра", "ядер")
THREAD_FORMS: tuple[str, str, str] = ("поток", "потока", "потоков")


# ────────────────────────────────────────────────────────────────────────────
# Характеристики машины
# ────────────────────────────────────────────────────────────────────────────

def collect_os_info() -> dict[str, Any]:
    """
    Собирает сведения об операционной системе и интерпретаторе Python.

    Returns:
        Словарь с полями system, release, machine, python и python_build.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_build": platform.python_implementation(),
    }


def read_cpu_model() -> str:
    """
    Определяет модель процессора.

    Источники в порядке опроса: поле model name в /proc/cpuinfo, ветвь реестра
    Windows, platform.processor(), platform.machine().

    Returns:
        Название модели процессора либо пометку «не определено».
    """
    # Linux
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for raw_line in handle:
                if raw_line.lower().startswith("model name"):
                    return raw_line.split(":", 1)[1].strip()
    except OSError:
        pass

    # Windows
    if platform.system() == "Windows":
        try:
            import winreg
            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if name:
                    return str(name).strip()
        except Exception:
            pass

    fallback = platform.processor() or platform.machine()
    return fallback or "не определено"


def collect_cpu_info() -> dict[str, Any]:
    """
    Собирает характеристики процессора.

    Частота берётся паспортная: поле max у psutil, при нулевом его значении
    текущая частота. Под нагрузкой процессор работает выше паспортной.

    Returns:
        Словарь с полями model, cores_physical, cores_logical, freq_base_mhz в
        мегагерцах и cv_threads. Недоступные значения равны None, кроме model.
    """
    info: dict[str, Any] = {
        "model": read_cpu_model(),
        "cores_physical": None,
        "cores_logical": None,
        "freq_base_mhz": None,
        "cv_threads": None,
    }

    try:
        import psutil
        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["cores_logical"] = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq is not None:
            info["freq_base_mhz"] = freq.max or freq.current
    except Exception:
        # Частота доступна не на всех платформах
        pass

    try:
        import cv2
        info["cv_threads"] = cv2.getNumThreads()
    except Exception:
        pass

    return info


def collect_memory_info() -> dict[str, Any]:
    """
    Собирает сведения об объёме оперативной памяти машины.

    Returns:
        Словарь с полями total и available в байтах, недоступные значения равны
        None.
    """
    info: dict[str, Any] = {"total": None, "available": None}
    try:
        import psutil
        virtual = psutil.virtual_memory()
        info["total"] = virtual.total
        info["available"] = virtual.available
    except Exception:
        pass
    return info


def collect_gpu_info() -> dict[str, Any]:
    """
    Собирает сведения о графическом процессоре с устройства с индексом 0.

    Основной источник PyTorch, NVML добавляет версию драйвера и текущую
    занятость видеопамяти.

    Returns:
        Словарь с полями cuda_available, device_count, name, vram_total,
        vram_free, capability, torch_cuda_version, driver_version и error.
        Недоступные значения равны None. Поле error заполняется при отказе
        импорта PyTorch, отказ NVML остальные поля сохраняет.
    """
    info: dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "name": None,
        "vram_total": None,
        "vram_free": None,
        "capability": None,
        "torch_cuda_version": None,
        "driver_version": None,
        "error": None,
    }

    try:
        import torch
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["name"] = props.name
            info["vram_total"] = props.total_memory
            info["capability"] = f"{props.major}.{props.minor}"
    except Exception as error:
        info["error"] = describe_exception(error)
        return info

    # Отдельный блок: отказ NVML не отменяет сведения от PyTorch
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["vram_free"] = memory.free
            if info["vram_total"] is None:
                info["vram_total"] = memory.total

            driver = pynvml.nvmlSystemGetDriverVersion()
            # В разных версиях NVML строка приходит как str либо как bytes
            info["driver_version"] = driver.decode() if isinstance(driver, bytes) else driver
        finally:
            # Библиотека остаётся проинициализированной и при отказе опроса
            pynvml.nvmlShutdown()
    except Exception:
        pass

    return info


def hardware_line() -> str:
    """
    Собирает описание машины в одну строку.

    Returns:
        Строку вида «процессор, ядра, потоки OpenCV, видеокарта». Недоступные
        части опускаются, при недоступной CUDA печатается «без видеокарты».
    """
    cpu_info = collect_cpu_info()
    gpu_info = collect_gpu_info()

    parts: list[str] = [cpu_info["model"]]

    physical, logical = cpu_info["cores_physical"], cpu_info["cores_logical"]
    if physical and logical:
        parts.append(f"{physical}/{logical} {plural(logical, CORE_FORMS)}")

    threads = cpu_info["cv_threads"]
    if threads is not None:
        parts.append(f"{threads} {plural(threads, THREAD_FORMS)} OpenCV")

    parts.append(gpu_info["name"] if gpu_info.get("cuda_available")
                 else "без видеокарты")

    return ", ".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# Состав окружения
# ────────────────────────────────────────────────────────────────────────────

def collect_library_versions() -> dict[str, str]:
    """
    Определяет версии библиотек из TRACKED_PACKAGES.

    Версия запрашивается у менеджера пакетов, при неудаче считывается из
    атрибута __version__ модуля.

    Returns:
        Словарь, сопоставляющий имени пакета его версию. Для отсутствующих
        пакетов значением служит NOT_INSTALLED.
    """
    from importlib.metadata import version as pkg_version

    versions: dict[str, str] = {}

    for package_name, module_name in TRACKED_PACKAGES.items():
        resolved = None

        try:
            resolved = pkg_version(package_name)
        except Exception:
            # Пакет без метаданных, версию ищет запасной путь
            pass

        if resolved is None:
            try:
                module = __import__(module_name)
                resolved = getattr(module, "__version__", None)
            except Exception:
                resolved = None

        versions[package_name] = resolved or NOT_INSTALLED

    return versions


def check_opencv_functions() -> dict[str, Any]:
    """
    Проверяет наличие в сборке OpenCV функций из REQUIRED_CV_ATTRS.

    Returns:
        Словарь с полями ok, missing_attrs и error. При отказе импорта ok равно
        False, missing_attrs пуст, а причина лежит в error.
    """
    result: dict[str, Any] = {
        "ok": False,
        "missing_attrs": [],
        "error": None,
    }

    try:
        import cv2
    except Exception as error:
        result["error"] = describe_exception(error)
        return result

    result["missing_attrs"] = [name for name in REQUIRED_CV_ATTRS if not hasattr(cv2, name)]
    result["ok"] = not result["missing_attrs"]
    return result


# ────────────────────────────────────────────────────────────────────────────
# Вывод отчёта
# ────────────────────────────────────────────────────────────────────────────

def format_cores(cpu_info: dict[str, Any]) -> str:
    """
    Собирает строку о ядрах процессора и его базовой частоте.

    Args:
        cpu_info: результат collect_cpu_info.

    Returns:
        Строку вида «8 физических, 16 логических, 3600 МГц базовая». Частота
        опускается, если недоступна, при недоступном числе ядер возвращает
        MISSING.
    """
    physical, logical = cpu_info["cores_physical"], cpu_info["cores_logical"]
    if not physical or not logical:
        return MISSING

    text = f"{physical} физических, {logical} логических"
    if cpu_info["freq_base_mhz"]:
        text += f", {cpu_info['freq_base_mhz']:.0f} МГц базовая"
    return text


def format_memory(memory_info: dict[str, Any]) -> str:
    """
    Собирает строку об объёме оперативной памяти.

    Args:
        memory_info: результат collect_memory_info.

    Returns:
        Строку вида «31.9 ГБ всего, 12.4 ГБ свободно», при недоступном объёме
        MISSING.
    """
    if memory_info["total"] is None and memory_info["available"] is None:
        return MISSING
    return (f"{format_bytes(memory_info['total'])} всего, "
            f"{format_bytes(memory_info['available'])} свободно")


def report_system(os_info: dict[str, Any], cpu_info: dict[str, Any],
                  memory_info: dict[str, Any]) -> None:
    """
    Выводит блок с характеристиками системы, процессора и оперативной памяти.

    Args:
        os_info: результат collect_os_info.
        cpu_info: результат collect_cpu_info.
        memory_info: результат collect_memory_info.
    """
    open_block("СИСТЕМА")

    block_line("Операционная система", f"{os_info['system']} {os_info['release']}, "
                                       f"{os_info['machine']}")
    block_line("Python", f"{os_info['python']}, {os_info['python_build']}")
    block_line("Процессор", cpu_info["model"])
    block_line("Ядра", format_cores(cpu_info))

    if cpu_info["cv_threads"] is not None:
        block_line("Потоков OpenCV", cpu_info["cv_threads"])

    block_line("Оперативная память", format_memory(memory_info))

    close_block()


def report_gpu(gpu_info: dict[str, Any]) -> None:
    """
    Выводит блок с характеристиками графического процессора.

    Печатается одна из трёх картин: отказ опроса, недоступная CUDA либо
    характеристики устройства.

    Args:
        gpu_info: результат collect_gpu_info.
    """
    open_block("ГРАФИЧЕСКИЙ ПРОЦЕССОР")

    if gpu_info.get("error"):
        block_line("Опрос устройства", gpu_info["error"], STATUS_FAIL)
    elif not gpu_info.get("cuda_available"):
        block_line("CUDA", "устройство недоступно", STATUS_NONE)
        block_line("CUDA в сборке PyTorch", gpu_info["torch_cuda_version"] or "отсутствует")
    else:
        block_line("Модель", gpu_info["name"])
        block_line("Видеопамять", f"{format_bytes(gpu_info['vram_total'])} всего, "
                                  f"{format_bytes(gpu_info['vram_free'])} свободно")
        block_line("Compute capability", gpu_info["capability"])
        block_line("CUDA в сборке PyTorch", gpu_info["torch_cuda_version"] or "отсутствует")
        block_line("Версия драйвера", gpu_info["driver_version"] or "не определена")

        # Число устройств печатается, только когда есть из чего выбирать
        if gpu_info["device_count"] > 1:
            block_line("Число устройств", gpu_info["device_count"])

    close_block()


def report_libraries(versions: dict[str, str], cv_check: dict[str, Any]) -> None:
    """
    Выводит блок с версиями библиотек и результатом проверки состава OpenCV.

    Пакеты раскладываются по LIB_COLUMNS колонкам. Состав OpenCV попадает в
    список ячеек только при отказе импорта или нехватке функций, следом
    печатается строка с подробностями.

    Args:
        versions: результат collect_library_versions.
        cv_check: результат check_opencv_functions.
    """
    open_block("БИБЛИОТЕКИ")

    missing_attrs = cv_check.get("missing_attrs") or []

    cells: list[str] = []
    for package_name, package_version in versions.items():
        # Примечание к версии имеет смысл только для установленного пакета
        note = VERSION_NOTES.get(package_name)
        shown = (f"{package_version} ({note})"
                 if note and package_version != NOT_INSTALLED else package_version)
        cells.append(f"{package_name:<{LIB_NAME_WIDTH}}"
                     f"{fit(shown, LIB_VERSION_WIDTH):<{LIB_VERSION_WIDTH}}")

    # Ячейкой, а не строкой block_line: та отбивает значение по иной ширине и
    # нарушила бы выравнивание колонок
    if cv_check.get("error"):
        cells.append(f"{'функции OpenCV':<{LIB_NAME_WIDTH}}"
                     f"{fit('модуль недоступен', LIB_VERSION_WIDTH):<{LIB_VERSION_WIDTH}}")
    elif missing_attrs:
        cells.append(f"{'функции OpenCV':<{LIB_NAME_WIDTH}}"
                     f"{fit('отсутствуют', LIB_VERSION_WIDTH):<{LIB_VERSION_WIDTH}}")

    print_columns(cells, LIB_COLUMNS)

    if cv_check.get("error"):
        block_line("Отказ импорта OpenCV", cv_check["error"], STATUS_FAIL)

    if missing_attrs:
        block_line("Отсутствуют функции", ", ".join(missing_attrs), STATUS_FAIL)

    close_block()


def report_summary(versions: dict[str, str], gpu_info: dict[str, Any],
                   cv_check: dict[str, Any]) -> None:
    """
    Выводит итоговое заключение о готовности окружения.

    Окружение считается неготовым при отсутствии пакета из CRITICAL_PACKAGES,
    при отказе импорта OpenCV и при нехватке функций в его сборке.
    Недоступность CUDA работу не блокирует и печатается отдельной строкой.

    Args:
        versions: результат collect_library_versions.
        gpu_info: результат collect_gpu_info.
        cv_check: результат check_opencv_functions.
    """
    problems: list[str] = []

    for package_name, affected in CRITICAL_PACKAGES.items():
        if versions.get(package_name, NOT_INSTALLED) == NOT_INSTALLED:
            problems.append(f"{package_name}: не установлен, недоступен {affected}")

    # Установленный пакет отказывает по двум разным причинам, подробности к ним
    # уже напечатал report_libraries
    if versions.get("opencv-python", NOT_INSTALLED) != NOT_INSTALLED:
        if cv_check.get("error"):
            problems.append("OpenCV: пакет установлен, но не импортируется")
        elif cv_check.get("missing_attrs"):
            problems.append("OpenCV: сборка не содержит требуемых функций")

    if problems:
        lines = ["ИТОГ: окружение к работе не готово", ""]
        lines += [f"  {problem}" for problem in problems]
        print_banner(lines)
        return

    lines = ["ИТОГ: окружение готово к работе"]
    if not gpu_info.get("cuda_available"):
        lines.append("CUDA недоступна, вычисления будут выполнены на процессоре")
    print_banner(lines)


# ────────────────────────────────────────────────────────────────────────────
# Точка входа
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Собирает сведения об окружении и выводит отчёт.
    """
    print_banner([
        "СВЕДЕНИЯ ОБ ОКРУЖЕНИИ",
        "Проект оценки собственного движения по данным ALTO",
    ])

    report_system(collect_os_info(), collect_cpu_info(), collect_memory_info())

    gpu_info = collect_gpu_info()
    versions = collect_library_versions()
    cv_check = check_opencv_functions()

    report_gpu(gpu_info)
    report_libraries(versions, cv_check)
    report_summary(versions, gpu_info, cv_check)

    print()


if __name__ == "__main__":
    main()
