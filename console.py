"""
Примитивы вывода в консоль.

Модуль содержит баннеры, блоки с боковой линейкой, таблицы с расшифровкой
обозначений, индикатор выполнения и приведение значений к читаемому виду.
Предметной области не знает.

Вывод состоит из символов Unicode, без управляющих последовательностей и
внешних зависимостей.
"""

# Стандартные библиотеки
import textwrap
from typing import Any, Sequence


# === ГЕОМЕТРИЯ БЛОКОВ ===
BOX_WIDTH = 78                  # Полная ширина рамки в символах
LABEL_WIDTH = 28                # Ширина колонки с названием параметра
STATUS_WIDTH = 8                # Ширина колонки статуса
PREFIX = "\u2502  "             # Вертикальная линейка слева и отступ

# Ширина колонки значения зависит от того, будет ли справа статус
VALUE_WIDTH_WITH_STATUS = BOX_WIDTH - len(PREFIX) - LABEL_WIDTH - 1 - STATUS_WIDTH
VALUE_WIDTH_PLAIN = BOX_WIDTH - len(PREFIX) - LABEL_WIDTH

# === СИМВОЛЫ РАМОК ===
LINE_SINGLE = "\u2500"          # Горизонтальная линия обычной рамки
LINE_DOUBLE = "\u2550"          # Горизонтальная линия баннера
CORNER_TOP = "\u250c"           # Левый верхний угол блока
CORNER_BOTTOM = "\u2514"        # Левый нижний угол блока

# === СТАТУСЫ ПРОВЕРОК ===
# Длина каждого статуса равна STATUS_WIDTH, правый край колонки не плавает
STATUS_OK = "[  OK  ]"          # Проверка пройдена
STATUS_FAIL = "[ОШИБКА]"        # Проверка не пройдена
STATUS_NONE = "[ НЕТ  ]"        # Значение недоступно, проверять нечего

# === ГЕОМЕТРИЯ ТАБЛИЦ ===
TABLE_INDENT = 2                # Отступ таблицы от левого края
TABLE_GAP = 2                   # Промежуток между колонками
TABLE_MAX_COL = 30              # Предельная ширина колонки, дальше обрезка
LEGEND_GAP = 3                  # Промежуток между обозначением и расшифровкой

# === ИНДИКАТОР ВЫПОЛНЕНИЯ ===
PROGRESS_WIDTH = 30             # Полная длина полосы в символах
PROGRESS_FILLED = "\u2501"      # Пройденная часть
PROGRESS_EMPTY = "\u2500"       # Оставшаяся часть
PROGRESS_MIN_WIDTH = 8          # Предел укорачивания полосы

# === ФОРМЫ СЧЁТНЫХ СЛОВ ===
# Порядок форм: одна пара, две пары, пять пар
PROGRESS_FORMS: tuple[str, str, str] = ("пара", "пары", "пар")
FRAME_FORMS: tuple[str, str, str] = ("кадр", "кадра", "кадров")

# === ЕДИНИЦЫ ИЗМЕРЕНИЯ ОБЪЁМА ===
BYTE_UNITS = ("Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ")   # По возрастанию
BYTE_STEP = 1024.0                                 # Множитель перехода к следующей

MISSING = "нет данных"          # Обозначение отсутствующего значения

# Признак того, что пустая строка перед следующим заголовком уже напечатана.
# Ставит clear_progress на месте погашенного индикатора, снимает begin_line
_separator_ready = False


# ────────────────────────────────────────────────────────────────────────────
# Приведение значений к читаемому виду
# ────────────────────────────────────────────────────────────────────────────

def fit(text: Any, width: int) -> str:
    """
    Укорачивает строку до заданной ширины, помечая обрезку многоточием.

    При ширине меньше четырёх символов многоточие не помещается, и строка
    обрезается по границе.

    Args:
        text: исходное значение, приводится к строке.
        width: предельная ширина в символах.

    Returns:
        Строку не длиннее width символов.
    """
    text = str(text)
    if len(text) <= width:
        return text
    if width < 4:
        return text[:max(width, 0)]
    return text[:width - 3] + "..."


def format_bytes(value: int | float | None) -> str:
    """
    Переводит число байт в читаемый вид с автоматическим выбором единицы.

    Args:
        value: количество байт либо None.

    Returns:
        Строку вида 31.9 ГБ, для None пометку MISSING.
    """
    if value is None:
        return MISSING

    amount = float(value)
    for unit in BYTE_UNITS[:-1]:
        if amount < BYTE_STEP:
            return f"{amount:.1f} {unit}"
        amount /= BYTE_STEP
    return f"{amount:.1f} {BYTE_UNITS[-1]}"


def format_number(value: Any, digits: int = 2, unit: str = "") -> str:
    """
    Приводит число к виду с фиксированным числом знаков после запятой.

    Целые печатаются без дробной части, логические значения как True и False,
    нечисловые значения возвращаются приведёнными к строке.

    Args:
        value: число либо None.
        digits: число знаков после запятой для дробных значений.
        unit: единица измерения, приписывается через пробел.

    Returns:
        Отформатированную строку, для None и NaN пометку MISSING.
    """
    if value is None:
        return MISSING

    suffix = f" {unit}" if unit else ""

    # Проверка идёт до целых чисел: bool наследуется от int,
    # и True иначе превратилось бы в 1.00
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return f"{value}{suffix}"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number != number:  # NaN не равен самому себе
        return MISSING
    return f"{number:.{digits}f}{suffix}"


def plural(count: int, forms: tuple[str, str, str]) -> str:
    """
    Выбирает форму счётного слова по числу.

    Args:
        count: количество.
        forms: формы для одного, двух и пяти.

    Returns:
        Подходящую форму слова.
    """
    tail = abs(count) % 10
    hundreds = abs(count) % 100

    if tail == 1 and hundreds != 11:
        return forms[0]
    if tail in (2, 3, 4) and hundreds not in (12, 13, 14):
        return forms[1]
    return forms[2]


def describe_exception(error: BaseException) -> str:
    """
    Сворачивает исключение в одну строку.

    Args:
        error: пойманное исключение.

    Returns:
        Строку с типом исключения и первой строкой его сообщения.
    """
    text = str(error).strip().splitlines()
    first_line = text[0] if text else "без описания"
    return f"{type(error).__name__}: {first_line}"


# ────────────────────────────────────────────────────────────────────────────
# Баннеры и заголовки
# ────────────────────────────────────────────────────────────────────────────

def begin_line() -> None:
    """
    Отбивает следующий заголовок пустой строкой, если её ещё нет.

    Отбивку после индикатора выполнения печатает clear_progress, и тогда
    функция её не повторяет.
    """
    global _separator_ready
    if _separator_ready:
        _separator_ready = False
    else:
        print()


def print_banner(lines: Sequence[str]) -> None:
    """
    Печатает рамку из двойных линий с текстом внутри.

    Строки печатаются с отступом, пустая строка печатается без отступа и
    работает как разделитель.

    Args:
        lines: строки внутри рамки.
    """
    rule = LINE_DOUBLE * BOX_WIDTH
    begin_line()
    print(rule)
    for line in lines:
        print(f"  {line}" if line else "")
    print(rule)


def print_section(title: str, width: int = BOX_WIDTH) -> None:
    """
    Печатает заголовок секции с подчёркиванием, без боковой линейки.

    Args:
        title: заголовок секции.
        width: длина подчёркивания.
    """
    begin_line()
    print(f"  {title}")
    print("  " + LINE_SINGLE * (width - 2))


# ────────────────────────────────────────────────────────────────────────────
# Индикатор выполнения
# ────────────────────────────────────────────────────────────────────────────

def compose_progress(label: str, filled_share: float, tail: str) -> str:
    """
    Собирает строку индикатора: подпись, полоса, текст справа.

    Полоса занимает свободное место до PROGRESS_WIDTH символов. При нехватке
    ширины сначала укорачивается полоса до PROGRESS_MIN_WIDTH, затем подпись.

    Args:
        label: пояснение слева от полосы.
        filled_share: доля закрашенной части, от нуля до единицы.
        tail: текст справа от полосы.

    Returns:
        Строку не длиннее BOX_WIDTH символов.
    """
    head = f"  {label} " if label else "  "
    available = BOX_WIDTH - len(head) - len(tail) - 1

    if available < PROGRESS_MIN_WIDTH:
        head = fit(head, max(len(head) - (PROGRESS_MIN_WIDTH - available), 3))
        available = BOX_WIDTH - len(head) - len(tail) - 1

    width = max(min(PROGRESS_WIDTH, available), 0)
    filled = int(round(min(max(filled_share, 0.0), 1.0) * width))
    bar = PROGRESS_FILLED * filled + PROGRESS_EMPTY * (width - filled)

    return fit(f"{head}{bar} {tail}", BOX_WIDTH)


def print_progress(current: int, total: int, label: str = "") -> None:
    """
    Перерисовывает индикатор выполнения поверх самого себя.

    Строка не переводится, каждая отрисовка ложится поверх предыдущей и
    добивается пробелами до полной ширины. Нулевой и отрицательный объём работы
    не печатается. Гасит индикатор clear_progress.

    Args:
        current: сколько единиц работы выполнено.
        total: сколько всего единиц работы.
        label: пояснение слева от полосы.
    """
    global _separator_ready
    if total <= 0:
        return

    _separator_ready = False

    text = compose_progress(label, current / total, f"{current}/{total}")
    print(f"\r{text:<{BOX_WIDTH}}", end="", flush=True)


def clear_progress(label: str, count: int, seconds: float,
                   forms: tuple[str, str, str] = PROGRESS_FORMS) -> None:
    """
    Заменяет индикатор итоговой строкой о завершённой работе.

    Полоса печатается заполненной целиком, справа от неё встаёт число
    обработанных единиц и затраченное время. Прежняя строка затирается
    пробелами, а не управляющей последовательностью.

    Время печатается с одним знаком после запятой, счётное слово при нём
    неизменяемо. Форму слова plural выбирает только для count.

    Args:
        label: пояснение слева от полосы.
        count: сколько единиц обработано.
        seconds: сколько времени это заняло.
        forms: формы счётного слова для одного, двух и пяти.
    """
    global _separator_ready

    tail = f"{count} {plural(count, forms)} за {seconds:.1f} секунды"
    print("\r" + " " * BOX_WIDTH, end="")
    print("\r" + compose_progress(label, 1.0, tail), flush=True)
    _separator_ready = True


# ────────────────────────────────────────────────────────────────────────────
# Блоки с боковой линейкой
# ────────────────────────────────────────────────────────────────────────────

def open_block(title: str, note: str = "") -> None:
    """
    Открывает блок: верхняя граница с заголовком, затем пустая строка линейки.

    Args:
        title: заголовок в левой части границы.
        note: примечание у правого края границы.
    """
    left = f"{CORNER_TOP}{LINE_SINGLE} {title} "
    right = f" {note} {LINE_SINGLE * 3}" if note else ""
    filler = max(BOX_WIDTH - len(left) - len(right), 0)
    begin_line()
    print(left + LINE_SINGLE * filler + right)
    print(PREFIX.rstrip())


def close_block() -> None:
    """
    Закрывает блок: пустая строка линейки, затем нижняя граница.
    """
    print(PREFIX.rstrip())
    print(CORNER_BOTTOM + LINE_SINGLE * (BOX_WIDTH - 1))


def block_line(label: str, value: Any, status: str = "") -> None:
    """
    Печатает строку блока: название, значение и необязательный статус.

    Статус встаёт у правого края блока, статусы разных строк выровнены в одну
    колонку. Название и значение обрезаются по ширине своих колонок.

    Args:
        label: название параметра.
        value: значение, приводится к строке.
        status: одна из констант STATUS_OK, STATUS_FAIL, STATUS_NONE.
    """
    label = fit(label, LABEL_WIDTH - 1)

    if status:
        text = fit(value, VALUE_WIDTH_WITH_STATUS)
        print(f"{PREFIX}{label:<{LABEL_WIDTH}}{text:<{VALUE_WIDTH_WITH_STATUS}} {status}")
    else:
        text = fit(value, VALUE_WIDTH_PLAIN)
        print(f"{PREFIX}{label:<{LABEL_WIDTH}}{text}")


def block_wrapped(label: str, value: Any) -> None:
    """
    Печатает значение с переносом по ширине колонки значения.

    Название стоит в первой строке, продолжение выравнивается под ней.

    Args:
        label: название параметра.
        value: значение, приводится к строке.
    """
    label = fit(label, LABEL_WIDTH - 1)
    lines = textwrap.wrap(str(value), width=VALUE_WIDTH_PLAIN) or [""]
    print(f"{PREFIX}{label:<{LABEL_WIDTH}}{lines[0]}")
    for line in lines[1:]:
        print(f"{PREFIX}{'':<{LABEL_WIDTH}}{line}")


def block_note(text: str) -> None:
    """
    Печатает свободный текст внутри блока с переносом по ширине рамки.

    Перед текстом печатается пустая строка линейки.

    Args:
        text: текст произвольной длины.
    """
    width = BOX_WIDTH - len(PREFIX)
    print(PREFIX.rstrip())
    for line in textwrap.wrap(text, width=width):
        print(f"{PREFIX}{line}")


# ────────────────────────────────────────────────────────────────────────────
# Таблицы
# ────────────────────────────────────────────────────────────────────────────

def looks_numeric(text: str) -> bool:
    """
    Определяет, начинается ли содержимое ячейки с числа.

    Числовой частью считается начальная последовательность из цифр, знака,
    точки и символа экспоненты. Пробелы отбрасываются, запятая читается как
    десятичный разделитель. Хвост из букв допускается и трактуется как единица
    измерения, хвост с цифрами означает составное значение вида 500x500 и
    числом не считается.

    Args:
        text: содержимое ячейки.

    Returns:
        True, если ячейка начинается с числа.
    """
    cleaned = text.strip().replace(",", ".").replace(" ", "")
    if not cleaned:
        return False

    head = ""
    for char in cleaned:
        if char.isdigit() or char in "+-.eE":
            head += char
        else:
            break

    tail = cleaned[len(head):]
    if any(char.isdigit() for char in tail):
        return False

    try:
        float(head)
        return True
    except ValueError:
        return False


def resolve_alignments(headers: Sequence[str],
                       rows: Sequence[Sequence[Any]]) -> list[str]:
    """
    Определяет выравнивание каждой колонки таблицы.

    Колонка выравнивается по правому краю, если все её непустые ячейки
    выглядят числами, иначе по левому.

    Args:
        headers: заголовки колонок, задают их число.
        rows: строки таблицы.

    Returns:
        Список символов выравнивания, r либо l, длиной по числу колонок.
    """
    resolved: list[str] = []
    for index in range(len(headers)):
        cells = [str(row[index]) for row in rows if index < len(row)]
        filled = [cell for cell in cells if cell.strip()]
        numeric = bool(filled) and all(looks_numeric(cell) for cell in filled)
        resolved.append("r" if numeric else "l")
    return resolved


def print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
                indent: int = TABLE_INDENT,
                max_col: int = TABLE_MAX_COL) -> None:
    """
    Печатает таблицу с автоподбором ширины колонок.

    Ширина колонки равна длине самой длинной её ячейки, включая заголовок.
    Выравнивание подбирает resolve_alignments. Строки короче заголовка
    дополняются пустыми ячейками. Пустой список заголовков не печатается.

    Args:
        headers: заголовки колонок.
        rows: строки таблицы, значения приводятся к строкам.
        indent: отступ таблицы слева в пробелах.
        max_col: предельная ширина колонки, дальше обрезка.
    """
    if not headers:
        return

    # Приведение к строкам идёт заранее, ширина считается по итоговому виду
    text_rows = [[fit(cell, max_col) for cell in row] for row in rows]
    text_headers = [fit(header, max_col) for header in headers]

    resolved_aligns = resolve_alignments(text_headers, text_rows)

    widths: list[int] = []
    for index, header in enumerate(text_headers):
        column = [len(row[index]) for row in text_rows if index < len(row)]
        widths.append(max([len(header)] + column))

    gap = " " * TABLE_GAP
    pad = " " * indent

    header_cells = [f"{header:<{widths[i]}}" for i, header in enumerate(text_headers)]
    print(pad + gap.join(header_cells).rstrip())
    print(pad + gap.join(LINE_SINGLE * width for width in widths))

    for row in text_rows:
        cells: list[str] = []
        for index, width in enumerate(widths):
            cell = row[index] if index < len(row) else ""
            if resolved_aligns[index] == "r":
                cells.append(f"{cell:>{width}}")
            else:
                cells.append(f"{cell:<{width}}")
        print(pad + gap.join(cells).rstrip())


def print_columns(cells: Sequence[str], columns: int, prefix: str = PREFIX) -> None:
    """
    Раскладывает готовые ячейки по строкам в несколько колонок.

    Ширину ячеек выравнивает вызывающий код, здесь они только соединяются.

    Args:
        cells: готовые строки одинаковой ширины.
        columns: число колонок.
        prefix: префикс строки, по умолчанию боковая линейка блока.
    """
    for start in range(0, len(cells), columns):
        row = "".join(cells[start:start + columns])
        print(f"{prefix}{row.rstrip()}")


def print_legend(entries: Sequence[Sequence[str]]) -> None:
    """
    Печатает расшифровку обозначений таблицы.

    Каждое обозначение занимает свою строку, описания выровнены по общей
    колонке. Длинное описание переносится по ширине отчёта с отступом под
    колонку описаний. Порядок записей задаёт вызывающий код.

    Args:
        entries: пары из обозначения и его расшифровки.

    Raises:
        ValueError: список записей пуст.
    """
    if not entries:
        raise ValueError("Легенду нельзя напечатать без записей")

    width = max(len(name) for name, _ in entries) + LEGEND_GAP
    text_width = BOX_WIDTH - TABLE_INDENT - width

    for name, meaning in entries:
        lines = textwrap.wrap(meaning, width=text_width) or [""]
        for index, line in enumerate(lines):
            head = f"{name:<{width}}" if index == 0 else " " * width
            print(" " * TABLE_INDENT + head + line)


def print_note(text: str) -> None:
    """
    Печатает пояснение под таблицей с переносом по ширине отчёта.

    В отличие от block_note не рисует боковую линейку слева: таблица выводится
    вне рамки блока.

    Args:
        text: текст произвольной длины.
    """
    for line in textwrap.wrap(text, width=BOX_WIDTH - TABLE_INDENT):
        print(" " * TABLE_INDENT + line)
