"""
Оформление и сохранение графиков.

Модуль собирает то, из чего складываются все картинки проекта: палитра,
приведение поля к общему виду, легенда, запись файла. Сами картинки строятся в
соседних модулях, а отсюда берут оформление.

Соглашение об обозначениях: цвет закреплён за связкой, начертание линии за
геометрией. Там, где связок нет и цвет свободен, он закрепляется за геометрией.
Два слоя одной связки различаются насыщенностью, а не новым цветом.

Заголовков внутри картинок нет: рисунок в статье подписывается снизу.
Освободившееся место занимает легенда, вынесенная из поля данных наверх.

Величины оформления заданы здесь и только здесь. У функций модуля нет настроек
размера: если величина не подходит, её меняют здесь для всех картинок сразу.

Предметной области модуль не знает, всё нужное приходит аргументами.
Единственная зависимость это config, и та ради пути к каталогу результатов и
списков связок и геометрий, задающих порядок цветов.
"""

# Стандартные библиотеки
import sys
from pathlib import Path
from typing import Any, Sequence

# Сторонние библиотеки
import matplotlib
import numpy as np

# Движок выбирается до первого обращения к pyplot. Без экранного вывода графики
# строятся быстрее и не требуют оконной подсистемы
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as Outline

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


# === ЛИСТ ===
# Соотношение сторон и разрешение рассчитаны на рисунок в две колонки
FIGURE_WIDTH = 9.0      # Ширина листа, дюймы
FIGURE_HEIGHT = 5.4     # Высота листа, дюймы
FIGURE_DPI = 300        # Разрешение при записи файла

# Поле данных в долях листа, четвёркой левый край, нижний край, ширина, высота.
# Левый край у всех наборов общий: по нему выравнивается легенда
AXES_RECT = (0.085, 0.135, 0.885, 0.710)        # Обычная картинка
AXES_RECT_WIDE = (0.085, 0.155, 0.885, 0.735)   # Двухстрочные подписи засечек
AXES_RECT_LOW = (0.085, 0.135, 0.885, 0.680)    # Две строки легенды
AXES_RECT_TWIN = (0.085, 0.135, 0.845, 0.710)   # Запас на подпись второй оси
AXES_RECT_PLAN = (0.085, 0.145, 0.885, 0.705)   # План маршрута, одна строка легенды

# Строки легенды в долях высоты листа
LEGEND_Y = 0.965
LEGEND_Y2 = 0.905

# === ЦВЕТ ===
BG = "#ffffff"          # Фон листа, он же фон поля данных
GRID = "#e4e8ee"        # Сетка, читается как фон
INK = "#151b26"         # Основной текст и опорные линии
SUB = "#47526a"         # Вторичный текст, подписи осей и засечек
MUTED = "#8791a3"       # Образцы начертаний в легенде
REFERENCE_COLOR = INK   # Эталон и опорные линии

# Цвет закреплён за связкой. Оттенки затемнены под белый фон: исходные экранные
# цвета на светлом листе выцветают и теряют контраст
PALETTE: tuple[str, ...] = (
    "#1d7fd6", "#e07a21", "#12a074", "#dc3a58", "#7d51d8",
    "#0f9bb5", "#c2477f", "#5b6577",
)

# Начертание закреплено за геометрией: сплошная у первой, пунктир у второй
LINE_STYLES: tuple[str, ...] = ("-", "--", ":")

# === ЛИНИИ ===
LINE_WIDTH = 1.9              # Кривая на обычной картинке
LINE_MARKER = 4.2             # Размер точки на линии, пункты
REFERENCE_WIDTH = 1.6         # Опорная линия
GRID_WIDTH = 0.9              # Линия сетки
TRAJECTORY_WIDTH = 1.3        # Кривая на плане: их много, они длинные и петлистые
TRAJECTORY_REFERENCE = 2.6    # Эталон на плане, находится взглядом без легенды

# Начертание опорных линий: эталонного значения и выбранной величины. Штрих
# длиннее и реже, чем у пунктира геометрии, а цвет тёмный вместо серого. Иначе
# образец опорной линии в легенде не отличить от образца геометрии
REFERENCE_DASH = (0.0, (7.0, 3.0))

# === СТОЛБЦЫ ===
BAR_WIDTH = 0.62              # Доля отрезка между засечками
BAR_PAIR_WIDTH = 0.34         # То же, когда столбца два на засечку
BAR_RADIUS_PT = 4.0           # Радиус скругления верхних углов, пункты
BAR_GAP = 1.06                # Просвет между столбцами группы, доли ширины

# Осветление цвета для второго слоя. Слои одной связки различаются
# насыщенностью: цвет остаётся за связкой, а не уходит на роль слоя
TINT = 0.72
TINT_PAIR = 0.55              # У столбцов рядом разница должна быть заметнее

# === ЗАЛИВКИ ===
# Полоса непригодных значений и заливка гистограмм читаются как фон: контур у
# них насыщенный, а заливка бледная, иначе два наложенных распределения дают
# третий цвет, которого в данных нет
SHADE_ALPHA = 0.10
FILL_ALPHA = 0.22

# === ТОЧКИ НА ДИАГРАММАХ РАССЕЯНИЯ ===
# Размер задан площадью, а не поперечником, так устроен matplotlib. Значение
# рассчитано на облако из сотен наблюдений, где точки перекрываются
CLOUD_MARKER = 16
CLOUD_ALPHA = 0.65

# === ГИСТОГРАММЫ ===
HIST_BINS = 30                # Число столбцов распределения

# === КЕГЛЬ ===
SIZE_VALUE = 15               # Число над столбцом
SIZE_CAPTION = 9              # Пояснение под числом и подписи на поле
SIZE_LABEL = 11               # Подписи осей
SIZE_TICK = 10                # Подписи засечек
SIZE_LEGEND = 10              # Подписи легенды
SIZE_LEGEND_DENSE = 9.5       # Когда в строке легенды пять связок и больше

# Длина подписи засечки, после которой она переносится на вторую строку
WRAP_LIMIT = 14

# === ЕДИНИЦЫ ИЗМЕРЕНИЯ ===
POINTS_PER_INCH = 72          # Перевод пунктов в дюймы
MIN_SPAN = 1e-9               # Наименьший охват данных, страхует деление на ноль


def _apply() -> None:
    """
    Задаёт общие для всех картинок настройки matplotlib.

    Вызывается при импорте модуля: настройки должны быть выставлены до
    построения первой картинки, а построение всегда идёт через этот модуль.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "axes.labelcolor": SUB,
        "axes.labelsize": SIZE_LABEL,
        "text.color": INK,
        "xtick.color": SUB,
        "ytick.color": SUB,
        "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "grid.color": GRID,
        "grid.linewidth": GRID_WIDTH,
        "legend.fontsize": SIZE_LEGEND,
    })


_apply()


# ────────────────────────────────────────────────────────────────────────────
# Палитра
# ────────────────────────────────────────────────────────────────────────────

def color_from(key: str, keys: Sequence[str]) -> str:
    """
    Возвращает цвет, закреплённый за элементом произвольного списка.

    Цвет определяется положением в списке, а не самим ключом, поэтому один и
    тот же набор всегда раскрашивается одинаково. Неизвестный ключ получает
    цвет, следующий за концом списка.

    Args:
        key: ключ, для которого нужен цвет.
        keys: список ключей, задающий порядок цветов.

    Returns:
        Цвет в шестнадцатеричной записи.
    """
    keys = list(keys)
    index = keys.index(key) if key in keys else len(keys)
    return PALETTE[index % len(PALETTE)]


def color_for(key: str) -> str:
    """
    Возвращает цвет, закреплённый за связкой.

    Args:
        key: ключ связки из config.FRONTENDS.

    Returns:
        Цвет в шестнадцатеричной записи.
    """
    return color_from(key, [item["key"] for item in config.FRONTENDS])


def color_for_geometry(key: str) -> str:
    """
    Возвращает цвет, закреплённый за геометрией.

    Нужен на картинках, где связок нет и цвет свободен.

    Args:
        key: ключ геометрии из config.GEOMETRIES.

    Returns:
        Цвет в шестнадцатеричной записи.
    """
    return color_from(key, [item["key"] for item in config.GEOMETRIES])


def style_from(key: str, keys: Sequence[str]) -> str:
    """
    Возвращает начертание линии, закреплённое за положением ключа в списке.

    Args:
        key: ключ, для которого нужно начертание.
        keys: список ключей, задающий порядок начертаний.

    Returns:
        Обозначение начертания для matplotlib.
    """
    keys = list(keys)
    index = keys.index(key) if key in keys else len(keys)
    return LINE_STYLES[index % len(LINE_STYLES)]


def style_for(key: str) -> str:
    """
    Возвращает начертание линии, закреплённое за геометрией.

    Args:
        key: ключ геометрии из config.GEOMETRIES.

    Returns:
        Обозначение начертания для matplotlib.
    """
    return style_from(key, [item["key"] for item in config.GEOMETRIES])


def label_for(key: str) -> str:
    """
    Возвращает читаемое название связки по её ключу.

    Args:
        key: ключ связки.

    Returns:
        Название из конфига либо сам ключ, если связка неизвестна.
    """
    for item in config.FRONTENDS:
        if item["key"] == key:
            return item["label"]
    return key


def tint(color: Any, amount: float = TINT) -> tuple[float, float, float]:
    """
    Осветляет цвет подмешиванием белого.

    Args:
        color: исходный цвет в любой записи, понятной matplotlib.
        amount: доля белого, от нуля до единицы.

    Returns:
        Цвет в виде тройки долей красного, зелёного и синего.
    """
    red, green, blue = to_rgb(color)
    return (red + (1 - red) * amount,
            green + (1 - green) * amount,
            blue + (1 - blue) * amount)


# ────────────────────────────────────────────────────────────────────────────
# Лист и поле данных
# ────────────────────────────────────────────────────────────────────────────

def figure(rect: Sequence[float] = AXES_RECT) -> tuple[Any, Any]:
    """
    Создаёт лист с единственным полем данных.

    Размер листа не настраивается, он один на весь проект. Поле ставится долями
    листа, а не автоподбором: одинаковые поля у всех картинок важнее плотной
    упаковки каждой отдельно.

    Args:
        rect: поле данных в долях листа.

    Returns:
        Пару из фигуры и осей.
    """
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
    return fig, fig.add_axes(list(rect))


def wrap(labels: Sequence[str], limit: int = WRAP_LIMIT) -> list[str]:
    """
    Переносит длинные подписи засечек на вторую строку по знаку плюс.

    Переносится последний плюс, чтобы строки вышли соразмерными. Наклон
    подписей не используется: наклонённый текст читается медленнее прямого.

    Args:
        labels: подписи засечек.
        limit: длина, после которой подпись переносится.

    Returns:
        Подписи, в которых длинные разбиты на две строки.
    """
    result: list[str] = []
    for label in labels:
        if len(label) > limit and " + " in label:
            head, _, tail = label.rpartition(" + ")
            result.append(f"{head} +\n{tail}")
        else:
            result.append(label)
    return result


def round_ticks(values: Sequence[float]) -> list[float]:
    """
    Подбирает круглые засечки логарифмической оси, покрывающие набор значений.

    Из каждой декады берутся деления 1, 2, 5: они делят её примерно поровну в
    логарифмическом масштабе и остаются круглыми числами, тогда как matplotlib
    по умолчанию размечает ось делениями вида 3x10^2. Оставляются только
    деления, попадающие в занятый значениями промежуток, с запасом по краям.

    Args:
        values: значения, по которым размечается ось. Нули, None и NaN
            пропускаются.

    Returns:
        Возрастающий список положений засечек. Пустой, если пригодных значений
        нет.
    """
    usable = [value for value in values if value and value > 0]
    if not usable:
        return []

    steps = [1.0, 2.0, 5.0]
    lowest = int(np.floor(np.log10(min(usable))))
    highest = int(np.ceil(np.log10(max(usable))))
    grid = [step * 10.0 ** power
            for power in range(lowest, highest + 1) for step in steps]

    inside = [value for value in grid
              if min(usable) / 1.5 <= value <= max(usable) * 1.5]
    return inside or grid


def decades(low: float, high: float) -> list[float]:
    """
    Возвращает степени десяти, покрывающие промежуток.

    Нужны там, где логарифмическая ось подписана редко: на картинке со
    столбцами значения подписаны над столбцами, и полная разметка делениями
    1, 2, 5 только загущает поле.

    Args:
        low: нижняя граница промежутка, больше нуля.
        high: верхняя граница промежутка.

    Returns:
        Возрастающий список засечек. Пустой, если границы непригодны.
    """
    if low <= 0 or high <= 0 or high < low:
        return []
    lowest = int(np.floor(np.log10(low)))
    highest = int(np.floor(np.log10(high)))
    return [10.0 ** power for power in range(lowest, highest + 1)]


def equal_limits(rect: Sequence[float], x_min: float, x_max: float,
                 y_min: float, y_max: float,
                 pad: float = 0.04) -> tuple[tuple[float, float],
                                             tuple[float, float]]:
    """
    Считает границы осей, при которых масштаб по обеим осям одинаков.

    Нужны там, где картинка это план местности: сжатие одной оси исказило бы
    форму пути. Штатное выравнивание matplotlib раздвигает границы под уже
    попавшие в поле данные, поэтому у двух картинок с разным охватом выходит
    разный масштаб. Здесь границы считаются от размеров поля, и при общих
    данных масштаб у всех картинок набора совпадает.

    Args:
        rect: поле данных в долях листа.
        x_min: наименьшее значение по горизонтали.
        x_max: наибольшее значение по горизонтали.
        y_min: наименьшее значение по вертикали.
        y_max: наибольшее значение по вертикали.
        pad: запас вокруг данных в долях их охвата.

    Returns:
        Пару из границ по горизонтали и границ по вертикали.
    """
    width = rect[2] * FIGURE_WIDTH
    height = rect[3] * FIGURE_HEIGHT

    span_x = max(x_max - x_min, MIN_SPAN) * (1.0 + 2.0 * pad)
    span_y = max(y_max - y_min, MIN_SPAN) * (1.0 + 2.0 * pad)

    # Единиц данных на дюйм берётся по той оси, которой теснее: иначе часть
    # данных вышла бы за поле
    scale = max(span_x / width, span_y / height)
    span_x, span_y = scale * width, scale * height

    middle_x = (x_min + x_max) / 2.0
    middle_y = (y_min + y_max) / 2.0
    return ((middle_x - span_x / 2.0, middle_x + span_x / 2.0),
            (middle_y - span_y / 2.0, middle_y + span_y / 2.0))


def bare(axis: Any) -> None:
    """
    Снимает рамку и засечки с поля данных.

    Границу поля обозначают подписи осей и сетка. Нужна и для второй оси,
    созданной через twinx: она приходит с собственной рамкой, и без этого поле
    оказалось бы обведено наполовину.

    Args:
        axis: оси matplotlib.
    """
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0, pad=7)


def finish(axis: Any, xlabel: str = "", ylabel: str = "",
           grid_axis: str = "y", log_x: bool = False, log_y: bool = False,
           equal: bool = False, xticks: Sequence[Any] | None = None,
           xticklabels: Sequence[str] | None = None,
           yticks: Sequence[Any] | None = None,
           yticklabels: Sequence[str] | None = None) -> None:
    """
    Приводит поле к общему для всех картинок виду.

    Вызывается последней при построении графика: часть настроек, в частности
    засечки, сбрасывается при смене масштаба оси, поэтому порядок здесь важен.

    Args:
        axis: оси matplotlib.
        xlabel: подпись горизонтальной оси.
        ylabel: подпись вертикальной оси.
        grid_axis: по каким осям вести сетку: "y", "x", "both" или "none". По
            умолчанию только горизонтальные линии: на столбцах и рядах
            вертикальные ничего не добавляют.
        log_x: логарифмическая ли горизонтальная ось.
        log_y: логарифмическая ли вертикальная ось.
        equal: одинаковый ли масштаб по осям, нужен для траекторий.
        xticks: положения засечек по горизонтали. Нужны там, где значения
            заданы редкой сеткой и автоматические засечки её не повторяют.
        xticklabels: подписи засечек. Без xticks не применяются, иначе подписи
            разъедутся с делениями при смене масштаба.
        yticks: положения засечек по вертикали. Нужны на логарифмической оси,
            где по умолчанию появляются деления вида 3x10^2.
        yticklabels: подписи засечек по вертикали. Без yticks не применяются.
    """
    # Масштаб меняется до засечек: переход к логарифму ставит свои деления и
    # затирает выставленные вручную
    if log_x:
        axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")
    if equal:
        axis.set_aspect("equal", adjustable="datalim")

    if xticks is not None:
        axis.set_xticks(list(xticks))
        if xticklabels is not None:
            axis.set_xticklabels(list(xticklabels))
        # Дополнительные деления между заданными сбивают с толку: они мельче
        # подписанных и на логарифмической оси стоят неравномерно
        axis.minorticks_off()

    if yticks is not None:
        axis.set_yticks(list(yticks))
        # Подписи задаются явно: на логарифмической оси matplotlib иначе
        # оставляет собственное оформление вида 2x10^1
        axis.set_yticklabels(list(yticklabels) if yticklabels is not None
                             else [f"{value:g}" for value in yticks])
        axis.minorticks_off()

    if xlabel:
        axis.set_xlabel(xlabel, labelpad=9)
    if ylabel:
        axis.set_ylabel(ylabel, labelpad=9)

    bare(axis)

    if grid_axis != "none":
        axis.grid(True, axis=grid_axis, zorder=0)
        # Сетка уходит под данные: поверх кривых она их перечёркивает
        axis.set_axisbelow(True)


# ────────────────────────────────────────────────────────────────────────────
# Легенда
# ────────────────────────────────────────────────────────────────────────────

def dot(color: Any, label: str) -> Line2D:
    """
    Собирает образец легенды в виде точки.

    Точкой обозначается то, за чем закреплён цвет: связка, детектор или
    геометрия там, где начертание занято другой величиной.

    Args:
        color: цвет образца.
        label: подпись.

    Returns:
        Образец для легенды.
    """
    return Line2D([], [], marker="o", linestyle="none", markersize=7,
                  color=color, label=label)


def solid(color: Any, label: str, width: float = LINE_WIDTH) -> Line2D:
    """
    Собирает образец легенды в виде сплошного отрезка заданного цвета.

    Нужен для эталона: он отличается от оценок не цветом, а толщиной, и точкой
    это не передать.

    Args:
        color: цвет образца.
        label: подпись.
        width: толщина линии.

    Returns:
        Образец для легенды.
    """
    return Line2D([], [], color=color, linewidth=width, label=label)


def reference(label: str) -> Line2D:
    """
    Собирает образец легенды для опорной линии.

    Повторяет линию на поле: тот же цвет, та же толщина, то же начертание.

    Args:
        label: подпись.

    Returns:
        Образец для легенды.
    """
    return Line2D([], [], color=REFERENCE_COLOR, linestyle=REFERENCE_DASH,
                  linewidth=REFERENCE_WIDTH, label=label)


def dash(style: str, label: str) -> Line2D:
    """
    Собирает образец легенды в виде отрезка заданного начертания.

    Цвет у образца нейтральный: начертание описывается отдельной записью именно
    потому, что цвет занят другой величиной.

    Args:
        style: обозначение начертания для matplotlib.
        label: подпись.

    Returns:
        Образец для легенды.
    """
    return Line2D([], [], color=MUTED, linestyle=style, linewidth=1.8,
                  label=label)


def legend_row(fig: Any, handles: Sequence[Any], y: float = LEGEND_Y,
               size: float = SIZE_LEGEND, spacing: float = 1.5) -> None:
    """
    Ставит легенду строкой над полем данных.

    Строка выравнивается по левому краю поля данных, общему у всех наборов
    AXES_RECT. Пустой список образцов ничего не печатает.

    Args:
        fig: фигура matplotlib.
        handles: образцы легенды.
        y: положение строки в долях высоты листа.
        size: кегль подписей. Уменьшается, когда записей много и строка
            перестаёт помещаться по ширине.
        spacing: просвет между записями в долях кегля.
    """
    if not handles:
        return
    fig.legend(handles=list(handles), ncols=len(handles), frameon=False,
               loc="upper left", bbox_to_anchor=(AXES_RECT[0], y),
               handletextpad=0.45, columnspacing=spacing, borderaxespad=0.0,
               labelcolor=SUB, fontsize=size)


# ────────────────────────────────────────────────────────────────────────────
# Столбцы
# ────────────────────────────────────────────────────────────────────────────

def fraction(axis: Any, x: float, y: float,
             log_y: bool = False) -> tuple[float, float]:
    """
    Переводит точку данных в доли поля.

    Нужен там, где рисуют не по данным, а по геометрии поля: скруглённые
    столбцы и подписи с постоянным отступом в пунктах. Границы осей должны быть
    выставлены до вызова.

    Args:
        axis: оси matplotlib с уже выставленными границами.
        x: положение по горизонтали в единицах данных.
        y: положение по вертикали в единицах данных.
        log_y: логарифмическая ли вертикальная ось.

    Returns:
        Пару долей поля по горизонтали и по вертикали.
    """
    left, right = axis.get_xlim()
    bottom, top = axis.get_ylim()
    part_x = (x - left) / (right - left)
    if log_y:
        part_y = ((np.log10(y) - np.log10(bottom))
                  / (np.log10(top) - np.log10(bottom)))
    else:
        part_y = (y - bottom) / (top - bottom)
    return part_x, part_y


def rounded_bar(axis: Any, centre: float, value: float, color: Any,
                width: float = BAR_WIDTH, log_y: bool = False,
                zorder: int = 3) -> None:
    """
    Рисует столбец со скруглёнными верхними углами.

    Путь строится в долях поля, а не в единицах данных: на логарифмической оси
    скругление, заданное в данных, растянулось бы вместе с масштабом. По той же
    причине радиус задан в пунктах и переводится в доли по горизонтали и
    вертикали по-разному: поле не квадратное, и общий радиус дал бы овал.

    Границы поля должны быть выставлены до вызова, иначе доли посчитаются по
    ещё не окончательному масштабу. Нечисловое и нулевое значение не рисуется.

    Args:
        axis: оси matplotlib.
        centre: положение середины столбца в единицах данных.
        value: высота столбца в единицах данных.
        color: цвет заливки.
        width: ширина столбца в единицах данных.
        log_y: логарифмическая ли вертикальная ось.
        zorder: порядок отрисовки, нужен при наложении слоёв.
    """
    if value is None or not np.isfinite(value):
        return

    left, _ = fraction(axis, centre - width / 2, value, log_y)
    right, height = fraction(axis, centre + width / 2, value, log_y)
    if height <= 0:
        return

    position = axis.get_position()
    radius_x = min(
        BAR_RADIUS_PT / (position.width * FIGURE_WIDTH * POINTS_PER_INCH),
        (right - left) / 2)
    radius_y = min(
        BAR_RADIUS_PT / (position.height * FIGURE_HEIGHT * POINTS_PER_INCH),
        height / 2)

    verts = [(left, 0), (left, height - radius_y),
             (left, height), (left + radius_x, height),
             (right - radius_x, height),
             (right, height), (right, height - radius_y),
             (right, 0), (left, 0)]
    codes = [Outline.MOVETO, Outline.LINETO,
             Outline.CURVE3, Outline.CURVE3,
             Outline.LINETO,
             Outline.CURVE3, Outline.CURVE3,
             Outline.LINETO, Outline.CLOSEPOLY]

    axis.add_patch(PathPatch(Outline(verts, codes), transform=axis.transAxes,
                             facecolor=color, edgecolor="none", zorder=zorder))


def bar_positions(count: int, groups: int = 1,
                  index: int = 0) -> tuple[np.ndarray, float]:
    """
    Считает положения и ширину столбцов для одной группы.

    Столбцы разных групп ставятся вокруг общей засечки. Группа занимает
    постоянную долю отрезка между засечками независимо от числа групп, поэтому
    просвет между наборами не зависит от того, сколько столбцов в наборе.

    Args:
        count: число засечек, то есть столбцов в одной группе.
        groups: сколько групп ставится вокруг каждой засечки.
        index: номер текущей группы, считая с нуля.

    Returns:
        Пару из массива положений и ширины столбца.
    """
    width = BAR_PAIR_WIDTH if groups > 1 else BAR_WIDTH
    span = width * groups * BAR_GAP
    shift = index * width * BAR_GAP - span / 2 + width / 2
    return np.arange(count) + shift, width


# ────────────────────────────────────────────────────────────────────────────
# Сохранение
# ────────────────────────────────────────────────────────────────────────────

def output_path(name: str) -> Path:
    """
    Собирает путь к файлу в каталоге результатов и создаёт сам каталог.

    Единственное место, где графики узнают, куда их кладут.

    Args:
        name: имя файла без расширения.

    Returns:
        Путь к файлу.
    """
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.PLOTS_DIR / f"{name}.png"


def save(fig: Any, name: str) -> Path:
    """
    Сохраняет фигуру в каталог результатов и закрывает её.

    Поля листа выставлены при создании, поэтому tight_layout здесь не
    вызывается: он подогнал бы поля под содержимое каждой картинки отдельно, и
    файлы вышли бы разной ширины.

    Закрытие обязательно: без него открытые фигуры накапливаются в памяти.

    Args:
        fig: фигура matplotlib.
        name: имя файла без расширения.

    Returns:
        Путь к сохранённому файлу.
    """
    path = output_path(name)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
