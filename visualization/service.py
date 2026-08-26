"""
Картинка подготовки к прогону.

Здесь одна фигура, которая строится до всякой одометрии: обоснование выбранного
шага прореживания кадров. Её смотрят один раз, вписывают полученное число в
config и больше к ней не возвращаются.

Калибровка своей фигуры не имеет: её результат это несколько констант конфига, и
показать их числами в таблице точнее, чем облаком точек.

Как и соседний модуль, этот только рисует. Ни отбор рабочего промежутка шагов,
ни пригодность отдельного значения здесь не считаются, всё приходит готовым.
"""

# Стандартные библиотеки
import sys
from pathlib import Path
from typing import Sequence

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from visualization import analysis as figures
from visualization import core


# === ПОДПИСИ ===
# Подпись выбранного значения. Стоит в легенде, а не на поле: образец там
# повторяет саму линию, а на поле подпись легла бы на кривые, которых в этом
# месте больше всего
CHOSEN_LABEL = "выбрано"

# === ЗАПАС НАД ДАННЫМИ ===
HEADROOM = 1.12   # Множитель верхней границы оси над самой высокой точкой


# ────────────────────────────────────────────────────────────────────────────
# Шаг прореживания
# ────────────────────────────────────────────────────────────────────────────

def frame_step(series: dict[str, dict[str, dict[str, Sequence[float]]]],
               dataset: str, span: int, chosen: int,
               usable: Sequence[int]) -> Path | None:
    """
    Рисует ошибку направления по шагам с отметкой рабочего промежутка.

    Отвечает на вопрос, почему выбрано именно это значение, и показывает то,
    чего не показывает таблица: форму кривых и границы промежутка сразу.

    Показаны обе геометрии: цвет за связкой, начертание за геометрией. Одной
    геометрии мало, потому что затенение слева объясняется именно второй. При
    короткой базе эпиполярная геометрия вырождается, и пунктирные кривые уходят
    заметно выше сплошных: результат там формально получается, но согласных
    точек так мало, что отбор признаёт его непригодным.

    Кривые не обрываются на границе пригодности: рисуется всё, что посчиталось,
    тогда как в таблицы попадает только прошедшее отбор. Иначе с картинки
    пропало бы ровно то, что объясняет затенение.

    Ось шага логарифмическая: сетка значений тоже логарифмическая, и при
    равномерной оси левая половина поля оказалась бы пустой. Засечки ставятся
    по самой сетке, иначе на оси появились бы деления, которых в исследовании
    не было.

    Args:
        series: по каждой геометрии ряды показателей каждой связки, нужны поля
            steps и direction_error. Пропуски задаются NaN.
        dataset: имя датасета, на котором велось исследование. В картинке не
            показывается: рисунок подписывается снизу.
        span: длина участка в кадрах, там же.
        chosen: выбранный шаг.
        usable: шаги, пригодные на показанном участке при обеих геометриях.
            Затенение считается по нему, а не по всему исследованию, иначе шаг
            оказался бы затенён из-за участка, которого на картинке нет.

    Returns:
        Путь к сохранённому файлу либо None, если рисовать нечего.
    """
    if not series:
        return None

    fig, axis = core.figure(core.AXES_RECT_LOW)
    ticks = list(config.FRAME_STEP_GRID)

    # Затенение непригодных шагов. Границы ставятся посередине между крайним
    # пригодным шагом и соседним по сетке, иначе край полосы накрыл бы точку,
    # которая как раз пригодна
    if usable:
        left, right = min(usable), max(usable)
        if left > ticks[0]:
            axis.axvspan(ticks[0], midpoint(ticks, left, -1), color=core.SUB,
                         alpha=core.SHADE_ALPHA, linewidth=0, zorder=1)
        if right < ticks[-1]:
            axis.axvspan(midpoint(ticks, right, 1), ticks[-1], color=core.SUB,
                         alpha=core.SHADE_ALPHA, linewidth=0, zorder=1)

    highest = 0.0
    for item in config.GEOMETRIES:
        rows = series.get(item["key"], {})
        for key, values in rows.items():
            axis.plot(values["steps"], values["direction_error"],
                      color=core.color_for(key),
                      linestyle=core.style_for(item["key"]),
                      linewidth=core.LINE_WIDTH, marker="o",
                      markersize=core.LINE_MARKER, markeredgewidth=0, zorder=3)
            # Пропуски отсеиваются сравнением с самим собой: NaN ему не равен
            highest = max([highest] + [value for value
                                       in values["direction_error"]
                                       if value == value])

    axis.axvline(chosen, color=core.REFERENCE_COLOR,
                 linewidth=core.REFERENCE_WIDTH,
                 linestyle=core.REFERENCE_DASH, zorder=5)

    core.finish(axis, "шаг прореживания, кадров", "ошибка направления, град",
                grid_axis="y", log_x=True, xticks=ticks,
                xticklabels=[str(value) for value in ticks])
    axis.set_xlim(ticks[0], ticks[-1])
    axis.set_ylim(0, highest * HEADROOM if highest else 1.0)

    figures.frontend_legend(fig)
    figures.geometry_legend(
        fig, extra=[core.reference(f"{CHOSEN_LABEL}: {chosen}")])

    return core.save(fig, "frame_step_choice")


def midpoint(ticks: Sequence[int], value: int, direction: int) -> float:
    """
    Находит середину между узлом сетки и соседним с ним.

    Сетка шагов логарифмическая, поэтому середина берётся геометрическая: на
    логарифмической оси она и окажется посередине между засечками.

    Args:
        ticks: узлы сетки по возрастанию.
        value: узел, от которого отсчитывается.
        direction: минус один для соседа слева, плюс один для соседа справа.

    Returns:
        Координату границы затенения. Сам узел, если соседа с этой стороны нет.
    """
    index = ticks.index(value)
    neighbour = index + direction
    if not 0 <= neighbour < len(ticks):
        return float(value)
    return float((value * ticks[neighbour]) ** 0.5)
