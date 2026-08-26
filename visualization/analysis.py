"""
Картинки прогона одометрии.

Здесь шесть фигур, которые строятся при каждом запуске main и идут в статью:
траектории и по одной картинке на каждый пункт задания. Каждая функция строит
одну фигуру целиком, сохраняет её и возвращает путь.

Модуль только рисует: он не считает метрики, не читает таблицы прогона и не
обращается к модулям анализа. Всё, что нужно нарисовать, приходит аргументами
готовыми числами. Из общего с проектом здесь config, откуда берутся порядок
связок и список геометрий, и core, откуда берётся всё оформление.

Заголовков внутри картинок нет: рисунок в статье подписывается снизу. Имя
датасета приходит аргументом, но идёт только в имя файла.

Величины оформления не выбираются по месту. Если для новой картинки нужен
размер точки или насыщенность, которых нет в core, их добавляют в core, а не
вписывают сюда числом.
"""

# Стандартные библиотеки
import sys
from pathlib import Path
from typing import Any, Sequence

# Сторонние библиотеки
import numpy as np

# Локальные импорты.
# При прямом запуске в путях поиска модулей оказывается каталог скрипта,
# а не корень проекта, поэтому корень добавляется явно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from visualization import core


# === ЕДИНИЦЫ ПЛАНА МАРШРУТА ===
# Охват, после которого координаты переводятся в километры. Пятизначные числа на
# засечках занимают половину поля и читаются медленнее двузначных, а точность
# плана от единиц не зависит
KILOMETRE_LIMIT = 5000.0
METRES_PER_KILOMETRE = 1000.0

# === ЗАПАС НАД ДАННЫМИ ===
HEADROOM = 1.12          # Множитель верхней границы оси над самым высоким столбцом
TIME_HEADROOM = 2.4      # То же для логарифмической оси времени: там нужны подписи

# === ГРАНИЦЫ ПОЛЯ СО СТОЛБЦАМИ ===
# Отступ от края поля до крайнего столбца в единицах засечек
BAR_MARGIN_LEFT = 0.62
BAR_MARGIN_RIGHT = 0.38

# === ПОДПИСИ НАД СТОЛБЦАМИ ===
VALUE_OFFSET_PT = 8.0    # Отступ подписи времени от вершины столбца, пункты
RATIO_OFFSET_PT = 20.0   # Отступ строки с отношением, пункты

# === РАЗМЕТКА ГИСТОГРАММЫ ОШИБКИ НАПРАВЛЕНИЯ ===
# Ошибка направления лежит в пределах полукруга в обе стороны
ANGLE_LIMIT = 180.0
ANGLE_TICKS = (-180, -120, -60, 0, 60, 120, 180)


# ────────────────────────────────────────────────────────────────────────────
# Общие приёмы
# ────────────────────────────────────────────────────────────────────────────

def compact(value: float) -> str:
    """
    Записывает величину так, чтобы близкие значения не слились.

    Ниже сотни оставляется десятая доля: без неё 18.1 и 18.4 печатались бы
    одинаково. От сотни доля уже не различима на глаз и только удлиняет
    подпись.

    Args:
        value: величина.

    Returns:
        Строку без единиц измерения.
    """
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}"


def ratio_text(records: list[dict[str, Any]], frontend: str,
               modes: Sequence[str], own: Sequence[tuple[str, float]]) -> str:
    """
    Записывает отношение времён двух режимов либо отказ от него.

    Отношение печатается только тогда, когда оно больше разброса самих замеров.
    Если повторы одной и той же связки расходятся сильнее, чем режимы между
    собой, отношение описывает шум измерения, а не свойство алгоритма.

    Args:
        records: результат run_all, нужны поля frontend, mode и spread.
        frontend: ключ связки.
        modes: названия режимов, первым основной.
        own: пары из названия режима и времени для этой связки.

    Returns:
        Готовую часть подписи. Пустая строка, если время основного режима
        нулевое.
    """
    times = dict(own)
    leading, trailing = times[modes[0]], times[modes[1]]
    if not leading:
        return ""

    spreads = [record.get("spread") for record in records
               if record["frontend"] == frontend and record["mode"] in modes]
    noise = max([value for value in spreads
                 if value and np.isfinite(value)] or [1.0])

    ratio = max(leading, trailing) / min(leading, trailing)
    if ratio <= noise:
        # Отношение прижимается к единице, а не опускается: пустое место уже
        # занято другим смыслом, там второй режим вовсе не замерен
        return "×1.0"
    return f"×{trailing / leading:.1f}"


def frontend_legend(fig: Any) -> None:
    """
    Ставит первой строкой легенды связки в закреплённых за ними цветах.

    Кегль уменьшен: связок пять, и при обычном названия перестают помещаться в
    строку по ширине листа.

    Args:
        fig: фигура matplotlib.
    """
    core.legend_row(fig, [core.dot(core.color_for(item["key"]), item["label"])
                          for item in config.FRONTENDS],
                    size=core.SIZE_LEGEND_DENSE, spacing=1.1)


def geometry_legend(fig: Any, y: float = core.LEGEND_Y2,
                    extra: Sequence[Any] = ()) -> None:
    """
    Ставит строкой легенды геометрии в закреплённых за ними начертаниях.

    Args:
        fig: фигура matplotlib.
        y: положение строки в долях высоты листа.
        extra: записи, которые ставятся перед геометриями. Сюда попадает
            опорная линия: она относится ко всему полю, а не к одной кривой.
    """
    core.legend_row(fig, list(extra)
                    + [core.dash(core.style_for(item["key"]), item["label"])
                       for item in config.GEOMETRIES],
                    y=y, size=core.SIZE_LEGEND_DENSE, spacing=1.1)


def draw_trajectory(axis: Any, reference: np.ndarray,
                    estimated: dict[str, np.ndarray],
                    divider: float = 1.0) -> None:
    """
    Рисует эталонную траекторию и оценки всех связок в плане.

    Эталон выделен тёмной и более толстой линией и лежит поверх оценок, чтобы
    отличать его без обращения к легенде: оценок пять, они петлистые и
    закрывали бы его на большей части пути.

    Args:
        axis: оси matplotlib.
        reference: эталонная траектория, массив (N, 3).
        estimated: ключ связки и её траектория, уже совмещённая с эталоном.
        divider: делитель координат, общий на все картинки набора.
    """
    for key, path in estimated.items():
        if path is None or not len(path):
            continue
        axis.plot(path[:, 0] / divider, path[:, 1] / divider,
                  color=core.color_for(key),
                  linewidth=core.TRAJECTORY_WIDTH, zorder=3)

    if len(reference):
        axis.plot(reference[:, 0] / divider, reference[:, 1] / divider,
                  color=core.REFERENCE_COLOR,
                  linewidth=core.TRAJECTORY_REFERENCE, zorder=5)


def trajectory_extent(reference: np.ndarray,
                      panels: Sequence[dict[str, Any]]) -> tuple[float, ...]:
    """
    Считает делитель координат и общий охват по всем панелям набора.

    Охват берётся сразу по всем геометриям, а не по каждой отдельно: панели
    идут в статье подряд и сравниваются глазами, поэтому масштаб у них должен
    быть один.

    Args:
        reference: эталонная траектория, массив (N, 3).
        panels: панели с полем whole, где лежат траектории связок.

    Returns:
        Пятёрку из делителя и наименьших и наибольших значений по востоку и по
        северу, уже поделённых. При отсутствии данных делитель равен единице, а
        границы вырожденному квадрату.
    """
    paths = [path for panel in panels for path in panel["whole"].values()
             if path is not None and len(path)]
    if len(reference):
        paths.append(np.asarray(reference)[:, :2])

    if not paths:
        return 1.0, 0.0, 1.0, 0.0, 1.0

    east = np.concatenate([path[:, 0] for path in paths])
    north = np.concatenate([path[:, 1] for path in paths])

    reach = max(float(np.ptp(east)), float(np.ptp(north)))
    divider = METRES_PER_KILOMETRE if reach > KILOMETRE_LIMIT else 1.0

    return (divider, float(east.min()) / divider, float(east.max()) / divider,
            float(north.min()) / divider, float(north.max()) / divider)


# ────────────────────────────────────────────────────────────────────────────
# Траектории
# ────────────────────────────────────────────────────────────────────────────

def trajectories(reference: np.ndarray, panels: Sequence[dict[str, Any]],
                 name: str) -> list[Path]:
    """
    Рисует траектории всех связок, по файлу на геометрию.

    Траектории приходят уже совмещёнными с эталоном: подбор преобразования
    зависит от того, восстанавливает ли метод длину перемещения, а это знание о
    методах, которого у модуля рисования нет. Совмещение идёт по всей длине,
    поэтому начало кривой не обязано совпадать с началом маршрута.

    Геометрии разнесены по файлам: десять кривых поверх эталона на одном поле
    неразличимы, а сравнивать надо связки внутри геометрии, а не поперёк.

    Легенда занимает одну строку, в отличие от остальных картинок: эталон это
    единственная добавка к списку связок.

    Args:
        reference: эталонная траектория, массив (N, 3).
        panels: по записи на геометрию, каждая с полями key, label и whole.
            Последнее это ключ связки и уже совмещённая траектория.
        name: имя датасета, идёт в имя файла.

    Returns:
        Пути к сохранённым файлам, по одному на геометрию.
    """
    saved: list[Path] = []
    rect = core.AXES_RECT_PLAN

    divider, east_min, east_max, north_min, north_max = trajectory_extent(
        reference, panels)
    horizontal, vertical = core.equal_limits(rect, east_min, east_max,
                                             north_min, north_max)
    unit = "км" if divider > 1.0 else "м"

    for panel in panels:
        fig, axis = core.figure(rect)
        draw_trajectory(axis, reference, panel["whole"], divider)

        core.finish(axis, f"восток, {unit}", f"север, {unit}",
                    grid_axis="both")
        # Границы выставляются готовыми, а не подбираются по данным панели:
        # они общие на весь набор
        axis.set_xlim(horizontal)
        axis.set_ylim(vertical)

        # Эталон стоит первым: он опорная величина, а не ещё одна оценка
        core.legend_row(
            fig,
            [core.solid(core.REFERENCE_COLOR, "эталон",
                        width=core.TRAJECTORY_REFERENCE)]
            + [core.dot(core.color_for(item["key"]), item["label"])
               for item in config.FRONTENDS],
            size=core.SIZE_LEGEND_DENSE, spacing=1.1)

        saved.append(core.save(fig, f"trajectories_{panel['key']}_{name}"))

    return saved


# ────────────────────────────────────────────────────────────────────────────
# Устойчивость ключевых точек
# ────────────────────────────────────────────────────────────────────────────

def keypoints(records: list[dict[str, Any]], summary: list[dict[str, Any]],
              name: str, detector_labels: Sequence[str]) -> Path:
    """
    Рисует повторяемость детекторов против насыщенности кадра деталями.

    Отвечает сразу на обе части пункта, показывая и уровень повторяемости у
    каждого детектора, и её связь с детализацией кадра.

    Цвет здесь закреплён за детектором, а не за связкой: сопоставление ещё не
    выполнялось, и связок как таковых нет. Сила связи вынесена в легенду, чтобы
    не занимать поле подписью.

    Args:
        records: наблюдения по парам, нужны поля detector, rate и gradient.
        summary: сводка по детекторам, нужны поля detector, label и
            texture_link.
        name: имя датасета, идёт в имя файла.
        detector_labels: ключи детекторов в порядке, задающем цвета.

    Returns:
        Путь к сохранённому файлу.
    """
    fig, axis = core.figure()
    detectors = list(detector_labels)

    for item in summary:
        own = [record for record in records
               if record["detector"] == item["detector"]]
        axis.scatter([record["gradient"] for record in own],
                     [record["rate"] for record in own],
                     s=core.CLOUD_MARKER, alpha=core.CLOUD_ALPHA,
                     color=core.color_from(item["detector"], detectors),
                     linewidths=0, zorder=3)

    core.finish(axis, "средний градиент кадра", "повторяемость, %",
                grid_axis="both")
    # Границы по данным с небольшим запасом: круглые границы оставляли бы
    # пустой треть поля
    axis.margins(x=0.04, y=0.08)

    core.legend_row(fig, [
        core.dot(core.color_from(item["detector"], detectors),
                 f"{item['label']}, связь {item['texture_link']:+.2f}")
        for item in summary])

    return core.save(fig, f"keypoints_{name}")


# ────────────────────────────────────────────────────────────────────────────
# Качество сопоставления
# ────────────────────────────────────────────────────────────────────────────

def matching(records: list[dict[str, Any]],
             methods: dict[str, list[dict[str, Any]]],
             thresholds: dict[str, list[dict[str, Any]]], name: str) -> Path:
    """
    Рисует влияние порога отбраковки на точность и на долю согласных точек.

    Обе части пункта идут по двум осям одной панели, и видна форма зависимости,
    которой в таблицах нет.

    Доля вынесена на вторую ось, а не подогнана множителем под первую: величины
    измеряются в разном. Цвет закреплён за геометрией, начертание за величиной,
    поэтому легенда описывает четыре вещи вместо четырёх безымянных кривых.

    Сетка ведётся только по левой оси: две сетки по двум осям встают в разных
    местах и образуют на поле клетку, к данным отношения не имеющую.

    Args:
        records: результат collect, не используется. Аргумент есть ради общего
            вида функций рисования.
        methods: результат перебора методов, не используется по той же причине.
        thresholds: результат перебора порогов по каждой геометрии, ключом
            служит её подпись либо ключ.
        name: имя датасета, идёт в имя файла.

    Returns:
        Путь к сохранённому файлу.
    """
    fig, axis = core.figure(core.AXES_RECT_TWIN)
    twin = axis.twinx()
    measured: list[float] = []
    shown: list[dict[str, str]] = []

    for item in config.GEOMETRIES:
        # Перебор приходит разложенным по подписи геометрии, а не по её ключу:
        # так его собирает разбор сопоставления. Ключ проверяется следом, чтобы
        # картинка не зависела от того, как раскладку соберут в дальнейшем
        outcome = (thresholds.get(item["label"])
                   or thresholds.get(item["key"]) or [])
        usable = [row for row in outcome if row.get("ok")]
        if not usable:
            continue

        values = [float(row["variant"]) for row in usable]
        color = core.color_for_geometry(item["key"])
        axis.plot(values, [row["error"] for row in usable], color=color,
                  linewidth=core.LINE_WIDTH, marker="o",
                  markersize=core.LINE_MARKER, markeredgewidth=0, zorder=4)
        twin.plot(values, [row["ratio"] for row in usable], color=color,
                  linewidth=core.LINE_WIDTH, linestyle="--", marker="s",
                  markersize=core.LINE_MARKER, markeredgewidth=0, zorder=3)
        measured += values
        shown.append(item)

    twin.set_ylim(0, 105)
    twin.set_yticks([0, 25, 50, 75, 100])
    twin.set_ylabel("согласных точек, %", labelpad=9)
    core.bare(twin)

    # Засечки ставятся по самой сетке порогов, а не круглыми числами: перебор
    # шёл по заданным значениям, и засечка на месте, где замера не было, увела
    # бы излом кривой в промежуток между делениями
    ticks = sorted(set(measured))
    core.finish(axis, "порог невязки, px", "ошибка направления, град",
                grid_axis="y", log_x=True, xticks=ticks,
                xticklabels=[f"{value:g}" for value in ticks])
    axis.set_ylim(bottom=0)

    core.legend_row(fig, [core.dot(core.color_for_geometry(item["key"]),
                                   item["label"]) for item in shown])
    core.legend_row(fig, [core.dash("-", "ошибка направления"),
                          core.dash("--", "доля согласных точек")],
                    y=core.LEGEND_Y2, size=core.SIZE_LEGEND_DENSE)

    return core.save(fig, f"matching_{name}")


# ────────────────────────────────────────────────────────────────────────────
# Геометрический анализ
# ────────────────────────────────────────────────────────────────────────────

def geometry(records: list[dict[str, Any]], spread_link: dict[str, Any],
             signed: dict[str, np.ndarray], name: str) -> list[Path]:
    """
    Рисует распределение знаковой ошибки направления по геометриям.

    Это единственная величина раздела, которую числами не передать. Медиана
    говорит, насколько модель точна в типичном случае, но не говорит, как
    ошибка устроена: симметрична ли она относительно нуля, насколько широка и
    есть ли отказы.

    Распределения даны контуром с бледной заливкой, а не двумя полупрозрачными
    заливками: при наложении полупрозрачных получается третий цвет, которого в
    данных нет.

    Восстановление позы распадается на перемещение и поворот, поэтому картинок
    здесь две. Вторая отвечает за поворот и строится отдельно: ошибка поворота
    лежит в единицах градусов, тогда как ошибка направления доходит до ста
    восьмидесяти, и на общей оси она была бы неотличима от нуля.

    Args:
        records: результат collect, идёт во вторую картинку раздела.
        spread_link: результат correlate_spread, не используется. Аргумент есть
            ради общего вида функций рисования.
        signed: знаковые ошибки по каждой геометрии, собранные по всем связкам.
        name: имя датасета, идёт в имя файла.

    Returns:
        Пути к сохранённым файлам: распределение ошибки направления и ошибка
        поворота по конфигурациям.
    """
    fig, axis = core.figure()
    edges = np.linspace(-ANGLE_LIMIT, ANGLE_LIMIT, core.HIST_BINS + 1)
    shown: list[dict[str, str]] = []

    for item in config.GEOMETRIES:
        values = signed.get(item["key"])
        if values is None or not len(values):
            continue
        color = core.color_for_geometry(item["key"])
        axis.hist(values, bins=edges, color=color, alpha=core.FILL_ALPHA,
                  zorder=3)
        axis.hist(values, bins=edges, histtype="step", color=color,
                  linewidth=1.7, zorder=4)
        shown.append(item)

    # Ноль отмечается линией: симметричность облака относительно него и есть
    # ответ на вопрос, систематична ли ошибка
    axis.axvline(0.0, color=core.REFERENCE_COLOR,
                 linewidth=core.REFERENCE_WIDTH, zorder=5)

    core.finish(axis, "знаковая ошибка направления, град", "число пар",
                grid_axis="y", xticks=ANGLE_TICKS)
    axis.set_xlim(-ANGLE_LIMIT, ANGLE_LIMIT)

    core.legend_row(fig, [core.dot(core.color_for_geometry(item["key"]),
                                   item["label"]) for item in shown])

    return [core.save(fig, f"geometry_{name}"),
            rotation_error(records, name)]


def rotation_error(records: list[dict[str, Any]], name: str) -> Path:
    """
    Рисует ошибку восстановления поворота рядом с величиной самого поворота.

    Ошибка поворота сама по себе ни о чём не говорит: полтора градуса это мало
    или много в зависимости от того, на сколько камера успевает повернуться за
    шаг. Поэтому эталонный поворот проводится линией: столбцы выше линии
    означают, что ошибка превышает измеряемую величину, то есть поворот не
    восстанавливается.

    Цвет закреплён за связкой, геометрия различается насыщенностью.

    Args:
        records: результат collect, нужны поля frontend, geometry,
            rotation_error и gt_rotation.
        name: имя датасета, идёт в имя файла.

    Returns:
        Путь к сохранённому файлу.
    """
    fig, axis = core.figure(core.AXES_RECT_WIDE)
    labels = [item["label"] for item in config.FRONTENDS]

    values: dict[tuple[str, str], float] = {}
    for frontend in config.FRONTENDS:
        for item in config.GEOMETRIES:
            found = next((record for record in records
                          if record["frontend"] == frontend["key"]
                          and record["geometry"] == item["key"]), None)
            values[(frontend["key"], item["key"])] = (
                found["rotation_error"] if found else np.nan)

    measured = [value for value in values.values() if np.isfinite(value)]
    reference = (float(np.median([record["gt_rotation"] for record in records]))
                 if records else 0.0)

    # Границы выставляются до столбцов: скруглённые вершины строятся по
    # геометрии поля, и при последующей смене масштаба они бы поплыли
    axis.set_xlim(-BAR_MARGIN_LEFT, len(labels) - BAR_MARGIN_RIGHT)
    axis.set_ylim(0, max(measured + [reference]) * HEADROOM
                  if measured else 1.0)

    for index, item in enumerate(config.GEOMETRIES):
        positions, width = core.bar_positions(len(labels),
                                              len(config.GEOMETRIES), index)
        for position, frontend in zip(positions, config.FRONTENDS):
            color = core.color_for(frontend["key"])
            core.rounded_bar(axis, position,
                             values[(frontend["key"], item["key"])],
                             color if index == 0
                             else core.tint(color, core.TINT_PAIR),
                             width=width)

    axis.axhline(reference, color=core.REFERENCE_COLOR,
                 linewidth=core.REFERENCE_WIDTH,
                 linestyle=core.REFERENCE_DASH, zorder=5)

    core.finish(axis, ylabel="ошибка поворота, град", grid_axis="y",
                xticks=np.arange(len(labels)), xticklabels=core.wrap(labels))

    # Величина эталонного поворота уходит в легенду: линия проходит низко, все
    # столбцы её пересекают, и подпись у линии легла бы поверх столбца.
    # Образцы геометрий нейтральны по цвету: цветной кружок читался бы как
    # связка, за которой цвет закреплён
    core.legend_row(fig, [
        core.reference(f"поворот за шаг: {reference:.2f}°")] + [
        core.dot(core.SUB if index == 0
                 else core.tint(core.SUB, core.TINT_PAIR), item["label"])
        for index, item in enumerate(config.GEOMETRIES)])

    return core.save(fig, f"geometry_rotation_{name}")


# ────────────────────────────────────────────────────────────────────────────
# Накопление ошибки
# ────────────────────────────────────────────────────────────────────────────

def drift(records: list[dict[str, Any]], name: str) -> Path:
    """
    Рисует относительную ошибку в зависимости от длины отрезка маршрута.

    Прямой ответ на вопрос пункта о влиянии длины траектории. Ошибка
    нормирована на длину отрезка, поэтому по наклону кривой видно, как копится:
    падающая кривая означает, что ошибки шагов гасят друг друга, растущая или
    ровная означает, что складываются.

    Отрезки берутся по всему маршруту, поэтому величина не зависит от того, где
    именно оценка сбилась, в отличие от расхождения к концу пути.

    Обе оси логарифмические и размечены круглыми числами: по умолчанию
    matplotlib ставит здесь деления вида 3x10^2.

    Args:
        records: показатели по связкам и геометриям, нужны поля frontend,
            geometry и relative.
        name: имя датасета, идёт в имя файла.

    Returns:
        Путь к сохранённому файлу.
    """
    fig, axis = core.figure()
    lengths: list[float] = []
    errors: list[float] = []

    for record in records:
        relative = record.get("relative") or {}
        own = sorted(relative)
        if not own:
            continue

        medians = [relative[length]["median"] for length in own]
        axis.plot(own, medians, color=core.color_for(record["frontend"]),
                  linestyle=core.style_for(record["geometry"]),
                  linewidth=core.LINE_WIDTH, marker="o",
                  markersize=core.LINE_MARKER, markeredgewidth=0,
                  zorder=3, clip_on=False)
        lengths += list(own)
        errors += [value for value in medians if value and value > 0]

    ticks = sorted(set(lengths))
    core.finish(axis, "длина отрезка, м", "расхождение, % длины",
                grid_axis="both", log_x=True, log_y=True,
                xticks=ticks, xticklabels=[f"{value:g}" for value in ticks],
                yticks=core.round_ticks(errors))

    frontend_legend(fig)
    geometry_legend(fig)

    return core.save(fig, f"drift_{name}")


# ────────────────────────────────────────────────────────────────────────────
# Нагрузка на устройство
# ────────────────────────────────────────────────────────────────────────────

def performance(records: list[dict[str, Any]], name: str,
                modes: Sequence[str]) -> Path:
    """
    Рисует время обработки пары в двух режимах по числу потоков процессора.

    Режимы наложены, а не поставлены рядом. Столбец медленного режима идёт
    фоном, столбец быстрого поверх него, и видимая светлая верхушка это и есть
    выигрыш от распараллеливания. Цвет остаётся закреплён за связкой, а режим
    различается насыщенностью.

    Ось времени логарифмическая: разница между связками более чем
    десятикратная, и на равномерной оси самые быстрые слились бы у нуля.
    Столбцы на такой оси отсчитываются не от нуля, которого на ней нет, а от
    нижней засечки, поэтому их высота не пропорциональна значению. Сами
    значения подписаны над столбцами.

    Крупно печатается отношение времён, а не время: картинка отвечает на вопрос
    о влиянии числа потоков, и ответ на него это именно отношение.

    Args:
        records: результат run_all, нужны поля frontend, mode, ok, spread и
            total_ms.
        name: имя датасета, идёт в имя файла.
        modes: названия режимов в порядке вывода, первым основной.

    Returns:
        Путь к сохранённому файлу. При отсутствии замеров сохраняется пустое
        поле с подписью оси.
    """
    fig, axis = core.figure(core.AXES_RECT_WIDE)
    labels = [item["label"] for item in config.FRONTENDS]
    modes = list(modes)

    values: dict[tuple[str, str], float] = {}
    for frontend in config.FRONTENDS:
        for mode in modes:
            found = next((record for record in records
                          if record.get("ok")
                          and record["frontend"] == frontend["key"]
                          and record["mode"] == mode), None)
            values[(frontend["key"], mode)] = (found["total_ms"] if found
                                               else np.nan)

    measured = [value for value in values.values()
                if np.isfinite(value) and value > 0]
    if not measured:
        core.finish(axis, ylabel="время на пару, мс")
        return core.save(fig, f"performance_{name}")

    # Нижняя граница берётся по данным, а не по засечкам: засечка может
    # оказаться выше самого низкого столбца, и тот ушёл бы за поле
    bottom = 10.0 ** np.floor(np.log10(min(measured)))
    top = max(measured) * TIME_HEADROOM
    ticks = core.decades(bottom, top)

    axis.set_xlim(-BAR_MARGIN_LEFT, len(labels) - BAR_MARGIN_RIGHT)
    axis.set_ylim(bottom, top)
    axis.set_yscale("log")

    for index, frontend in enumerate(config.FRONTENDS):
        own = [(mode, values[(frontend["key"], mode)]) for mode in modes]
        own = [(mode, value) for mode, value in own if np.isfinite(value)]
        if not own:
            continue

        by_mode = dict(own)
        color = core.color_for(frontend["key"])
        placed: dict[str, float] = {}

        for position, mode in enumerate(modes):
            place, width = core.bar_positions(len(labels), len(modes), position)
            placed[mode] = place[index]
            core.rounded_bar(axis, place[index], by_mode.get(mode, np.nan),
                             color if position == 0
                             else core.tint(color, core.TINT_PAIR),
                             width=width, log_y=True)

        # Подписи времён стоят на общей высоте, взятой по более высокому
        # столбцу пары: на логарифмической оси столбцы пары различаются в разы,
        # и подписи по своим вершинам разошлись бы лесенкой
        highest = max(value for _, value in own)
        _, height = core.fraction(axis, index, highest, True)

        for mode, value in own:
            axis.annotate(compact(value), xy=(placed[mode], height),
                          xycoords=("data", "axes fraction"),
                          xytext=(0.0, VALUE_OFFSET_PT),
                          textcoords="offset points",
                          ha="center", va="bottom",
                          fontsize=core.SIZE_CAPTION, color=core.SUB, zorder=6,
                          bbox=dict(facecolor=core.BG, edgecolor="none",
                                    pad=1.5))

        # Крупная строка пуста только тогда, когда второй режим не замерен:
        # сравнивать нечего, и отношения не существует. Отсутствие выигрыша
        # пустотой не обозначается, у него есть своя запись
        if len(modes) == 2 and len(own) == 2:
            ratio = ratio_text(records, frontend["key"], modes, own)
            if ratio:
                axis.annotate(ratio, xy=(index, height),
                              xycoords=("data", "axes fraction"),
                              xytext=(0.0, RATIO_OFFSET_PT),
                              textcoords="offset points",
                              ha="center", va="bottom",
                              fontsize=core.SIZE_VALUE, fontweight="bold",
                              color=core.INK, zorder=6)

    core.finish(axis, ylabel="время на пару, мс", grid_axis="y", log_y=True,
                xticks=np.arange(len(labels)), xticklabels=core.wrap(labels),
                yticks=ticks)

    core.legend_row(fig, [
        core.dot(core.SUB if position == 0 else core.tint(core.SUB), mode)
        for position, mode in enumerate(modes)])

    return core.save(fig, f"performance_{name}")
