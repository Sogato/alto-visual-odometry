"""
Загрузка датасета: кадры, телеметрия и эталонное движение между кадрами.

Dataset связывает кадры с телеметрией по колонке name и даёт доступ к ним по
порядковому номеру внутри получившегося набора. Отсюда же берутся пары кадров с
заданным шагом прореживания и эталонное перемещение между кадрами пары.

Высота отдаётся такой, какой записана в телеметрии. Трактовать её по
altitude_mode должен вызывающий код.

Рядом с классом лежит перевод кватернионов ориентации в матрицы поворота и
величина поворота по его матрице.
"""

# Стандартные библиотеки
import sys
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


# === КЕШ ИЗОБРАЖЕНИЙ ===
# При последовательном обходе каждый кадр нужен дважды: как второй в одной паре
# и как первый в следующей
IMAGE_CACHE_SIZE = 4     # Сколько кадров держать в памяти


class Dataset:
    """
    Кадры датасета вместе с их эталонными позициями и ориентациями.

    При создании читает телеметрию, сканирует каталог с кадрами и оставляет
    только те кадры, для которых есть строка телеметрии. Дальше работа идёт по
    порядковому номеру внутри этого набора, от нуля до len(dataset) минус один.

    Attributes:
        name: имя датасета, ключ в config.DATASETS.
        images_dir: каталог с кадрами.
        frames: таблица совпавших кадров, отсортированная по номеру кадра.
        names: имена файлов кадров в том же порядке.
        positions: координаты в проекции UTM, массив (N, 2).
        altitudes: высоты из телеметрии, массив (N,).
        rotations: матрицы ориентации, массив (N, 3, 3).
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name: имя датасета, ключ в config.DATASETS.

        Raises:
            KeyError: датасета нет в конфиге.
            FileNotFoundError: нет файла телеметрии или каталога с кадрами.
            ValueError: в телеметрии нет обязательных колонок либо ни один кадр
                не совпал с телеметрией.
        """
        if name not in config.DATASETS:
            raise KeyError(f"датасет {name} не описан в config.DATASETS")

        paths = config.DATASETS[name]
        self.name = name
        self.images_dir: Path = paths["images"]

        if not paths["telemetry"].is_file():
            raise FileNotFoundError(f"нет файла телеметрии: {paths['telemetry']}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"нет каталога с кадрами: {self.images_dir}")

        telemetry = pd.read_csv(paths["telemetry"])
        required = [config.COLUMN_NAME, config.COLUMN_EASTING, config.COLUMN_NORTHING,
                    config.COLUMN_ALTITUDE, *config.COLUMNS_QUATERNION]
        missing = [column for column in required if column not in telemetry.columns]
        if missing:
            raise ValueError(f"в телеметрии нет колонок: {', '.join(missing)}")

        self.frames = self._join(telemetry)
        if self.frames.empty:
            raise ValueError(f"в датасете {name} ни один кадр не совпал с телеметрией")

        self.names: list[str] = self.frames[config.COLUMN_NAME].tolist()
        self.positions: np.ndarray = self.frames[
            [config.COLUMN_EASTING, config.COLUMN_NORTHING]].to_numpy(dtype=float)
        self.altitudes: np.ndarray = self.frames[
            config.COLUMN_ALTITUDE].to_numpy(dtype=float)
        self.rotations: np.ndarray = quaternions_to_matrices(
            self.frames[list(config.COLUMNS_QUATERNION)].to_numpy(dtype=float))

        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self._image_size: tuple[int, int] | None = None

    def _join(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """
        Оставляет строки телеметрии, для которых на диске есть кадр.

        Кадры с нечисловым именем отбрасываются: по имени восстанавливается
        порядок обхода.

        Args:
            telemetry: полная таблица телеметрии.

        Returns:
            Таблицу совпавших кадров, отсортированную по номеру кадра, с
            добавленной колонкой frame_number.
        """
        files = {path.name for path in
                 self.images_dir.glob(f"*{config.IMAGE_EXTENSION}")}

        table = telemetry.copy()
        table[config.COLUMN_NAME] = table[config.COLUMN_NAME].astype(str)
        table = table[table[config.COLUMN_NAME].isin(files)]

        numbers = table[config.COLUMN_NAME].map(lambda value: Path(value).stem)
        numeric = numbers.str.isdigit()
        table = table[numeric].assign(frame_number=numbers[numeric].astype(int))

        # Сортировка по номеру кадра, а не по имени: при именах разной длины
        # порядок строк разошёлся бы с числовым
        return table.sort_values("frame_number").reset_index(drop=True)

    def __len__(self) -> int:
        """
        Returns:
            Число кадров, для которых есть и файл, и телеметрия.
        """
        return len(self.frames)

    # ────────────────────────────────────────────────────────────────────────
    # Доступ к изображениям
    # ────────────────────────────────────────────────────────────────────────

    def gray(self, index: int) -> np.ndarray:
        """
        Возвращает кадр в градациях серого.

        Отдаётся копия, поэтому правка кадра на месте у вызывающего кеш не
        портит.

        Args:
            index: порядковый номер кадра внутри датасета.

        Returns:
            Массив формы (высота, ширина) типа uint8.

        Raises:
            IndexError: индекс за пределами датасета.
            OSError: файл не читается или не декодируется.
        """
        return self._load(index).copy()

    def _load(self, index: int) -> np.ndarray:
        """
        Читает кадр с диска либо достаёт его из кеша.

        Массив отдаётся как есть, без копирования: копию делает gray.

        Args:
            index: порядковый номер кадра внутри датасета.

        Returns:
            Кадр в градациях серого.

        Raises:
            IndexError: индекс за пределами датасета.
            OSError: файл не читается или не декодируется.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"кадр {index} вне диапазона 0...{len(self) - 1}")

        if index in self._cache:
            return self._cache[index]

        gray = self._decode(index)
        self._remember(index, gray)
        return gray

    def _decode(self, index: int) -> np.ndarray:
        """
        Читает кадр с диска, минуя кеш.

        Обращения, которым кадр нужен разово, не вытесняют из кеша те кадры, с
        которыми идёт работа. Альфа-канал отбрасывается.

        Args:
            index: порядковый номер кадра внутри датасета.

        Returns:
            Кадр в градациях серого.

        Raises:
            OSError: файл не читается или не декодируется.
        """
        path = self.images_dir / self.names[index]
        # Штатный cv2.imread не работает с путями вне latin-1, поэтому файл
        # читается как байты и декодируется отдельно
        raw = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"не удалось прочитать кадр: {path}")

        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _remember(self, index: int, gray: np.ndarray) -> None:
        """
        Кладёт кадр в кеш, вытесняя самый давний при переполнении.

        Args:
            index: порядковый номер кадра.
            gray: кадр в градациях серого.
        """
        self._cache[index] = gray
        self._cache_order.append(index)

        while len(self._cache_order) > IMAGE_CACHE_SIZE:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Размер кадра в пикселях, определённый по первому кадру датасета.

        Значение запоминается при первом обращении, кадр читается мимо кеша.

        Returns:
            Пару из ширины и высоты в пикселях.
        """
        if self._image_size is None:
            height, width = self._decode(0).shape[:2]
            self._image_size = (width, height)
        return self._image_size

    # ────────────────────────────────────────────────────────────────────────
    # Пары кадров
    # ────────────────────────────────────────────────────────────────────────

    def pairs(self, step: int, start: int = 0, stop: int | None = None,
              limit: int | None = None) -> list[tuple[int, int]]:
        """
        Составляет последовательность пар кадров с заданным шагом прореживания.

        Пары идут встык: конец одной пары служит началом следующей, поэтому
        перемещения складываются в непрерывную траекторию.

        Шаг отсчитывается по порядковому номеру кадра внутри датасета, а не по
        номеру в имени файла. При разрывах нумерации шаг по именам оказался бы
        больше заказанного.

        Args:
            step: шаг прореживания, берётся каждый step-й кадр.
            start: индекс первого кадра последовательности.
            stop: индекс, до которого идти, не включая. По умолчанию до конца.
            limit: предельное число пар. По умолчанию без ограничения.

        Returns:
            Список пар индексов кадров. Пустой, если кадров на пару не хватило.

        Raises:
            ValueError: шаг меньше единицы.
        """
        if step < 1:
            raise ValueError(f"шаг прореживания должен быть не меньше 1, "
                             f"получен {step}")

        last = len(self) if stop is None else min(stop, len(self))
        indices = list(range(start, last, step))
        result = list(zip(indices[:-1], indices[1:]))

        return result[:limit] if limit is not None else result

    # ────────────────────────────────────────────────────────────────────────
    # Эталонное движение
    # ────────────────────────────────────────────────────────────────────────

    def relative_motion(self, first: int, second: int) -> dict[str, Any]:
        """
        Считает эталонное перемещение и поворот между двумя кадрами.

        Перемещение отдаётся в двух видах. В мировых координатах это разность
        позиций в проекции UTM, где оси направлены на восток, на север и вверх.
        В системе носителя это та же разность, повёрнутая в оси первого кадра.

        Перевод в оси носителя опирается на допущение, что кватернион записан в
        тех же осях восток, север, вверх. Допущение проверяет
        service/calibration.py, и до проверки delta_body нельзя сравнивать с
        результатом одометрии: при ином соглашении об осях вектор окажется
        повёрнут на постоянный угол. Расстояние и величина поворота от
        соглашения об осях не зависят.

        Args:
            first: индекс первого кадра.
            second: индекс второго кадра.

        Returns:
            Словарь с полями delta_world, delta_body, distance по горизонтали в
            метрах, rotation в виде матрицы 3x3 и rotation_angle в градусах.

        Raises:
            IndexError: индекс за пределами датасета.
        """
        delta_world = np.array([
            self.positions[second, 0] - self.positions[first, 0],
            self.positions[second, 1] - self.positions[first, 1],
            self.altitudes[second] - self.altitudes[first],
        ], dtype=float)

        rotation_first = self.rotations[first]
        rotation_second = self.rotations[second]

        # Транспонирование матрицы поворота равносильно обратному повороту
        relative_rotation = rotation_first.T @ rotation_second
        delta_body = rotation_first.T @ delta_world

        return {
            "delta_world": delta_world,
            "delta_body": delta_body,
            "distance": float(np.hypot(delta_world[0], delta_world[1])),
            "rotation": relative_rotation,
            "rotation_angle": rotation_angle(relative_rotation),
        }

    def path_length(self, start: int = 0, stop: int | None = None) -> float:
        """
        Считает длину эталонного маршрута по горизонтали.

        Args:
            start: индекс первого кадра.
            stop: индекс, до которого идти, не включая.

        Returns:
            Длину пути в метрах. Ноль, если кадров меньше двух.
        """
        last = len(self) if stop is None else min(stop, len(self))
        segment = self.positions[start:last]
        if len(segment) < 2:
            return 0.0
        return float(np.hypot(*np.diff(segment, axis=0).T).sum())


# ────────────────────────────────────────────────────────────────────────────
# Работа с поворотами
# ────────────────────────────────────────────────────────────────────────────

def quaternions_to_matrices(quaternions: np.ndarray) -> np.ndarray:
    """
    Переводит кватернионы в матрицы поворота.

    Кватернионы нормируются перед переводом.

    Args:
        quaternions: массив формы (N, 4) с компонентами в порядке x, y, z, w.

    Returns:
        Массив формы (N, 3, 3) с матрицами поворота.

    Raises:
        ValueError: среди кватернионов есть вырожденный, с нулевой нормой.
            Молчаливое деление обратило бы матрицу в NaN, и порча разошлась бы
            по всем последующим расчётам.
    """
    values = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)

    degenerate = np.flatnonzero(norms.ravel() == 0.0)
    if degenerate.size:
        raise ValueError(f"кватернион с нулевой нормой в строках: "
                         f"{degenerate[:5].tolist()}, всего {degenerate.size}")

    values = values / norms
    x, y, z, w = values[:, 0], values[:, 1], values[:, 2], values[:, 3]

    matrices = np.empty((len(values), 3, 3), dtype=float)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def rotation_angle(rotation: np.ndarray) -> float:
    """
    Возвращает величину поворота в градусах.

    След матрицы поворота связан с углом соотношением trace = 1 + 2 cos(угол).
    Величина не зависит от того, в каких осях записана матрица.

    Args:
        rotation: матрица поворота 3x3.

    Returns:
        Угол поворота в градусах, от нуля до ста восьмидесяти.
    """
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
