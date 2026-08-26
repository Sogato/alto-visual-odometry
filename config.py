"""
Константы проекта.

Здесь собраны пути, характеристики датасетов и параметры алгоритмов. Модуль
состоит только из значений: он ничего не вычисляет, не проверяет и не печатает.

Значения бывают трёх видов. Часть задана заранее и определяет условия
эксперимента: пороги, размеры окон, число итераций. Часть взята из описания
датасета ALTO со ссылкой на источник. Часть измеряется модулями из service и
вписывается сюда руками, до измерения такое поле хранит None.

Измеряемые величины и модуль, которым получают каждое значение:

    FRAME_STEP                          service/frame_step.py
    CALIBRATION[*].focal_px             service/calibration.py, проверка паспорта
    CALIBRATION[*].principal_point      service/calibration.py
    CALIBRATION[*].altitude_mode        service/calibration.py
    CALIBRATION[*].ground_elevation_m   service/calibration.py, только при msl
    CALIBRATION[*].rotation_cam_to_gt   service/calibration.py

Обязательные поля калибровки перечисляет CALIBRATION_REQUIRED.

Порядок запуска: service/system_info.py, service/dataset_validator.py,
service/calibration.py, заполнение калибровки, service/frame_step.py,
заполнение шага, main.py.

Связки в FRONTENDS отвечают трём подходам задания:

    SIFT/ORB + Essential Matrix     sift_bf, orb_bf
    SIFT/ORB + Lucas-Kanade         sift_lk, orb_lk
    SuperPoint + LightGlue          sp_lg

GEOMETRIES перечисляет две геометрии восстановления движения. Обе считаются
поверх одних и тех же сопоставлений, поэтому в состав связки не входят.
"""

# Стандартные библиотеки
from pathlib import Path
from typing import Any


# === ПУТИ ===
PROJECT_ROOT = Path(__file__).resolve().parent   # Корень проекта, файл лежит в нём
DATA_DIR = PROJECT_ROOT / "data"                 # Кадры и телеметрия датасетов
RESULTS_DIR = PROJECT_ROOT / "results"           # Результаты прогонов
PLOTS_DIR = RESULTS_DIR / "plots"                # Готовые фигуры анализа

# === ДАТАСЕТЫ ===
# Ключ верхнего уровня это имя выборки, оно же имя каталога в data и ключ в
# CALIBRATION
DATASETS: dict[str, dict[str, Path]] = {
    "train_sample": {
        "images": DATA_DIR / "train_sample" / "query_images",
        "telemetry": DATA_DIR / "train_sample" / "query.csv",
    },
    "val_sample": {
        "images": DATA_DIR / "val_sample" / "query_images",
        "telemetry": DATA_DIR / "val_sample" / "query.csv",
    },
}

IMAGE_EXTENSION = ".png"   # Расширение файлов кадров в query_images

# === КОЛОНКИ ТЕЛЕМЕТРИИ ===
# Координаты в метрах в проекции UTM, зона 17N. Ориентация задана кватернионом.
# Кадр связывается с телеметрией по COLUMN_NAME, а не по номеру строки
COLUMN_EASTING = "easting"     # Координата на восток, м
COLUMN_NORTHING = "northing"   # Координата на север, м
COLUMN_ALTITUDE = "altitude"   # Высота, м, трактуется по altitude_mode
COLUMN_NAME = "name"           # Имя файла кадра
COLUMNS_QUATERNION = ("orient_x", "orient_y", "orient_z", "orient_w")   # Ориентация

# === КАМЕРА ===
# Паспортные характеристики камеры ALTO и её установки на носителе. Источник:
# Cisneros et al., ALTO, arXiv:2207.12317, раздел The Platform.
#
# Частное фокусного расстояния и размера пикселя даёт фокусное в пикселях. Оно
# относится к полному кадру камеры и сохраняется, если кадры датасета вырезаны
# из него без сжатия, что проверяет service/calibration.py.
LENS_FOCAL_MM = 3.5        # Фокусное расстояние объектива, мм
PIXEL_PITCH_MM = 0.0045    # Размер пикселя матрицы, мм

# Частота съёмки, кадров в секунду. Колонки со временем в телеметрии нет, время
# кадра восстанавливается по его номеру
CAMERA_FPS = 20.0

# Ось камеры в системе носителя. Камера закреплена жёстко и смотрит вниз, её
# оптической осью принимается третья ось системы носителя
CAMERA_AXIS_IN_BODY = 2

# === ОБЩИЕ ПАРАМЕТРЫ ===
# Начальное состояние генератора случайных чисел OpenCV. Задаёт выборки точек в
# робастной оценке, без него повторный прогон даёт другие числа
RANDOM_SEED = 42

# === КАЛИБРОВКА ДАТАСЕТОВ ===
# Заполняется руками по выводу service/calibration.py. Состав полей:
#
#   focal_px            фокусное расстояние в пикселях
#   principal_point     главная точка, пара координат в пикселях
#   altitude_mode       что означает колонка altitude, msl либо agl
#   ground_elevation_m  высота рельефа, вычитается из altitude при режиме msl
#   rotation_cam_to_gt  поворот из системы камеры в систему эталона, матрица 3x3
#
# Поля из CALIBRATION_REQUIRED обязательны для прогона. Высота рельефа
# становится обязательной при altitude_mode, равном msl
CALIBRATION_REQUIRED: tuple[str, ...] = (
    "focal_px",
    "principal_point",
    "altitude_mode",
    "rotation_cam_to_gt",
)

CALIBRATION: dict[str, dict[str, Any]] = {
    "train_sample": {
        "focal_px": LENS_FOCAL_MM / PIXEL_PITCH_MM,
        "principal_point": (249.5, 249.5),
        "altitude_mode": "agl",
        "ground_elevation_m": 0.0,
        "rotation_cam_to_gt": [
            [0.984497, -0.175402, 0.000000],
            [0.175402, 0.984497, 0.000000],
            [0.000000, 0.000000, 1.000000],
        ],
    },
    "val_sample": {
        "focal_px": LENS_FOCAL_MM / PIXEL_PITCH_MM,
        "principal_point": (249.5, 249.5),
        "altitude_mode": "agl",
        "ground_elevation_m": 0.0,
        "rotation_cam_to_gt": [
            [0.984497, -0.175402, 0.000000],
            [0.175402, 0.984497, 0.000000],
            [0.000000, 0.000000, 1.000000],
        ],
    },
}

# === ШАГ ПРОРЕЖИВАНИЯ КАДРОВ ===
# Пара кадров составляется через FRAME_STEP кадров исходной съёмки. Значение
# выбирается по выводу service/frame_step.py и вписывается руками
FRAME_STEP: int | None = 12

# Сетка значений, которые перебирает service/frame_step.py. Логарифмическая:
# соседние значения дают почти неотличимый результат
FRAME_STEP_GRID: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20, 30, 50)

# === ПАРАМЕТРЫ ДЕТЕКТОРОВ ===
MAX_KEYPOINTS = 1024   # Предел числа точек, общий для всех детекторов

SIFT_PARAMS: dict[str, Any] = {
    "nfeatures": MAX_KEYPOINTS,      # Предел числа точек
    "contrastThreshold": 0.04,       # Порог отбраковки слабоконтрастных точек
    "edgeThreshold": 10,             # Порог отбраковки точек на рёбрах
    "sigma": 1.6,                    # Сглаживание нулевого уровня пирамиды
}

ORB_PARAMS: dict[str, Any] = {
    "nfeatures": MAX_KEYPOINTS,      # Предел числа точек
    "scaleFactor": 1.2,              # Отношение масштабов соседних уровней
    "nlevels": 8,                    # Число уровней пирамиды
    "fastThreshold": 20,             # Порог детектора FAST внутри ORB
}

SUPERPOINT_PARAMS: dict[str, Any] = {
    "max_num_keypoints": MAX_KEYPOINTS,   # Предел числа точек
    "detection_threshold": 0.0005,        # Порог отклика детектора
    "nms_radius": 4,                      # Радиус подавления немаксимумов, px
}

# === ПАРАМЕТРЫ СОПОСТАВЛЕНИЯ ===
# Порог теста отношения расстояний Лоу: пара принимается, если ближайший
# дескриптор ближе второго по близости в BF_RATIO_THRESHOLD раз
BF_RATIO_THRESHOLD = 0.75

# Параметры оптического потока Лукаса и Канаде. Первые два ключа названы как
# параметры cv2.calcOpticalFlowPyrLK, последние два собираются в его аргумент
# criteria при вызове
LK_PARAMS: dict[str, Any] = {
    "winSize": (21, 21),         # Размер окна поиска, px
    "maxLevel": 3,               # Число уровней пирамиды сверх нулевого
    "criteria_max_iter": 30,     # Предел итераций уточнения позиции точки
    "criteria_epsilon": 0.01,    # Порог сходимости по смещению, px
}

# Порог обратной проверки трекинга, px: точка прогоняется вперёд и назад, при
# расхождении больше порога отбрасывается
LK_BACKWARD_THRESHOLD = 1.0

# Пороги адаптивной работы LightGlue оставлены на умолчаниях библиотеки: сеть
# останавливается до прохода всех слоёв и по ходу отбрасывает часть точек.
# Число сопоставлений при этом зависит от сложности пары. Адаптация отключается
# добавлением сюда depth_confidence и width_confidence, равных -1
LIGHTGLUE_PARAMS: dict[str, Any] = {
    "features": "superpoint",   # Тип точек, под который обучены веса
}

# === ПАРАМЕТРЫ РОБАСТНОЙ ОЦЕНКИ ===
# Метод передаётся явно во всех вызовах: в OpenCV 5.0 умолчания изменились
RANSAC_METHOD = "RANSAC"        # Метод оценки в основных прогонах
RANSAC_CONFIDENCE = 0.999       # Требуемая вероятность найти верную модель
RANSAC_MAX_ITERS = 5000         # Предел числа итераций

# Пороги невязки, px. У гомографии это ошибка перепроекции, у Essential
# расстояние до эпиполярной линии
RANSAC_THRESHOLD_HOMOGRAPHY = 3.0
RANSAC_THRESHOLD_ESSENTIAL = 1.0

# Методы, перебираемые в анализе влияния RANSAC
RANSAC_METHODS_COMPARED: tuple[str, ...] = (
    "RANSAC",
    "LMEDS",
    "USAC_DEFAULT",
    "USAC_MAGSAC",
)

# Нижняя граница числа сопоставлений, при котором считается геометрия.
# Гомографии нужно четыре точки, Essential пять, остальное запас на выбросы
MIN_MATCHES_FOR_GEOMETRY = 12

# === МАТРИЦА КОНФИГУРАЦИЙ ПРОГОНА ===
# Связка это пара детектора и матчера. Состав полей:
#
#   key         короткий идентификатор, ключ результатов и имя в фигурах
#   label       подпись для отчёта и легенд
#   detector    детектор точек
#   matcher     способ сопоставления точек между кадрами
FRONTENDS: tuple[dict[str, str], ...] = (
    {"key": "sift_bf", "label": "SIFT + BF", "detector": "sift", "matcher": "bf"},
    {"key": "orb_bf", "label": "ORB + BF", "detector": "orb", "matcher": "bf"},
    {"key": "sift_lk", "label": "SIFT + LK", "detector": "sift", "matcher": "lk"},
    {"key": "orb_lk", "label": "ORB + LK", "detector": "orb", "matcher": "lk"},
    {"key": "sp_lg", "label": "SuperPoint + LightGlue",
     "detector": "superpoint", "matcher": "lightglue"},
)

# Геометрии восстановления движения. Поля те же, что у связки: key и label
GEOMETRIES: tuple[dict[str, str], ...] = (
    {"key": "homography", "label": "Homography"},
    {"key": "essential", "label": "Essential"},
)
