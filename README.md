## PeachGuard AI

Выращивание персика - сложная задача: болезни листьев и плодов быстро снижают качество урожая.
**PeachGuard AI** использует модель компьютерного зрения, чтобы находить признаки заболеваний персика на изображении.

## Возможности

- **Детекция объектов** - модель возвращает класс, уверенность и координаты найденной области.
- **Обучение YOLO-модели** из YAML-конфига с возможностью переопределять параметры из CLI.
- **Инференс на одном изображении** с выводом результата в JSON.
- **Определение 5 классов**:
  - `bacterial_spot`
  - `brown_rot`
  - `healthy_peach`
  - `shot_hole`
  - `shot_hole_leaf`

## Структура проекта

- `configs/train.yaml` - параметры обучения.
- `configs/dataset.yaml` - описание датасета в формате YOLO.
- `data/processed` - изображения и разметка для `train`, `val` и `test`.
- `peachguard/cli` - CLI-команды обучения, инференса, отчёта и бота.
- `peachguard/detection` - обертка над YOLO для предсказаний (`PeachDetector`).
- `peachguard/training` - загрузка конфига, запуск обучения и отчёт по `results.csv`.
- `peachguard/bot` - Telegram-бот поверх `PeachDetector`.
- `artifacts/runs` - директория для результатов обучения по умолчанию.
- `artifacts/models` - сюда кладите `best.pt` после обучения в Colab.

## Установка

Проект использует Python `>=3.13`. Зависимости описаны в `pyproject.toml` и зафиксированы в `uv.lock`.

```bash
uv sync
```

Если `uv` не используется, зависимости можно установить через `pip`:

```bash
python -m pip install -e .
```

## CLI-команды

Команды запускаются из корня проекта через `uv run python -m`.

### Обучение

Базовый запуск использует настройки из `configs/train.yaml`:

```bash
uv run python -m peachguard.cli.train
```

Эквивалентно можно явно передать путь к конфигу:

```bash
uv run python -m peachguard.cli.train --config configs/train.yaml
```

Пример отладочного запуска на части датасета:

```bash
uv run python -m peachguard.cli.train \
  --epochs 5 \
  --batch 2 \
  --fraction 0.1 \
  --run-name debug \
  --exist-ok
```

Доступные параметры:

| Параметр | Описание |
| --- | --- |
| `--config` | Путь к YAML-конфигу обучения. По умолчанию: `configs/train.yaml`. |
| `--epochs` | Переопределить количество эпох. |
| `--batch` | Переопределить размер batch. |
| `--workers` | Переопределить количество dataloader workers. |
| `--patience` | Переопределить early stopping patience. |
| `--image-size` | Переопределить размер изображения для модели. |
| `--output-dir` | Переопределить директорию результатов. |
| `--run-name` | Переопределить имя запуска. |
| `--fraction` | Доля train-датасета для отладочного запуска. |
| `--save-period` | Сохранять checkpoint каждые N эпох. |
| `--exist-ok` | Разрешить перезапись директории запуска. |

После завершения команда печатает путь к результатам обучения. По умолчанию артефакты сохраняются в `artifacts/runs/train`.

### Инференс

Команда принимает путь к весам модели `.pt` и путь к изображению:

```bash
uv run python -m peachguard.cli.predict \
  --model artifacts/runs/train/weights/best.pt \
  --image data/processed/images/test/example.jpg
```

Можно настроить размер изображения и минимальную уверенность детекции:

```bash
uv run python -m peachguard.cli.predict \
  --model artifacts/runs/train/weights/best.pt \
  --image data/processed/images/test/example.jpg \
  --image-size 640 \
  --confidence 0.5
```

Доступные параметры:

| Параметр | Описание |
| --- | --- |
| `--model` | Обязательный путь к весам модели `.pt`. |
| `--image` | Обязательный путь к изображению. |
| `--image-size` | Размер изображения для модели. По умолчанию: `640`. |
| `--confidence` | Минимальная уверенность детекции. По умолчанию: `0.5`. |

Результат выводится в JSON:

```json
[
  {
    "class_id": 0,
    "class_name": "bacterial_spot",
    "confidence": 0.87,
    "box": {
      "x1": 12.4,
      "y1": 35.1,
      "x2": 220.8,
      "y2": 310.6
    }
  }
]
```

## Как это работает

1. Пользователь отправляет фото.
2. Изображение подается в модель детекции.
3. Предобученная модель выполняет инференс.
4. Система возвращает найденные объекты:
   - название класса;
   - уверенность модели;
   - координаты рамки.
