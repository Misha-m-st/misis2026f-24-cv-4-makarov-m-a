# Воспроизведение

Установка: `uv sync --locked`. Основные платформы — Linux и Ubuntu в WSL2;
официальное обучение использует NCCL и CUDA. Нативная Windows не квалифицирована.
[PyTorch distributed](https://docs.pytorch.org/docs/2.14/distributed.html),
[CUDA в WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

```bash
uv run document-vos setup
uv run document-vos download --plan
uv run document-vos download
uv run document-vos reproduce
```

`download` только получает архивы. `prepare datasets` нормализует их локально,
строит общий манифест, назначает фиксированные выборки и создаёт training-export.
При существующих нормализованных данных задайте `datasets.roots` в config.yaml;
пути относительны к YAML. Исходные папки не перемещаются.

Ресурсы пакета `recipe.json` фиксируют 2836 train, 717 validation, 200 106 кадров,
21 исключённое DLC-видео (1144 кадра), причины и хеши аннотаций. Назначение по
идентификаторам переносится независимо от прошлых запусков. Научная конфигурация
и полный resolved-конфиг находятся в ресурсах пакета и каждом training-run.

Seed 42, 30 000 итераций, batch 2, последовательность 8, crop 480×480,
3 reference frames, AMP, LR 1e-5. Один объект, merge_probability=0,
pre_training выключен. Связанные document/template/group не пересекаются.
Парный инференс использует одни кадры и первые маски, внутренний размер 480;
PNG и метрики сохраняют исходное разрешение. Протокол J, DAVIS F, J&F,
Boundary IoU, decay и усреднения сохранён. Первый кадр инициализации исключён
из метрик. Произвольные изменения рецепта допускаются через train/infer;
`reproduce` их отклоняет до начала работы.

`reproduce` никогда не скачивает данные или веса. Если локальных входов нет,
он сообщает нужную команду. Запуск создаёт новый каталог и HTML-индекс.

```bash
uv run document-vos reproduce --resume results/ПАПКА
```

Продолжение явно проверяет входы, код и все завершённые артефакты. Повреждение
блокирует продолжение. FAILED/RUNNING-этап повторяется в новой папке попытки;
предыдущая попытка сохраняется. Внутреннее продолжение обучения не реализовано.
Изменившийся вход требует нового запуска. Полное обучение при проверке
рефакторинга не повторяется.

Диагностики: `uv run document-vos tools --help`; например `tools validate-data`,
`tools probe-training`, `tools evaluate`, `tools compare`, `tools render`.

Сборка будущего Release доступна через `tools build-release --help`: требует
явно переданных исходного разбиения, метаданных, завершённого training-run и
baseline. Она сохраняет 12 ZIP, карточку модели, веса, каталоги и SHA256SUMS
в новый каталог; публикацию не выполняет.
