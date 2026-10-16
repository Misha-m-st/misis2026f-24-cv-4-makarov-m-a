# Данные: только явное скачивание

Проверено по описаниям авторов, GitHub release metadata и FTP LIST/маленьким readme/md5-файлам 2026-10-09. Кадры, видео и model checkpoints при реализации проекта не скачивались. ГБ ниже — десятичные 10⁹ байт. Используйте ограничения лицензий и цитируйте авторов наборов.

| Набор | Обязательная часть | Размер скачивания | Источник/контроль |
|---|---|---:|---|
| SmartDoc 2015 challenge 1 | `frames.tar.gz` v2.0.0 (кадры + CSV) | 1,019,404,933 B | [авторы](https://github.com/jchazalon/smartdoc15-ch1-dataset), SHA-256 `3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae` |
| MIDV-2020 | `dataset/clips.tar` (кадры + VIA) | 10,303,651,840 B | [авторы](https://arxiv.org/abs/2107.00396), MD5 `752dc7770ef555d4e16273045c2f1726` |
| MIDV-500 | выбранные `dataset/*.zip` | все 50: 32,853,003,590 B | [авторы](https://arxiv.org/abs/1807.05786); [архивы/MD5](midv500-archives.csv) |
| MIDV-2019 | выбранные `extra/midv-2019/dataset/*.zip` | все 50: 39,229,146,301 B | [авторы](https://arxiv.org/abs/1910.04009); [архивы/MD5](midv2019-archives.csv) |
| MIDV-Holo | `images.tar`, `markup.tar`, `metadata.tar` | 14,198,179,840 B | [авторы](https://github.com/SmartEngines/midv-holo); checksum в просмотренном каталоге не опубликован |
| DLC-2021 | `clips.tar` | 17,768,312,320 B | [авторы](https://www.mdpi.com/2313-433X/8/7/181); MD5 `0758a65d3ddccdd24eba25a98e4ba3c6` |
| MIDV-LAIT | выбранные каталоги `dataset/<document>/` | НЕ ПОДТВЕРЖДЁН для всего набора | [страница авторов](https://smartengines.ru/science/dataset/), FTP readme; на FTP 180 каталогов, единого архива не найдено |
| SIDTD | только новые физические видео + документная GT + подтверждённый источник | НЕ ПОДТВЕРЖДЁН | [авторы/loader](https://github.com/Oriolrt/SIDTD_Dataset), [описание](https://tc11.cvc.uab.es/datasets/SIDTD_1) |

Стартовый профиль SmartDoc + MIDV-2020 = **11,323,056,773 B ≈ 11,3 ГБ**. **40–60 ГБ свободного места** — оценка для архивов, распаковки и экспортов, до локального измерения. Полный выбранный комплект **около 166 ГБ без SIDTD** — оценка из исходного плана, а не подтверждённая сумма всех источников. Проверенная сумма перечисленных архивов без LAIT/SIDTD — 115,371,698,824 B. MIDV-LAIT и размер SIDTD требуют отдельной проверки; не заменяйте неизвестный объём нулём.

Приоритет расширения: MIDV-2019 и MIDV-Holo, затем другие выбранные наборы. Для MIDV-500/2019 можно скачивать отдельные документы; отчёт должен явно описывать это подмножество. ZIP этих двух наборов содержат и оригинальные видео; если авторы не предлагают архив только кадров, это неизбежная часть конкретного ZIP. Для MIDV-2020/Holo/DLC отдельные архивы исходных видео, сканов и фотографий **не нужны**.

## Команды

Для стартового профиля проще использовать скрипт (нужны только Python 3 и `curl`, установка проекта не требуется):

```bash
# Из корня проекта: оба датасета, с проверкой сумм и распаковкой.
python3 scripts/download_datasets.py

# Только MIDV-2020:
python3 scripts/download_datasets.py --datasets midv2020

# Предварительно посмотреть пути и объём, ничего не скачивая:
python3 scripts/download_datasets.py --plan

# Другой диск или только архивы:
python3 scripts/download_datasets.py --data-dir /mnt/data/document-vos
python3 scripts/download_datasets.py --download-only
```

Скрипт определяет корень проекта по своему расположению, а не текущему каталогу терминала. Например, из `data/raw` запускайте `python3 ../../scripts/download_datasets.py --datasets midv2020`. После обрыва повторите ту же команду: `.part` будет докачан, включая неполный файл от прежнего ручного `curl`. Готовый архив проверяется и повторно не скачивается. При неверной сумме файл сохраняется, распаковка запрещается. Распаковка выполняется во временный каталог; вложенная папка `clips/frames` нормализуется под `configs/starter.yaml`, внешние README/LICENSE сохраняются в `.archive-extras`. Завершённая распаковка отмечается `.document-vos-download.json`, повторный запуск её пропускает. Существующие непустые каталоги без этой отметки не перезаписываются: используйте `--download-only` или новый `--data-dir`. После выбора другого диска исправьте `root` в конфигурации подготовки. Архивы сохраняются после распаковки.

Скрипт также поддерживает MIDV-2019 и MIDV-500 (все 50 ZIP каждого набора), MIDV-Holo (три TAR) и DLC-2021 (`clips.tar`). По умолчанию скачиваются только два стартовых набора. Для Holo проверяются размеры архивов, вычисляется SHA-256 и проверяется распаковка; совпадение с контрольной суммой автора не заявляется, поскольку она не опубликована. DLC CSV с условиями съёмки можно сохранить отдельной командой ниже.

Простой следующий шаг — скачать MIDV-2019 и MIDV-Holo одной командой, затем подготовить новую версию общего набора:

```bash
python3 scripts/download_datasets.py --datasets midv2019 midvholo
uv run --locked document-vos prepare --config configs/starter.yaml --datasets smartdoc2015 midv2020 midv2019 midvholo --output data/prepared/manifest-v2.jsonl
uv run --locked document-vos split --manifest data/prepared/manifest-v2.jsonl --previous data/split-v1 --output data/split-v2 --seed 42 --val-fraction 0.2
uv run --locked document-vos export-training --manifest data/split-v2/train.jsonl --split-dir data/split-v2 --output data/training-v2 --seq-length 8 --workers 4
```

При ошибке остановитесь перед следующим шагом. `v1` сохраняется; существующие результаты `v2` не перезаписываются. Эти команды скачивания поддерживают повторный запуск и докачку. Поддержка проверена на маленьких локальных TAR/ZIP; полное скачивание и подготовка расширенных наборов ещё не выполнены.

Ниже — ручные команды. Не смешивайте внутренние `images/annotations` разных архивов.

```bash
mkdir -p data/archives data/raw/smartdoc2015 data/raw/midv2020/clips
curl -fL https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/frames.tar.gz -o data/archives/smartdoc-frames.tar.gz
curl -fL https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/sha256.chksum -o data/archives/smartdoc-sha256.chksum
# Файл models.tar.gz для нашей задачи не нужен.
printf '3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae  data/archives/smartdoc-frames.tar.gz\n' | sha256sum -c -
tar -xzf data/archives/smartdoc-frames.tar.gz -C data/raw/smartdoc2015

curl -f ftp://smartengines.com/midv-2020/dataset/clips.tar -o data/archives/midv2020-clips.tar
printf '752dc7770ef555d4e16273045c2f1726  data/archives/midv2020-clips.tar\n' | md5sum -c -
tar -xf data/archives/midv2020-clips.tar -C data/raw/midv2020/clips
```

После распаковки MIDV-2020 найдите `images/<type>/<clip>/` и `annotations/<type>/<clip>.json`; задайте root их **общего** родителя. Если tar добавил ещё один `clips/`, исправьте root в YAML. `models.tar.gz`, `photo.tar`, `scan_*.tar`, `clips_video.tar` не используются. MD5 проверяет транспортную целостность опубликованного архива; проект дополнительно сохраняет SHA-256 собственных входов и результатов.

```bash
# Пример одного документа; размеры и MD5 всех 50 в CSV выше.
mkdir -p data/raw/midv500 data/raw/midv2019
curl -f ftp://smartengines.com/midv-500/dataset/01_alb_id.zip -o data/archives/midv500-01_alb_id.zip
unzip data/archives/midv500-01_alb_id.zip -d data/raw/midv500
curl -f ftp://smartengines.com/midv-500/extra/midv-2019/dataset/01_alb_id.zip -o data/archives/midv2019-01_alb_id.zip
unzip data/archives/midv2019-01_alb_id.zip -d data/raw/midv2019
# До распаковки сверьте MD5 с CSV/ftp://smartengines.com/.../md5.txt.
```

```bash
mkdir -p data/raw/midvholo
curl -f ftp://smartengines.com/midv-holo/images.tar -o data/archives/holo-images.tar
curl -f ftp://smartengines.com/midv-holo/markup.tar -o data/archives/holo-markup.tar
curl -f ftp://smartengines.com/midv-holo/metadata.tar -o data/archives/holo-metadata.tar
tar -xf data/archives/holo-images.tar -C data/raw/midvholo
tar -xf data/archives/holo-markup.tar -C data/raw/midvholo
tar -xf data/archives/holo-metadata.tar -C data/raw/midvholo
```

Holo: `images.tar` = 14,166,804,480 B; `markup.tar` = 31,324,160 B; `metadata.tar` = 51,200 B. `video.tar` и `templates.tar` не обязательны. Если конкретная разметка использует другой ключ/раскладку вместо покадрового `quad`, перенесите опубликованные координаты в явный индекс без интерполяции. Метаданные происхождения/освещения сохраняются с SHA-256 в provenance.

```bash
mkdir -p data/raw/dlc2021/clips
curl -f ftp://smartengines.com/dlc-2021/clips.tar -o data/archives/dlc-clips.tar
curl -f ftp://smartengines.com/dlc-2021/dlc-2021.csv -o data/raw/dlc2021/dlc-2021.csv
printf '0758a65d3ddccdd24eba25a98e4ba3c6  data/archives/dlc-clips.tar\n' | md5sum -c -
tar -xf data/archives/dlc-clips.tar -C data/raw/dlc2021/clips
```

DLC: названия клипов `<document>.or0001`, `.cc0001`, `.cg0001` принимаются; `.re0001` исключаются. Физические копии/оригиналы наследуют связи MIDV-2020. CSV даёт условия съёмки; его также сохраните рядом с исходными данными.

```bash
# MIDV-LAIT: один выбранный физический документ вместе с покадровой GT.
# wget рекурсивно получает именно этот каталог. Сумму размеров сначала проверяйте FTP LIST.
wget -r -np -nH --cut-dirs=2 -P data/raw/midvlait/dataset ftp://smartengines.com/midv-lait/dataset/afg_id_01/
```

Результат должен содержать `afg_id_01/afg_id_01_images/*.jpg` и `afg_id_01/afg_id_01_groundtruth/*.jpg.json`. Идеальный исходный шаблон `<doc>.png` в этой папке training-кадром не является. Для полного набора повторите по **явно выбранным** каталогам, не скачивайте весь FTP без оценки диска.

SIDTD не имеет здесь выдуманной команды для неизвестного архива. Используйте [официальный loader/инструкции](https://github.com/Oriolrt/SIDTD_Dataset), выберите дополнительную физическую видео-часть, проверьте наличие документной quad/бинарной разметки и запишите `sequences.jsonl`. Повторные MIDV-2020, синтетические статичные изображения и экранные предъявления не входят в профиль. Если данные требуют доступа/не содержат локализацию, не выбирайте SIDTD, пока необходимые материалы не получены.

Файлы контрольных сумм меняются авторами: при расхождении размеров/MD5 не продолжайте подготовку автоматически; зафиксируйте новую версию источника. Перепаковка или непроверенные зеркала не считаются тем же архивом.
