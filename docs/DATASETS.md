# Данные

Основной рецепт: SmartDoc2015, MIDV-2020, MIDV-500, MIDV-2019, MIDV-Holo,
DLC-2021. Порядок, состав, контрольные суммы архивов и назначение
последовательностей находятся в ресурсах Python-пакета, поэтому работают и из wheel.

```bash
uv run document-vos download --plan
uv run document-vos download --dataset smartdoc2015 midv2020
uv run document-vos prepare datasets --dataset smartdoc2015 midv2020
```

Один датасет, связанный в единственную группу документов/шаблонов, нельзя
безопасно разделить; подготовка сообщает об этом. Полный рецепт использует
фиксированное переносимое назначение из recipe.json, без зависимости от старых
разбиений. Допускаются физические оригиналы и копии DLC; replay исключается.
Пустая опубликованная VIA-разметка DLC исключается с причиной и SHA-256.

Нормализация TIFF в PNG сохраняет пиксели. Подготовка строит бинарные маски из
исходных четырёхугольников, затем training-export только из train. Валидация
проверяет document/template/group и отсутствие утечек. Данные samples/ и
собственные videos/ автоматически в обучение не входят.

```bash
uv run document-vos download --sample --plan
uv run document-vos download --sample --dataset smartdoc2015 --count 2
uv run document-vos infer --sample --count 2
```

Небольшие ZIP содержат только одну validation-последовательность каждый,
первую GT-маску, порядок, доступное время и атрибуцию. Скачивание sample никогда
не заменяется загрузкой полного архива. Повторный запуск проверяет содержимое
и переиспользует его; .part поддерживает докачку с HTTP Range. Сведения об
условиях данных: [THIRD_PARTY.md](THIRD_PARTY.md).
