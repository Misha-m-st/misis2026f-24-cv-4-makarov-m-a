# Сторонние компоненты и данные

Лицензия собственного кода проекта пока не назначена. Это не меняет условий
сторонних компонентов и датасетов. Веса и примеры подготавливаются локально;
ничего автоматически не публикуется.

- Cutie: Ho Kei Cheng et al., CVPR 2024; MIT, commit
  ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a.
  [Источник и лицензия](https://github.com/hkchengrex/Cutie/blob/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a/LICENSE).
- PyAV 16.1.0: [BSD и условия включённых библиотек FFmpeg](https://github.com/PyAV-Org/PyAV/blob/v16.1.0/LICENSE.txt).
  [Готовые wheels](https://github.com/PyAV-Org/PyAV/blob/v16.1.0/docs/overview/installation.rst).
- SAM3: необязательный компонент Facebook Research,
  [код и условия](https://github.com/facebookresearch/sam3).
- SmartDoc2015: Burie, Chazalon, Coustaty, Eskenazi, Luqman, Mehri, Nayef, Ogier,
  Prum, Rusinol, ICDAR 2015. CC BY 4.0;
  [официальное описание и атрибуция](https://github.com/jchazalon/smartdoc15-ch1-dataset).
  В ZIP сохранена исходная LICENCE. Изменение: первая GT-четырёхугольная
  аннотация растеризована в бинарную PNG; кадры скопированы без изменения.
- MIDV-500: Bulatov et al., 2018,
  [статья авторов](https://arxiv.org/abs/1807.05786),
  [распространение](ftp://smartengines.com/midv-500/).
  Исходные изображения документов имеют собственные public-domain/public-license
  условия. Лицензия сторонних Python-обёрток не используется как лицензия данных.
- MIDV-2019: Bulatov et al., 2019,
  [статья авторов](https://arxiv.org/abs/1910.04009),
  [распространение](ftp://smartengines.com/midv-500/extra/midv-2019/).
  Сохраняются условия исходных документов MIDV-500.
- MIDV-2020: Bulatov et al.,
  [статья авторов](https://arxiv.org/abs/2107.00396),
  [распространение](ftp://smartengines.com/midv-2020/).
  CC BY-SA 2.5 Generic по [лицензии авторов](https://docs.google.com/forms/d/e/1FAIpQLSdxB1gvdVlRcARUlMolTJzyqY93XBZHhwiBwkDx8BDyMIPWIg/viewform).
  Лица: Generated Photos; исходные документы — Wikimedia Commons, атрибуция
  в [таблице авторов](ftp://smartengines.com/midv-500/documents.pdf).
- MIDV-Holo: Smart Engines, CC BY-SA 2.5 Generic;
  [условия авторов](https://github.com/SmartEngines/midv-holo).
  Лица: [Generated Photos](https://generated.photos/), атрибуция сохранена.
- DLC-2021: Polevoy, Sigareva, Ershova, Arlazarov, Nikolaev, Ming, Luqman, Burie,
  J. Imaging 8(7):181, 2022. CC BY-SA 2.5 Generic по
  [Data Availability Statement авторов](https://pmc.ncbi.nlm.nih.gov/articles/PMC9323793/).
  [Распространение](ftp://smartengines.com/dlc-2021/), лица: Generated Photos.

Каталоги сохраняют URL, библиографию, доступные сведения об исходных условиях и
контрольные суммы. Условия данных отличаются от лицензий статей и обёрток.
Индивидуальные условия исходных документов MIDV-500/2019 сохранены ссылкой на
[таблицу авторов](ftp://smartengines.com/midv-500/documents.pdf). Маленькие
примеры сохраняют исходные пиксели (нормализованные TIFF могут храниться в PNG),
первую GT-маску, порядок и реально доступное время. Остальная разметка не входит.
