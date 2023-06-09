https://ods.ai/competitions/mtsmlcup

Данное соревнование является первым для меня. В дальнейшем буду принимать участие в подобных соревнованиях, чтобы набираться опыта и получать новые знания.

**Leaderboard score 1,5629284013**

**Итоговое место 210 из 510**

## Задача соревнования
- Определение пола и возраста владельца HTTP cookie по истории активности пользователя в интернете на основе синтетических данных.

## Метрики соревнования:
* ROC-AUC – для определения пола, f1 weighted – для определения возраста.
* Все решения рассчитываются по формуле -  2 * f1_weighted(по 6 возрастным бакетам) + gini по полу.
* Возрастные бакеты (Класс 1 — 19-25, Класс 2 — 26-35, Класс 3 — 36-45, Класс 4 — 46-55, Класс 5 — 56-65, Класс 6 — 66+).

## Описание предобработанных данных
Предобработка, аггрегация и создание новых фич произведена в 0.Data_preparing.ipng.
Описание колонок аггрегированного файла с данными:
* 'part_of_day_day' – кол-во визитов пользователя днем
* 'part_of_day_evening' – кол-во визитов пользователя вечером
* 'part_of_day_morning' – кол-во визитов пользователя утром
* 'part_of_day_night' – кол-во визитов пользователя ночью
* 'sum_visits' – кол-во визитов пользователя
* 'day_pct' – доля визитов пользователя днем
* 'evening_pct' – доля визитов пользователя вечером
* 'morning_pct' – доля визитов пользователя утром
* 'night_pct' – доля визитов пользователя ночью
* 'act_days' – кол-во дней, в которые пользователь совершил визит пользователя
* 'request_cnt' - кол-во запросов пользователя
* 'avg_req_per_day' - среднее кол-во запросов пользователя
* 'period_days' - кол-во дней между первым и последним визитом пользователя
* 'request_std' - стандартное отклонение по количеству запросов
* 'act_days_pct' - доля дней, когда пользователь совершал визит
* 'cpe_type_cd - тип устройства
* 'cpe_model_os_type' - операционная система устройства
* 'cpe_manufacturer_name' -производитель устройства
* 'price' - цена устройства пользователя
* 'region_cnt' - кол-во уникальных регионов, из которых был совершен визит
* 'city_cnt' - кол-во уникальных городов, из которых был совершен визит
* 'url_host_cnt' - кол-во уникальных ссылок, с которых был совершен визит
* 'user_id' – ID пользователя

* также сгенерированы 300 признаков - эмбеддингов на основании региона, города, url и модели телефона

Описание колонок файла с таргетами:

* 'age' – Возраст пользователя
* 'is_male' – Признак пользователя : мужчина (1-Да, 0-Нет)
* 'user_id' – ID пользователя

## Структура проекта

Данная работа была разделена на несколько jupyter ноутбуков:

0. Data_preparing.ipnb - аггрегация отдельных файлов по user_id и склейка в финальный датасет
1. EDA.ipynb - исследовательская часть
2. baseline.ipynb - бейзлайн модели
3. create_embeddings.ipynb - создание эмбеддингов для дальнейшего их использования в качестве фич
4. baseline_embeddings.ipynb - бейзлан модели с эмбеддингами
5. model_tuning.ipynb - тюнинг наиболее перспективных моделей
6. gender_prediction_stacking.ipynb - стекинг моделей для предсказания пола
