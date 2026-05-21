# EDA для датасета апскейлинга белков

Скрипт `run_eda.py` отвечает на 4 ключевых вопроса о качестве датасета:

1. Парсятся ли все файлы и валиден ли каждый pair как supervision? (Blocks A, B)
2. Однозначен ли target — нет ли разноголосых good-структур у одного uniprot_id? (Block C)
3. Честный ли train/val split (нет ли data leakage)? (Block D)
4. Лучше ли модель тривиального baseline `pred = coords_bad`? (Block E)

## Настройка путей

В начале `run_eda.py` есть три заглушки:

```python
CSV_PATH = "PATH/TO/pdb_df.csv"
DATA_DIR = "PATH/TO/data"
RESULTS_DIR = "eda/results"
```

Замените на свои либо передайте через флаги `--csv`, `--data-dir`, `--output`.

## Запуск

Первый запуск (парсит все PDB/CIF — самый долгий шаг, далее кэшируется в pickle):

```bash
python eda/run_eda.py --workers 8
```

Повторные запуски (использует кэш парсинга):

```bash
python eda/run_eda.py --blocks BCDE --workers 8
```

Для запуска отдельного блока:

```bash
python eda/run_eda.py --blocks E
```

## Что попадает в `results/`

- `block_a_inventory.csv`, `block_a_summary.json`, `block_a_distributions.png` — статистика по файлам и парсингу
- `block_b_pairs.csv`, `block_b_summary.json`, `block_b_distributions.png` — coverage, kabsch RMSD, Δresolution для каждой пары
- `block_c_good_good.csv`, `block_c_uniprot_summary.csv`, `block_c_good_good.png`, `block_c_summary.json` — амбивалентность таргета (good-good RMSD)
- `block_d_train_pairs.csv`, `block_d_val_pairs.csv`, `block_d_split.json`, `block_d_distributions.png` — split-по-uniprot, KS-тесты на распределения
- `block_e_baseline.json`, `block_e_baseline.png` — identity-baseline RMSD: модель **обязана** опуститься ниже median, иначе не учится
- `_parse_cache.pkl` — кэш распарсенных структур (можно удалить, тогда Block A пройдёт заново)

## Эффективность

- Парсинг файлов и попарные RMSD выполняются в `ProcessPoolExecutor` параллельно.
- Каждая структура парсится один раз и хранится в кэше.
- Все попарные сравнения работают только над общими атомами (хеш-пересечение, без полного pairwise).
- На датасете ~300 файлов с ~50k атомов в среднем все 5 блоков выполняются за минуты на 8 ядрах.

## Чтение результатов — короткая шпаргалка

| Метрика | Норма | Тревога |
|---|---|---|
| `block_a.failed` | 0–5% | >5% |
| `block_b.low_coverage_<0.6` | <10% | >20% |
| `block_b.high_rmsd_>5A` | <15% | >30% |
| `block_c.ambiguous_>3A` | <30% uniprot_id | >50% |
| `block_d.uniprot_id_overlap` | **0** | >0 (баг split-а) |
| `block_d.ks_tests.*.p_value` | >0.05 | <0.01 (distribution shift) |
| `block_e.identity_rmsd_median` | — | floor, модель должна опуститься ниже |
