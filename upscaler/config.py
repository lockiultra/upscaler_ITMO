import os

import torch



CSV_FILE = os.environ.get(
    "UPSCALER_CSV",
    os.path.join(os.getcwd(), "pdb_df.csv"),
)
DATA_FOLDER = os.environ.get(
    "UPSCALER_DATA_DIR",
    os.path.join(os.getcwd(), "data"),
)

RESOLUTION_GOOD = float(os.environ.get("UPSCALER_RES_GOOD", 2.0))
RESOLUTION_BAD = float(os.environ.get("UPSCALER_RES_BAD", 3.5))

# --- Фильтрация пар по качеству supervision (см. data/dataset.py) ---
# Порог identity при sequence-выравнивании цепей bad↔good.
SEQ_IDENTITY_MIN = float(os.environ.get("UPSCALER_SEQ_IDENTITY_MIN", 0.8))
# Минимальная доля совпавших остатков относительно меньшей структуры.
COVERAGE_MIN = float(os.environ.get("UPSCALER_COVERAGE_MIN", 0.6))
# Максимальный допустимый post-Kabsch RMSD пары (Å). Выше — пара считается
# несовместимой конформацией и в обучение не берётся.
PAIR_RMSD_MAX = float(os.environ.get("UPSCALER_PAIR_RMSD_MAX", 15.0))
# Кап на число атомов в образце (защита от OOM на гигантских комплексах).
MAX_ATOMS = int(os.environ.get("UPSCALER_MAX_ATOMS", 3000))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
