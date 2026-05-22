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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
