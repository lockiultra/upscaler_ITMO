import torch


# CSV_FILE = '/Users/lockiultra/Desktop/prog/Upscaler/pdb_df.csv'
# DATA_FOLDER = '/Users/lockiultra/Desktop/prog/Upscaler/data'

CSV_FILE = '/Users/lockiultra/Desktop/prog/upscaler_ITMO/mini_data/pdb_df.csv'
DATA_FOLDER = '/Users/lockiultra/Desktop/prog/upscaler_ITMO/mini_data'

RESOLUTION_GOOD = 2.0
RESOLUTION_BAD = 3.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
