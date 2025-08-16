from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from upscaler.data.align import align_structures


logger = logging.getLogger(__name__)


BATCH_SIZE = 4
PAD_TOKEN = 0


class ProteinUpscalingDataset(Dataset):
    """Датасет пар «плохое разрешение → хорошее» для белков."""
    def __init__(
        self,
        csv_file: str | os.PathLike,
        data_folder: str | os.PathLike,
        resolution_good: float = 2.0,
        resolution_bad: float = 3.5,
        prefilter_cache: str | None = None,
    ) -> None:
        self.data_folder = Path(data_folder)
        self.atom_type_map = self._build_map(
            ['C', 'N', 'O', 'S', 'H', 'P', 'SE',
             'FE', 'ZN', 'MG', 'CA', 'MN', 'CU', 'NI', 'CO',
             'CL', 'BR', 'I', 'F']
        )
        self.residue_type_map = self._build_map(
            ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
             'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        )
        df = pd.read_csv(csv_file)
        grouped = df.groupby('uniprot_id')
        self.all_pairs = self._create_pairs(grouped, resolution_good, resolution_bad)
        
        self.prefilter_cache = Path(prefilter_cache) if prefilter_cache else None
        self.good_pair_indices = self._load_or_create_good_indices()
        logger.info(f"Dataset initialized with {len(self.good_pair_indices)} good pairs out of {len(self.all_pairs)} total pairs.")

    @staticmethod
    def _build_map(items: List[str]) -> Dict[str, int]:
        return {'PAD': PAD_TOKEN, **{k: idx + 1 for idx, k in enumerate(items)}}

    def _create_pairs(
        self,
        grouped,
        resolution_good: float,
        resolution_bad: float,
    ) -> List[Tuple[pd.Series, pd.Series]]:
        pairs = []
        for _, group in grouped:
            good = group[group['resolution'] < resolution_good]
            bad  = group[group['resolution'] > resolution_bad]
            pairs.extend((g, b) for _, g in good.iterrows() for _, b in bad.iterrows())
        return sorted(pairs, key=lambda x: x[1]['resolution'])

    def _find_structure_file(self, pdb_id: str) -> Path:
        """
        Находит файл структуры по ID, проверяя разные расширения.
        """
        possible_suffixes = ['.pdb', '.cif']
        for suffix in possible_suffixes:
            path = self.data_folder / f"{pdb_id}{suffix}"
            if path.exists():
                return path
        tried_paths = [str(self.data_folder / f"{pdb_id}{s}") for s in possible_suffixes]
        raise FileNotFoundError(f"Structure file for {pdb_id} not found. Tried: {tried_paths}")

    def _is_pair_alignable(self, idx: int) -> bool:
        """Проверяет, можно ли выровнять пару с заданным индексом."""
        try:
            good_row, bad_row = self.all_pairs[idx]
            good_path = self._find_structure_file(good_row['pdb'])
            bad_path = self._find_structure_file(bad_row['pdb'])
            align_structures(str(bad_path), str(good_path))
            return True
        except (FileNotFoundError, ValueError) as e:
            logger.debug(f"Pair {idx} is not alignable: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking pair {idx}: {e}")
            return False

    def _load_or_create_good_indices(self) -> List[int]:
        """Загружает индексы хороших пар из кэша или создает их."""
        if self.prefilter_cache and self.prefilter_cache.exists():
            logger.info(f"Loading good pair indices from cache: {self.prefilter_cache}")
            try:
                # Загружаем список индексов из текстового файла
                with open(self.prefilter_cache, 'r') as f:
                    indices = [int(line.strip()) for line in f if line.strip().isdigit()]
                logger.info(f"Loaded {len(indices)} indices from cache.")
                return indices
            except Exception as e:
                logger.error(f"Failed to load cache, regenerating: {e}")

        logger.info("Generating list of good pair indices...")
        good_indices = []
        total_pairs = len(self.all_pairs)
        
        # report_every = max(1, total_pairs // 20) # Отчет каждые 5%
        
        for i in tqdm(range(total_pairs)):
            if self._is_pair_alignable(i):
                good_indices.append(i)
            
            # if (i + 1) % report_every == 0 or i == total_pairs - 1:
            #     logger.info(f"Processed {i+1}/{total_pairs} pairs. Good pairs so far: {len(good_indices)}")

        if self.prefilter_cache:
            logger.info(f"Saving good pair indices to cache: {self.prefilter_cache}")
            try:
                with open(self.prefilter_cache, 'w') as f:
                    for idx in good_indices:
                        f.write(f"{idx}\n")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
        
        return good_indices

    def __len__(self) -> int:
        return len(self.good_pair_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Получаем реальный индекс из списка хороших индексов
        actual_idx = self.good_pair_indices[idx]
        good_row, bad_row = self.all_pairs[actual_idx]
        
        try:
            good_path = self._find_structure_file(good_row['pdb'])
            bad_path = self._find_structure_file(bad_row['pdb'])
        except FileNotFoundError as e:
            logger.error(f"File not found for pair {actual_idx}: {e}")
            raise e

        try:
            low_coords, high_coords, atom_strs, res_strs = align_structures(
                str(bad_path), str(good_path)
            )
        except ValueError as e:
            logger.error(f"Alignment failed for pair {actual_idx} ({bad_row['pdb']}, {good_row['pdb']}): {e}")
            raise e

        atom_ids = [self.atom_type_map.get(a, 0) for a in atom_strs] # 0 для неизвестных
        res_ids  = [self.residue_type_map.get(r, 0) for r in res_strs] # 0 для неизвестных

        return {
            "coords_bad":   torch.from_numpy(low_coords),
            "coords_good":  torch.from_numpy(high_coords),
            "atom_types":   torch.tensor(atom_ids, dtype=torch.long),
            "residue_types": torch.tensor(res_ids, dtype=torch.long),
        }


def collate_batch(batch):
    """Pad variable-length samples to the longest in the batch."""
    return {
        'coords_bad':   pad_sequence([b['coords_bad'] for b in batch], batch_first=True),
        'coords_good':  pad_sequence([b['coords_good'] for b in batch], batch_first=True),
        'atom_types':   pad_sequence([b['atom_types'] for b in batch],
                                     batch_first=True, padding_value=0),
        'residue_types': pad_sequence([b['residue_types'] for b in batch],
                                      batch_first=True, padding_value=0),
    }

def main() -> None:
    from upscaler.config import CSV_FILE, DATA_FOLDER
    csv_file = Path(CSV_FILE)
    data_folder = Path(DATA_FOLDER)
    
    if not csv_file.exists():
        print(f"CSV file not found: {csv_file}")
        return
    if not data_folder.exists():
        print(f"Data folder not found: {data_folder}")
        return

    try:
        logging.basicConfig(level=logging.INFO)
        cache_file = data_folder / "good_pairs_cache.txt"
        dataset = ProteinUpscalingDataset(csv_file, data_folder, prefilter_cache=cache_file)
        print(f"Dataset loaded successfully. Number of good pairs: {len(dataset)}")
        
        if len(dataset) == 0:
            print("Dataset is empty. Check your CSV file and data folder.")
            return

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
        )
        for batch in loader:
            print("First batch shapes:")
            for k, v in batch.items():
                print(f"  {k}: {v.shape}")
            break
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
