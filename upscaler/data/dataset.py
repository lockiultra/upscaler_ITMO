from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import math
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from upscaler.data.align import align_structures, kabsch_superimpose


logger = logging.getLogger(__name__)


BATCH_SIZE = 4
PAD_TOKEN = 0


class ProteinUpscalingDataset(Dataset):
    """Датасет пар «плохое разрешение → хорошее» для белков.

    Изменения:
      - добавлен параметр max_atoms: ограничивает число атомов в выборке (subsample если слишком много)
      - prefilter_cache теперь хранит строки `idx,length,rmsd`
      - внутренние словари _lengths_by_idx, _rmsd_by_idx
      - __getitem__ возвращает поле 'length'
    """
    def __init__(
        self,
        csv_file: str | os.PathLike,
        data_folder: str | os.PathLike,
        resolution_good: float = 2.0,
        resolution_bad: float = 3.5,
        prefilter_cache: str | None = None,
        max_atoms: int | None = 3000,
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
        self.max_atoms = max_atoms

        # caches filled by _load_or_create_good_indices
        self.good_pair_indices: List[int] = []
        self._lengths_by_idx: Dict[int, int] = {}
        self._rmsd_by_idx: Dict[int, float] = {}

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
        possible_suffixes = ['.pdb', '.cif']
        for suffix in possible_suffixes:
            path = self.data_folder / f"{pdb_id}{suffix}"
            if path.exists():
                return path
        tried_paths = [str(self.data_folder / f"{pdb_id}{s}") for s in possible_suffixes]
        raise FileNotFoundError(f"Structure file for {pdb_id} not found. Tried: {tried_paths}")

    def _is_pair_alignable(self, idx: int) -> tuple[bool, int, float]:
        """Проверяет, можно ли выровнять пару с заданным индексом.

        Возвращает: (is_alignable, length, rmsd)
        rmsd = float('inf') при ошибке.
        """
        try:
            good_row, bad_row = self.all_pairs[idx]
            good_path = self._find_structure_file(good_row['pdb'])
            bad_path = self._find_structure_file(bad_row['pdb'])
            low_coords, high_coords, atom_strs, res_strs = align_structures(str(bad_path), str(good_path))
            if low_coords.shape[0] == 0:
                return False, 0, float('inf')
            # Вычисляем RMSD после Kabsch (superimpose high -> low)
            _, rmsd, _, _ = kabsch_superimpose(low_coords, high_coords)
            return True, int(low_coords.shape[0]), float(rmsd)
        except (FileNotFoundError, ValueError) as e:
            logger.debug(f"Pair {idx} is not alignable: {e}")
            return False, 0, float('inf')
        except Exception as e:
            logger.warning(f"Unexpected error checking pair {idx}: {e}")
            return False, 0, float('inf')

    def _load_or_create_good_indices(self) -> List[int]:
        """Загружает индексы хороших пар из кэша или создаёт их.

        Кеш в файле хранит строки вида: idx,length,rmsd
        """
        if self.prefilter_cache and self.prefilter_cache.exists():
            logger.info(f"Loading good pair indices from cache: {self.prefilter_cache}")
            try:
                indices = []
                with open(self.prefilter_cache, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(',') if p.strip()]
                        if not parts:
                            continue
                        idx = int(parts[0])
                        indices.append(idx)
                        if len(parts) > 1:
                            try:
                                length = int(parts[1])
                                self._lengths_by_idx[idx] = length
                            except Exception:
                                pass
                        if len(parts) > 2:
                            try:
                                rmsd = float(parts[2])
                                self._rmsd_by_idx[idx] = rmsd
                            except Exception:
                                pass
                logger.info(f"Loaded {len(indices)} indices from cache.")
                return indices
            except Exception as e:
                logger.error(f"Failed to load cache, regenerating: {e}")

        logger.info("Generating list of good pair indices...")
        good_indices: List[int] = []
        total_pairs = len(self.all_pairs)

        for i in tqdm(range(total_pairs)):
            ok, length, rmsd = self._is_pair_alignable(i)
            if ok:
                good_indices.append(i)
                self._lengths_by_idx[i] = int(length)
                self._rmsd_by_idx[i] = float(rmsd)

        if self.prefilter_cache:
            logger.info(f"Saving good pair indices to cache: {self.prefilter_cache}")
            try:
                with open(self.prefilter_cache, 'w') as f:
                    for idx in good_indices:
                        length = self._lengths_by_idx.get(idx, 0)
                        rmsd = self._rmsd_by_idx.get(idx, float('inf'))
                        f.write(f"{idx},{length},{rmsd:.4f}\n")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

        return good_indices

    def __len__(self) -> int:
        return len(self.good_pair_indices)

    def get_num_atoms(self, dataset_index: int) -> int:
        """Возвращает число атомов для образца в датасете (по позиции dataset_index)."""
        actual_idx = self.good_pair_indices[dataset_index]
        return int(self._lengths_by_idx.get(actual_idx, 0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

        N = low_coords.shape[0]
        # ограничение по количеству атомов: случайная подвыборка если очень много
        if self.max_atoms is not None and N > self.max_atoms:
            keep = np.random.choice(N, size=self.max_atoms, replace=False)
            keep.sort()
            low_coords = low_coords[keep]
            high_coords = high_coords[keep]
            atom_strs = [atom_strs[i] for i in keep]
            res_strs = [res_strs[i] for i in keep]
            N = self.max_atoms

        atom_ids = [self.atom_type_map.get(a, 0) for a in atom_strs]
        res_ids  = [self.residue_type_map.get(r, 0) for r in res_strs]

        return {
            "coords_bad":   torch.from_numpy(low_coords).float(),
            "coords_good":  torch.from_numpy(high_coords).float(),
            "atom_types":   torch.tensor(atom_ids, dtype=torch.long),
            "residue_types": torch.tensor(res_ids, dtype=torch.long),
            "length": torch.tensor(N, dtype=torch.long),
        }


def collate_batch(batch):
    """Pad variable-length samples to the longest in the batch and produce mask/lengths."""
    coords_bad_list  = [b['coords_bad'] for b in batch]
    coords_good_list = [b['coords_good'] for b in batch]
    atom_list = [b['atom_types'] for b in batch]
    res_list = [b['residue_types'] for b in batch]
    lengths = torch.tensor([int(b.get('length', t.shape[0])) for b, t in zip(batch, coords_bad_list)], dtype=torch.long)

    coords_bad = pad_sequence(coords_bad_list, batch_first=True)   # [B, Lmax, 3]
    coords_good = pad_sequence(coords_good_list, batch_first=True) # [B, Lmax, 3]
    atom_types = pad_sequence(atom_list, batch_first=True, padding_value=0)
    residue_types = pad_sequence(res_list, batch_first=True, padding_value=0)

    max_len = coords_bad.shape[1]
    arange = torch.arange(max_len).unsqueeze(0)  # [1, Lmax]
    mask = (arange < lengths.unsqueeze(1))  # [B, Lmax] bool

    return {
        'coords_bad': coords_bad,
        'coords_good': coords_good,
        'atom_types': atom_types,
        'residue_types': residue_types,
        'lengths': lengths,
        'mask': mask,
    }


# ----------------- BucketBatchSampler -----------------
class BucketBatchSampler(Sampler):
    """Группирует индексы по длине и формирует батчи, чтобы минимизировать padding внутри батча."""
    def __init__(self, dataset: ProteinUpscalingDataset, batch_size: int, num_buckets: int = 20, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle

        lengths = [dataset.get_num_atoms(i) for i in range(len(dataset))]
        idxs = list(range(len(dataset)))
        sorted_by_len = sorted(idxs, key=lambda i: lengths[i])
        bucket_size = max(1, math.ceil(len(sorted_by_len) / num_buckets))
        self.buckets = [sorted_by_len[i:i+bucket_size] for i in range(0, len(sorted_by_len), bucket_size)]

    def __iter__(self):
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i+self.batch_size]

    def __len__(self):
        total = 0
        for b in self.buckets:
            total += math.ceil(len(b) / self.batch_size)
        return total


# ----------------- CurriculumSampler -----------------
class CurriculumSampler(Sampler):
    """
    Простейшая реализация curriculum sampler по RMSD: на ранних эпохах выдаёт только "простиые" примеры
    (малый rmsd). Метод get_epoch_iterator(epoch) возвращает последовательность батч-индексов.
    """
    def __init__(self, dataset: ProteinUpscalingDataset, batch_size: int, schedule_fn=None, shuffle_within_bucket: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_within_bucket = shuffle_within_bucket

        # индексы внутри датасета (0..len-1)
        self._idxs = list(range(len(dataset)))
        # сортируем по сложности (rmsd)
        def _rmsd_for_dataset_idx(i):
            actual_idx = dataset.good_pair_indices[i]
            return dataset._rmsd_by_idx.get(actual_idx, float('inf'))
        self.sorted_by_difficulty = sorted(self._idxs, key=_rmsd_for_dataset_idx)

        # schedule_fn(epoch, max_rmsd) -> threshold
        self.schedule_fn = schedule_fn or (lambda epoch, max_rmsd: max_rmsd * min(1.0, (epoch + 1) / 20.0))

    def __iter__(self):
        raise RuntimeError("CurriculumSampler requires calling get_epoch_iterator(epoch).")

    def get_epoch_iterator(self, epoch: int):
        max_rmsd_all = max(self.dataset._rmsd_by_idx.values()) if self.dataset._rmsd_by_idx else 0.0
        thr = self.schedule_fn(epoch, max_rmsd_all)
        allowed = [i for i in self.sorted_by_difficulty if self.dataset._rmsd_by_idx.get(self.dataset.good_pair_indices[i], float('inf')) <= thr]
        if not allowed:
            allowed = self.sorted_by_difficulty[:max(1, len(self.sorted_by_difficulty)//100)]
        if self.shuffle_within_bucket:
            random.shuffle(allowed)
        for i in range(0, len(allowed), self.batch_size):
            yield allowed[i:i+self.batch_size]

    def __len__(self):
        return math.ceil(len(self.sorted_by_difficulty) / self.batch_size)


# ----------------- main (debug) -----------------
if __name__ == '__main__':
    from upscaler.config import CSV_FILE, DATA_FOLDER
    csv_file = Path(CSV_FILE)
    data_folder = Path(DATA_FOLDER)

    if not csv_file.exists():
        print(f"CSV file not found: {csv_file}")
        raise SystemExit(1)
    if not data_folder.exists():
        print(f"Data folder not found: {data_folder}")
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO)
    cache_file = data_folder / "good_pairs_cache.txt"
    dataset = ProteinUpscalingDataset(csv_file, data_folder, prefilter_cache=cache_file)
    print(f"Dataset loaded successfully. Number of good pairs: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty. Check your CSV file and data folder.")
        raise SystemExit(1)

    sampler = BucketBatchSampler(dataset, batch_size=BATCH_SIZE, num_buckets=20)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=list(sampler), collate_fn=collate_batch)
    for batch in loader:
        print("First batch shapes:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")
        break
