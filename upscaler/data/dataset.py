from __future__ import annotations
import os
import logging
from pathlib import Path

import math
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from upscaler.data.align import align_structures, kabsch_superimpose
from upscaler.model.geometric import frames_from_bb_coords


logger = logging.getLogger(__name__)


BATCH_SIZE = 4
PAD_TOKEN = 0


def _get_residue_data(coords, atom_names, res_names, res_seqs):
    residues = {}
    last_res_key = None
    for i in range(len(atom_names)):
        res_key = (res_names[i], res_seqs[i])
        if res_key != last_res_key:
            res_id = len(residues)
            residues[res_id] = {'atoms': {}, 'indices': []}
            last_res_key = res_key
        residues[res_id]['atoms'][atom_names[i]] = coords[i]
        residues[res_id]['indices'].append(i)

    bb_coords_list = []
    res_map = torch.full((len(coords),), -1, dtype=torch.long)
    atom_mask = torch.zeros(len(coords), dtype=torch.bool)  # NEW: маска валидных атомов
    
    new_res_idx = 0
    for res_id in sorted(residues.keys()):
        res = residues[res_id]
        if 'N' in res['atoms'] and 'CA' in res['atoms'] and 'C' in res['atoms']:
            n_coord = res['atoms']['N']
            ca_coord = res['atoms']['CA']
            c_coord = res['atoms']['C']
            bb_coords_list.append(np.stack([n_coord, ca_coord, c_coord]))
            
            for atom_original_index in res['indices']:
                res_map[atom_original_index] = new_res_idx
                atom_mask[atom_original_index] = True  # NEW: отмечаем валидные атомы
            new_res_idx += 1

    if not bb_coords_list:
        return None, None, None

    final_bb_coords = torch.from_numpy(np.stack(bb_coords_list, axis=0)).float()
    return final_bb_coords, res_map, atom_mask  # NEW: возвращаем маску


class ProteinUpscalingDataset(Dataset):
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
        self.atom_name_map = self._build_map(
            ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 
             'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', 'ND1', 'ND2', 
             'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OD1', 'OD2', 'OE1', 'OE2', 
             'OG', 'OG1', 'OH', 'SD', 'SE', 'SG']
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

        self.good_pair_indices: list[int] = []
        self._lengths_by_idx: dict[int, int] = {}
        self._rmsd_by_idx: dict[int, float] = {}

        self.good_pair_indices = self._load_or_create_good_indices()
        logger.info(f"Dataset initialized with {len(self.good_pair_indices)} good pairs out of {len(self.all_pairs)} total pairs.")

    @staticmethod
    def _build_map(items: list[str]) -> dict[str, int]:
        return {'PAD': PAD_TOKEN, **{k: idx + 1 for idx, k in enumerate(items)}}

    def _create_pairs(
        self,
        grouped,
        resolution_good: float,
        resolution_bad: float,
    ) -> list[tuple[pd.Series, pd.Series]]:
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
        try:
            good_row, bad_row = self.all_pairs[idx]
            good_path = self._find_structure_file(good_row['pdb'])
            bad_path = self._find_structure_file(bad_row['pdb'])
            low_coords, high_coords, _, _, _, _ = align_structures(str(bad_path), str(good_path))
            if low_coords.shape[0] == 0:
                return False, 0, float('inf')
            _, rmsd, _, _ = kabsch_superimpose(low_coords, high_coords)
            return True, int(low_coords.shape[0]), float(rmsd)
        except (FileNotFoundError, ValueError) as e:
            logger.debug(f"Pair {idx} is not alignable: {e}")
            return False, 0, float('inf')
        except Exception as e:
            logger.warning(f"Unexpected error checking pair {idx}: {e}")
            return False, 0, float('inf')

    def _load_or_create_good_indices(self) -> list[int]:
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
        good_indices: list[int] = []
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
        actual_idx = self.good_pair_indices[dataset_index]
        return int(self._lengths_by_idx.get(actual_idx, 0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        actual_idx = self.good_pair_indices[idx]
        good_row, bad_row = self.all_pairs[actual_idx]

        try:
            good_path = self._find_structure_file(good_row['pdb'])
            bad_path = self._find_structure_file(bad_row['pdb'])
        except FileNotFoundError as e:
            logger.error(f"File not found for pair {actual_idx}: {e}")
            raise e

        try:
            low_coords, high_coords, atom_elements, atom_names, res_names, res_seqs = align_structures(
                str(bad_path), str(good_path)
            )
        except ValueError as e:
            logger.error(f"Alignment failed for pair {actual_idx} ({bad_row['pdb']}, {good_row['pdb']}): {e}")
            raise e

        if low_coords.shape[0] > 0:
            aligned_high_coords, _, _, _ = kabsch_superimpose(low_coords, high_coords)
            high_coords = aligned_high_coords

        bb_coords_bad, res_map_bad, atom_mask_bad = _get_residue_data(
            low_coords, atom_names, res_names, res_seqs
        )
        bb_coords_good, res_map_good, atom_mask_good = _get_residue_data(
            high_coords, atom_names, res_names, res_seqs
        )

        if bb_coords_bad is None or bb_coords_good is None:
            return self.__getitem__((idx + 1) % len(self))

        if not torch.equal(atom_mask_bad, atom_mask_good):
            logger.warning(f"Atom masks mismatch for pair {actual_idx}, trying next sample")
            return self.__getitem__((idx + 1) % len(self))

        rots_bad, trans_bad = frames_from_bb_coords(
            bb_coords_bad[:,0], bb_coords_bad[:,1], bb_coords_bad[:,2]
        )
        rots_good, trans_good = frames_from_bb_coords(
            bb_coords_good[:,0], bb_coords_good[:,1], bb_coords_good[:,2]
        )

        atom_ids = [self.atom_type_map.get(a, 0) for a in atom_elements]
        res_ids  = [self.residue_type_map.get(r, 0) for r in res_names]
        atom_name_ids = [self.atom_name_map.get(name, 0) for name in atom_names]

        valid_indices = atom_mask_bad.nonzero(as_tuple=True)[0]
        
        return {
            "coords_bad": torch.from_numpy(low_coords[valid_indices]).float(),
            "coords_good": torch.from_numpy(high_coords[valid_indices]).float(),
            "rots_bad": rots_bad,
            "trans_bad": trans_bad,
            "rots_good": rots_good,
            "trans_good": trans_good,
            "res_map": res_map_bad[valid_indices],
            "atom_names": torch.tensor([atom_name_ids[i] for i in valid_indices], dtype=torch.long),
            "atom_types": torch.tensor([atom_ids[i] for i in valid_indices], dtype=torch.long),
            "residue_types": torch.tensor([res_ids[i] for i in valid_indices], dtype=torch.long),
            "length": torch.tensor(len(valid_indices), dtype=torch.long),
            "num_residues": torch.tensor(bb_coords_bad.shape[0], dtype=torch.long),
        }


def collate_batch(batch):
    keys = ['coords_bad', 'coords_good', 'atom_names', 'atom_types', 'residue_types', 'res_map']
    out = {}
    
    # Padding для атомных данных
    for k in keys:
        out[k] = pad_sequence([b[k] for b in batch], batch_first=True, padding_value=0)
    
    # Padding для фреймов остатков
    out['rots_bad'] = pad_sequence([b['rots_bad'] for b in batch], batch_first=True, padding_value=0)
    out['trans_bad'] = pad_sequence([b['trans_bad'] for b in batch], batch_first=True, padding_value=0)
    out['rots_good'] = pad_sequence([b['rots_good'] for b in batch], batch_first=True, padding_value=0)
    out['trans_good'] = pad_sequence([b['trans_good'] for b in batch], batch_first=True, padding_value=0)
    
    # Длины и маски
    out['lengths'] = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    max_len = out['coords_bad'].shape[1]
    arange = torch.arange(max_len).unsqueeze(0)
    out['mask'] = (arange < out['lengths'].unsqueeze(1))
    
    num_residues = torch.tensor([b['num_residues'] for b in batch], dtype=torch.long)
    max_res_len = out['rots_bad'].shape[1]
    res_arange = torch.arange(max_res_len).unsqueeze(0)
    out['res_mask'] = (res_arange < num_residues.unsqueeze(1))
    
    return out


class BucketBatchSampler(Sampler):
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
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        total = 0
        for b in self.buckets:
            total += math.ceil(len(b) / self.batch_size)
        return total


class CurriculumSampler(Sampler):
    def __init__(self, dataset: ProteinUpscalingDataset, batch_size: int, schedule_fn=None, shuffle_within_bucket: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_within_bucket = shuffle_within_bucket

        self._idxs = list(range(len(dataset)))
        
        def _rmsd_for_dataset_idx(i):
            actual_idx = dataset.good_pair_indices[i]
            return dataset._rmsd_by_idx.get(actual_idx, float('inf'))
        
        if not dataset._rmsd_by_idx:
            self.sorted_by_difficulty = list(range(len(dataset)))
        else:
            self.sorted_by_difficulty = sorted(self._idxs, key=_rmsd_for_dataset_idx)

        self.schedule_fn = schedule_fn or (lambda epoch, max_rmsd: max_rmsd * min(1.0, (epoch + 1) / 20.0))

    def __iter__(self):
        raise RuntimeError("CurriculumSampler requires calling get_epoch_iterator(epoch).")

    def get_epoch_iterator(self, epoch: int):
        max_rmsd_all = max(self.dataset._rmsd_by_idx.values()) if self.dataset._rmsd_by_idx else 0.0
        max_rmsd_all = max_rmsd_all or 1.0 
        thr = self.schedule_fn(epoch, max_rmsd_all)
        allowed = [i for i in self.sorted_by_difficulty 
                   if self.dataset._rmsd_by_idx.get(self.dataset.good_pair_indices[i], float('inf')) <= thr]
        if not allowed:
            allowed = self.sorted_by_difficulty[:max(1, len(self.sorted_by_difficulty)//100)]
        if self.shuffle_within_bucket:
            random.shuffle(allowed)
        for i in range(0, len(allowed), self.batch_size):
            yield allowed[i:i+self.batch_size]

    def __len__(self):
        return math.ceil(len(self.sorted_by_difficulty) / self.batch_size)
