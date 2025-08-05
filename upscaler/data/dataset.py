from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from upscaler.data.align import align_structures


# Константы
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
        self.pairs = self._create_pairs(grouped, resolution_good, resolution_bad)

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
        # Curriculum: сортировка по resolution «плохой» структуры
        return sorted(pairs, key=lambda x: x[1]['resolution'])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        good_row, bad_row = self.pairs[idx]

        good_path = self.data_folder / f"{good_row['pdb']}.pdb"
        bad_path  = self.data_folder / f"{bad_row['pdb']}.pdb"

        # good = self._load_structure(good_path)
        # bad  = self._load_structure(bad_path)

        low_coords, high_coords, atom_strs, res_strs = align_structures(
            str(bad_path), str(good_path)
        )

        atom_ids = [self.atom_type_map[a] for a in atom_strs]
        res_ids  = [self.residue_type_map[r] for r in res_strs]

        return {
            "coords_bad":   torch.from_numpy(low_coords),
            "coords_good":  torch.from_numpy(high_coords),
            "atom_types":   torch.tensor(atom_ids, dtype=torch.long),
            "residue_types": torch.tensor(res_ids, dtype=torch.long),
        }

    def _load_structure(self, path: Path) -> Dict[str, torch.Tensor]:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('s', path)

        coords, atom_types, residue_types = [], [], []
        for atom in structure.get_atoms():
            parent = atom.get_parent()
            if parent.get_id()[0] != ' ':
                continue  # skip hetatm/water

            elem = atom.element.strip().upper()
            res  = parent.resname

            if elem not in self.atom_type_map:
                continue
            atom_idx = self.atom_type_map[elem]
            res_idx  = self.residue_type_map.get(res, len(self.residue_type_map))

            coords.append(atom.coord)
            atom_types.append(atom_idx)
            residue_types.append(res_idx)

        return {
            'coords':       torch.tensor(np.array(coords, dtype=np.float32)),
            'atom_types':   torch.tensor(atom_types, dtype=torch.long),
            'residue_types': torch.tensor(residue_types, dtype=torch.long),
        }


def collate_batch(batch):
    return {
        'coords_bad':   pad_sequence([b['coords_bad'] for b in batch], batch_first=True),
        'coords_good':  pad_sequence([b['coords_good'] for b in batch], batch_first=True),
        'atom_types':   pad_sequence([b['atom_types'] for b in batch],
                                     batch_first=True, padding_value=0),
        'residue_types': pad_sequence([b['residue_types'] for b in batch],
                                      batch_first=True, padding_value=0),
    }


def main() -> None:
    csv_file = Path('/Users/lockiultra/Desktop/prog/Upscaler/pdb_df.csv')
    data_folder = Path('/Users/lockiultra/Desktop/prog/Upscaler/data')

    dataset = ProteinUpscalingDataset(csv_file, data_folder)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    for batch in loader:
        for k, v in batch.items():
            print(f"{k}: {v.shape}")
        break


if __name__ == '__main__':
    main()
