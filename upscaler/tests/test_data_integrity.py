"""Regression tests for UniProt splits and chain-safe PDB/mmCIF alignment."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFIO, PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
from torch.utils.data import Dataset, Subset
import torch

from quick_train import build_mini_dataloaders
from upscaler.data.align import align_structures, kabsch_superimpose
from upscaler.data.dataset import ProteinUpscalingDataset, _get_residue_data
from upscaler.tests import ComponentTestCase
from upscaler.tests.test_data_components import BACKBONE, ELEMENTS, _residue_atom_coords
from upscaler.tests.test_training_components import _make_dummy_sample
from upscaler.training.cross_validation import ProteinFoldCrossValidation
from upscaler.training.split import split_indices_by_uniprot, uniprot_id_for_index
from upscaler.training.train import build_dataloaders


class GroupedDataset(Dataset):
    def __init__(self, uids):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        return _make_dummy_sample()

    def uniprot_id_for_pair(self, index):
        return self.uids[index]

    def get_num_atoms(self, index):
        return 12


class UniProtSplitTest(ComponentTestCase):
    def assert_proteins_disjoint(self, dataset, train, val):
        train_uids = {uniprot_id_for_index(dataset, i) for i in train} - {None}
        val_uids = {uniprot_id_for_index(dataset, i) for i in val}
        self.assertTrue(train_uids.isdisjoint(val_uids))
        self.assertNotIn(None, val_uids)
        self.assertTrue(set(train).isdisjoint(val))

    def test_grouped_split_is_complete_reproducible_and_keeps_unknowns_in_train(self):
        dataset = GroupedDataset(['A'] * 8 + ['B'] * 2 + ['C'] * 3 + [None])
        train, val = split_indices_by_uniprot(dataset)
        self.assert_proteins_disjoint(dataset, train, val)
        self.assertEqual(sorted(train + val), list(range(len(dataset))))
        self.assertIn(13, train)
        self.assertEqual((train, val), split_indices_by_uniprot(dataset))
        train, val = split_indices_by_uniprot(dataset, val_fraction=0.999)
        self.assertTrue(train and val)
        self.assert_proteins_disjoint(dataset, train, val)

    def test_insufficient_groups_never_fall_back_to_pair_split(self):
        for uids in ([], [None, None], ['A'] * 4, ['A', None]):
            with self.subTest(uids=uids):
                dataset = GroupedDataset(uids)
                self.assertEqual(split_indices_by_uniprot(dataset), (list(range(len(uids))), []))
                with self.assertRaisesRegex(ValueError, 'distinct uniprot_id'):
                    build_dataloaders(dataset, 1, torch.device('cpu'), num_workers=0)
        for fraction in (0, 1, -1, float('nan')):
            with self.assertRaises(ValueError):
                split_indices_by_uniprot(GroupedDataset(['A', 'B']), val_fraction=fraction)

    def test_nested_subsets_preserve_protein_ids(self):
        full = GroupedDataset(['A', 'B', 'A', 'C', 'B', 'C'])
        dataset = Subset(Subset(full, [5, 0, 4, 2, 3, 1]), [0, 1, 2, 3, 4])
        train, val = split_indices_by_uniprot(dataset)
        self.assert_proteins_disjoint(dataset, train, val)
        self.assertEqual(sorted(train + val), list(range(len(dataset))))

    def test_bucket_plain_and_curriculum_loaders_use_same_split(self):
        dataset = GroupedDataset(['A', 'B', 'C'] * 3)
        for bucket, curriculum in ((True, False), (False, False), (True, True)):
            with self.subTest(bucket=bucket, curriculum=curriculum):
                train_loader, val_loader, train, val = build_dataloaders(
                    dataset, 2, torch.device('cpu'), num_workers=0,
                    use_bucket=bucket, use_curriculum=curriculum,
                )
                self.assert_proteins_disjoint(dataset, train.indices, val.indices)
                self.assertEqual(next(iter(val_loader))['coords_bad'].shape[-1], 3)
                if curriculum:
                    self.assertIsNone(train_loader)
                elif bucket:
                    actual_indices = [i for batch in train_loader.batch_sampler for i in batch]
                    self.assertEqual(sorted(actual_indices), train.indices)
                    self.assertEqual(next(iter(train_loader))['coords_bad'].shape[-1], 3)
                else:
                    self.assertEqual(train_loader.dataset.indices, train.indices)

    def test_quick_experiment_splits_before_limiting_pairs(self):
        dataset = GroupedDataset(['A'] * 20 + ['B'] * 20 + ['C'] * 20)
        train, val = build_mini_dataloaders(dataset, max_pairs=6)
        self.assert_proteins_disjoint(dataset, train.dataset.indices, val.dataset.indices)
        self.assertLessEqual(len(train.dataset) + len(val.dataset), 6)
        self.assertTrue(len(train.dataset) and len(val.dataset))
        with self.assertRaises(ValueError):
            build_mini_dataloaders(GroupedDataset(['A'] * 20))
        with self.assertRaises(ValueError):
            build_mini_dataloaders(dataset, max_pairs=1)

    def test_cross_validation_groups_proteins_and_excludes_unknowns_from_validation(self):
        dataset = GroupedDataset(['A'] * 5 + ['B'] * 2 + ['C'] * 4 + [None])
        folds = ProteinFoldCrossValidation(dataset, num_folds=3).create_folds()
        validation_indices = []
        for train, val in folds:
            self.assert_proteins_disjoint(dataset, train.indices, val.indices)
            self.assertIn(11, train.indices)
            validation_indices.extend(val.indices)
        self.assertEqual(sorted(validation_indices), list(range(11)))
        with self.assertRaises(ValueError):
            ProteinFoldCrossValidation(dataset, num_folds=4)
        with self.assertRaises(ValueError):
            ProteinFoldCrossValidation(dataset, num_folds=1)

    def test_dataset_ids_respect_prefilter_mapping_and_missing_values(self):
        dataset = ProteinUpscalingDataset.__new__(ProteinUpscalingDataset)
        dataset.all_pairs = [
            (pd.Series({'uniprot_id': ' A '}), pd.Series({'uniprot_id': 'A'})),
            (pd.Series({'uniprot_id': np.nan}), pd.Series({'uniprot_id': 'B'})),
            (pd.Series({'uniprot_id': ' '}), pd.Series({'uniprot_id': None})),
        ]
        dataset.good_pair_indices = [1, 2, 0]
        self.assertEqual([dataset.uniprot_id_for_pair(i) for i in range(3)], ['B', None, 'A'])

    def test_invalid_pair_does_not_return_next_protein(self):
        dataset = ProteinUpscalingDataset.__new__(ProteinUpscalingDataset)
        dataset.all_pairs = [(pd.Series({'pdb': 'good', 'uniprot_id': uid}),
                              pd.Series({'pdb': 'bad', 'uniprot_id': uid})) for uid in ('A', 'B')]
        dataset.good_pair_indices = [0, 1]
        dataset.seq_identity_min = 0.8
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        arrays = (coords, coords, ['N', 'C', 'C'], ['N', 'CA', 'C'],
                  ['ALA'] * 3, [1] * 3, ['A'] * 3, [''] * 3)
        with patch.object(dataset, '_find_structure_file', return_value=Path('fixture.pdb')), \
                patch('upscaler.data.dataset.align_structures', return_value=arrays) as align, \
                patch('upscaler.data.dataset._get_residue_data', return_value=(None,) * 4):
            with self.assertRaisesRegex(ValueError, 'pair 0'):
                dataset[0]
            self.assertEqual(align.call_count, 1)
        with self.assertRaises(IndexError):
            dataset[2]


def write_multichain_structure(path, chain_ids, shift=None):
    builder = StructureBuilder()
    builder.init_structure('fixture')
    sequence = [('VAL', 1, ''), ('GLY', 2, ''), ('ALA', 3, ''), ('SER', 3, 'A')]
    shift = np.zeros(3) if shift is None else shift
    for model_index in range(2):
        builder.init_model(model_index, serial_num=model_index + 1)
        for chain_index, cid in enumerate(chain_ids):
            builder.init_chain(cid)
            builder.init_seg(' ')
            # Deliberately unsorted records: extraction must use residue numbers.
            for index in (2, 0, 3, 1):
                name, number, icode = sequence[index]
                builder.init_residue(name, ' ', number, icode or ' ')
                for atom_name, xyz in _residue_atom_coords(index).items():
                    offset = np.array([model_index * 100., chain_index * 20., 0.])
                    builder.init_atom(atom_name, (xyz + offset + shift).astype('f'),
                                      0., 1., ' ', atom_name, element=ELEMENTS[atom_name])
    writer = PDBIO() if path.suffix == '.pdb' else MMCIFIO()
    writer.set_structure(builder.get_structure())
    writer.save(str(path))


class ChainAlignmentTest(ComponentTestCase):
    def test_csv_dataset_and_loaders_preserve_chains_and_protein_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for uid in ('A', 'B'):
                for label, resolution, shift in (('good', 1.5, [10., 0., 0.]),
                                                  ('bad', 4.5, [0., 0., 0.])):
                    pdb_id = f'{uid}_{label}'
                    write_multichain_structure(root / (pdb_id + '.pdb'), ('A', 'B'), np.array(shift))
                    rows.append({'uniprot_id': uid, 'pdb': pdb_id, 'resolution': resolution})
            csv_path = root / 'pairs.csv'
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            dataset = ProteinUpscalingDataset(csv_path, root)
            self.assertEqual(len(dataset), 2)
            train_loader, val_loader, train, val = build_dataloaders(
                dataset, 1, torch.device('cpu'), num_workers=0,
            )
            self.assertNotEqual(dataset.uniprot_id_for_pair(train.indices[0]),
                                dataset.uniprot_id_for_pair(val.indices[0]))
            for loader in (train_loader, val_loader):
                batch = next(iter(loader))
                self.assertEqual(batch['coords_bad'].shape, (1, 32, 3))
                self.assertEqual(batch['res_chain'].tolist(), [[0] * 4 + [1] * 4])
                torch.testing.assert_close(batch['coords_bad'], batch['coords_good'], atol=1e-4, rtol=0)

    def test_chains_insertions_numeric_order_and_first_model_in_pdb_and_cif(self):
        for suffixes in (('.pdb', '.pdb'), ('.pdb', '.cif'), ('.mmcif', '.pdb')):
            with self.subTest(suffixes=suffixes), tempfile.TemporaryDirectory() as directory:
                low = Path(directory) / ('low' + suffixes[0])
                high = Path(directory) / ('high' + suffixes[1])
                write_multichain_structure(low, ('A', 'B'))
                write_multichain_structure(high, ('X', 'Y'), np.array([10., 0., 0.]))
                lc, hc, _, names, residues, numbers, chains, icodes = align_structures(str(low), str(high))
                self.assertEqual(lc.shape, (32, 3))
                self.assertEqual(hc.shape, lc.shape)
                self.assertEqual(chains, ['A'] * 16 + ['B'] * 16)
                self.assertEqual(numbers, ([1] * 4 + [2] * 4 + [3] * 8) * 2)
                self.assertEqual(residues[:16], ['VAL'] * 4 + ['GLY'] * 4 + ['ALA'] * 4 + ['SER'] * 4)
                self.assertEqual(icodes[:16], [''] * 12 + ['A'] * 4)
                self.assertEqual(len(set(zip(chains, numbers, icodes, names))), 32)
                self.assertLess(lc[:, 0].max(), 20)  # no second-model coordinates
                np.testing.assert_allclose(hc - lc, np.tile([10., 0., 0.], (32, 1)), atol=1e-5)
                backbone, res_map, atom_mask, res_chain = _get_residue_data(
                    lc, names, residues, numbers, chains, icodes
                )
                self.assertEqual(backbone.shape, (8, 3, 3))
                self.assertEqual(res_chain.tolist(), [0] * 4 + [1] * 4)
                self.assertEqual(res_map.unique().numel(), 8)
                self.assertTrue(atom_mask.all())

    def test_kabsch_aligns_rotation_and_translation(self):
        points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moved = points @ rotation.T + np.array([10., -2., 4.])
        aligned, rmsd, result_rotation, translation = kabsch_superimpose(points, moved)
        np.testing.assert_allclose(aligned, points, atol=1e-6)
        self.assertLess(rmsd, 1e-6)
        np.testing.assert_allclose(result_rotation, rotation.T, atol=1e-6)
        np.testing.assert_allclose(result_rotation @ moved.T + translation[:, None], points.T, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
