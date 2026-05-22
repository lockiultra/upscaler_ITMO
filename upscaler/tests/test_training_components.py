"""Тесты тренировочного пайплайна под новый dict-API."""
import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from upscaler.training.pipeline import TrainingPipeline
from upscaler.training.cross_validation import ProteinFoldCrossValidation
from upscaler.model.upscaler import ProteinUpscaler
from upscaler.model.geometric import frames_from_bb_coords
from upscaler.loss.loss import ProteinUpscalingLoss, ATOM_NAME_MAP
from upscaler.utils.metrics import QualityMetrics


def _make_dummy_sample(n_res: int = 4):
    n_atoms = n_res * 3
    coords_bad = torch.randn(n_atoms, 3)
    coords_good = coords_bad + 0.05 * torch.randn_like(coords_bad)

    bb = coords_bad.view(n_res, 3, 3)
    rots_bad, trans_bad = frames_from_bb_coords(bb[:, 0], bb[:, 1], bb[:, 2])
    bb_g = coords_good.view(n_res, 3, 3)
    rots_good, trans_good = frames_from_bb_coords(bb_g[:, 0], bb_g[:, 1], bb_g[:, 2])

    atom_names = torch.tensor(
        [ATOM_NAME_MAP['N'], ATOM_NAME_MAP['CA'], ATOM_NAME_MAP['C']] * n_res,
        dtype=torch.long,
    )

    return {
        'coords_bad': coords_bad,
        'coords_good': coords_good,
        'rots_bad': rots_bad,
        'trans_bad': trans_bad,
        'rots_good': rots_good,
        'trans_good': trans_good,
        'atom_types': torch.randint(1, 5, (n_atoms,)),
        'residue_types': torch.randint(1, 10, (n_atoms,)),
        'atom_names': atom_names,
        'res_map': torch.arange(n_res).repeat_interleave(3),
        'length': torch.tensor(n_atoms, dtype=torch.long),
        'num_residues': torch.tensor(n_res, dtype=torch.long),
    }


class DummyDataset(Dataset):
    def __init__(self, size: int = 4, n_res: int = 4):
        self.size = size
        self.n_res = n_res

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return _make_dummy_sample(self.n_res)


def _collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    out = {}
    for k in ('coords_bad', 'coords_good', 'atom_types', 'residue_types',
              'atom_names', 'res_map'):
        out[k] = pad_sequence([b[k] for b in batch], batch_first=True, padding_value=0)
    for k in ('rots_bad', 'trans_bad', 'rots_good', 'trans_good'):
        out[k] = pad_sequence([b[k] for b in batch], batch_first=True, padding_value=0)
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    max_len = out['coords_bad'].shape[1]
    out['mask'] = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1))
    n_res = torch.tensor([b['num_residues'] for b in batch], dtype=torch.long)
    max_res = out['rots_bad'].shape[1]
    out['res_mask'] = (torch.arange(max_res).unsqueeze(0) < n_res.unsqueeze(1))
    return out


def test_training_pipeline_train_and_validate():
    print("Testing TrainingPipeline train_epoch + validate...")
    device = torch.device('cpu')
    model = ProteinUpscaler(num_iterations=1)
    loss_fn = ProteinUpscalingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics_calc = QualityMetrics(device=device)

    pipeline = TrainingPipeline(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=1.0,
        metrics_calculator=metrics_calc,
        use_amp=False,
    )

    dataset = DummyDataset(size=2, n_res=4)
    loader = DataLoader(dataset, batch_size=1, collate_fn=_collate)

    train_metrics = pipeline.train_epoch(loader)
    assert isinstance(train_metrics, dict)
    assert 'loss' in train_metrics and train_metrics['loss'] >= 0
    print("train_epoch OK")

    val_metrics = pipeline.validate(loader)
    assert set(val_metrics.keys()) == {'val_rmsd', 'val_lddt', 'val_clash'}
    for v in val_metrics.values():
        assert isinstance(v, float)
        assert v >= 0
    print("validate OK")


def test_protein_fold_cross_validation():
    print("Testing ProteinFoldCrossValidation...")
    dataset = DummyDataset(size=20)
    cv = ProteinFoldCrossValidation(dataset, num_folds=5, seed=42)
    folds = cv.create_folds()
    assert len(folds) == 5
    for train_subset, val_subset in folds:
        assert isinstance(train_subset, Subset)
        assert isinstance(val_subset, Subset)
        assert set(train_subset.indices).isdisjoint(val_subset.indices)
    print("Cross-validation test passed.")


if __name__ == "__main__":
    test_training_pipeline_train_and_validate()
    test_protein_fold_cross_validation()
    print("All training component tests passed!")
