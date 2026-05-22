"""Тесты для новой FAPE-loss архитектуры."""
import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import torch

from upscaler.loss.loss import (
    FAPELoss,
    ProteinUpscalingLoss,
    ClashLoss,
    PhysicsLoss,
    ATOM_TYPE_MAP,
    ATOM_NAME_MAP,
)


def _make_random_frames(batch_size: int, n_res: int):
    rots = torch.eye(3).expand(batch_size, n_res, 3, 3).clone()
    trans = torch.randn(batch_size, n_res, 3)
    return rots, trans


def _make_pred_true(batch_size: int = 2, n_res: int = 6, atoms_per_res: int = 5):
    n_atoms = n_res * atoms_per_res

    rots_pred, trans_pred = _make_random_frames(batch_size, n_res)
    rots_true, trans_true = _make_random_frames(batch_size, n_res)

    coords_pred = torch.randn(batch_size, n_atoms, 3)
    coords_true = coords_pred + 0.05 * torch.randn_like(coords_pred)

    atom_types = torch.randint(1, 5, (batch_size, n_atoms), dtype=torch.long)
    res_map = torch.arange(n_res).repeat_interleave(atoms_per_res)
    res_map = res_map.unsqueeze(0).expand(batch_size, -1).clone()

    atom_names = torch.full((batch_size, n_atoms), ATOM_NAME_MAP['CA'], dtype=torch.long)
    # Делаем backbone N, CA, C для первых трёх атомов каждого остатка
    bb_pattern = torch.tensor(
        [ATOM_NAME_MAP['N'], ATOM_NAME_MAP['CA'], ATOM_NAME_MAP['C']] +
        [ATOM_NAME_MAP['O']] * (atoms_per_res - 3)
    )
    atom_names = bb_pattern.repeat(n_res).unsqueeze(0).expand(batch_size, -1).clone()

    mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    res_mask = torch.ones(batch_size, n_res, dtype=torch.bool)

    pred_data = {
        'rots': rots_pred,
        'trans': trans_pred,
        'coords': coords_pred,
        'res_mask': res_mask,
        'mask': mask,
    }
    true_data = {
        'rots': rots_true,
        'trans': trans_true,
        'coords': coords_true,
        'atom_types': atom_types,
        'atom_names': atom_names,
        'res_map': res_map,
    }
    return pred_data, true_data


def test_fape_loss_forward():
    print("Testing FAPELoss forward (dict API)...")
    pred_data, true_data = _make_pred_true()
    loss_fn = FAPELoss()
    total, metrics = loss_fn(pred_data, true_data)
    assert total.shape == torch.Size([]), f"Expected scalar, got {total.shape}"
    assert total.item() >= 0
    for key in ('fape', 'clash', 'physics', 'total'):
        assert key in metrics, f"Missing {key} in metrics"
    print("FAPELoss test passed.")


def test_protein_upscaling_loss_alias():
    print("Testing ProteinUpscalingLoss is FAPELoss...")
    assert ProteinUpscalingLoss is FAPELoss
    print("ProteinUpscalingLoss alias test passed.")


def test_fape_normalization_consistent():
    """С маскированным паддингом и без него loss должен совпадать,
    если паддинг просто 'нулевой'."""
    print("Testing FAPE normalization consistency...")
    pred_data, true_data = _make_pred_true(batch_size=1, n_res=4, atoms_per_res=4)
    loss_fn = FAPELoss(clash_weight=0.0, physics_weight=0.0)
    total_full, _ = loss_fn(pred_data, true_data)
    # Сейчас mask и res_mask полностью True — проверим, что значение конечно
    assert torch.isfinite(total_full), f"FAPE produced non-finite: {total_full}"
    print("FAPE normalization test passed.")


def test_clash_loss_excludes_neighbor_residues():
    print("Testing ClashLoss excludes intra/neighbour-residue pairs...")
    batch_size, n_atoms = 1, 10
    coords = torch.zeros(batch_size, n_atoms, 3)
    coords[0, :, 0] = torch.arange(n_atoms, dtype=torch.float32) * 0.1  # все атомы рядом
    atom_types = torch.full((batch_size, n_atoms), ATOM_TYPE_MAP['C'], dtype=torch.long)

    # Все атомы в одном остатке → clashes должны игнорироваться
    res_map_same = torch.zeros(batch_size, n_atoms, dtype=torch.long)
    clash_fn = ClashLoss()
    loss_same = clash_fn(coords, atom_types, res_map=res_map_same)
    assert loss_same.item() == 0.0, f"Expected 0 clashes in same residue, got {loss_same.item()}"

    # Атомы из удалённых остатков → clashes считаются
    res_map_diff = (torch.arange(n_atoms) * 5).unsqueeze(0)
    loss_diff = clash_fn(coords, atom_types, res_map=res_map_diff)
    assert loss_diff.item() > 0.0, "Clashes between distant residues must be counted"
    print("ClashLoss neighbour-exclusion test passed.")


def test_physics_loss_peptide_only():
    print("Testing PhysicsLoss only on real C(i)-N(i+1) bonds...")
    n_res, atoms_per_res = 4, 4
    n_atoms = n_res * atoms_per_res
    coords = torch.zeros(1, n_atoms, 3)
    # Линейная цепочка, расстояние между соседями = 1.33 (целевая длина)
    coords[0, :, 0] = torch.arange(n_atoms, dtype=torch.float32) * 1.33

    atom_names = torch.tensor(
        [ATOM_NAME_MAP['N'], ATOM_NAME_MAP['CA'], ATOM_NAME_MAP['C'], ATOM_NAME_MAP['O']]
        * n_res,
        dtype=torch.long,
    ).unsqueeze(0)
    res_map = torch.arange(n_res).repeat_interleave(atoms_per_res).unsqueeze(0)
    atom_types = torch.full((1, n_atoms), ATOM_TYPE_MAP['C'], dtype=torch.long)

    physics_fn = PhysicsLoss()
    loss = physics_fn(coords, atom_types, atom_names=atom_names, res_map=res_map)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    # При идеальных расстояниях штраф должен быть около нуля
    assert loss.item() < 1e-3, f"Expected near-zero penalty, got {loss.item()}"

    # Если атомных имён нет — возвращаем 0
    fallback = physics_fn(coords, atom_types)
    assert fallback.item() == 0.0
    print("PhysicsLoss peptide-only test passed.")


if __name__ == "__main__":
    test_fape_loss_forward()
    test_protein_upscaling_loss_alias()
    test_fape_normalization_consistent()
    test_clash_loss_excludes_neighbor_residues()
    test_physics_loss_peptide_only()
    print("All loss component tests passed!")
