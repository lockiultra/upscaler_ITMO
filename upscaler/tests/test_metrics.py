import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import torch
from upscaler.utils.metrics import QualityMetrics


def test_compute_rmsd():
    print("Testing QualityMetrics.compute_rmsd...")
    batch_size, num_atoms = 2, 100
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.1 * torch.randn_like(pred_coords)

    rmsd = QualityMetrics.compute_rmsd(pred_coords, true_coords)

    assert rmsd.shape == torch.Size([]), f"Expected scalar RMSD, got {rmsd.shape}"
    assert rmsd.item() >= 0, "RMSD should be non-negative"
    print("QualityMetrics.compute_rmsd test passed.")


def test_compute_rmsd_with_mask():
    print("Testing QualityMetrics.compute_rmsd with mask...")
    batch_size, num_atoms = 2, 30
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.1 * torch.randn_like(pred_coords)
    mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
    mask[:, 20:] = False  # выключаем последние атомы как паддинг

    rmsd = QualityMetrics.compute_rmsd(pred_coords, true_coords, mask=mask)
    assert rmsd.shape == torch.Size([]), f"Expected scalar RMSD, got {rmsd.shape}"
    assert rmsd.item() >= 0, "RMSD should be non-negative"
    print("QualityMetrics.compute_rmsd with mask test passed.")


def test_compute_lddt():
    print("Testing QualityMetrics.compute_lddt...")
    batch_size, num_atoms = 2, 50
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.05 * torch.randn_like(pred_coords)

    lddt = QualityMetrics.compute_lddt(pred_coords, true_coords)
    assert lddt.shape == torch.Size([]), f"Expected scalar lDDT, got {lddt.shape}"
    assert 0 <= lddt.item() <= 1, f"lDDT should be in [0, 1], got {lddt.item()}"

    # Также проверяем mask-aware вариант
    mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
    mask[:, 40:] = False
    lddt_m = QualityMetrics.compute_lddt(pred_coords, true_coords, mask=mask)
    assert lddt_m.shape == torch.Size([]), f"Expected scalar lDDT, got {lddt_m.shape}"
    assert 0 <= lddt_m.item() <= 1
    print("QualityMetrics.compute_lddt test passed.")


def test_compute_clash_score():
    print("Testing QualityMetrics.compute_clash_score...")
    batch_size, num_atoms = 2, 20
    coords = torch.randn(batch_size, num_atoms, 3)
    coords_with_clashes = coords.clone()
    coords_with_clashes[:, 0] = coords_with_clashes[:, 1]

    atom_types = torch.randint(1, 5, (batch_size, num_atoms))
    metrics_calculator = QualityMetrics()

    clash_score_no_clash = metrics_calculator.compute_clash_score(coords, atom_types)
    clash_score_with_clash = metrics_calculator.compute_clash_score(coords_with_clashes, atom_types)
    assert clash_score_no_clash.item() >= 0
    assert clash_score_with_clash.item() >= 0

    mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)
    mask[:, 15:] = False
    clash_score_masked = metrics_calculator.compute_clash_score(
        coords_with_clashes, atom_types, mask=mask
    )
    assert clash_score_masked.shape == torch.Size([])
    assert clash_score_masked.item() >= 0
    print("QualityMetrics.compute_clash_score test passed.")


def test_quality_metrics_class():
    print("Testing QualityMetrics class...")
    batch_size, num_atoms = 2, 30
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.1 * torch.randn_like(pred_coords)
    atom_types = torch.randint(1, 10, (batch_size, num_atoms))

    metrics_calculator = QualityMetrics()
    rmsd = metrics_calculator.compute_rmsd(pred_coords, true_coords)
    lddt = metrics_calculator.compute_lddt(pred_coords, true_coords)
    clash_score = metrics_calculator.compute_clash_score(pred_coords, atom_types)

    assert rmsd.item() >= 0
    assert 0 <= lddt.item() <= 1
    assert clash_score.item() >= 0
    print("QualityMetrics class test passed.")


if __name__ == "__main__":
    test_compute_rmsd()
    test_compute_rmsd_with_mask()
    test_compute_lddt()
    test_compute_clash_score()
    test_quality_metrics_class()
    print("All metrics tests passed!")
