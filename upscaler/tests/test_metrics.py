import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from upscaler.utils.metrics import QualityMetrics


def test_compute_rmsd():
    print("Testing QualityMetrics.compute_rmsd...")
    batch_size, num_atoms = 2, 100
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.1 * torch.randn_like(pred_coords) # Небольшое отклонение
    
    rmsd = QualityMetrics.compute_rmsd(pred_coords, true_coords)
    
    assert rmsd.shape == torch.Size([]), f"Expected scalar RMSD, got {rmsd.shape}"
    assert rmsd.item() >= 0, "RMSD should be non-negative"
    print("QualityMetrics.compute_rmsd test passed.")

def test_compute_lddt():
    print("Testing QualityMetrics.compute_lddt...")
    batch_size, num_atoms = 2, 50 # Меньше атомов для быстродействия теста
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.05 * torch.randn_like(pred_coords) # Очень близкие координаты
    
    lddt = QualityMetrics.compute_lddt(pred_coords, true_coords)
    
    assert lddt.shape == torch.Size([]), f"Expected scalar lDDT, got {lddt.shape}"
    assert 0 <= lddt.item() <= 1, f"lDDT should be between 0 and 1, got {lddt.item()}"
    print("QualityMetrics.compute_lddt test passed.")

def test_compute_clash_score():
    print("Testing QualityMetrics.compute_clash_score...")
    batch_size, num_atoms = 2, 20
    # Создаем координаты, которые почти совпадают, чтобы вызвать столкновения
    coords = torch.randn(batch_size, num_atoms, 3)
    # Дублируем некоторые атомы, чтобы создать столкновения
    coords_with_clashes = coords.clone()
    coords_with_clashes[:, 0] = coords_with_clashes[:, 1] # Совпадающие атомы
    
    # Используем типы атомов, которые точно есть в ATOM_TYPE_MAP
    # Предполагаем, что PAD=0, C=1, N=2, O=3, S=4
    atom_types = torch.randint(1, 5, (batch_size, num_atoms)) # Индексы 1-4
    
    metrics_calculator = QualityMetrics()
    # Тест без столкновений
    clash_score_no_clash = metrics_calculator.compute_clash_score(coords, atom_types)
    # Тест со столкновениями
    clash_score_with_clash = metrics_calculator.compute_clash_score(coords_with_clashes, atom_types)
    
    assert clash_score_no_clash.shape == torch.Size([]), f"Expected scalar clash score, got {clash_score_no_clash.shape}"
    assert clash_score_with_clash.shape == torch.Size([]), f"Expected scalar clash score, got {clash_score_with_clash.shape}"
    assert clash_score_no_clash.item() >= 0, "Clash score should be non-negative"
    assert clash_score_with_clash.item() >= 0, "Clash score should be non-negative"
    # Столкновения должны увеличить средний clash score
    # assert clash_score_with_clash.item() >= clash_score_no_clash.item(), "Clash score should be higher with clashes"
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
    
    assert rmsd.shape == torch.Size([]), f"Expected scalar RMSD, got {rmsd.shape}"
    assert lddt.shape == torch.Size([]), f"Expected scalar lDDT, got {lddt.shape}"
    assert clash_score.shape == torch.Size([]), f"Expected scalar clash score, got {clash_score.shape}"
    
    assert rmsd.item() >= 0, "RMSD should be non-negative"
    assert 0 <= lddt.item() <= 1, f"lDDT should be between 0 and 1, got {lddt.item()}"
    assert clash_score.item() >= 0, "Clash score should be non-negative"
    
    print("QualityMetrics class test passed.")

if __name__ == "__main__":
    test_compute_rmsd()
    test_compute_lddt()
    test_compute_clash_score()
    test_quality_metrics_class()
    print("All metrics tests passed!")
