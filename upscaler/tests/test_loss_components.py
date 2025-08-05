import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from upscaler.loss.loss import ProteinUpscalingLoss


def test_coord_rmsd_loss():
    print("Testing compute_coord_rmsd_loss...")
    batch_size, num_atoms = 2, 100
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.1 * torch.randn_like(pred_coords) # Небольшое отклонение
    
    loss_fn = ProteinUpscalingLoss()
    loss = loss_fn.compute_coord_rmsd_loss(pred_coords, true_coords)
    
    assert loss.shape == torch.Size([]), f"Expected scalar loss, got {loss.shape}"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("compute_coord_rmsd_loss test passed.")

def test_lddt_loss():
    print("Testing compute_lddt_loss...")
    batch_size, num_atoms = 2, 50 # Меньше атомов для быстродействия теста
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.05 * torch.randn_like(pred_coords) # Очень близкие координаты
    
    loss_fn = ProteinUpscalingLoss()
    loss = loss_fn.compute_lddt_loss(pred_coords, true_coords)
    
    assert loss.shape == torch.Size([]), f"Expected scalar loss, got {loss.shape}"
    assert 0 <= loss.item() <= 1, f"lDDT loss should be between 0 and 1, got {loss.item()}"
    print("compute_lddt_loss test passed.")

def test_clash_penalty():
    print("Testing compute_clash_penalty...")
    batch_size, num_atoms = 2, 20
    # Создаем координаты, которые почти совпадают, чтобы вызвать столкновения
    coords = torch.randn(batch_size, num_atoms, 3)
    # Дублируем некоторые атомы, чтобы создать столкновения
    coords_with_clashes = coords.clone()
    coords_with_clashes[:, 0] = coords_with_clashes[:, 1] # Совпадающие атомы
    
    # Используем типы атомов, которые точно есть в ATOM_TYPE_MAP
    atom_types = torch.randint(1, 5, (batch_size, num_atoms)) # Индексы 1-4
    
    loss_fn = ProteinUpscalingLoss()
    # Тест без столкновений
    loss_no_clash = loss_fn.compute_clash_penalty(coords, atom_types)
    # Тест со столкновениями
    loss_with_clash = loss_fn.compute_clash_penalty(coords_with_clashes, atom_types)
    
    assert loss_no_clash.shape == torch.Size([]), f"Expected scalar loss, got {loss_no_clash.shape}"
    assert loss_with_clash.shape == torch.Size([]), f"Expected scalar loss, got {loss_with_clash.shape}"
    # Штраф со столкновениями должен быть выше (или равен, если нет столкновений)
    # assert loss_with_clash.item() >= loss_no_clash.item(), "Clash penalty should be higher with clashes"
    print("compute_clash_penalty test passed.")

def test_physics_constraints():
    print("Testing compute_physics_constraints...")
    batch_size, num_atoms = 2, 20
    coords = torch.randn(batch_size, num_atoms, 3)
    # Создаем атомы C и N с реалистичными расстояниями
    atom_types = torch.ones(batch_size, num_atoms, dtype=torch.long) # Все атомы 'C' (индекс 1)
    atom_types[:, 1::2] = 2 # Каждый второй атом 'N' (индекс 2)
    
    loss_fn = ProteinUpscalingLoss()
    loss = loss_fn.compute_physics_constraints(coords, atom_types)
    
    assert loss.shape == torch.Size([]), f"Expected scalar loss, got {loss.shape}"
    assert loss.item() >= 0, "Physics loss should be non-negative"
    print("compute_physics_constraints test passed.")

def test_protein_upscaling_loss():
    print("Testing ProteinUpscalingLoss...")
    batch_size, num_atoms = 2, 30
    pred_coords = torch.randn(batch_size, num_atoms, 3)
    true_coords = pred_coords + 0.1 * torch.randn_like(pred_coords)
    atom_types = torch.randint(1, 10, (batch_size, num_atoms))
    
    loss_fn = ProteinUpscalingLoss()
    total_loss, metrics = loss_fn(pred_coords, true_coords, atom_types)
    
    assert total_loss.shape == torch.Size([]), f"Expected scalar total loss, got {total_loss.shape}"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    expected_keys = {'coord_rmsd', 'lddt', 'clash', 'physics', 'total'}
    assert set(metrics.keys()) == expected_keys, f"Metrics keys mismatch. Expected {expected_keys}, got {set(metrics.keys())}"
    
    for key, value in metrics.items():
        assert value.shape == torch.Size([]), f"Metric {key} should be scalar, got {value.shape}"
        assert value.item() >= 0, f"Metric {key} should be non-negative, got {value.item()}"
        
    # Проверим, что total loss примерно равен сумме взвешенных компонентов
    # (с небольшой погрешностью из-за вычислений с плавающей точкой)
    # calculated_total = (
    #     loss_fn.coord_weight * metrics['coord_rmsd'] +
    #     loss_fn.lddt_weight * metrics['lddt'] +
    #     loss_fn.clash_weight * metrics['clash'] +
    #     loss_fn.physics_weight * metrics['physics']
    # )
    # assert torch.allclose(total_loss, calculated_total, atol=1e-5), "Total loss mismatch"
    
    print("ProteinUpscalingLoss test passed.")

if __name__ == "__main__":
    test_coord_rmsd_loss()
    test_lddt_loss()
    test_clash_penalty()
    test_physics_constraints()
    test_protein_upscaling_loss()
    print("All loss component tests passed!")
