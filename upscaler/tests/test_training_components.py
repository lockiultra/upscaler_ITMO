import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from upscaler.training.pipeline import TrainingPipeline
from upscaler.training.cross_validation import ProteinFoldCrossValidation
from upscaler.model.upscaler import ProteinUpscaler
from upscaler.loss.loss import ProteinUpscalingLoss
from upscaler.utils.metrics import QualityMetrics
from upscaler.data.dataset import ProteinUpscalingDataset, collate_batch

# Создаем фиктивный датасет для тестирования
class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Возвращаем фиктивные данные
        num_atoms = 50 # Фиксированное количество атомов для простоты теста
        return {
            'coords_bad': torch.randn(num_atoms, 3),
            'coords_good': torch.randn(num_atoms, 3) + 0.1 * torch.randn(num_atoms, 3),
            'atom_types': torch.randint(1, 10, (num_atoms,)),
            'residue_types': torch.randint(1, 20, (num_atoms,)),
        }

def dummy_collate_batch(batch):
    """Простая функция для объединения образцов в батч."""
    return {
        'coords_bad': torch.stack([b['coords_bad'] for b in batch]),
        'coords_good': torch.stack([b['coords_good'] for b in batch]),
        'atom_types': torch.stack([b['atom_types'] for b in batch]),
        'residue_types': torch.stack([b['residue_types'] for b in batch]),
    }

def test_training_pipeline():
    print("Testing TrainingPipeline...")
    
    # Инициализируем компоненты
    device = torch.device('cpu')
    model = ProteinUpscaler()
    loss_fn = ProteinUpscalingLoss(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics_calculator = QualityMetrics(device=device)
    
    # Инициализируем пайплайн
    pipeline = TrainingPipeline(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=1.0,
        metrics_calculator=metrics_calculator,
    )
    
    # Создаем фиктивный даталоадер
    dummy_dataset = DummyDataset(size=4) # Маленький датасет для теста
    dummy_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=True, collate_fn=dummy_collate_batch)
    
    # Тест train_epoch
    train_metrics = pipeline.train_epoch(dummy_loader)
    assert isinstance(train_metrics, dict), "train_epoch should return a dict"
    assert 'loss' in train_metrics, "train_metrics should contain 'loss'"
    assert train_metrics['loss'] >= 0, "Train loss should be non-negative"
    print("TrainingPipeline.train_epoch test passed.")
    
    # Тест validate
    val_metrics = pipeline.validate(dummy_loader)
    assert isinstance(val_metrics, dict), "validate should return a dict"
    expected_keys = {'val_rmsd', 'val_lddt', 'val_clash'}
    assert set(val_metrics.keys()) == expected_keys, f"Val metrics keys mismatch. Expected {expected_keys}, got {set(val_metrics.keys())}"
    for key, value in val_metrics.items():
        assert isinstance(value, float), f"Val metric {key} should be float, got {type(value)}"
        assert value >= 0, f"Val metric {key} should be non-negative, got {value}"
    print("TrainingPipeline.validate test passed.")
    
    print("TrainingPipeline test passed.")

def test_protein_fold_cross_validation():
    print("Testing ProteinFoldCrossValidation...")
    
    # Создаем фиктивный датасет
    dummy_dataset = DummyDataset(size=20)
    
    # Инициализируем кросс-валидацию
    cv = ProteinFoldCrossValidation(dummy_dataset, fold_strategy='random')
    
    # Создаем фолды
    folds = cv.create_folds(n_folds=5)
    
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
    for i, (train_subset, val_subset) in enumerate(folds):
        assert isinstance(train_subset, Subset), f"Fold {i} train subset should be Subset"
        assert isinstance(val_subset, Subset), f"Fold {i} val subset should be Subset"
        assert len(train_subset) > 0, f"Fold {i} train subset should not be empty"
        assert len(val_subset) > 0, f"Fold {i} val subset should not be empty"
        # Проверим, что нет пересечения между train и val
        train_indices = set(train_subset.indices)
        val_indices = set(val_subset.indices)
        assert train_indices.isdisjoint(val_indices), f"Fold {i} train and val subsets should not overlap"
    
    print("ProteinFoldCrossValidation test passed.")

# Примечание: Тест для train.py будет интеграционным и требует наличия данных и модели.
# Для модульного тестирования мы протестировали отдельные компоненты.

if __name__ == "__main__":
    test_training_pipeline()
    test_protein_fold_cross_validation()
    print("All training component tests passed!")
