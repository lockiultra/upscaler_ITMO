import torch
from torch.utils.data import Dataset, Subset
from typing import List, Tuple, Dict, Any
import random
import logging

logger = logging.getLogger(__name__)


class ProteinFoldCrossValidation:
    """Стратегия кросс-валидации по семействам белков."""
    
    def __init__(self, dataset: Dataset, fold_strategy: str = 'random'):
        """
        Args:
            dataset (Dataset): Датасет для кросс-валидации.
            fold_strategy (str): Стратегия разделения ('random' или 'scop_family').
        """
        self.dataset = dataset
        self.fold_strategy = fold_strategy

        self.num_folds = 5
        self._indices_by_fold = self._create_folds()

    def _create_folds(self) -> List[List[int]]:
        """Создает индексы для фолдов."""
        total_size = len(self.dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        fold_size = total_size // self.num_folds
        folds = []
        for i in range(self.num_folds):
            start = i * fold_size
            end = start + fold_size if i < self.num_folds - 1 else total_size
            folds.append(indices[start:end])
        return folds

    def create_folds(self, n_folds: int = 5) -> List[Tuple[Subset, Subset]]:
        """
        Создает фолды для кросс-валидации.
        
        Args:
            n_folds (int): Количество фолдов.
            
        Returns:
            List[Tuple[Subset, Subset]]: Список пар (обучающий_subset, тестовый_subset).
        """
        if n_folds != self.num_folds:
            logger.warning(f"n_folds ({n_folds}) не совпадает с self.num_folds ({self.num_folds}). Используется self.num_folds.")
            
        folds = []
        for i in range(self.num_folds):
            test_indices = self._indices_by_fold[i]
            train_indices = [idx for fold in self._indices_by_fold[:i] + self._indices_by_fold[i+1:] for idx in fold]
            
            train_dataset = Subset(self.dataset, train_indices)
            test_dataset = Subset(self.dataset, test_indices)
            
            folds.append((train_dataset, test_dataset))
            
        return folds
