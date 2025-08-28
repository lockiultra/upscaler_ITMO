from __future__ import annotations

import logging
import random
from typing import List, Tuple

from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


class ProteinFoldCrossValidation:
    """
    Create folds for cross-validation.

    Args:
        dataset: instance of torch.utils.data.Dataset
        num_folds: number of folds (default 5)
        seed: random seed for reproducibility
    """

    def __init__(self, dataset: Dataset, num_folds: int = 5, seed: int | None = 42):
        self.dataset = dataset
        self.num_folds = int(num_folds)
        self.seed = seed

        self._indices_by_fold = self._create_folds()

    def _create_folds(self) -> List[List[int]]:
        total = len(self.dataset)
        indices = list(range(total))
        if self.seed is not None:
            rnd = random.Random(self.seed)
            rnd.shuffle(indices)
        else:
            random.shuffle(indices)

        fold_size = total // self.num_folds
        folds: List[List[int]] = []
        for i in range(self.num_folds):
            start = i * fold_size
            end = start + fold_size if i < self.num_folds - 1 else total
            folds.append(indices[start:end])
        return folds

    def create_folds(self) -> List[Tuple[Subset, Subset]]:
        """
        Returns:
            list of (train_subset, val_subset) pairs as Subset objects
        """
        folds: List[Tuple[Subset, Subset]] = []
        for i in range(self.num_folds):
            test_indices = self._indices_by_fold[i]
            train_indices = [idx for j, fold in enumerate(self._indices_by_fold) if j != i for idx in fold]
            folds.append((Subset(self.dataset, train_indices), Subset(self.dataset, test_indices)))
        return folds
