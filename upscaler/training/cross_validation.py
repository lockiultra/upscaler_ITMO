from __future__ import annotations

import logging
import random
from typing import List, Tuple

from torch.utils.data import Dataset, Subset
from upscaler.training.split import group_indices_by_uniprot

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
        if self.num_folds < 2:
            raise ValueError('num_folds must be at least 2')
        self.seed = seed

        self._indices_by_fold = self._create_folds()

    def _create_folds(self) -> List[List[int]]:
        groups, self._train_only = group_indices_by_uniprot(self.dataset)
        uids = sorted(groups)
        if len(uids) < self.num_folds:
            raise ValueError('Each fold requires a distinct uniprot_id')
        if self.seed is not None:
            rnd = random.Random(self.seed)
            rnd.shuffle(uids)
        else:
            random.shuffle(uids)

        folds: List[List[int]] = [[] for _ in range(self.num_folds)]
        for index, uid in enumerate(uids):
            folds[index % self.num_folds].extend(groups[uid])
        return folds

    def create_folds(self) -> List[Tuple[Subset, Subset]]:
        """
        Returns:
            list of (train_subset, val_subset) pairs as Subset objects
        """
        folds: List[Tuple[Subset, Subset]] = []
        for i in range(self.num_folds):
            test_indices = self._indices_by_fold[i]
            train_indices = list(self._train_only) + [
                idx for j, fold in enumerate(self._indices_by_fold) if j != i for idx in fold
            ]
            folds.append((Subset(self.dataset, train_indices), Subset(self.dataset, test_indices)))
        return folds
