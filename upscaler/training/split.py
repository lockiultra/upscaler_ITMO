"""UniProt grouping shared by training, quick experiments and cross-validation."""
from __future__ import annotations

import random

from torch.utils.data import Dataset, Subset


def uniprot_id_for_index(dataset: Dataset, index: int) -> str | None:
    if isinstance(dataset, Subset):
        return uniprot_id_for_index(dataset.dataset, dataset.indices[index])
    return dataset.uniprot_id_for_pair(index)


def group_indices_by_uniprot(dataset: Dataset) -> tuple[dict[str, list[int]], list[int]]:
    groups: dict[str, list[int]] = {}
    unknown: list[int] = []
    for index in range(len(dataset)):
        uid = uniprot_id_for_index(dataset, index)
        if uid is None:
            unknown.append(index)
        else:
            groups.setdefault(uid, []).append(index)
    return groups, unknown


def split_indices_by_uniprot(
    dataset: Dataset, val_fraction: float = 0.2, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Keep whole proteins together and leave unidentified pairs in training.

    With fewer than two identified proteins there is no independent validation
    set; callers must reject that case instead of moving an individual pair.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must be between 0 and 1 (exclusive)')
    groups, unknown = group_indices_by_uniprot(dataset)
    uids = sorted(groups)
    random.Random(seed).shuffle(uids)
    n_val = min(len(uids) - 1, max(1, round(val_fraction * len(uids)))) if len(uids) > 1 else 0
    val_uids = set(uids[:n_val])
    train_indices = list(unknown)
    val_indices: list[int] = []
    for uid, indices in groups.items():
        (val_indices if uid in val_uids else train_indices).extend(indices)
    return sorted(train_indices), sorted(val_indices)
