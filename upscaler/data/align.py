from __future__ import annotations

import logging

import numpy as np

from upscaler.data.seq_align import (
    SeqAlignment,
    align_structures_by_sequence,
)


LOGGER = logging.getLogger(__name__)


def _atom_arrays_from_alignment(
    aln: SeqAlignment,
) -> tuple[
    np.ndarray, np.ndarray,
    list[str], list[str], list[str], list[int],
    list[str], list[str],
]:
    """Строит параллельные массивы атомов из соответствий остатков.

    Атомы матчатся ВНУТРИ соответствующих остатков по ``atom_name`` (общие
    имена). Метаданные (resname/resseq/icode/chain) берутся со стороны low —
    они одинаково используются для группировки и low, и high координат в
    ``dataset._get_residue_data``.
    """
    low_coords: list[np.ndarray] = []
    high_coords: list[np.ndarray] = []
    atom_elements: list[str] = []
    atom_names: list[str] = []
    res_names: list[str] = []
    res_seqs: list[int] = []
    chain_ids: list[str] = []
    icodes: list[str] = []

    for low_r, high_r in aln.matched_pairs:
        # Стабильный порядок: порядок атомов в low-остатке (обычно N, CA, C, ...)
        for name, low_atom in low_r.atoms.items():
            high_atom = high_r.atoms.get(name)
            if high_atom is None:
                continue
            low_coords.append(low_atom.coord)
            high_coords.append(high_atom.coord)
            atom_elements.append(low_atom.element.strip().upper())
            atom_names.append(name.strip().upper())
            res_names.append(low_r.resname.strip())
            res_seqs.append(low_r.resseq)
            chain_ids.append(low_r.chain_id)
            icodes.append(low_r.icode)

    if not low_coords:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            [], [], [], [], [], [],
        )

    low_arr = np.stack(low_coords).astype(np.float32)
    high_arr = np.stack(high_coords).astype(np.float32)
    return (
        low_arr, high_arr, atom_elements, atom_names,
        res_names, res_seqs, chain_ids, icodes,
    )


def align_structures(
    low_path: str,
    high_path: str,
    identity_min: float = 0.8,
) -> tuple[
    np.ndarray, np.ndarray,
    list[str], list[str], list[str], list[int],
    list[str], list[str],
]:
    """Возвращает выровненные по атомам координаты low/high структур.

    В отличие от прежней exact-match версии, соответствие остатков строится
    через sequence alignment (см. ``seq_align``), а атомы матчатся внутри
    соответствующих остатков по имени. Это убирает ложные совпадения из-за
    несовпадающей нумерации/букв цепей между разными PDB-депозициями.

    Returns
    -------
    low_coords    : np.ndarray (N, 3)
    high_coords   : np.ndarray (N, 3)
    atom_elements : list[str]   элементы (C, N, O, ...)
    atom_names    : list[str]   имена атомов (CA, CB, ...)
    res_names     : list[str]   названия остатков (ALA, VAL, ...)
    res_seqs      : list[int]   номера остатков
    chain_ids     : list[str]   идентификаторы цепей
    icodes        : list[str]   insertion codes (обычно "")
    """
    aln = align_structures_by_sequence(low_path, high_path, identity_min=identity_min)
    result = _atom_arrays_from_alignment(aln)
    if result[0].shape[0] == 0:
        raise ValueError(
            "Нет общих остатков между low и high структурами после "
            "sequence-выравнивания."
        )
    LOGGER.debug("Выравнено %d атомов", result[0].shape[0])
    return result


def align_structures_stats(
    low_path: str,
    high_path: str,
    identity_min: float = 0.8,
) -> tuple[int, float, float]:
    """Лёгкая статистика пары для prefilter: (n_atoms, rmsd, coverage).

    ``coverage`` — доля совпавших остатков относительно меньшей структуры
    (см. :pyattr:`SeqAlignment.coverage`). ``rmsd`` — post-Kabsch RMSD по
    совпавшим атомам. Если совпавших атомов нет, возвращает
    ``(0, inf, 0.0)``.
    """
    aln = align_structures_by_sequence(low_path, high_path, identity_min=identity_min)
    low_coords, high_coords, *_ = _atom_arrays_from_alignment(aln)
    n_atoms = int(low_coords.shape[0])
    coverage = aln.coverage
    if n_atoms == 0:
        return 0, float("inf"), coverage
    _, rmsd, _, _ = kabsch_superimpose(low_coords, high_coords)
    return n_atoms, float(rmsd), coverage


def kabsch_superimpose(P: np.ndarray, Q: np.ndarray):
    """
    Проецирует Q на P: находит R, t такие что R @ Q + t ≈ P (по методу Кабша).
    P, Q: (N,3) numpy arrays
    Возвращает: Q_aligned (N,3), rmsd (float), R (3,3), t (3,)
    """
    assert P.shape == Q.shape and P.ndim == 2 and P.shape[1] == 3
    N = P.shape[0]
    if N == 0:
        raise ValueError("Empty coordinates for Kabsch.")

    # центры масс
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P_centered = P - Pc
    Q_centered = Q - Qc

    # ковариация
    C = np.dot(Q_centered.T, P_centered)
    V, S, Wt = np.linalg.svd(C)

    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1.0, 1.0, d])
    R = np.dot(V, np.dot(D, Wt))
    t = Pc - R.dot(Qc)

    Q_aligned = (R.dot(Q.T)).T + t
    diff = P - Q_aligned
    rmsd = float(np.sqrt((diff**2).sum() / N))
    return Q_aligned, rmsd, R, t
