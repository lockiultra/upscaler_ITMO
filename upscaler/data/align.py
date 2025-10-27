from __future__ import annotations

import logging

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


LOGGER = logging.getLogger(__name__)


def _atom_id_in_chain(atom: Atom) -> tuple[str, int, str]:
    """
    Уникальный ключ атома внутри одной цепи:
    (resname, resseq, atom.name)
    """
    res: Residue = atom.get_parent()
    return (res.resname.strip(), res.id[1], atom.name.strip())


def align_structures(
    low_path: str,
    high_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Возвращает координаты low и high, выровненные по атомам.
    Если атома нет в одной из структур – пропускаем.

    Returns
    -------
    low_coords : np.ndarray (N, 3)
    high_coords: np.ndarray (N, 3)
    atom_types : list[str]        элементы (C, N, O, ...)
    res_types  : list[str]        остатки (ALA, VAL, ...)
    """
    parser = PDBParser(QUIET=True)
    low_s  = parser.get_structure("low",  low_path)
    high_s = parser.get_structure("high", high_path)

    # словарь: ключ -> Atom
    def _build_map(struct: Structure) -> dict[tuple[str, int, str], Atom]:
        return {
            _atom_id_in_chain(at): at
            for at in struct.get_atoms()
            if at.get_parent().get_id()[0] == " "
        }

    low_map  = _build_map(low_s)
    high_map = _build_map(high_s)

    # пересечение атомов
    common_keys = sorted(low_map.keys() & high_map.keys())

    if not common_keys:
        raise ValueError("Нет общих атомов между low и high структурами.")

    low_coords  = np.stack([low_map[k].coord  for k in common_keys], dtype=np.float32)
    high_coords = np.stack([high_map[k].coord for k in common_keys], dtype=np.float32)
    atom_elements = [low_map[k].element.strip().upper() for k in common_keys]
    atom_names = [k[2].strip().upper() for k in common_keys]
    res_names = [k[0].strip() for k in common_keys]
    res_seqs = [k[1] for k in common_keys]
    
    LOGGER.debug("Выравнено %d атомов", len(common_keys))

    return low_coords, high_coords, atom_elements, atom_names, res_names, res_seqs
    

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
