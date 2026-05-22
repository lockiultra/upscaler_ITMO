from __future__ import annotations

import logging

import os

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


LOGGER = logging.getLogger(__name__)


def _atom_id_in_chain(atom: Atom) -> tuple[str, str, int, str, str, str]:
    """
    Уникальный ключ атома, учитывающий цепь, hetflag, insertion code, altloc:
    (chain_id, resname, resseq, icode, atom.name, altloc)
    """
    res: Residue = atom.get_parent()
    chain_id = res.get_parent().id
    hetflag, resseq, icode = res.id
    altloc = atom.get_altloc() or ""
    return (
        str(chain_id),
        res.resname.strip(),
        int(resseq),
        str(icode).strip(),
        atom.name.strip(),
        str(altloc).strip(),
    )


def _make_parser(path: str):
    """Выбираем парсер исходя из расширения файла структуры."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def align_structures(
    low_path: str,
    high_path: str,
) -> tuple[
    np.ndarray, np.ndarray,
    list[str], list[str], list[str], list[int],
    list[str], list[str],
]:
    """Возвращает выровненные по атомам координаты low/high структур.

    Атом, отсутствующий в одной из структур, пропускается. Ключ атома
    учитывает ``chain_id``, ``resname``, ``resseq``, ``icode``, ``atom_name``,
    ``altloc`` — это важно для multi-chain и неоднозначных residues.

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
    low_s  = _make_parser(low_path).get_structure("low",  low_path)
    high_s = _make_parser(high_path).get_structure("high", high_path)

    # словарь: ключ -> Atom (только полипептидные остатки, hetflag == ' ')
    def _build_map(struct: Structure) -> dict[tuple, Atom]:
        out: dict[tuple, Atom] = {}
        for at in struct.get_atoms():
            res = at.get_parent()
            if res.get_id()[0] != " ":
                continue
            key = _atom_id_in_chain(at)
            # Если для одного atom.name есть несколько altloc, оставляем первый
            # стабильный (обычно "A" или "").
            if key not in out:
                out[key] = at
        return out

    low_map  = _build_map(low_s)
    high_map = _build_map(high_s)

    # пересечение атомов. Сортируем по биологически-значимому порядку
    # (chain_id, resseq, icode, atom_name, altloc) — но НЕ по resname,
    # иначе атомы соседних остатков перемешаются.
    # key layout: (chain_id, resname, resseq, icode, atom_name, altloc)
    common = low_map.keys() & high_map.keys()
    common_keys = sorted(
        common,
        key=lambda k: (k[0], k[2], k[3], k[4], k[5]),
    )

    if not common_keys:
        raise ValueError("Нет общих атомов между low и high структурами.")

    low_coords  = np.stack([low_map[k].coord  for k in common_keys], dtype=np.float32)
    high_coords = np.stack([high_map[k].coord for k in common_keys], dtype=np.float32)
    atom_elements = [low_map[k].element.strip().upper() for k in common_keys]
    atom_names = [k[4].strip().upper() for k in common_keys]
    res_names = [k[1].strip() for k in common_keys]
    res_seqs = [k[2] for k in common_keys]
    chain_ids = [k[0] for k in common_keys]
    icodes = [k[3] for k in common_keys]

    LOGGER.debug("Выравнено %d атомов", len(common_keys))

    return (
        low_coords, high_coords, atom_elements, atom_names,
        res_names, res_seqs, chain_ids, icodes,
    )
    

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
