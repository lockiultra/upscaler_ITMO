from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def _atom_id_in_chain(atom: Atom) -> Tuple[str, int, str]:
    """
    Уникальный ключ атома внутри одной цепи:
    (resname, resseq, atom.name)
    """
    res: Residue = atom.get_parent()
    return (res.resname.strip(), res.id[1], atom.name.strip())


def align_structures(
    low_path: str,
    high_path: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Возвращает координаты low и high, выровненные по атомам.
    Если атома нет в одной из структур – пропускаем.

    Returns
    -------
    low_coords : np.ndarray (N, 3)
    high_coords: np.ndarray (N, 3)
    atom_types : List[str]        элементы (C, N, O, ...)
    res_types  : List[str]        остатки (ALA, VAL, ...)
    """
    parser = PDBParser(QUIET=True)
    low_s  = parser.get_structure("low",  low_path)
    high_s = parser.get_structure("high", high_path)

    # словарь: ключ -> Atom
    def _build_map(struct: Structure) -> Dict[Tuple[str, int, str], Atom]:
        return {
            _atom_id_in_chain(at): at
            for at in struct.get_atoms()
            if at.get_parent().get_id()[0] == " "  # ATOM, не HETATM
        }

    low_map  = _build_map(low_s)
    high_map = _build_map(high_s)

    # пересечение атомов
    common_keys = sorted(low_map.keys() & high_map.keys())

    if not common_keys:
        raise ValueError("Нет общих атомов между low и high структурами.")

    low_coords  = np.stack([low_map[k].coord  for k in common_keys], dtype=np.float32)
    high_coords = np.stack([high_map[k].coord for k in common_keys], dtype=np.float32)
    atom_types  = [low_map[k].element.strip().upper() for k in common_keys]
    res_types   = [low_map[k].get_parent().resname.strip() for k in common_keys]

    LOGGER.debug("Выравнено %d атомов", len(common_keys))
    return low_coords, high_coords, atom_types, res_types
