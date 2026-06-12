"""Тесты sequence-based выравнивания (upscaler.data.seq_align / align).

Ключевая проверка: пары bad↔good с РАЗНЫМИ буквами цепей и СДВИНУТОЙ
нумерацией остатков должны корректно сопоставляться по последовательности.
Прежняя exact-match логика на таких парах матчила 0 атомов.
"""
import sys
import os
import tempfile

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import numpy as np
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB import PDBIO

from upscaler.data.align import align_structures, align_structures_stats
from upscaler.data.seq_align import align_structures_by_sequence


SEQUENCE = ["ALA", "GLY", "SER", "VAL", "LEU", "THR", "PRO", "LYS"]
BACKBONE = ["N", "CA", "C", "O"]
ELEMENTS = {"N": "N", "CA": "C", "C": "C", "O": "O"}


def _residue_atom_coords(i: int):
    """Детерминированные, не-вырожденные координаты backbone для остатка i."""
    base = i * 3.8
    return {
        "N": np.array([base, 0.0, 0.0]),
        "CA": np.array([base + 1.0, 1.0, 0.0]),
        "C": np.array([base + 2.0, 0.0, 1.0]),
        "O": np.array([base + 2.5, 0.0, 1.5]),
    }


def _write_structure(path: str, chain_id: str, start_resseq: int, offset: np.ndarray):
    sb = StructureBuilder()
    sb.init_structure("s")
    sb.init_model(0)
    sb.init_chain(chain_id)
    sb.init_seg(" ")
    for i, resname in enumerate(SEQUENCE):
        resseq = start_resseq + i
        sb.init_residue(resname, " ", resseq, " ")
        coords = _residue_atom_coords(i)
        for atom_name in BACKBONE:
            xyz = (coords[atom_name] + offset).astype("f")
            sb.init_atom(
                atom_name, xyz, 0.0, 1.0, " ", atom_name,
                element=ELEMENTS[atom_name],
            )
    io = PDBIO()
    io.set_structure(sb.get_structure())
    io.save(path)


def test_seq_align_matches_renumbered_relabeled_chains():
    print("Testing sequence alignment across renumbering + chain relabel...")
    with tempfile.TemporaryDirectory() as tmp:
        low_path = os.path.join(tmp, "low.pdb")
        high_path = os.path.join(tmp, "high.pdb")
        # low: цепь A, нумерация с 1; high: цепь B, нумерация со 101,
        # плюс жёсткий сдвиг координат на [10,0,0].
        _write_structure(low_path, "A", 1, np.zeros(3))
        _write_structure(high_path, "B", 101, np.array([10.0, 0.0, 0.0]))

        aln = align_structures_by_sequence(low_path, high_path, identity_min=0.8)
        # Все 8 остатков должны сопоставиться, несмотря на разные цепи/нумерацию.
        assert len(aln.matched_pairs) == len(SEQUENCE), len(aln.matched_pairs)
        assert abs(aln.coverage - 1.0) < 1e-6, aln.coverage

        low_coords, high_coords, elements, names, resn, resseq, chains, icodes = \
            align_structures(low_path, high_path, identity_min=0.8)
        # 8 остатков × 4 backbone-атома.
        assert low_coords.shape == (len(SEQUENCE) * len(BACKBONE), 3), low_coords.shape
        assert high_coords.shape == low_coords.shape
        assert set(names) == set(BACKBONE)

        n_atoms, rmsd, coverage = align_structures_stats(low_path, high_path)
        assert n_atoms == len(SEQUENCE) * len(BACKBONE), n_atoms
        # После Kabsch жёсткий сдвиг убирается → RMSD ~ 0.
        assert rmsd < 1e-3, rmsd
        assert abs(coverage - 1.0) < 1e-6, coverage
    print("Renumbering/relabel alignment test passed.")


def test_seq_align_partial_overlap_coverage():
    print("Testing partial-overlap coverage...")
    with tempfile.TemporaryDirectory() as tmp:
        low_path = os.path.join(tmp, "low.pdb")
        high_path = os.path.join(tmp, "high.pdb")
        # high покрывает всю последовательность, low — только первые 4 остатка.
        _write_full(low_path, "A", 1, np.zeros(3), n_res=4)
        _write_structure(high_path, "A", 1, np.zeros(3))

        aln = align_structures_by_sequence(low_path, high_path, identity_min=0.8)
        # Совпасть могут максимум 4 остатка; coverage относительно МЕНЬШЕЙ
        # структуры (low, 4 остатка) → ~1.0.
        assert len(aln.matched_pairs) == 4, len(aln.matched_pairs)
        assert abs(aln.coverage - 1.0) < 1e-6, aln.coverage
    print("Partial-overlap coverage test passed.")


def _write_full(path: str, chain_id: str, start_resseq: int, offset: np.ndarray, n_res: int):
    sb = StructureBuilder()
    sb.init_structure("s")
    sb.init_model(0)
    sb.init_chain(chain_id)
    sb.init_seg(" ")
    for i in range(n_res):
        resname = SEQUENCE[i]
        sb.init_residue(resname, " ", start_resseq + i, " ")
        coords = _residue_atom_coords(i)
        for atom_name in BACKBONE:
            xyz = (coords[atom_name] + offset).astype("f")
            sb.init_atom(atom_name, xyz, 0.0, 1.0, " ", atom_name, element=ELEMENTS[atom_name])
    io = PDBIO()
    io.set_structure(sb.get_structure())
    io.save(path)


if __name__ == "__main__":
    test_seq_align_matches_renumbered_relabeled_chains()
    test_seq_align_partial_overlap_coverage()
    print("All data component tests passed!")
