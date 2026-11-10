"""Sequence-based выравнивание двух белковых структур.

Зачем это нужно
---------------
Раньше атомы low/high структур сопоставлялись по *точному* ключу
``(chain_id, resname, resseq, icode, atom_name, altloc)``. Между разными
депозициями одного и того же белка (например, cryo-EM комплекс vs рентген
одного домена) буквы цепей и нумерация остатков почти никогда не совпадают,
поэтому матчилось ~6 % атомов, причём часто это были ложные совпадения
(одинаковая буква «A» + номер, но физически разные остатки). EDA показывал
медианное покрытие 0.064 и baseline RMSD ~30 Å — фактически мусорный
supervision-сигнал.

Здесь мы вместо этого:
  1. извлекаем по цепям последовательность остатков;
  2. глобально выравниваем цепи bad↔good (BLOSUM62, global alignment);
  3. жадно сопоставляем цепи по числу совпавших остатков;
  4. возвращаем соответствия НА УРОВНЕ ОСТАТКОВ, а атомы уже матчатся
     внутри соответствующих остатков по ``atom_name`` (см. ``align.py``).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure
from Bio.Align import PairwiseAligner, substitution_matrices


LOGGER = logging.getLogger(__name__)


# 3-буквенный → 1-буквенный код для 20 стандартных аминокислот.
# Нестандартные/модифицированные остатки кодируем как 'X' (BLOSUM62 его
# содержит, так что aligner не падает; identity по 'X' мы не засчитываем).
THREE_TO_ONE: dict[str, str] = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


@dataclass
class ResidueInfo:
    """Лёгкая обёртка над остатком: метаданные + словарь атомов по имени."""
    chain_id: str
    resname: str
    resseq: int
    icode: str
    one_letter: str
    atoms: dict[str, Atom] = field(default_factory=dict)


@dataclass
class SeqAlignment:
    """Результат sequence-выравнивания пары структур."""
    matched_pairs: list[tuple[ResidueInfo, ResidueInfo]]
    n_res_low: int
    n_res_high: int

    @property
    def coverage(self) -> float:
        """Доля совпавших остатков относительно меньшей структуры.

        Берём ``min`` намеренно: если в low — огромный комплекс, а в high —
        один домен, покрытие считается относительно домена (того, что вообще
        можно совместить), а не относительно всего комплекса.
        """
        denom = min(self.n_res_low, self.n_res_high)
        if denom <= 0:
            return 0.0
        return len(self.matched_pairs) / denom


def _make_parser(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.mode = "global"
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    return aligner


def _extract_chain_residues(structure: Structure) -> dict[str, list[ResidueInfo]]:
    """Группирует полипептидные остатки по цепям (только первая model).

    Берётся первый altloc для каждого имени атома. Гетероатомы/вода
    (``res.id[0] != ' '``) пропускаются.
    """
    chains: dict[str, list[ResidueInfo]] = {}
    try:
        model = next(iter(structure))
    except StopIteration:
        return chains

    for chain in model:
        cid = str(chain.id)
        residues: list[ResidueInfo] = []
        for res in chain:
            if res.id[0] != " ":
                continue
            resname = res.resname.strip()
            atoms: dict[str, Atom] = {}
            for at in res:
                name = at.name.strip()
                if name not in atoms:  # стабильно оставляем первый altloc
                    atoms[name] = at
            if not atoms:
                continue
            residues.append(
                ResidueInfo(
                    chain_id=cid,
                    resname=resname,
                    resseq=int(res.id[1]),
                    icode=str(res.id[2]).strip(),
                    one_letter=THREE_TO_ONE.get(resname, "X"),
                    atoms=atoms,
                )
            )
        residues.sort(key=lambda residue: (residue.resseq, residue.icode))
        if residues:
            chains[cid] = residues
    return chains


def _align_residue_pairs(
    low_res: list[ResidueInfo],
    high_res: list[ResidueInfo],
    aligner: PairwiseAligner,
) -> tuple[list[tuple[ResidueInfo, ResidueInfo]], float]:
    """Выравнивает последовательности двух цепей.

    Возвращает (matched_pairs, identity), где matched_pairs — только колонки,
    в которых оба остатка не gap И совпадают по типу (исключая 'X'). identity
    считается по всем выровненным (не-gap) колонкам.
    """
    low_seq = "".join(r.one_letter for r in low_res)
    high_seq = "".join(r.one_letter for r in high_res)
    if not low_seq or not high_seq:
        return [], 0.0

    try:
        alignment = aligner.align(low_seq, high_seq)[0]
    except Exception as exc:  # очень редкий случай несовместимого алфавита
        LOGGER.debug("Alignment failed for chain pair: %s", exc)
        return [], 0.0

    pairs: list[tuple[ResidueInfo, ResidueInfo]] = []
    n_aligned = 0
    n_identical = 0
    # alignment.aligned — два массива gap-free блоков одинаковой длины,
    # внутри блока позиции low/high соответствуют один-к-одному.
    low_blocks, high_blocks = alignment.aligned
    for (l_start, l_end), (h_start, h_end) in zip(low_blocks, high_blocks):
        for offset in range(l_end - l_start):
            li = l_start + offset
            hi = h_start + offset
            n_aligned += 1
            lr = low_res[li]
            hr = high_res[hi]
            if lr.one_letter == hr.one_letter and lr.one_letter != "X":
                n_identical += 1
                pairs.append((lr, hr))

    identity = n_identical / n_aligned if n_aligned else 0.0
    return pairs, identity


def match_chains(
    low_chains: dict[str, list[ResidueInfo]],
    high_chains: dict[str, list[ResidueInfo]],
    identity_min: float = 0.8,
) -> list[tuple[ResidueInfo, ResidueInfo]]:
    """Жадно сопоставляет цепи low↔high и возвращает соответствия остатков.

    Работает с абстрактным представлением ``dict[chain_id -> list[ResidueInfo]]``,
    поэтому переиспользуется и из ``align_structures_by_sequence`` (Bio.PDB),
    и из EDA (numpy-массивы атомов) — без дублирования логики выравнивания.

    Считаем выравнивание для всех пар (low_chain, high_chain), берём пары с
    identity >= ``identity_min`` и назначаем их по убыванию числа совпавших
    остатков, не переиспользуя цепь дважды (корректно для гомо-олигомеров).
    """
    if not low_chains or not high_chains:
        return []

    aligner = _make_aligner()

    # Кандидаты: (n_pairs, identity, lcid, hcid, pairs)
    candidates: list[tuple[int, float, str, str, list]] = []
    for lcid, lres in low_chains.items():
        for hcid, hres in high_chains.items():
            pairs, identity = _align_residue_pairs(lres, hres, aligner)
            if pairs and identity >= identity_min:
                candidates.append((len(pairs), identity, lcid, hcid, pairs))

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    used_low: set[str] = set()
    used_high: set[str] = set()
    matched_pairs: list[tuple[ResidueInfo, ResidueInfo]] = []
    for _, _, lcid, hcid, pairs in candidates:
        if lcid in used_low or hcid in used_high:
            continue
        used_low.add(lcid)
        used_high.add(hcid)
        matched_pairs.extend(pairs)

    # Сортируем по цепи и номеру low-остатка, сохраняя соответствие с high.
    # Приоритет кандидатов цепей не должен менять порядок остатков в датасете.
    matched_pairs.sort(key=lambda pair: (pair[0].chain_id, pair[0].resseq, pair[0].icode))
    return matched_pairs


def align_structures_by_sequence(
    low_path: str,
    high_path: str,
    identity_min: float = 0.8,
) -> SeqAlignment:
    """Sequence-based выравнивание low/high структур (из файлов Bio.PDB).

    Цепи извлекаются из структур и сопоставляются через :func:`match_chains`.
    """
    low_struct = _make_parser(low_path).get_structure("low", low_path)
    high_struct = _make_parser(high_path).get_structure("high", high_path)

    low_chains = _extract_chain_residues(low_struct)
    high_chains = _extract_chain_residues(high_struct)

    n_res_low = sum(len(v) for v in low_chains.values())
    n_res_high = sum(len(v) for v in high_chains.values())

    matched_pairs = match_chains(low_chains, high_chains, identity_min=identity_min)
    return SeqAlignment(matched_pairs, n_res_low, n_res_high)
