"""EDA для датасета апскейлинга белков.

Пять блоков:
    A — инвентаризация и парсимость PDB/CIF файлов
    B — валидность каждой good/bad пары как supervision
    C — однозначность таргета (good-good RMSD внутри одного uniprot_id)
    D — честность train/val split (overlap, distribution shift)
    E — identity-baseline (RMSD(coords_bad, coords_good) после Kabsch)

Эффективность:
    - каждый PDB парсится ровно один раз, результат кэшируется в pickle —
      повторный запуск пропускает Block A;
    - парсинг файлов и попарные RMSD считаются параллельно
      через ProcessPoolExecutor (по числу CPU);
    - попарные RMSD считаются только по общим атомам (без полного cdist).

Запуск:
    python eda/run_eda.py                    # с дефолтами ниже
    python eda/run_eda.py --workers 8        # параллелизм
    python eda/run_eda.py --blocks BCDE      # пропустить парсинг (использовать кэш)

Перед первым запуском замените CSV_PATH / DATA_DIR ниже на свои.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Bio.PDB import MMCIFParser, PDBParser  # noqa: E402


# ---------------------------------------------------------------------------
Пути
# ---------------------------------------------------------------------------
CSV_PATH = "PATH/TO/pdb_df.csv"
DATA_DIR = "PATH/TO/data"
RESULTS_DIR = "eda/results"
CACHE_PATH = "eda/results/_parse_cache.pkl"
# ---------------------------------------------------------------------------


LOG = logging.getLogger("eda")


# ---------------------------------------------------------------------------
# Парсинг структур
# ---------------------------------------------------------------------------

ATOMS_DTYPE = np.dtype([
    ("chain_id", "U6"), ("resseq", "i4"), ("icode", "U2"),
    ("resname", "U6"), ("atom_name", "U6"), ("altloc", "U2"),
    ("element", "U4"), ("x", "f4"), ("y", "f4"), ("z", "f4"),
])


def _parser_for(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def parse_structure(pdb_id: str, data_dir: str) -> dict:
    """Парсит один файл; возвращает словарь с numpy-массивом атомов или ошибкой."""
    path = None
    base = Path(data_dir) / pdb_id
    for ext in (".pdb", ".cif"):
        candidate = Path(str(base) + ext)
        if candidate.exists():
            path = candidate
            break
    if path is None:
        return {"pdb_id": pdb_id, "ok": False, "error": "file_not_found", "path": None}

    try:
        s = _parser_for(str(path)).get_structure(pdb_id, str(path))
    except Exception as e:
        return {
            "pdb_id": pdb_id, "ok": False,
            "error": f"parse:{type(e).__name__}", "path": str(path),
        }

    rows: list[tuple] = []
    chains: set[str] = set()
    residues: set[tuple] = set()
    for atom in s.get_atoms():
        res = atom.get_parent()
        if res.get_id()[0] != " ":  # только полипептидные остатки
            continue
        chain = res.get_parent()
        chain_id = str(chain.id)
        _, resseq, icode = res.id
        rows.append((
            chain_id, int(resseq), str(icode).strip(),
            res.resname.strip(), atom.name.strip(),
            (atom.get_altloc() or "").strip(),
            (atom.element or "").strip().upper(),
            float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2]),
        ))
        chains.add(chain_id)
        residues.add((chain_id, int(resseq), str(icode).strip()))

    if not rows:
        return {
            "pdb_id": pdb_id, "ok": False,
            "error": "no_polymer_atoms", "path": str(path),
        }

    atoms = np.array(rows, dtype=ATOMS_DTYPE)
    return {
        "pdb_id": pdb_id, "ok": True, "atoms": atoms, "path": str(path),
        "n_atoms": int(len(rows)), "n_chains": int(len(chains)),
        "n_residues": int(len(residues)),
    }


# ---------------------------------------------------------------------------
# Геометрия: общие атомы + Kabsch RMSD
# ---------------------------------------------------------------------------

def _atom_keys(atoms: np.ndarray) -> list[tuple]:
    """(chain_id, resseq, icode, atom_name, altloc) — без resname для устойчивости к мутантам."""
    return list(zip(
        atoms["chain_id"].tolist(),
        atoms["resseq"].tolist(),
        atoms["icode"].tolist(),
        atoms["atom_name"].tolist(),
        atoms["altloc"].tolist(),
    ))


def _common_xyz(atoms_a: np.ndarray, atoms_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys_a = {k: i for i, k in enumerate(_atom_keys(atoms_a))}
    keys_b = {k: i for i, k in enumerate(_atom_keys(atoms_b))}
    common = sorted(
        set(keys_a) & set(keys_b),
        key=lambda k: (k[0], k[1], k[2], k[3], k[4]),
    )
    if not common:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    idx_a = np.fromiter((keys_a[k] for k in common), dtype=np.int64, count=len(common))
    idx_b = np.fromiter((keys_b[k] for k in common), dtype=np.int64, count=len(common))
    A = np.stack(
        [atoms_a["x"][idx_a], atoms_a["y"][idx_a], atoms_a["z"][idx_a]], axis=-1
    )
    B = np.stack(
        [atoms_b["x"][idx_b], atoms_b["y"][idx_b], atoms_b["z"][idx_b]], axis=-1
    )
    return A.astype(np.float32), B.astype(np.float32)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD после оптимального наложения по методу Кабша."""
    n = P.shape[0]
    if n < 1:
        return float("nan")
    Pc = P.mean(0)
    Qc = Q.mean(0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = Q0.T @ P0
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, float(d)]) @ Vt
    Q_aligned = (R @ Q0.T).T + Pc
    return float(np.sqrt(((P - Q_aligned) ** 2).sum() / n))


# ---------------------------------------------------------------------------
# Block A — инвентаризация
# ---------------------------------------------------------------------------

def block_a(df: pd.DataFrame, data_dir: str, workers: int, results_dir: Path
            ) -> tuple[dict, pd.DataFrame]:
    LOG.info("Block A: parsing %d unique structures...", df["pdb"].nunique())
    pdbs = df["pdb"].unique().tolist()
    parsed: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(parse_structure, p, data_dir): p for p in pdbs}
        for fut in tqdm(as_completed(futures), total=len(pdbs), desc="parse"):
            r = fut.result()
            parsed[r["pdb_id"]] = r

    rows = []
    for pdb_id, r in parsed.items():
        meta = df[df["pdb"] == pdb_id].iloc[0]
        rows.append({
            "pdb": pdb_id, "uniprot_id": meta["uniprot_id"],
            "resolution": meta["resolution"], "method": meta["method"],
            "class": meta["class"], "ok": r["ok"], "error": r.get("error", ""),
            "n_atoms": r.get("n_atoms", 0), "n_chains": r.get("n_chains", 0),
            "n_residues": r.get("n_residues", 0),
        })
    inv = pd.DataFrame(rows).sort_values("pdb").reset_index(drop=True)
    inv.to_csv(results_dir / "block_a_inventory.csv", index=False)

    inv_ok = inv[inv["ok"]]
    summary = {
        "n_files": int(len(inv)),
        "ok": int(inv["ok"].sum()),
        "failed": int((~inv["ok"]).sum()),
        "by_error": inv[~inv["ok"]]["error"].value_counts().to_dict(),
        "by_method": inv["method"].value_counts().to_dict(),
        "by_class": inv["class"].value_counts().to_dict(),
        "resolution": {
            "good": inv[inv["class"] == "good"]["resolution"].describe().to_dict(),
            "bad":  inv[inv["class"] == "bad"]["resolution"].describe().to_dict(),
        },
        "n_residues_per_structure": (
            inv_ok["n_residues"].describe().to_dict() if len(inv_ok) else {}
        ),
        "n_chains_per_structure": (
            inv_ok["n_chains"].describe().to_dict() if len(inv_ok) else {}
        ),
    }
    with open(results_dir / "block_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for klass, color in (("good", "tab:green"), ("bad", "tab:red")):
        sub = inv[inv["class"] == klass]["resolution"].dropna()
        if len(sub):
            axes[0].hist(sub, bins=40, alpha=0.6, label=klass, color=color)
    axes[0].set_xlabel("Resolution, Å")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].set_title(f"Resolution by class (n={len(inv)})")

    if len(inv_ok):
        axes[1].hist(inv_ok["n_residues"], bins=40, color="tab:blue")
        axes[1].set_xlabel("Residues per structure")
        axes[1].set_title("Structure size")
    fig.tight_layout()
    fig.savefig(results_dir / "block_a_distributions.png", dpi=100)
    plt.close(fig)

    LOG.info(
        "Block A done. ok=%d failed=%d (errors: %s)",
        summary["ok"], summary["failed"], summary["by_error"],
    )
    return parsed, inv


# ---------------------------------------------------------------------------
# Block B — валидность пар
# ---------------------------------------------------------------------------

def _pair_validity_one(args):
    (atoms_good, atoms_bad, good_pdb, bad_pdb, uid,
     good_res, bad_res, good_method, bad_method) = args

    A, B = _common_xyz(atoms_good, atoms_bad)
    n_common = int(A.shape[0])
    rmsd = kabsch_rmsd(A, B) if n_common >= 4 else float("nan")

    res_keys_good = set(zip(
        atoms_good["chain_id"].tolist(),
        atoms_good["resseq"].tolist(),
        atoms_good["icode"].tolist(),
    ))
    res_keys_bad = set(zip(
        atoms_bad["chain_id"].tolist(),
        atoms_bad["resseq"].tolist(),
        atoms_bad["icode"].tolist(),
    ))
    n_common_res = len(res_keys_good & res_keys_bad)

    n_atoms_good = int(len(atoms_good))
    n_atoms_bad = int(len(atoms_bad))
    coverage = n_common / max(n_atoms_good, n_atoms_bad, 1)

    chains_good = set(atoms_good["chain_id"].tolist())
    chains_bad = set(atoms_bad["chain_id"].tolist())

    return {
        "uniprot_id": uid, "good_pdb": good_pdb, "bad_pdb": bad_pdb,
        "good_resolution": good_res, "bad_resolution": bad_res,
        "delta_resolution": bad_res - good_res,
        "good_method": good_method, "bad_method": bad_method,
        "n_atoms_good": n_atoms_good, "n_atoms_bad": n_atoms_bad,
        "n_common_atoms": n_common, "n_common_residues": n_common_res,
        "coverage": coverage, "kabsch_rmsd_initial": rmsd,
        "n_chains_good": len(chains_good), "n_chains_bad": len(chains_bad),
        "chain_overlap": len(chains_good & chains_bad),
    }


def block_b(df: pd.DataFrame, parsed: dict, results_dir: Path, workers: int
            ) -> pd.DataFrame:
    LOG.info("Block B: pair-level validity...")
    pair_args = []
    for uid, group in df.groupby("uniprot_id"):
        good = group[group["class"] == "good"]
        bad = group[group["class"] == "bad"]
        for _, g in good.iterrows():
            if not parsed.get(g["pdb"], {}).get("ok"):
                continue
            for _, b in bad.iterrows():
                if not parsed.get(b["pdb"], {}).get("ok"):
                    continue
                pair_args.append((
                    parsed[g["pdb"]]["atoms"], parsed[b["pdb"]]["atoms"],
                    g["pdb"], b["pdb"], uid,
                    float(g["resolution"]), float(b["resolution"]),
                    str(g["method"]), str(b["method"]),
                ))

    if not pair_args:
        LOG.warning("Block B: no pairs to evaluate")
        return pd.DataFrame()

    LOG.info("Total pairs: %d", len(pair_args))
    rows = []
    chunk = max(1, min(64, len(pair_args) // (workers * 4) or 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(
            ex.map(_pair_validity_one, pair_args, chunksize=chunk),
            total=len(pair_args), desc="pairs",
        ):
            rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "block_b_pairs.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    out["coverage"].hist(bins=40, ax=axes[0, 0], color="tab:blue")
    axes[0, 0].axvline(0.6, color="red", linestyle="--", label="0.6")
    axes[0, 0].set_title("Coverage = common_atoms / max(n_a, n_b)")
    axes[0, 0].legend()

    out["kabsch_rmsd_initial"].hist(bins=40, ax=axes[0, 1], color="tab:orange")
    axes[0, 1].axvline(5.0, color="red", linestyle="--", label="5 Å")
    axes[0, 1].set_title("Kabsch RMSD initial (= identity baseline)")
    axes[0, 1].set_xlabel("RMSD, Å")
    axes[0, 1].legend()

    out["delta_resolution"].hist(bins=40, ax=axes[1, 0], color="tab:green")
    axes[1, 0].set_title("Δresolution (bad − good)")
    axes[1, 0].set_xlabel("Å")

    out["n_common_atoms"].hist(bins=40, ax=axes[1, 1], color="tab:purple")
    axes[1, 1].set_title("Common atoms per pair")
    fig.tight_layout()
    fig.savefig(results_dir / "block_b_distributions.png", dpi=100)
    plt.close(fig)

    summary = {
        "n_pairs": int(len(out)),
        "low_coverage_<0.6": int((out["coverage"] < 0.6).sum()),
        "low_coverage_<0.3": int((out["coverage"] < 0.3).sum()),
        "high_rmsd_>10A": int((out["kabsch_rmsd_initial"] > 10).sum()),
        "high_rmsd_>5A": int((out["kabsch_rmsd_initial"] > 5).sum()),
        "no_chain_overlap": int((out["chain_overlap"] == 0).sum()),
        "rmsd": out["kabsch_rmsd_initial"].describe().to_dict(),
        "coverage": out["coverage"].describe().to_dict(),
    }
    with open(results_dir / "block_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    LOG.info(
        "Block B done. n_pairs=%d, low_coverage<0.6=%d, high_rmsd>5Å=%d",
        summary["n_pairs"], summary["low_coverage_<0.6"], summary["high_rmsd_>5A"],
    )
    return out


# ---------------------------------------------------------------------------
# Block C — однозначность таргета
# ---------------------------------------------------------------------------

def _good_good_one(args):
    a_atoms, b_atoms = args
    A, B = _common_xyz(a_atoms, b_atoms)
    if A.shape[0] < 4:
        return float("nan"), int(A.shape[0])
    return kabsch_rmsd(A, B), int(A.shape[0])


def block_c(df: pd.DataFrame, parsed: dict, results_dir: Path, workers: int
            ) -> pd.DataFrame:
    LOG.info("Block C: target ambiguity (good-good RMSD)...")
    pair_args = []
    pair_meta = []
    for uid, group in df.groupby("uniprot_id"):
        goods = [
            p for p in group[group["class"] == "good"]["pdb"].tolist()
            if parsed.get(p, {}).get("ok")
        ]
        if len(goods) < 2:
            continue
        for a, b in combinations(goods, 2):
            pair_args.append((parsed[a]["atoms"], parsed[b]["atoms"]))
            pair_meta.append((uid, a, b))

    if not pair_args:
        LOG.warning("Block C: <2 good per uniprot — nothing to compare")
        return pd.DataFrame()

    LOG.info("good-good pairs: %d", len(pair_args))
    chunk = max(1, min(64, len(pair_args) // (workers * 4) or 1))
    rmsds = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(
            ex.map(_good_good_one, pair_args, chunksize=chunk),
            total=len(pair_args), desc="good-good",
        ):
            rmsds.append(r)

    rows = [
        {"uniprot_id": uid, "good_a": a, "good_b": b, "rmsd": rmsd, "n_common_atoms": n}
        for (uid, a, b), (rmsd, n) in zip(pair_meta, rmsds)
    ]
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "block_c_good_good.csv", index=False)

    uniprot_stats = (
        out.groupby("uniprot_id")
        .agg(
            n_good_pairs=("rmsd", "size"),
            mean_rmsd=("rmsd", "mean"),
            max_rmsd=("rmsd", "max"),
            p90_rmsd=("rmsd", lambda x: x.quantile(0.9)),
        )
        .reset_index()
        .sort_values("max_rmsd", ascending=False)
    )
    uniprot_stats.to_csv(results_dir / "block_c_uniprot_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    out["rmsd"].hist(bins=40, ax=axes[0], color="tab:blue")
    axes[0].axvline(3.0, color="red", linestyle="--", label="3 Å")
    axes[0].set_title(f"good-good RMSD across all pairs (n={len(out)})")
    axes[0].set_xlabel("RMSD, Å")
    axes[0].legend()

    uniprot_stats["max_rmsd"].hist(bins=40, ax=axes[1], color="tab:orange")
    axes[1].axvline(3.0, color="red", linestyle="--", label="3 Å")
    axes[1].set_title(f"Max good-good RMSD per uniprot (n={len(uniprot_stats)})")
    axes[1].set_xlabel("Max RMSD, Å")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "block_c_good_good.png", dpi=100)
    plt.close(fig)

    summary = {
        "n_good_good_pairs": int(len(out)),
        "n_uniprots_with_>=2_good": int(len(uniprot_stats)),
        "ambiguous_>3A": int((uniprot_stats["max_rmsd"] > 3).sum()),
        "ambiguous_>5A": int((uniprot_stats["max_rmsd"] > 5).sum()),
        "median_max_good_good_rmsd": (
            float(uniprot_stats["max_rmsd"].median()) if len(uniprot_stats) else None
        ),
        "top10_ambiguous_uniprots": uniprot_stats.head(10).to_dict(orient="records"),
    }
    with open(results_dir / "block_c_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    LOG.info(
        "Block C done. uniprots with ambiguity>3Å: %d / %d",
        summary["ambiguous_>3A"], summary["n_uniprots_with_>=2_good"],
    )
    return out


# ---------------------------------------------------------------------------
# Block D — честность train/val split
# ---------------------------------------------------------------------------

def _maybe_ks(x: np.ndarray, y: np.ndarray):
    """Возвращает (stat, pvalue) или None, если scipy недоступен / выборок мало."""
    if len(x) < 5 or len(y) < 5:
        return None
    try:
        from scipy.stats import ks_2samp
    except Exception:
        return None
    stat, pval = ks_2samp(x, y)
    return float(stat), float(pval)


def block_d(pair_df: pd.DataFrame, results_dir: Path,
            val_fraction: float = 0.2, seed: int = 42) -> dict:
    LOG.info("Block D: split fairness...")
    if pair_df.empty:
        LOG.warning("Block D skipped: no pairs")
        return {}

    uids = sorted(pair_df["uniprot_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(uids)
    n_val = max(1, int(round(val_fraction * len(uids))))
    val_uids = set(uids[:n_val])
    train_pairs = pair_df[~pair_df["uniprot_id"].isin(val_uids)].copy()
    val_pairs = pair_df[pair_df["uniprot_id"].isin(val_uids)].copy()

    overlap = {
        "val_fraction": val_fraction,
        "seed": seed,
        "n_train_pairs": int(len(train_pairs)),
        "n_val_pairs": int(len(val_pairs)),
        "n_train_uniprots": int(train_pairs["uniprot_id"].nunique()),
        "n_val_uniprots": int(val_pairs["uniprot_id"].nunique()),
        "uniprot_id_overlap": int(len(
            set(train_pairs["uniprot_id"]) & set(val_pairs["uniprot_id"])
        )),
        "good_pdb_overlap": int(len(
            set(train_pairs["good_pdb"]) & set(val_pairs["good_pdb"])
        )),
        "bad_pdb_overlap": int(len(
            set(train_pairs["bad_pdb"]) & set(val_pairs["bad_pdb"])
        )),
    }

    cols = ["bad_resolution", "delta_resolution", "kabsch_rmsd_initial",
            "n_common_atoms", "coverage"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 3 * len(cols)))
    if len(cols) == 1:
        axes = [axes]
    ks_results = {}
    for ax, col in zip(axes, cols):
        t = train_pairs[col].dropna().to_numpy()
        v = val_pairs[col].dropna().to_numpy()
        if len(t) and len(v):
            lo = min(t.min(), v.min())
            hi = max(t.max(), v.max())
            bins = np.linspace(lo, hi, 40) if hi > lo else 20
        else:
            bins = 20
        ax.hist(t, bins=bins, alpha=0.5, label=f"train (n={len(t)})",
                color="tab:blue", density=True)
        ax.hist(v, bins=bins, alpha=0.5, label=f"val (n={len(v)})",
                color="tab:orange", density=True)
        title = col
        ks = _maybe_ks(t, v)
        if ks is not None:
            stat, pval = ks
            ks_results[col] = {"ks_stat": stat, "p_value": pval}
            title = f"{col} | KS={stat:.3f}, p={pval:.3g}"
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "block_d_distributions.png", dpi=100)
    plt.close(fig)

    overlap["ks_tests"] = ks_results
    with open(results_dir / "block_d_split.json", "w") as f:
        json.dump(overlap, f, indent=2, default=str)

    train_pairs.to_csv(results_dir / "block_d_train_pairs.csv", index=False)
    val_pairs.to_csv(results_dir / "block_d_val_pairs.csv", index=False)

    LOG.info(
        "Block D done. train=%d, val=%d, leakage(uniprot/good_pdb/bad_pdb)=%d/%d/%d",
        overlap["n_train_pairs"], overlap["n_val_pairs"],
        overlap["uniprot_id_overlap"],
        overlap["good_pdb_overlap"], overlap["bad_pdb_overlap"],
    )
    return overlap


# ---------------------------------------------------------------------------
# Block E — identity baseline
# ---------------------------------------------------------------------------

def block_e(pair_df: pd.DataFrame, results_dir: Path) -> dict:
    """Identity baseline = RMSD(coords_bad, coords_good) после Kabsch.

    Эта цифра — обязательный floor, ниже которого модель должна опускаться,
    чтобы считаться полезной.
    """
    LOG.info("Block E: identity baseline...")
    if pair_df.empty:
        LOG.warning("Block E skipped: empty pair_df")
        return {}

    rmsd = pair_df["kabsch_rmsd_initial"].dropna()
    if rmsd.empty:
        LOG.warning("Block E: all kabsch_rmsd_initial are NaN")
        return {}

    summary = {
        "n_pairs_used": int(len(rmsd)),
        "identity_rmsd_mean": float(rmsd.mean()),
        "identity_rmsd_median": float(rmsd.median()),
        "identity_rmsd_std": float(rmsd.std()),
        "identity_rmsd_p10": float(rmsd.quantile(0.10)),
        "identity_rmsd_p25": float(rmsd.quantile(0.25)),
        "identity_rmsd_p75": float(rmsd.quantile(0.75)),
        "identity_rmsd_p90": float(rmsd.quantile(0.90)),
        "identity_rmsd_max": float(rmsd.max()),
        "identity_rmsd_min": float(rmsd.min()),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rmsd, bins=40, color="tab:orange")
    ax.axvline(summary["identity_rmsd_median"], color="black",
               linestyle="--", label=f"median {summary['identity_rmsd_median']:.2f} Å")
    ax.set_xlabel("Identity RMSD (Kabsch), Å")
    ax.set_ylabel("Count")
    ax.set_title("Identity baseline distribution — модель должна быть лучше этого")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "block_e_baseline.png", dpi=100)
    plt.close(fig)

    with open(results_dir / "block_e_baseline.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    LOG.info(
        "Block E done. identity RMSD: median=%.3f Å, p90=%.3f Å",
        summary["identity_rmsd_median"], summary["identity_rmsd_p90"],
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=CSV_PATH, help=f"Default: {CSV_PATH}")
    parser.add_argument("--data-dir", default=DATA_DIR, help=f"Default: {DATA_DIR}")
    parser.add_argument("--output", default=RESULTS_DIR, help=f"Default: {RESULTS_DIR}")
    parser.add_argument("--cache", default=CACHE_PATH, help=f"Default: {CACHE_PATH}")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", default="ABCDE",
                        help="Подмножество блоков для запуска, например 'BCE'")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.csv).exists():
        LOG.error(
            "CSV не найден: %s — поправьте CSV_PATH в начале скрипта или флаг --csv",
            args.csv,
        )
        return 1
    if not Path(args.data_dir).exists():
        LOG.error(
            "Data dir не найден: %s — поправьте DATA_DIR в начале скрипта или флаг --data-dir",
            args.data_dir,
        )
        return 1

    df = pd.read_csv(args.csv)
    LOG.info("Loaded CSV: %d rows, %d uniprot_id, %d unique pdb",
             len(df), df["uniprot_id"].nunique(), df["pdb"].nunique())

    cache_path = Path(args.cache)
    parsed: dict | None = None
    if cache_path.exists() and "A" not in args.blocks:
        try:
            with open(cache_path, "rb") as f:
                parsed = pickle.load(f)
            LOG.info("Loaded parse cache: %d structures", len(parsed))
        except Exception as e:
            LOG.warning("Cache load failed: %s — will re-parse", e)
            parsed = None

    if parsed is None:
        parsed, _ = block_a(df, args.data_dir, args.workers, results_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(parsed, f)
    elif "A" in args.blocks:
        LOG.info("Block A: cache already present — to force re-parse delete %s", cache_path)

    pair_df = pd.DataFrame()
    if "B" in args.blocks:
        pair_df = block_b(df, parsed, results_dir, args.workers)
    elif (results_dir / "block_b_pairs.csv").exists():
        pair_df = pd.read_csv(results_dir / "block_b_pairs.csv")
        LOG.info("Block B: re-using cached %d pairs", len(pair_df))

    if "C" in args.blocks:
        block_c(df, parsed, results_dir, args.workers)

    if "D" in args.blocks:
        block_d(pair_df, results_dir, args.val_fraction, args.seed)

    if "E" in args.blocks:
        block_e(pair_df, results_dir)

    LOG.info("Done. Results in %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
