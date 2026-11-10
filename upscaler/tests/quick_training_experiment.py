#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from upscaler.config import (
    COVERAGE_MIN, CSV_FILE, DATA_FOLDER, DEVICE, PAIR_RMSD_MAX,
    RESOLUTION_BAD, RESOLUTION_GOOD, SEQ_IDENTITY_MIN,
)
from upscaler.data.dataset import ProteinUpscalingDataset, collate_batch
from upscaler.loss.loss import ATOM_NAME_MAP, ATOM_TYPE_MAP, ProteinUpscalingLoss
from upscaler.model.encoders import RESIDUE_TYPE_MAP
from upscaler.model.geometric import apply_frames, frames_from_bb_coords, rots_from_vec
from upscaler.model.upscaler import ProteinUpscaler
from upscaler.training.pipeline import TrainingPipeline
from upscaler.training.split import split_indices_by_uniprot, uniprot_id_for_index
from upscaler.training.train import get_optimizer, load_checkpoint, save_checkpoint
from upscaler.utils.metrics import QualityMetrics

LOGGER = logging.getLogger('quick_experiment')


class CachedPairs(Dataset):
    """Materialize only the chosen pairs, avoiding PDB parsing on every step."""
    def __init__(self, samples, uids):
        self.samples, self.uids = samples, uids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def uniprot_id_for_pair(self, index):
        return self.uids[index]


def place_atom(a, b, c, length, angle, torsion):
    """Internal coordinates for a synthetic peptide with valid bond lengths."""
    direction = torch.nn.functional.normalize(c - b, dim=0)
    normal = torch.nn.functional.normalize(torch.cross(b - a, direction, dim=0), dim=0)
    tangent = torch.cross(normal, direction, dim=0)
    angle, torsion = math.radians(angle), math.radians(torsion)
    return c + length * (-math.cos(angle) * direction + math.sin(angle) *
                         (math.cos(torsion) * tangent + math.sin(torsion) * normal))


def synthetic_dataset(count, residues, seed):
    """Distinct synthetic backbones; degradation moves each residue rigidly."""
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(count):
        n = torch.tensor([0., 0., 0.])
        ca = torch.tensor([1.458, 0., 0.])
        angle = math.radians(111.2)
        c = ca + 1.525 * torch.tensor([-math.cos(angle), math.sin(angle), 0.])
        backbone = []
        for residue in range(residues):
            o = place_atom(n, ca, c, 1.231, 120.8, 135.)
            backbone.append(torch.stack([n, ca, c, o]))
            if residue + 1 < residues:
                psi = -45. + float(torch.randn((), generator=generator)) * 8.
                phi = -60. + float(torch.randn((), generator=generator)) * 8.
                next_n = place_atom(n, ca, c, 1.329, 116.2, psi)
                next_ca = place_atom(ca, c, next_n, 1.458, 121.7, 180.)
                next_c = place_atom(c, next_n, next_ca, 1.525, 111.2, phi)
                n, ca, c = next_n, next_ca, next_c
        good = torch.stack(backbone)
        rg, tg = frames_from_bb_coords(good[:, 0], good[:, 1], good[:, 2])
        rotation_noise = rots_from_vec(0.12 * torch.randn(residues, 3, generator=generator))
        translation_noise = 0.35 * torch.randn(residues, 3, generator=generator)
        bad = apply_frames(rotation_noise[:, None], (tg + translation_noise)[:, None],
                           good - tg[:, None])
        rb, tb = frames_from_bb_coords(bad[:, 0], bad[:, 1], bad[:, 2])
        samples.append({
            'coords_bad': bad.reshape(-1, 3), 'coords_good': good.reshape(-1, 3),
            'rots_bad': rb, 'trans_bad': tb, 'rots_good': rg, 'trans_good': tg,
            'res_map': torch.arange(residues).repeat_interleave(4),
            'res_chain': torch.zeros(residues, dtype=torch.long),
            'atom_names': torch.tensor([ATOM_NAME_MAP[x] for x in ('N', 'CA', 'C', 'O')]).repeat(residues),
            'atom_types': torch.tensor([ATOM_TYPE_MAP[x] for x in ('N', 'C', 'C', 'O')]).repeat(residues),
            'residue_types': torch.full((residues * 4,), RESIDUE_TYPE_MAP['ALA'], dtype=torch.long),
            'length': residues * 4, 'num_residues': residues,
        })
    return CachedPairs(samples, [f'synthetic_{i}' for i in range(count)])


def select_pairs(dataset, args):
    if args.skip_phase2:
        train, val = list(range(len(dataset))), []
        train_count, val_count = args.overfit_pairs, 0
    else:
        train, val = split_indices_by_uniprot(
            dataset, val_fraction=args.val_pairs / (args.train_pairs + args.val_pairs), seed=args.seed,
        )
        train_count, val_count = args.train_pairs, args.val_pairs
    if len(train) < train_count or len(val) < val_count:
        raise ValueError(f'Independent split provides {len(train)} train / {len(val)} val pairs; '
                         'reduce --train-pairs/--val-pairs or provide more UniProt groups')
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)
    train, val = train[:train_count], val[:val_count]
    train_uids = {uniprot_id_for_index(dataset, i) for i in train}
    val_uids = {uniprot_id_for_index(dataset, i) for i in val}
    if train_uids & val_uids or None in val_uids:
        raise RuntimeError('Selected train and validation proteins are not independent')
    selections, manifest = [], []
    for split, indices in (('train', train), ('val', val)):
        samples, uids = [], []
        for index in indices:
            sample = dataset[index]
            if int(sample['length']) > args.max_atoms:
                raise ValueError(f'Pair {index} exceeds --max-atoms')
            for key in ('coords_bad', 'coords_good', 'rots_bad', 'rots_good', 'trans_bad', 'trans_good'):
                if not torch.isfinite(sample[key]).all():
                    raise ValueError(f'Non-finite {key} in pair {index}')
            uid = uniprot_id_for_index(dataset, index)
            row = {'split': split, 'dataset_index': index, 'uniprot_id': uid,
                   'atoms': int(sample['length']), 'residues': int(sample['num_residues'])}
            if isinstance(dataset, ProteinUpscalingDataset):
                good, bad = dataset.all_pairs[dataset.good_pair_indices[index]]
                row.update(good_pdb=str(good['pdb']), bad_pdb=str(bad['pdb']))
            manifest.append(row)
            samples.append(sample)
            uids.append(uid)
        selections.append(CachedPairs(samples, uids))
    return selections[0], selections[1], manifest


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def true_data(batch):
    return {'rots': batch['rots_good'], 'trans': batch['trans_good'],
            'coords': batch['coords_good'], 'atom_types': batch['atom_types'],
            'atom_names': batch['atom_names'], 'res_map': batch['res_map'],
            'res_chain': batch['res_chain']}


def geometry_metrics(coords, batch):
    """Chain-aware peptide lengths and preservation of all intra-residue distances."""
    mask, res_map, names = batch['mask'][0], batch['res_map'][0], batch['atom_names'][0]
    chain = batch['res_chain'][0]
    bonds, internal_errors = [], []
    previous_c, previous_chain, previous_residue = None, None, None
    for residue in torch.nonzero(batch['res_mask'][0], as_tuple=False).flatten().tolist():
        indices = torch.nonzero(mask & (res_map == residue), as_tuple=False).flatten()
        n_indices = indices[names[indices] == ATOM_NAME_MAP['N']]
        c_indices = indices[names[indices] == ATOM_NAME_MAP['C']]
        current_chain = int(chain[residue])
        if (previous_c is not None and n_indices.numel() and current_chain >= 0
                and current_chain == previous_chain and residue == previous_residue + 1):
            bonds.append(float(torch.norm(coords[0, previous_c] - coords[0, n_indices[0]])))
        previous_c = int(c_indices[0]) if c_indices.numel() else None
        previous_chain, previous_residue = current_chain, residue
        if indices.numel() > 1:
            before = torch.cdist(batch['coords_bad'][0, indices][None].float(),
                                 batch['coords_bad'][0, indices][None].float())
            after = torch.cdist(coords[0, indices][None], coords[0, indices][None])
            internal_errors.append(float((after - before).abs().max()))
    return {
        'peptide_bonds': len(bonds),
        'peptide_mean': float(np.mean(bonds)) if bonds else None,
        'peptide_mae': float(np.mean(np.abs(np.asarray(bonds) - 1.329))) if bonds else None,
        'peptide_valid_fraction': float(np.mean(np.abs(np.asarray(bonds) - 1.329) <= 0.2)) if bonds else None,
        'internal_distance_error_max': max(internal_errors, default=0.),
    }


@torch.no_grad()
def evaluate(pipeline, dataset, baseline=False, retain=0):
    model, device = pipeline.model, pipeline.device
    model.eval()
    metrics_calculator = pipeline.metrics_calculator
    rows, saved, updates = [], [], []
    current_mask = None

    def update_hook(module, inputs, output):
        updates.extend(output[1].detach().float().norm(dim=-1)[current_mask].cpu().tolist())

    handle = model.update_head.register_forward_hook(update_hook) if not baseline else None
    try:
        for index, sample in enumerate(dataset):
            batch = to_device(collate_batch([sample]), device)
            current_mask = batch['res_mask']
            if baseline:
                pred = {'rots': batch['rots_bad'], 'trans': batch['trans_bad'],
                        'coords': batch['coords_bad'], 'mask': batch['mask'], 'res_mask': batch['res_mask']}
            else:
                with torch.autocast(device_type=device.type, enabled=pipeline.use_amp):
                    pred = model(batch)
            pred = {key: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value for key, value in pred.items()}
            loss, components = pipeline.loss_fn(pred, true_data(batch))
            coords = pred['coords']
            rotations = pred['rots'][batch['res_mask']]
            row = {
                'loss': float(loss), 'fape': float(components['fape']),
                'rmsd': float(metrics_calculator.compute_rmsd(coords, batch['coords_good'], mask=batch['mask'])),
                'lddt': float(metrics_calculator.compute_lddt(coords, batch['coords_good'], mask=batch['mask'])),
                'clash': float(metrics_calculator.compute_clash_score(coords, batch['atom_types'], mask=batch['mask'])),
                'clash_loss': float(components['clash']),
                'rotation_orthogonality_error': float((rotations.transpose(-1, -2) @ rotations
                                                      - torch.eye(3, device=device)).abs().max()),
                'rotation_determinant_error': float((torch.det(rotations) - 1.).abs().max()),
                'identity_error': float((coords - batch['coords_bad']).abs()[batch['mask']].max()),
                **geometry_metrics(coords, batch),
            }
            if any(value is not None and not math.isfinite(value) for value in row.values()):
                raise FloatingPointError(f'Non-finite evaluation metric for sample {index}')
            rows.append(row)
            if index < retain:
                saved.append((sample, coords[0].cpu()))
    finally:
        if handle is not None:
            handle.remove()
    if not rows:
        raise ValueError('Cannot evaluate an empty selection')
    aggregate = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        aggregate[key] = (sum(values) if key == 'peptide_bonds' else
                          max(values) if key.endswith('_max') or key.endswith('_error')
                          else float(np.mean(values))) if values else None
    aggregate['translation_update_mean'] = float(np.mean(updates)) if updates else 0.
    aggregate['translation_update_max'] = max(updates, default=0.)
    aggregate['head_weight_norm'] = float(model.update_head.predictor.weight.float().norm())
    return aggregate, saved


def gradient_metrics(model):
    norm_sq, result = 0., {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f'Non-finite gradient: {name}')
            norm_sq += float(parameter.grad.detach().double().square().sum())
    result['clipped_grad_norm'] = math.sqrt(norm_sq)
    for index, block in enumerate(model.local_blocks):
        grad = block.equivariant_layer.tp.weight.grad
        result[f'local_{index}_grad_norm'] = float(grad.norm()) if grad is not None else 0.
    for index, block in enumerate(model.ipa_blocks):
        grad = block.to_out.weight.grad
        result[f'ipa_{index}_grad_norm'] = float(grad.norm()) if grad is not None else 0.
    return result


def make_pipeline(args, device, seed):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = ProteinUpscaler(num_iterations=args.num_iterations).to(device)
    return TrainingPipeline(
        model, ProteinUpscalingLoss(clash_weight=0.05, physics_weight=0.0).to(device),
        get_optimizer(model, lr=args.lr), device, metrics_calculator=QualityMetrics(device=device),
        use_amp=args.amp, clip_grad_norm=1., log_every=args.log_every,
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def physical_checks(baseline, final):
    bonds_available = baseline['peptide_mae'] is not None and final['peptide_mae'] is not None
    return {
        'peptide_bonds_available': bonds_available,
        'peptide_geometry_preserved': bool(bonds_available
            and final['peptide_mae'] <= baseline['peptide_mae'] + 0.01
            and final['peptide_valid_fraction'] >= baseline['peptide_valid_fraction'] - 1e-6
            and abs(final['peptide_mean'] - 1.329) <= 0.2),
        'clashes_not_increased': final['clash'] <= baseline['clash'] + 1e-6,
        'internal_geometry_preserved': final['internal_distance_error_max'] <= 2e-3,
        'rotations_valid': max(final['rotation_orthogonality_error'], final['rotation_determinant_error']) <= 2e-3,
    }


def run_overfit(pipeline, dataset, args, output):
    baseline, _ = evaluate(pipeline, dataset, baseline=True)
    initial, _ = evaluate(pipeline, dataset)
    if initial['identity_error'] > 1e-3:
        raise RuntimeError(f"Zero-initialized model changes coordinates by {initial['identity_error']:.6g} Å")
    LOGGER.info('Overfit baseline: FAPE %.4f, RMSD %.4f Å, lDDT %.4f, clash %.4f',
                baseline['fape'], baseline['rmsd'], baseline['lddt'], baseline['clash'])
    history, evaluations = [], [{'step': 0, **initial}]
    maximum_gradients = {}
    started = time.perf_counter()
    for step in range(1, args.overfit_steps + 1):
        batch = collate_batch([dataset[(step - 1) % len(dataset)]])
        pipeline.model.train()
        loss, components = pipeline.train_one_batch(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite overfit loss at step {step}')
        gradients = gradient_metrics(pipeline.model)
        for key, value in gradients.items():
            maximum_gradients[key] = max(maximum_gradients.get(key, 0.), value)
        history.append({'step': step, **{key: float(value) for key, value in components.items()}, **gradients})
        if step == 1 or step % args.log_every == 0 or step == args.overfit_steps:
            metrics, _ = evaluate(pipeline, dataset)
            evaluations.append({'step': step, **metrics})
            LOGGER.info('Overfit %d/%d: FAPE %.4f, RMSD %.4f Å, lDDT %.4f, clipped |grad| %.4f',
                        step, args.overfit_steps, metrics['fape'], metrics['rmsd'], metrics['lddt'],
                        gradients['clipped_grad_norm'])
            write_csv(output / 'overfit_steps.csv', history)
            write_csv(output / 'overfit_evaluation.csv', evaluations)
    final, structures = evaluate(pipeline, dataset, retain=2)
    fape_drop = (baseline['fape'] - final['fape']) / max(baseline['fape'], 1e-8)
    checks = {
        'identity_at_initialization': initial['identity_error'] <= 1e-3,
        'fape_drop_at_least_20_percent': fape_drop >= 0.2,
        'structure_metrics_improved': final['rmsd'] < baseline['rmsd'] or final['lddt'] > baseline['lddt'],
        'head_weights_changed': final['head_weight_norm'] > 0.,
        'all_backbone_blocks_receive_gradients': all(
            value > 0. for key, value in maximum_gradients.items() if key != 'clipped_grad_norm'),
        'translation_updates_bounded': final['translation_update_max'] <= 1.5,
        **physical_checks(baseline, final),
    }
    result = {'baseline': baseline, 'initial_model': initial, 'final': final, 'checks': checks,
              'fape_drop_fraction': fape_drop, 'maximum_gradients': maximum_gradients,
              'seconds': time.perf_counter() - started,
              'fape_decrease_fraction_at_logged_steps': float(np.mean([
                  b['fape'] <= a['fape'] for a, b in zip(evaluations, evaluations[1:])])),
              'passed': all(checks.values())}
    return result, history, evaluations, structures


def run_mini_training(pipeline, train, val, args, output):
    baseline, _ = evaluate(pipeline, val, baseline=True)
    initial, _ = evaluate(pipeline, val)
    if initial['identity_error'] > 1e-3:
        raise RuntimeError('Fresh phase-2 model does not preserve input coordinates')
    generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate_batch,
                        num_workers=0, pin_memory=pipeline.device.type == 'cuda', generator=generator)
    history = [{'epoch': 0, 'train_loss': None, 'train_fape': None, **initial}]
    best_rmsd, best_epoch = float('inf'), None
    checkpoint_dir = output / 'checkpoint'
    checkpoint = checkpoint_dir / 'best.pt'
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        # Use the existing single-batch pipeline while retaining diagnostics for
        # every step instead of implementing another backward/optimizer loop.
        totals = {}
        for batch in loader:
            pipeline.model.train()
            loss, components = pipeline.train_one_batch(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite mini-training loss at epoch {epoch}')
            gradient_metrics(pipeline.model)
            for key, value in components.items():
                totals[key] = totals.get(key, 0.) + float(value)
        metrics, _ = evaluate(pipeline, val)
        history.append({'epoch': epoch, 'train_loss': totals['total'] / len(loader),
                        'train_fape': totals['fape'] / len(loader), **metrics})
        if metrics['rmsd'] < best_rmsd:
            best_rmsd, best_epoch = metrics['rmsd'], epoch
            save_checkpoint(pipeline.model, pipeline.optimizer, epoch, best_rmsd,
                            str(checkpoint_dir), pipeline.state_dict())
            (checkpoint_dir / f'checkpoint_epoch_{epoch}.pt').replace(checkpoint)
        LOGGER.info('Epoch %d/%d: train FAPE %.4f, val RMSD %.4f Å, lDDT %.4f, clash %.4f',
                    epoch, args.epochs, history[-1]['train_fape'], metrics['rmsd'], metrics['lddt'], metrics['clash'])
        write_csv(output / 'mini_epochs.csv', history)
    load_checkpoint(pipeline.model, pipeline.optimizer, str(checkpoint), pipeline=pipeline)
    best, structures = evaluate(pipeline, val, retain=2)
    checkpoint_valid = all(math.isclose(best[key], history[best_epoch][key], rel_tol=1e-5, abs_tol=1e-5)
                           for key in ('fape', 'rmsd', 'lddt', 'clash'))
    checks = {'checkpoint_roundtrip_valid': checkpoint_valid,
              'validation_rmsd_improved': best['rmsd'] < baseline['rmsd'],
              'validation_lddt_improved': best['lddt'] > baseline['lddt'],
              **physical_checks(baseline, best)}
    return {'baseline': baseline, 'best': best, 'best_epoch': best_epoch,
            'last_epoch': history[-1], 'checks': checks, 'passed': all(checks.values()),
            'seconds': time.perf_counter() - started, 'checkpoint': str(checkpoint)}, history, structures


def export_structures(output, phase, structures):
    from Bio.PDB import MMCIFIO
    from Bio.PDB.StructureBuilder import StructureBuilder
    atom_names = {index: name for name, index in ATOM_NAME_MAP.items()}
    elements = {index: name for name, index in ATOM_TYPE_MAP.items()}
    residue_names = {index: name for name, index in RESIDUE_TYPE_MAP.items()}
    directory = output / 'structures'
    directory.mkdir(exist_ok=True)
    for index, (sample, predicted) in enumerate(structures):
        for label, coords in (('input', sample['coords_bad']), ('target', sample['coords_good']), ('predicted', predicted)):
            builder = StructureBuilder()
            builder.init_structure(f'{phase}_{index}_{label}')
            builder.init_model(0, serial_num=1)
            previous_chain, previous_residue = None, None
            for atom in range(int(sample['length'])):
                residue = int(sample['res_map'][atom])
                chain = int(sample['res_chain'][residue])
                if chain != previous_chain:
                    builder.init_chain(str(chain))
                    builder.init_seg(' ')
                    previous_chain, previous_residue = chain, None
                if residue != previous_residue:
                    builder.init_residue(residue_names[int(sample['residue_types'][atom])], ' ', residue + 1, ' ')
                    previous_residue = residue
                name = atom_names[int(sample['atom_names'][atom])]
                builder.init_atom(name, coords[atom].numpy().astype('f'), 0., 1., ' ',
                                  name, element=elements[int(sample['atom_types'][atom])])
            writer = MMCIFIO()
            writer.set_structure(builder.get_structure())
            writer.save(str(directory / f'{phase}_{index + 1}_{label}.cif'))


def write_report(output, summary):
    lines = ['# Quick Training Experiment', '', f"Data source: {summary['data_source']}", '',
             f"Ready for full training on real data: **{summary['ready_for_full_training']}**", '']
    rows = []
    for name, result in (('Overfit', summary['overfit']), ('Validation', summary.get('mini_training'))):
        if result is None:
            continue
        final = result.get('final', result.get('best'))
        lines.extend([f'## {name}', '', '| Metric | Input baseline | Model |', '|---|---:|---:|'])
        for key in ('fape', 'rmsd', 'lddt', 'clash', 'peptide_mean', 'peptide_valid_fraction', 'internal_distance_error_max'):
            before, after = result['baseline'][key], final[key]
            lines.append(f'| {key} | {before:.6f} | {after:.6f} |' if before is not None and after is not None
                         else f'| {key} | unavailable | unavailable |')
            rows.append({'phase': name, 'metric': key, 'baseline': before, 'model': after})
        lines.extend(['', 'Criteria:', ''])
        lines.extend(f"- {'PASS' if passed else 'FAIL'}: {check}" for check, passed in result['checks'].items())
        lines.append('')
    lines.extend(['Metrics use the existing project QualityMetrics definitions.',
                  'The clash score includes bonded contacts; clash_loss uses the project bonded-pair exclusions.',
                  'CIF exports contain matched atoms with residue and chain indices assigned by the dataset.',
                  'Synthetic runs check the pipeline and cannot establish generalization to real proteins.', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_csv(output / 'baseline_vs_model.csv', rows)


def plot_results(output, overfit_evaluations, mini_history):
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric in zip(axes, ('fape', 'rmsd', 'lddt')):
        axis.plot([r['step'] for r in overfit_evaluations], [r[metric] for r in overfit_evaluations])
        axis.set(xlabel='Optimizer step', ylabel=metric, title=f'Overfit {metric}')
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / 'overfit.png', dpi=160)
    plt.close(fig)
    if mini_history:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        trained = mini_history[1:]
        axes[0, 0].plot([r['epoch'] for r in trained], [r['train_loss'] for r in trained], label='Train loss')
        axes[0, 0].plot([r['epoch'] for r in trained], [r['train_fape'] for r in trained], label='Train FAPE')
        axes[0, 0].legend()
        axes[0, 0].set(xlabel='Epoch', title='Training')
        for axis, metric in zip((axes[0, 1], axes[1, 0], axes[1, 1]), ('rmsd', 'lddt', 'clash')):
            axis.plot([r['epoch'] for r in mini_history], [r[metric] for r in mini_history])
            axis.axhline(mini_history[0][metric], linestyle='--', color='gray', label='Input baseline')
            axis.set(xlabel='Epoch', ylabel=metric, title=f'Validation {metric}')
            axis.legend()
        fig.tight_layout()
        fig.savefig(output / 'mini_training.png', dpi=160)
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv-file', type=Path, default=Path(CSV_FILE))
    parser.add_argument('--data-folder', type=Path, default=Path(DATA_FOLDER))
    parser.add_argument('--synthetic', action='store_true', help='Explicit pipeline diagnostics without real PDB files')
    parser.add_argument('--synthetic-residues', type=int, default=12)
    parser.add_argument('--overfit-pairs', type=int, default=2)
    parser.add_argument('--overfit-steps', type=int, default=80)
    parser.add_argument('--train-pairs', type=int, default=16)
    parser.add_argument('--val-pairs', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-iterations', type=int, default=3)
    parser.add_argument('--max-atoms', type=int, default=1500)
    parser.add_argument('--device', default=str(DEVICE), choices=('cpu', 'cuda'))
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU threads')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, help='New or empty output directory')
    parser.add_argument('--skip-phase2', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--strict', action='store_true', help='Exit 2 when experiment criteria fail')
    args = parser.parse_args(argv)
    for key in ('overfit_pairs', 'overfit_steps', 'train_pairs', 'val_pairs', 'epochs', 'num_iterations',
                'max_atoms', 'threads', 'log_every'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if not args.skip_phase2 and args.overfit_pairs > args.train_pairs:
        parser.error('--overfit-pairs cannot exceed --train-pairs')
    if args.synthetic and (args.synthetic_residues < 2 or args.synthetic_residues * 4 > args.max_atoms):
        parser.error('--synthetic-residues must be at least 2 and fit within --max-atoms')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir or PROJECT_ROOT / 'artifacts' / 'quick_training' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'run.log', encoding='utf-8')])
    started = time.perf_counter()
    summary = {'status': 'running', 'arguments': {key: str(value) if isinstance(value, Path) else value
                                                for key, value in vars(args).items()},
               'data_source': 'synthetic' if args.synthetic else 'real', 'ready_for_full_training': False,
               'torch_version': torch.__version__, 'device': args.device, 'output_dir': str(output.resolve())}
    try:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu')
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if args.synthetic:
            LOGGER.info('Synthetic mode: results assess pipeline behavior, not biological generalization')
            dataset = synthetic_dataset(args.train_pairs + args.val_pairs, args.synthetic_residues, args.seed)
        else:
            if not args.csv_file.is_file() or not args.data_folder.is_dir():
                raise FileNotFoundError('Provide --csv-file and --data-folder for real pairs, or use --synthetic')
            dataset = ProteinUpscalingDataset(
                args.csv_file, args.data_folder, max_atoms=args.max_atoms, prefilter_cache=None,
                resolution_good=RESOLUTION_GOOD, resolution_bad=RESOLUTION_BAD,
                coverage_min=COVERAGE_MIN, rmsd_max=PAIR_RMSD_MAX, seq_identity_min=SEQ_IDENTITY_MIN,
            )
        train, val, manifest = select_pairs(dataset, args)
        write_json(output / 'selection.json', manifest)
        LOGGER.info('Selected %d train / %d val pairs; overfit uses %d training pairs', len(train), len(val), args.overfit_pairs)
        overfit = CachedPairs(train.samples[:args.overfit_pairs], train.uids[:args.overfit_pairs])
        pipeline = make_pipeline(args, device, args.seed)
        result, _, overfit_evaluations, structures = run_overfit(pipeline, overfit, args, output)
        summary['overfit'] = result
        export_structures(output, 'overfit', structures)
        write_json(output / 'summary.json', summary)
        del pipeline
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        mini_history = None
        if not args.skip_phase2:
            pipeline = make_pipeline(args, device, args.seed + 1)
            mini, mini_history, structures = run_mini_training(pipeline, train, val, args, output)
            summary['mini_training'] = mini
            export_structures(output, 'validation', structures)
            del pipeline
        evaluated_criteria_pass = result['passed'] and (args.skip_phase2 or summary['mini_training']['passed'])
        summary.update(status='completed', criteria_passed=evaluated_criteria_pass,
                       ready_for_full_training=not args.synthetic and not args.skip_phase2 and evaluated_criteria_pass,
                       seconds=time.perf_counter() - started,
                       peak_cuda_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None)
        write_report(output, summary)
        if not args.no_plots:
            plot_results(output, overfit_evaluations, mini_history)
        write_json(output / 'summary.json', summary)
        LOGGER.info('Experiment completed: criteria_passed=%s, ready_for_full_training=%s; outputs: %s',
                    evaluated_criteria_pass, summary['ready_for_full_training'], output.resolve())
        print((output / 'report.md').read_text(encoding='utf-8'))
        return 2 if args.strict and not evaluated_criteria_pass else 0
    except Exception as error:
        summary.update(status='failed', ready_for_full_training=False,
                       error=f'{type(error).__name__}: {error}', seconds=time.perf_counter() - started)
        write_json(output / 'summary.json', summary)
        LOGGER.exception('Experiment failed; diagnostics saved to %s', output.resolve())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
