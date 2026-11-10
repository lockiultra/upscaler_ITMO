"""Regression checks for the mandatory fixes in fixes.pdf.

Run with: venv/bin/python -m unittest upscaler.tests.test_prelaunch_fixes -v
"""
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from upscaler.data.dataset import collate_batch
from upscaler.loss.loss import ATOM_TYPE_MAP, ClashLoss, FAPELoss
from upscaler.model.encoders import SE3PositionalEncoder
from upscaler.model.geometric import apply_frames, frames_from_bb_coords, rots_from_vec
from upscaler.model.layers import LocalRefinementBlock
from upscaler.model.upscaler import ProteinUpscaler
from upscaler.training.pipeline import TrainingPipeline
from upscaler.training.train import load_checkpoint, save_checkpoint
from upscaler.utils.metrics import QualityMetrics


def make_batch():
    generator = torch.Generator().manual_seed(42)
    samples = []
    for n_res in (4, 6):
        local = torch.tensor([[-1.4, 0.5, 0.0], [0.0, 0.0, 0.0],
                              [1.5, 0.0, 0.0], [2.0, 1.0, 0.0]])
        rotations = rots_from_vec(torch.randn(n_res, 3, generator=generator))
        centers = torch.randn(n_res, 3, generator=generator) * 3.0
        bad = apply_frames(rotations[:, None], centers[:, None], local[None])
        target_rotations = rots_from_vec(
            0.2 * torch.randn(n_res, 3, generator=generator)
        ) @ rotations
        target_centers = centers + 0.3 * torch.randn(n_res, 3, generator=generator)
        good = apply_frames(target_rotations[:, None], target_centers[:, None], local[None])
        rb, tb = frames_from_bb_coords(bad[:, 0], bad[:, 1], bad[:, 2])
        rg, tg = frames_from_bb_coords(good[:, 0], good[:, 1], good[:, 2])
        samples.append({
            'coords_bad': bad.reshape(-1, 3), 'coords_good': good.reshape(-1, 3),
            'rots_bad': rb, 'trans_bad': tb, 'rots_good': rg, 'trans_good': tg,
            'res_map': torch.arange(n_res).repeat_interleave(4),
            'res_chain': torch.zeros(n_res, dtype=torch.long),
            'atom_names': torch.tensor([1, 2, 3, 4]).repeat(n_res),
            'atom_types': torch.tensor([2, 1, 1, 3]).repeat(n_res),
            'residue_types': torch.ones(n_res * 4, dtype=torch.long),
            'length': n_res * 4, 'num_residues': n_res,
        })
    return collate_batch(samples)


class PrelaunchFixesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)

    def test_positional_encoder_has_trainable_paths(self):
        encoder = SE3PositionalEncoder(irreps_out='4x0e + 2x1e + 1x2e', k=3)
        coords = torch.randn(1, 5, 3)
        features = encoder(coords)
        self.assertGreater(encoder.tp.weight_numel, 0)
        for start, stop in ((0, 4), (4, 10), (10, 15)):
            self.assertGreater(features[..., start:stop].abs().sum().item(), 0)
        features.square().sum().backward()
        self.assertGreater(encoder.tp.weight.grad.abs().sum().item(), 0)

    def test_local_refinement_has_trainable_paths(self):
        block = LocalRefinementBlock('4x0e + 2x1e + 1x2e', k=3)
        features = torch.randn(1, 5, 15, requires_grad=True)
        out = block(features, torch.randn(1, 5, 3))
        self.assertGreater(block.equivariant_layer.tp.weight_numel, 0)
        self.assertGreater(out.abs().sum().item(), 0)
        out.square().sum().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertGreater(block.equivariant_layer.tp.weight.grad.abs().sum().item(), 0)

    def test_padding_does_not_change_geometric_features(self):
        coords = torch.randn(1, 3, 3)
        padded_coords = torch.cat([coords, torch.zeros(1, 4, 3)], dim=1)
        mask = torch.tensor([[True, True, True, False, False, False, False]])
        encoder = SE3PositionalEncoder(irreps_out='4x0e + 2x1e + 1x2e', k=8)
        torch.testing.assert_close(encoder(coords), encoder(padded_coords, mask=mask)[:, :3])
        block = LocalRefinementBlock('4x0e + 2x1e + 1x2e', k=8)
        features = torch.randn(1, 3, 15)
        padded_features = torch.cat([features, torch.randn(1, 4, 15)], dim=1)
        torch.testing.assert_close(
            block(features, coords),
            block(padded_features, padded_coords, mask=mask)[:, :3],
        )
        self.assertEqual(encoder(padded_coords, mask=mask)[:, 3:].abs().sum().item(), 0)

    def test_fape_normalization_for_batch_and_padding(self):
        loss_fn = FAPELoss()
        rp = rots_from_vec(torch.randn(1, 3, 3))
        rt = rots_from_vec(torch.randn(1, 3, 3))
        tp, tt = torch.randn(1, 3, 3), torch.randn(1, 3, 3)
        single = loss_fn.compute_fape((rp, tp), (rt, tt))
        duplicated = loss_fn.compute_fape(
            (rp.repeat(2, 1, 1, 1), tp.repeat(2, 1, 1)),
            (rt.repeat(2, 1, 1, 1), tt.repeat(2, 1, 1)),
        )
        torch.testing.assert_close(single, duplicated)
        padded = loss_fn.compute_fape(
            (torch.cat([rp, torch.zeros(1, 2, 3, 3)], 1),
             torch.cat([tp, torch.zeros(1, 2, 3)], 1)),
            (torch.cat([rt, torch.zeros(1, 2, 3, 3)], 1),
             torch.cat([tt, torch.zeros(1, 2, 3)], 1)),
            res_mask=torch.tensor([[True, True, True, False, False]]),
        )
        torch.testing.assert_close(single, padded)

    def test_clash_loss_gradient_and_chunking(self):
        coords = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [1.5, 0., 0.],
                                [0.2, 0., 0.]]], requires_grad=True)
        atom_types = torch.full((1, 4), ATOM_TYPE_MAP['C'], dtype=torch.long)
        kwargs = {'res_map': torch.tensor([[0, 0, 3, 5]]),
                  'mask': torch.tensor([[True, True, True, False]])}
        loss_fn = ClashLoss()
        loss = loss_fn(coords, atom_types, chunk_size=2, **kwargs)
        torch.testing.assert_close(loss, loss_fn(coords, atom_types, chunk_size=10, **kwargs))
        # Only (0, 2) and (1, 2) contribute; the first two atoms are bonded.
        self.assertAlmostEqual(loss.item(), ((3.4 - 1.5) + (3.4 - 0.5)) / 3, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(coords.grad).all())
        self.assertGreater(coords.grad[:, :3].abs().sum().item(), 0)
        self.assertEqual(coords.grad[:, 3:].abs().sum().item(), 0)

    def test_physics_is_disabled_and_not_evaluated_by_default(self):
        batch = make_batch()
        loss_fn = FAPELoss()
        pred = {'rots': batch['rots_bad'], 'trans': batch['trans_bad'],
                'coords': batch['coords_bad'], 'res_mask': batch['res_mask'],
                'mask': batch['mask']}
        true = {'rots': batch['rots_good'], 'trans': batch['trans_good'],
                'coords': batch['coords_good'], 'atom_types': batch['atom_types'],
                'atom_names': batch['atom_names'], 'res_map': batch['res_map'],
                'res_chain': batch['res_chain']}
        with patch.object(loss_fn.physics_loss_fn, 'forward', side_effect=AssertionError):
            loss, metrics = loss_fn(pred, true)
        self.assertEqual(loss_fn.physics_weight, 0)
        self.assertEqual(metrics['physics'].item(), 0)
        torch.testing.assert_close(loss, metrics['fape'] + 0.05 * metrics['clash'])

    def test_rotation_derivative_at_zero_and_orthogonality(self):
        vec = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        rots_from_vec(vec)[0, 1].backward()
        torch.testing.assert_close(vec.grad, torch.tensor([0., 0., -1.], dtype=torch.float64))
        self.assertTrue(torch.autograd.gradcheck(rots_from_vec, (vec,)))
        vectors = torch.randn(10, 3, dtype=torch.float64)
        vectors[0] = 0
        vectors[1] *= 1e-9
        rotations = rots_from_vec(vectors)
        torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations,
                                   torch.eye(3, dtype=torch.float64).expand(10, 3, 3))
        torch.testing.assert_close(torch.det(rotations), torch.ones(10, dtype=torch.float64))

    def test_model_training_metrics_and_checkpoint(self):
        batch = make_batch()
        model = ProteinUpscaler(num_iterations=1)
        model.eval()
        with torch.no_grad():
            initial = model(batch)
        torch.testing.assert_close(initial['coords'], batch['coords_bad'], atol=1e-5, rtol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        pipeline = TrainingPipeline(model, FAPELoss(), optimizer, torch.device('cpu'),
                                    metrics_calculator=QualityMetrics(), use_amp=False)
        # The zero-initialized head blocks backbone gradients on the first step.
        # After one update, both the backbone and rotation head must receive them.
        for _ in range(2):
            loss, metrics = pipeline.train_one_batch(batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertIn('fape', metrics)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for index, block in enumerate(model.local_blocks):
            self.assertGreater(block.equivariant_layer.tp.weight.grad.abs().sum().item(), 0,
                               f'local block {index}')
        for index, block in enumerate(model.ipa_blocks):
            self.assertGreater(block.to_out.weight.grad.abs().sum().item(), 0, f'IPA block {index}')
        self.assertGreater(model.update_head.predictor.weight[:3].abs().sum().item(), 0)
        epoch_metrics = pipeline.train_epoch([batch])
        self.assertGreater(epoch_metrics['fape'], 0)
        for value in pipeline.validate([batch]).values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(model, optimizer, 0, loss.item(), directory, pipeline.state_dict())
            saved_weights = {key: value.clone() for key, value in model.state_dict().items()}
            with torch.no_grad():
                model.update_head.predictor.weight.add_(1)
            epoch, metric = load_checkpoint(
                model, optimizer, str(Path(directory) / 'checkpoint_epoch_0.pt'), pipeline
            )
            self.assertEqual(epoch, 0)
            self.assertAlmostEqual(metric, loss.item())
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, saved_weights[name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
