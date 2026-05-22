"""Тесты под новый dict-API модели и frame-based архитектуру."""
import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import torch

from upscaler.model.encoders import (
    ProteinEncoder,
    SE3PositionalEncoder,
    AtomTypeEmbedder,
    ResidueTypeEmbedder,
    ATOM_TYPE_MAP,
    RESIDUE_TYPE_MAP,
)
from upscaler.model.layers import LocalRefinementBlock, FrameUpdateHead
from upscaler.model.geometric import (
    frames_from_bb_coords,
    invert_frames,
    apply_frames,
    rots_from_vec,
)
from upscaler.model.upscaler import ProteinUpscaler


def test_se3_positional_encoder():
    print("Testing SE3PositionalEncoder...")
    batch_size, num_atoms = 2, 12
    coords = torch.randn(batch_size, num_atoms, 3)
    encoder = SE3PositionalEncoder()
    feats = encoder(coords)
    assert feats.shape[0] == batch_size and feats.shape[1] == num_atoms
    assert feats.shape[-1] == encoder.irreps_out.dim
    print("SE3PositionalEncoder test passed.")


def test_atom_residue_embedders():
    print("Testing AtomTypeEmbedder / ResidueTypeEmbedder...")
    batch_size, num_atoms = 2, 10
    atom_types = torch.randint(0, len(ATOM_TYPE_MAP), (batch_size, num_atoms))
    res_types = torch.randint(0, len(RESIDUE_TYPE_MAP), (batch_size, num_atoms))
    atom_embed = AtomTypeEmbedder()
    res_embed = ResidueTypeEmbedder()
    a_feats = atom_embed(atom_types)
    r_feats = res_embed(res_types)
    assert a_feats.shape == (batch_size, num_atoms, atom_embed.irreps_out.dim)
    assert r_feats.shape == (batch_size, num_atoms, res_embed.irreps_out.dim)
    print("Embedders test passed.")


def test_protein_encoder():
    print("Testing ProteinEncoder...")
    batch_size, num_atoms = 2, 10
    coords = torch.randn(batch_size, num_atoms, 3)
    atom_types = torch.randint(0, len(ATOM_TYPE_MAP), (batch_size, num_atoms))
    res_types = torch.randint(0, len(RESIDUE_TYPE_MAP), (batch_size, num_atoms))
    enc = ProteinEncoder()
    feats = enc(coords, atom_types, res_types)
    assert feats.shape == (batch_size, num_atoms, enc.d_out)
    print("ProteinEncoder test passed.")


def test_local_refinement_block():
    print("Testing LocalRefinementBlock...")
    enc = ProteinEncoder()
    batch_size, n_res = 1, 6
    res_feats = torch.randn(batch_size, n_res, enc.d_out)
    trans = torch.randn(batch_size, n_res, 3)
    block = LocalRefinementBlock(irreps_model=enc.irreps_out, radius=5.0, k=4)
    out = block(res_feats, trans)
    assert out.shape == res_feats.shape
    print("LocalRefinementBlock test passed.")


def test_frame_update_head():
    print("Testing FrameUpdateHead...")
    batch_size, n_res, d = 2, 5, 256
    feats = torch.randn(batch_size, n_res, d)
    head = FrameUpdateHead(d_model=d)
    rot_vec, trans_vec = head(feats)
    assert rot_vec.shape == (batch_size, n_res, 3)
    assert trans_vec.shape == (batch_size, n_res, 3)
    print("FrameUpdateHead test passed.")


def test_geometric_helpers_roundtrip():
    print("Testing apply/invert frames round-trip...")
    batch_size, n_res = 2, 4
    n_atoms = n_res * 3
    coords = torch.randn(batch_size, n_atoms, 3)

    # Сгенерируем фреймы из backbone
    bb = coords.view(batch_size, n_res, 3, 3)
    rots, trans = frames_from_bb_coords(bb[:, :, 0], bb[:, :, 1], bb[:, :, 2])

    inv_rots, inv_trans = invert_frames(rots, trans)
    # Точка в локальной системе остатка 0: к world и обратно
    p_world = coords[:, 0:1, :]
    p_local = apply_frames(inv_rots[:, 0], inv_trans[:, 0], p_world.squeeze(1))
    p_back = apply_frames(rots[:, 0], trans[:, 0], p_local)
    assert torch.allclose(p_back, p_world.squeeze(1), atol=1e-4), "Round-trip failed"
    print("Geometric helpers round-trip test passed.")


def test_rots_from_vec_identity():
    print("Testing rots_from_vec with zero vector → identity...")
    vec = torch.zeros(2, 3)
    R = rots_from_vec(vec)
    I = torch.eye(3).expand(2, 3, 3)
    assert torch.allclose(R, I, atol=1e-5)
    print("rots_from_vec identity test passed.")


def test_protein_upscaler_forward():
    print("Testing ProteinUpscaler forward (dict-API)...")
    batch_size, n_res = 1, 5
    n_atoms = n_res * 3  # упрощённо: N, CA, C на остаток

    coords_bad = torch.randn(batch_size, n_atoms, 3)
    bb = coords_bad.view(batch_size, n_res, 3, 3)
    rots_bad, trans_bad = frames_from_bb_coords(bb[:, :, 0], bb[:, :, 1], bb[:, :, 2])

    atom_types = torch.randint(1, len(ATOM_TYPE_MAP), (batch_size, n_atoms))
    residue_types = torch.randint(1, len(RESIDUE_TYPE_MAP), (batch_size, n_atoms))
    res_map = torch.arange(n_res).repeat_interleave(3).unsqueeze(0).expand(batch_size, -1).clone()
    mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    res_mask = torch.ones(batch_size, n_res, dtype=torch.bool)

    batch = {
        'coords_bad': coords_bad,
        'atom_types': atom_types,
        'residue_types': residue_types,
        'rots_bad': rots_bad,
        'trans_bad': trans_bad,
        'res_map': res_map,
        'mask': mask,
        'res_mask': res_mask,
    }

    model = ProteinUpscaler(num_iterations=1)
    out = model(batch)

    assert isinstance(out, dict)
    assert {'rots', 'trans', 'coords', 'res_mask', 'mask'}.issubset(out.keys())
    assert out['coords'].shape == coords_bad.shape
    assert out['rots'].shape == rots_bad.shape
    assert out['trans'].shape == trans_bad.shape
    print("ProteinUpscaler forward test passed.")


if __name__ == "__main__":
    test_se3_positional_encoder()
    test_atom_residue_embedders()
    test_protein_encoder()
    test_local_refinement_block()
    test_frame_update_head()
    test_geometric_helpers_roundtrip()
    test_rots_from_vec_identity()
    test_protein_upscaler_forward()
    print("All model component tests passed!")
