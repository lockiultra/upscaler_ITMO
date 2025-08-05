# upscaler/model/tests/test_model_components.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from upscaler.model.encoders import ProteinEncoder, SE3PositionalEncoder, ATOM_TYPE_MAP, RESIDUE_TYPE_MAP, AtomTypeEmbedder, ResidueTypeEmbedder
from upscaler.model.layers import SE3EquivariantLayer, LocalRefinementBlock, GlobalAttentionBlock, CoordinatePredictor
from upscaler.model.upscaler import ProteinUpscaler


def test_se3_positional_encoder():
    print("Testing SE3PositionalEncoder...")
    batch_size, num_atoms = 2, 10
    coords = torch.randn(batch_size, num_atoms, 3)
    
    encoder = SE3PositionalEncoder(irreps_out="128x0e")
    features = encoder(coords)
    
    assert features.shape == (batch_size, num_atoms, 128), f"Expected {(batch_size, num_atoms, 128)}, got {features.shape}"
    print("SE3PositionalEncoder test passed.")

def test_atom_residue_embedders():
    print("Testing AtomTypeEmbedder and ResidueTypeEmbedder...")
    batch_size, num_atoms = 2, 10
    atom_types = torch.randint(0, len(ATOM_TYPE_MAP), (batch_size, num_atoms))
    residue_types = torch.randint(0, len(RESIDUE_TYPE_MAP), (batch_size, num_atoms))
    
    atom_embedder = AtomTypeEmbedder()
    res_embedder = ResidueTypeEmbedder()
    
    atom_feats = atom_embedder(atom_types)
    res_feats = res_embedder(residue_types)
    
    assert atom_feats.shape == (batch_size, num_atoms, 64), f"Expected {(batch_size, num_atoms, 64)}, got {atom_feats.shape}"
    assert res_feats.shape == (batch_size, num_atoms, 64), f"Expected {(batch_size, num_atoms, 64)}, got {res_feats.shape}"
    print("AtomTypeEmbedder and ResidueTypeEmbedder tests passed.")

def test_protein_encoder():
    print("Testing ProteinEncoder...")
    batch_size, num_atoms = 2, 10
    coords = torch.randn(batch_size, num_atoms, 3)
    atom_types = torch.randint(0, len(ATOM_TYPE_MAP), (batch_size, num_atoms))
    residue_types = torch.randint(0, len(RESIDUE_TYPE_MAP), (batch_size, num_atoms))
    
    encoder = ProteinEncoder()
    features = encoder(coords, atom_types, residue_types)
    
    expected_d_out = 128 + 64 + 64 # d_pos + d_atom + d_res
    assert features.shape == (batch_size, num_atoms, expected_d_out), f"Expected {(batch_size, num_atoms, expected_d_out)}, got {features.shape}"
    print("ProteinEncoder test passed.")

def test_se3_equivariant_layer():
    print("Testing SE3EquivariantLayer...")
    from e3nn import o3
    num_nodes, num_edges = 10, 40
    irreps_input = o3.Irreps("256x0e")
    irreps_output = o3.Irreps("256x0e")
    irreps_edge_attr = o3.Irreps("1x1o")
    
    node_feats = torch.randn(num_nodes, irreps_input.dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, irreps_edge_attr.dim)
    
    layer = SE3EquivariantLayer(irreps_input, irreps_output, irreps_edge_attr)
    out = layer(node_feats, edge_index, edge_attr)
    
    assert out.shape == (num_nodes, irreps_output.dim), f"Expected {(num_nodes, irreps_output.dim)}, got {out.shape}"
    print("SE3EquivariantLayer test passed.")

def test_local_refinement_block():
    print("Testing LocalRefinementBlock...")
    batch_size, num_atoms, d_model = 2, 10, 256
    node_feats = torch.randn(batch_size, num_atoms, d_model)
    coords = torch.randn(batch_size, num_atoms, 3)
    
    block = LocalRefinementBlock(d_model=d_model)
    refined_feats = block(node_feats, coords)
    
    assert refined_feats.shape == (batch_size, num_atoms, d_model), f"Expected {(batch_size, num_atoms, d_model)}, got {refined_feats.shape}"
    print("LocalRefinementBlock test passed.")

def test_global_attention_block():
    print("Testing GlobalAttentionBlock...")
    batch_size, num_atoms, d_model = 2, 10, 256
    node_feats = torch.randn(batch_size, num_atoms, d_model)
    
    block = GlobalAttentionBlock(d_model=d_model)
    attended_feats = block(node_feats)
    
    assert attended_feats.shape == (batch_size, num_atoms, d_model), f"Expected {(batch_size, num_atoms, d_model)}, got {attended_feats.shape}"
    print("GlobalAttentionBlock test passed.")

def test_coordinate_predictor():
    print("Testing CoordinatePredictor...")
    batch_size, num_atoms, d_model = 2, 10, 256
    node_feats = torch.randn(batch_size, num_atoms, d_model)
    
    predictor = CoordinatePredictor(d_model=d_model)
    updates = predictor(node_feats)
    
    assert updates.shape == (batch_size, num_atoms, 3), f"Expected {(batch_size, num_atoms, 3)}, got {updates.shape}"
    print("CoordinatePredictor test passed.")

def test_protein_upscaler():
    print("Testing ProteinUpscaler...")
    batch_size, num_atoms = 2, 10
    coords = torch.randn(batch_size, num_atoms, 3)
    atom_types = torch.randint(0, len(ATOM_TYPE_MAP), (batch_size, num_atoms))
    residue_types = torch.randint(0, len(RESIDUE_TYPE_MAP), (batch_size, num_atoms))
    
    model = ProteinUpscaler()
    refined_coords = model(coords, atom_types, residue_types)
    
    assert refined_coords.shape == (batch_size, num_atoms, 3), f"Expected {(batch_size, num_atoms, 3)}, got {refined_coords.shape}"
    print("ProteinUpscaler test passed.")

if __name__ == "__main__":
    test_se3_positional_encoder()
    test_atom_residue_embedders()
    test_protein_encoder()
    test_se3_equivariant_layer()
    test_local_refinement_block()
    test_global_attention_block()
    test_coordinate_predictor()
    test_protein_upscaler()
    print("All model component tests passed!")
