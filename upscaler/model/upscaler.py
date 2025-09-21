import torch.nn as nn

from upscaler.model.encoders import ProteinEncoder
from upscaler.model.layers import LocalRefinementBlock, GlobalAttentionBlock, CoordinatePredictor


class MultiScaleProteinUpscaler(nn.Module):
    def __init__(self, num_iterations=3):
        super().__init__()
        self.encoder = ProteinEncoder()
        
        irreps_model = self.encoder.irreps_out
        d_model = self.encoder.d_out
        
        # Многомасштабные блоки
        self.local_blocks = nn.ModuleList([
            LocalRefinementBlock(irreps_model=irreps_model, radius=2.5, k=4),
            LocalRefinementBlock(irreps_model=irreps_model, radius=5.0, k=8),
            LocalRefinementBlock(irreps_model=irreps_model, radius=7.5, k=12),
            LocalRefinementBlock(irreps_model=irreps_model, radius=10.0, k=16),
            LocalRefinementBlock(irreps_model=irreps_model, radius=12.5, k=20),
        ])
        self.global_blocks = nn.ModuleList([
            GlobalAttentionBlock(d_model=d_model, n_heads=8),
            GlobalAttentionBlock(d_model=d_model, n_heads=8),
            GlobalAttentionBlock(d_model=d_model, n_heads=8),
            GlobalAttentionBlock(d_model=d_model, n_heads=8),
        ])
        self.coordinate_predictor = CoordinatePredictor(d_model=d_model)

        self.num_iterations = num_iterations

    def forward(self, coords, atom_types, residue_types, mask=None):
        refined_coords = coords
        
        for _ in range(self.num_iterations):
            node_feats = self.encoder(refined_coords, atom_types, residue_types)
            
            if mask is not None:
                maskf = mask.unsqueeze(-1).to(node_feats.dtype)
                node_feats = node_feats * maskf
            
            residual = node_feats
            for block in self.local_blocks:
                node_feats = block(node_feats, refined_coords) + residual
                if mask is not None:
                    node_feats = node_feats * maskf
                residual = node_feats
            
            residual = node_feats
            for block in self.global_blocks:
                try:
                    node_feats = block(node_feats, mask=mask) + residual
                except TypeError:
                    node_feats = block(node_feats) + residual
                residual = node_feats
            
            coord_updates = self.coordinate_predictor(node_feats)
            
            if mask is not None:
                mask3 = mask.unsqueeze(-1).to(coord_updates.dtype)
                coord_updates = coord_updates * mask3
            
            refined_coords = refined_coords + coord_updates
        
        return refined_coords


ProteinUpscaler = MultiScaleProteinUpscaler
