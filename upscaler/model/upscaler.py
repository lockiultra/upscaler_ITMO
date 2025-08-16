import torch.nn as nn

from upscaler.model.encoders import ProteinEncoder
from upscaler.model.layers import LocalRefinementBlock, GlobalAttentionBlock, CoordinatePredictor


class MultiScaleProteinUpscaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ProteinEncoder()
        d_model = self.encoder.d_out # 256
        
        # Многомасштабные блоки
        self.local_blocks = nn.ModuleList([
            LocalRefinementBlock(d_model=d_model, radius=5.0, k=8),
            LocalRefinementBlock(d_model=d_model, radius=10.0, k=16),
        ])
        self.global_blocks = nn.ModuleList([
            GlobalAttentionBlock(d_model=d_model),
            GlobalAttentionBlock(d_model=d_model),
        ])
        self.coordinate_predictor = CoordinatePredictor(d_model=d_model)

    def forward(self, coords, atom_types, residue_types):
        """
        Args:
            coords: Tensor of shape [B, N, 3]
            atom_types: LongTensor of shape [B, N]
            residue_types: LongTensor of shape [B, N]
        Returns:
            refined_coords: Tensor of shape [B, N, 3]
        """
        # Кодирование
        node_feats = self.encoder(coords, atom_types, residue_types) # [B, N, 256]
        
        # Локальная обработка
        for block in self.local_blocks:
            node_feats = block(node_feats, coords) # [B, N, 256]
            
        # Глобальная обработка
        for block in self.global_blocks:
            node_feats = block(node_feats) # [B, N, 256]
            
        # Предсказание координат
        coord_updates = self.coordinate_predictor(node_feats) # [B, N, 3]
        refined_coords = coords + coord_updates
        return refined_coords

ProteinUpscaler = MultiScaleProteinUpscaler
