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

    def forward(self, coords, atom_types, residue_types, mask=None):
        node_feats = self.encoder(coords, atom_types, residue_types) # [B, N, d_model]

        # применяем маску к признакам, чтобы скрыть паддинг
        if mask is not None:
            maskf = mask.unsqueeze(-1).to(node_feats.dtype)  # [B, N, 1]
            node_feats = node_feats * maskf

        # Локальная обработка
        for block in self.local_blocks:
            node_feats = block(node_feats, coords) # [B, N, d_model]
            if mask is not None:
                node_feats = node_feats * maskf

        # Глобальная обработка
        for block in self.global_blocks:
            try:
                node_feats = block(node_feats, mask=mask)
            except TypeError:
                node_feats = block(node_feats)

        # Предсказание координат
        coord_updates = self.coordinate_predictor(node_feats) # [B, N, 3]

        # Наносим обновления только на валидные позиции
        if mask is not None:
            mask3 = mask.unsqueeze(-1).to(coord_updates.dtype)
            coord_updates = coord_updates * mask3

        refined_coords = coords + coord_updates
        return refined_coords


ProteinUpscaler = MultiScaleProteinUpscaler
