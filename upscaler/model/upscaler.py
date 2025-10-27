import torch
import torch.nn as nn

from upscaler.model.encoders import ProteinEncoder
from upscaler.model.layers import LocalRefinementBlock, GlobalAttentionBlock, FrameUpdateHead
from upscaler.model.geometric import rots_from_vec, apply_frames, invert_frames

class MultiScaleProteinUpscaler(nn.Module):
    def __init__(self, num_iterations=3):
        super().__init__()
        self.encoder = ProteinEncoder()
        
        irreps_model = self.encoder.irreps_out
        d_model = self.encoder.d_out
        
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
        self.update_head = FrameUpdateHead(d_model=d_model)

        self.num_iterations = num_iterations

    def forward(self, batch):
        coords_bad = batch['coords_bad']
        atom_types = batch['atom_types']
        res_types = batch['residue_types']
        mask = batch['mask']
        res_mask = batch['res_mask']
        res_map = batch['res_map']
        
        # Начальные фреймы из low-res структуры
        current_rots, current_trans = batch['rots_bad'], batch['trans_bad']

        for _ in range(self.num_iterations):
            # 1. Собираем признаки остатков из признаков атомов (усредняем)
            # Это упрощение, в реальных моделях здесь стоит GNN
            res_feats = self._pool_atom_feats_to_res(
                self.encoder(coords_bad, atom_types, res_types), 
                res_map, 
                res_mask
            )

            # 2. Обновляем признаки остатков
            residual = res_feats
            for block in self.local_blocks: # Local/Global блоки теперь работают на уровне остатков
                res_feats = block(res_feats, current_trans) + residual
                residual = res_feats
            
            for block in self.global_blocks:
                res_feats = block(res_feats, mask=res_mask) + residual
                residual = res_feats

            # 3. Предсказываем обновления фреймов
            rot_update_vec, trans_update_vec = self.update_head(res_feats)
            
            # 4. Применяем обновления
            rot_update_mat = rots_from_vec(rot_update_vec)
            
            # Композиция фреймов
            current_rots = torch.einsum('...ij,...jk->...ik', rot_update_mat, current_rots)
            current_trans = current_trans + torch.einsum('...ij,...j->...i', current_rots, trans_update_vec)

            # Маскируем, чтобы паддинг не "улетал"
            current_trans = current_trans * res_mask.unsqueeze(-1)
        
        # 5. Генерируем финальные координаты атомов
        inv_rots_bad, inv_trans_bad = invert_frames(batch['rots_bad'], batch['trans_bad'])
        
        # Собираем инвертированные фреймы для каждого атома
        atom_inv_rots = inv_rots_bad.gather(1, res_map.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        atom_inv_trans = inv_trans_bad.gather(1, res_map.unsqueeze(-1).expand(-1, -1, 3))
        
        # Локальные координаты
        local_coords = apply_frames(atom_inv_rots, atom_inv_trans, coords_bad)
        
        # Собираем финальные фреймы для каждого атома
        atom_rots_final = current_rots.gather(1, res_map.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        atom_trans_final = current_trans.gather(1, res_map.unsqueeze(-1).expand(-1, -1, 3))

        # Применяем финальные фреймы
        final_coords = apply_frames(atom_rots_final.transpose(-1,-2), atom_trans_final, local_coords)
        
        return {
            'rots': current_rots,
            'trans': current_trans,
            'coords': final_coords,
            'res_mask': res_mask,
            'mask': mask
        }

    def _pool_atom_feats_to_res(self, atom_feats, res_map, res_mask):
        B, N_res = res_mask.shape
        D = atom_feats.shape[-1]
        
        res_feats = torch.zeros(B, N_res, D, device=atom_feats.device)
        res_map_exp = res_map.unsqueeze(-1).expand(-1, -1, D)
        
        res_feats.scatter_add_(1, res_map_exp, atom_feats)
        
        # Нормализуем
        counts = torch.zeros(B, N_res, 1, device=atom_feats.device)
        counts.scatter_add_(1, res_map.unsqueeze(-1), torch.ones_like(atom_feats[..., :1]))
        
        return res_feats / (counts + 1e-8)



ProteinUpscaler = MultiScaleProteinUpscaler
