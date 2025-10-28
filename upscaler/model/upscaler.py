import torch
import torch.nn as nn
from invariant_point_attention import InvariantPointAttention

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

        self.pair_proj = nn.Linear(4, d_model)

        self.ipa_blocks = nn.ModuleList([
            InvariantPointAttention(
                dim = d_model,
                heads = 8,
                scalar_key_dim = max(8, d_model // 8),
                scalar_value_dim = max(8, d_model // 8),
                point_key_dim = 4,
                point_value_dim = 4 
            )
            for _ in range(4)
        ])
        # self.global_blocks = nn.ModuleList([
        #     GlobalAttentionBlock(d_model=d_model, n_heads=8),
        #     GlobalAttentionBlock(d_model=d_model, n_heads=8),
        #     GlobalAttentionBlock(d_model=d_model, n_heads=8),
        #     GlobalAttentionBlock(d_model=d_model, n_heads=8),
        # ])

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
            res_feats = self._pool_atom_feats_to_res(
                self.encoder(coords_bad, atom_types, res_types), 
                res_map, 
                res_mask
            )

            # 2. Локальные блоки
            residual = res_feats
            for block in self.local_blocks:
                res_feats = block(res_feats, current_trans) + residual
                residual = res_feats

            B, N_res, _ = current_trans.shape
            t_i = current_trans.unsqueeze(2)  # (B, N, 1, 3)
            t_j = current_trans.unsqueeze(1)  # (B, 1, N, 3)
            rel = t_i - t_j                    # (B, N, N, 3)
            dist = torch.norm(rel, dim=-1, keepdim=True)  # (B, N, N, 1)
            pair_feats = torch.cat([rel, dist], dim=-1)   # (B, N, N, 4)

            # Проецируем в d_model
            pairwise_repr = self.pair_proj(pair_feats)    # (B, N, N, d_model)

            # Маска: приводим к булевскому виду (True для валидных остатков)
            mask_bool = res_mask.bool()

            # 3. IPA-блоки (замена global attention)
            for ipa in self.ipa_blocks:
                res_out = ipa(
                    res_feats,            # single_repr (B, N, d_model)
                    pairwise_repr,        # pairwise_repr (B, N, N, d_model)
                    rotations = current_rots,      # (B, N, 3, 3)
                    translations = current_trans,  # (B, N, 3)
                    mask = mask_bool
                )
                if isinstance(res_out, tuple) or isinstance(res_out, list):
                    res_feats = res_out[0]
                else:
                    res_feats = res_out

                # residual connection
                res_feats = res_feats + residual
                residual = res_feats

            # 4. Предсказываем обновления фреймов
            rot_update_vec, trans_update_vec = self.update_head(res_feats)
            
            # 5. Применяем обновления фреймов
            rot_update_mat = rots_from_vec(rot_update_vec)            
            current_rots = torch.einsum('...ij,...jk->...ik', rot_update_mat, current_rots)
            current_trans = current_trans + torch.einsum('...ij,...j->...i', current_rots, trans_update_vec)

            # Маскируем
            current_trans = current_trans * res_mask.unsqueeze(-1)
        
        # Финальная генерация координат атомов
        inv_rots_bad, inv_trans_bad = invert_frames(batch['rots_bad'], batch['trans_bad'])
        
        atom_inv_rots = inv_rots_bad.gather(1, res_map.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        atom_inv_trans = inv_trans_bad.gather(1, res_map.unsqueeze(-1).expand(-1, -1, 3))
        
        local_coords = apply_frames(atom_inv_rots, atom_inv_trans, coords_bad)
        
        atom_rots_final = current_rots.gather(1, res_map.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        atom_trans_final = current_trans.gather(1, res_map.unsqueeze(-1).expand(-1, -1, 3))

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
