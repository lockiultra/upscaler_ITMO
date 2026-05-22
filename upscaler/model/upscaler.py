import torch
import torch.nn as nn
from invariant_point_attention import InvariantPointAttention

from upscaler.model.encoders import ProteinEncoder
from upscaler.model.layers import LocalRefinementBlock, FrameUpdateHead
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

        self.update_head = FrameUpdateHead(d_model=d_model)
        self.num_iterations = num_iterations

    def forward(self, batch):
        device = next(self.parameters()).device
        coords_bad = batch['coords_bad'].to(device)
        atom_types = batch['atom_types'].to(device)
        res_types = batch['residue_types'].to(device)
        if 'mask' in batch:
            mask = batch['mask'].to(device)
        else:
            mask = None
        if 'res_mask' in batch:
            res_mask = batch['res_mask'].to(device)
        else:
            res_mask = None
        if 'res_map' in batch:
            res_map = batch['res_map'].to(device)
        else:
            res_map = None
        
        B, N_atoms = coords_bad.shape[:2]
        N_res = batch['rots_bad'].shape[1]

        # Защита от невалидных индексов остатков (только если есть res_map и mask)
        if res_map is not None:
            invalid_mask = (res_map < 0) | (res_map >= N_res)
            if invalid_mask.any():
                res_map = res_map.clone()
                res_map[invalid_mask] = 0
                if mask is not None:
                    mask = mask & (~invalid_mask)

        initial_rots = batch['rots_bad'].to(device)
        initial_trans = batch['trans_bad'].to(device)

        # Маска валидных residue-фреймов: для паддинговых остатков
        # принудительно ставим identity-rotation и нулевой translation, чтобы
        # rots_from_vec×current_rots не разъезжалось в NaN.
        if res_mask is not None:
            rm = res_mask.to(dtype=initial_rots.dtype).unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(3, device=device, dtype=initial_rots.dtype).expand_as(initial_rots)
            initial_rots = initial_rots * rm + eye * (1.0 - rm)
            initial_trans = initial_trans * res_mask.unsqueeze(-1).to(initial_trans.dtype)

        current_rots = initial_rots.clone()
        current_trans = initial_trans.clone()

        for iteration in range(self.num_iterations):
            # 1. Собираем признаки остатков из признаков атомов (усредняем)
            res_feats = self._pool_atom_feats_to_res(
                self.encoder(coords_bad, atom_types, res_types, mask=mask),
                res_map,
                res_mask,
                mask
            )

            # 2. Локальные блоки: классический residual — складываем с входом блока,
            # а не с накапливающейся суммой (иначе активации растут как 2^k).
            local_in = res_feats
            for block in self.local_blocks:
                res_feats = block(res_feats, current_trans, mask=res_mask) + local_in

            B, N_res, _ = current_trans.shape
            t_i = current_trans.unsqueeze(2)
            t_j = current_trans.unsqueeze(1)
            rel = t_i - t_j
            dist = torch.norm(rel, dim=-1, keepdim=True)
            pair_feats = torch.cat([rel, dist], dim=-1)
            pairwise_repr = self.pair_proj(pair_feats)
            mask_bool = res_mask.bool() if res_mask is not None else None

            # 3. IPA-блоки: тот же фикс residual.
            ipa_in = res_feats
            for ipa in self.ipa_blocks:
                res_out = ipa(
                    res_feats,
                    pairwise_repr,
                    rotations=current_rots,
                    translations=current_trans,
                    mask=mask_bool,
                )
                if isinstance(res_out, (tuple, list)):
                    res_feats = res_out[0]
                else:
                    res_feats = res_out
                res_feats = res_feats + ipa_in

            # 4. Предсказываем обновления фреймов
            rot_update_vec, trans_update_vec = self.update_head(res_feats)

            # 5. Применяем обновления фреймов
            rot_update_mat = rots_from_vec(rot_update_vec)
            current_rots = torch.einsum('...ij,...jk->...ik', rot_update_mat, current_rots)
            current_trans = current_trans + torch.einsum(
                '...ij,...j->...i', current_rots, trans_update_vec
            )

            # Маскируем паддинг и в rotations, и в translations
            if res_mask is not None:
                rm = res_mask.to(dtype=current_rots.dtype).unsqueeze(-1).unsqueeze(-1)
                eye = torch.eye(3, device=device, dtype=current_rots.dtype).expand_as(current_rots)
                current_rots = current_rots * rm + eye * (1.0 - rm)
                current_trans = current_trans * res_mask.unsqueeze(-1).to(current_trans.dtype)
        
        # Финальная генерация координат атомов
        final_coords = self._compute_final_coords(
            coords_bad, 
            res_map, 
            current_rots, 
            current_trans,
            initial_rots,
            initial_trans,
            mask
        )
        
        return {
            'rots': current_rots,
            'trans': current_trans,
            'coords': final_coords,
            'res_mask': res_mask,
            'mask': mask
        }

    def _pool_atom_feats_to_res(self, atom_feats, res_map, res_mask, atom_mask=None):
        B, N_res = res_mask.shape
        D = atom_feats.shape[-1]

        # Все буферы создаём с dtype/device от atom_feats — иначе под AMP
        # (float16 / bfloat16) scatter_add_ выдаст type-mismatch.
        res_feats = atom_feats.new_zeros(B, N_res, D)
        counts = atom_feats.new_zeros(B, N_res, 1)

        res_map_clamped = torch.clamp(res_map, 0, N_res - 1)
        res_map_exp = res_map_clamped.unsqueeze(-1).expand(-1, -1, D)

        # Маска атомов: float того же dtype, что и features.
        if atom_mask is not None:
            atom_mask_f = atom_mask.to(dtype=atom_feats.dtype).unsqueeze(-1)
            masked_feats = atom_feats * atom_mask_f
        else:
            masked_feats = atom_feats
            atom_mask_f = atom_feats.new_ones(B, atom_feats.shape[1], 1)

        res_feats.scatter_add_(1, res_map_exp, masked_feats)
        counts.scatter_add_(1, res_map_clamped.unsqueeze(-1), atom_mask_f)

        counts = torch.clamp(counts, min=1.0)
        return res_feats / counts

    def _compute_final_coords(
        self, 
        coords_bad, 
        res_map, 
        final_rots, 
        final_trans,
        initial_rots,
        initial_trans,
        atom_mask
    ):
        B, N_atoms, _ = coords_bad.shape
        N_res = final_rots.shape[1]
        
        # Клампим индексы
        res_map_clamped = torch.clamp(res_map, 0, N_res - 1)
        
        # Инвертируем начальные фреймы
        inv_rots_bad, inv_trans_bad = invert_frames(initial_rots, initial_trans)
        
        atom_inv_rots = torch.gather(
            inv_rots_bad.unsqueeze(1).expand(-1, N_atoms, -1, -1, -1),
            2,
            res_map_clamped.view(B, N_atoms, 1, 1, 1).expand(-1, -1, 1, 3, 3)
        ).squeeze(2)
        
        atom_inv_trans = torch.gather(
            inv_trans_bad.unsqueeze(1).expand(-1, N_atoms, -1, -1),
            2,
            res_map_clamped.view(B, N_atoms, 1, 1).expand(-1, -1, 1, 3)
        ).squeeze(2)
        
        # Переводим в локальные координаты остатков
        local_coords = apply_frames(atom_inv_rots, atom_inv_trans, coords_bad)
        
        # Получаем финальные фреймы для каждого атома
        atom_rots_final = torch.gather(
            final_rots.unsqueeze(1).expand(-1, N_atoms, -1, -1, -1),
            2,
            res_map_clamped.view(B, N_atoms, 1, 1, 1).expand(-1, -1, 1, 3, 3)
        ).squeeze(2)
        
        atom_trans_final = torch.gather(
            final_trans.unsqueeze(1).expand(-1, N_atoms, -1, -1),
            2,
            res_map_clamped.view(B, N_atoms, 1, 1).expand(-1, -1, 1, 3)
        ).squeeze(2)

        # Применяем финальные фреймы (local -> world): world = R @ local + t
        final_coords = apply_frames(
            atom_rots_final,
            atom_trans_final,
            local_coords
        )
        
        # Применяем маску атомов
        if atom_mask is not None:
            final_coords = final_coords * atom_mask.unsqueeze(-1).float()
        
        return final_coords


ProteinUpscaler = MultiScaleProteinUpscaler
