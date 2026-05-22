import torch
import torch.nn as nn
import torch.nn.functional as F

from upscaler.data.dataset import ProteinUpscalingDataset
from upscaler.model.geometric import invert_frames, apply_frames


# Определяем радиусы Ван-дер-Ваальса для атомов из dataset.py
VDW_RADII_ANGSTROM = {
    'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'H': 1.2, 'P': 1.8,
    'SE': 1.9, 'FE': 2.0, 'ZN': 2.1, 'MG': 1.73, 'CA': 1.75, 'MN': 2.0,
    'CU': 1.89, 'NI': 1.84, 'CO': 1.8, 'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'F': 1.47
}

# Создаем тензор радиусов, соответствующий atom_type_map из dataset.py
ATOM_TYPE_MAP = ProteinUpscalingDataset._build_map(
    ['C', 'N', 'O', 'S', 'H', 'P', 'SE',
     'FE', 'ZN', 'MG', 'CA', 'MN', 'CU', 'NI', 'CO',
     'CL', 'BR', 'I', 'F']
)

# Карта имён атомов согласована с ProteinUpscalingDataset
ATOM_NAME_MAP = ProteinUpscalingDataset._build_map(
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',
     'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', 'ND1', 'ND2',
     'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OD1', 'OD2', 'OE1', 'OE2',
     'OG', 'OG1', 'OH', 'SD', 'SE', 'SG']
)

# Создаем тензор радиусов Ван-дер-Ваальса
vdw_radii_list = [0.0]  # PAD
for atom_symbol, _ in sorted(ATOM_TYPE_MAP.items(), key=lambda item: item[1]):
    if atom_symbol == 'PAD':
        continue
    vdw_radii_list.append(VDW_RADII_ANGSTROM.get(atom_symbol, 1.7))

# Преобразуем в тензор
VDW_RADII_TENSOR = torch.tensor(vdw_radii_list, dtype=torch.float32)


class FAPELoss(nn.Module):
    """
    Frame-Aligned Point Error (FAPE) Loss.
    """
    def __init__(self, clash_weight=0.05, physics_weight=0.3, clamp_dist=30.0):
        super().__init__()
        self.clash_weight = clash_weight
        self.physics_weight = physics_weight
        self.clamp_dist = clamp_dist
        # Сохраняем старые компоненты как вспомогательные
        self.clash_loss_fn = ClashLoss()
        self.physics_loss_fn = PhysicsLoss()

    def forward(self, pred_data, true_data):
        """
        Args:
            pred_data (dict): 'rots', 'trans', 'coords', 'mask', 'res_mask'
            true_data (dict): 'rots', 'trans', 'coords', 'atom_types',
                              'atom_names', 'res_map', 'res_chain'
        """
        fape_loss = self.compute_fape(
            (pred_data['rots'], pred_data['trans']),
            (true_data['rots'], true_data['trans']),
            res_mask=pred_data.get('res_mask'),
        )

        clash_loss = self.clash_loss_fn(
            pred_data['coords'],
            true_data['atom_types'],
            res_map=true_data.get('res_map'),
            res_chain=true_data.get('res_chain'),
            mask=pred_data.get('mask'),
        )
        physics_loss = self.physics_loss_fn(
            pred_data['coords'],
            true_data['atom_types'],
            atom_names=true_data.get('atom_names'),
            res_map=true_data.get('res_map'),
            res_chain=true_data.get('res_chain'),
            mask=pred_data.get('mask'),
        )

        total_loss = fape_loss + \
                     self.clash_weight * clash_loss + \
                     self.physics_weight * physics_loss

        metrics = {
            'fape': fape_loss.detach(),
            'clash': clash_loss.detach(),
            'physics': physics_loss.detach(),
            'total': total_loss.detach()
        }
        
        return total_loss, metrics

    def compute_fape(self, pred_frames, true_frames, res_mask=None, eps=1e-8):
        """CA-FAPE: ошибка предсказанных позиций CA-атомов (= translations
        фреймов) в локальных системах координат каждого residue.

        Тензор-память: ``[B, N_res, N_res, 3]`` вместо ``[B, N_res, N_atoms, 3]``.
        Для N_atoms ≫ N_res это разница в десятки раз: на длинных белках
        полная all-atom версия упирается в OOM.
        """
        rots_pred, trans_pred = pred_frames  # trans это CA-позиция
        rots_true, trans_true = true_frames

        inv_rots_pred, inv_trans_pred = invert_frames(rots_pred, trans_pred)
        inv_rots_true, inv_trans_true = invert_frames(rots_true, trans_true)

        # local_coords[b, i, j] = inv_R_i @ (CA_j - t_i): CA остатка j в
        # локальной системе остатка i.
        local_coords_pred = apply_frames(
            inv_rots_pred.unsqueeze(2),
            inv_trans_pred.unsqueeze(2),
            trans_pred.unsqueeze(1),
        )
        local_coords_true = apply_frames(
            inv_rots_true.unsqueeze(2),
            inv_trans_true.unsqueeze(2),
            trans_true.unsqueeze(1),
        )

        error_dist = torch.sqrt(
            torch.sum((local_coords_pred - local_coords_true) ** 2, dim=-1) + eps
        )
        clamped_error = torch.clamp(error_dist, max=self.clamp_dist)

        # Маска валидных пар (residue_i, residue_j)
        if res_mask is not None:
            m = res_mask.to(dtype=clamped_error.dtype)
            pair_mask = m.unsqueeze(2) * m.unsqueeze(1)
        else:
            pair_mask = torch.ones_like(clamped_error)

        return torch.sum(clamped_error * pair_mask) / (torch.sum(pair_mask) + eps)


class ClashLoss(nn.Module):
    """
    Штраф за стерические столкновения.

    Исключаем из учёта:
      • атомы в одном остатке (внутри-residue связи);
      • атомы соседних остатков (i, i+1) — содержат пептидную связь C–N;
      • паддинг (по mask).
    """

    def forward(
        self,
        coords,
        atom_types,
        vdw_radii=VDW_RADII_TENSOR,
        res_map=None,
        res_chain=None,
        mask=None,
        eps=1e-8,
    ):
        vdw_radii = vdw_radii.to(atom_types.device)
        radii = vdw_radii[atom_types]

        # Попарные расстояния и пороги
        distances = torch.cdist(coords, coords, p=2)
        min_distances = radii.unsqueeze(-1) + radii.unsqueeze(-2)

        clashes = (distances < min_distances) & (distances > eps)

        # Исключаем "bonded" пары: |i-j|<=1 в ОДНОЙ цепи.
        # На chain break (разные цепи) пара остаётся как clash-кандидат.
        if res_map is not None:
            diff_res = (res_map.unsqueeze(-1) - res_map.unsqueeze(-2)).abs()
            same_chain = torch.ones_like(diff_res, dtype=torch.bool)
            if res_chain is not None:
                # chain per atom = res_chain.gather(res_map_clamped)
                rmap_c = res_map.clamp(min=0)
                chain_per_atom = torch.gather(res_chain, 1, rmap_c)
                same_chain = (
                    chain_per_atom.unsqueeze(-1) == chain_per_atom.unsqueeze(-2)
                ) & (chain_per_atom.unsqueeze(-1) >= 0)
            bonded = (diff_res <= 1) & same_chain
            clashes = clashes & (~bonded)

        # Маска паддинга
        if mask is not None:
            m = mask.to(dtype=torch.bool)
            pair_mask = m.unsqueeze(-1) & m.unsqueeze(-2)
            clashes = clashes & pair_mask
            valid_atoms = m.float().sum(dim=-1)
        else:
            valid_atoms = torch.full(
                coords.shape[:-2], float(coords.shape[-2]),
                device=coords.device, dtype=coords.dtype
            )

        num_clashes = torch.sum(clashes.float(), dim=(-1, -2)) / 2.0
        valid_atoms = torch.clamp(valid_atoms, min=1.0)
        normalized_clashes = num_clashes / valid_atoms

        return torch.mean(normalized_clashes)


class PhysicsLoss(nn.Module):
    """
    Штраф за отклонение длины пептидной связи C(i)–N(i+1) от ожидаемой ~1.33 Å.

    В отличие от наивной all-pairs C–N версии, здесь учитываются ТОЛЬКО реальные
    пептидные связи между остатком i и i+1, что требует atom_names и res_map.
    """

    def forward(
        self,
        coords,
        atom_types,
        atom_names=None,
        res_map=None,
        res_chain=None,
        mask=None,
        expected_length: float = 1.33,
        tolerance: float = 0.2,
        eps: float = 1e-8,
    ):
        # Без сведений об именах атомов / индексах остатков честно посчитать
        # peptide-bond loss не получится → возвращаем 0.
        if atom_names is None or res_map is None:
            return coords.new_zeros(())

        c_name_idx = ATOM_NAME_MAP['C']
        n_name_idx = ATOM_NAME_MAP['N']

        B, N = atom_names.shape

        if mask is not None:
            m = mask.to(dtype=torch.bool)
        else:
            m = torch.ones_like(atom_names, dtype=torch.bool)

        is_bb_c = (atom_names == c_name_idx) & m
        is_bb_n = (atom_names == n_name_idx) & m

        penalties = []
        for b in range(B):
            c_idx = torch.nonzero(is_bb_c[b], as_tuple=False).flatten()
            n_idx = torch.nonzero(is_bb_n[b], as_tuple=False).flatten()
            if c_idx.numel() == 0 or n_idx.numel() == 0:
                continue

            c_res = res_map[b, c_idx]
            n_res = res_map[b, n_idx]

            n_lookup: dict[int, int] = {}
            for ni in range(n_idx.numel()):
                r = int(n_res[ni].item())
                if r not in n_lookup:
                    n_lookup[r] = int(n_idx[ni].item())

            pairs_c, pairs_n = [], []
            for ci in range(c_idx.numel()):
                r = int(c_res[ci].item())
                nxt = n_lookup.get(r + 1)
                if nxt is None:
                    continue
                # Пропускаем границу цепей: если residue r и r+1 в разных
                # цепях, это НЕ пептидная связь.
                if res_chain is not None:
                    chain_r = int(res_chain[b, r].item()) if 0 <= r < res_chain.shape[1] else -1
                    chain_n = int(res_chain[b, r + 1].item()) if 0 <= r + 1 < res_chain.shape[1] else -1
                    if chain_r != chain_n or chain_r < 0:
                        continue
                pairs_c.append(int(c_idx[ci].item()))
                pairs_n.append(nxt)

            if not pairs_c:
                continue

            c_xyz = coords[b, pairs_c]
            n_xyz = coords[b, pairs_n]
            dist = torch.norm(c_xyz - n_xyz, dim=-1)
            deviation = torch.abs(dist - expected_length)
            penalty = F.relu(deviation - tolerance)
            penalty = torch.clamp(penalty, max=10.0)
            penalties.append(penalty.mean())

        if not penalties:
            return coords.new_zeros(())

        return torch.stack(penalties).mean()


class LddtLoss(nn.Module):
    def forward(self, pred_coords, true_coords, cutoff=15.0, eps=1e-8):
        """Вычисляет lDDT (Local Distance Difference Test) loss."""
        # Вычисление всех попарных расстояний
        pred_distances = torch.cdist(pred_coords, pred_coords, p=2)
        true_distances = torch.cdist(true_coords, true_coords, p=2)
        
        # Маска для локальных взаимодействий
        local_mask = (true_distances < cutoff) & (true_distances > eps)
        
        # Вычисление разницы расстояний
        diff = torch.abs(pred_distances - true_distances)
        
        # lDDT score: доля сохраненных расстояний в пределах определенных порогов
        score_05 = (diff < 0.5).float() * 1.0
        score_10 = ((diff >= 0.5) & (diff < 1.0)).float() * 0.5
        score_20 = ((diff >= 1.0) & (diff < 2.0)).float() * 0.25
        score_40 = ((diff >= 2.0) & (diff < 4.0)).float() * 0.125
        # Суммируем все вклады
        preserved = score_05 + score_10 + score_20 + score_40
        
        # Суммируем preserved * local_mask по всем парам
        numerator = torch.sum(preserved * local_mask, dim=(-1, -2)) 
        # Суммируем local_mask по всем парам
        denominator = torch.sum(local_mask.float(), dim=(-1, -2)) 
        
        # Избегаем деления на ноль
        denominator = torch.clamp(denominator, min=eps)
        
        # lDDT для каждого примера в батче
        lddt_per_sample = numerator / denominator
        
        # lDDT
        # Усредняем по всем примерам в батче
        return torch.mean(1.0 - lddt_per_sample)


class RmsdLoss(nn.Module):
    def forward(self, pred_coords, true_coords):
        """Вычисляет RMSD между предсказанными и истинными координатами."""
        diff = pred_coords - true_coords
        # torch.norm(diff, dim=-1) дает расстояния для каждого атома
        # torch.mean по всем атомам и батчам
        return torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1)) + 1e-8)


ProteinUpscalingLoss = FAPELoss
