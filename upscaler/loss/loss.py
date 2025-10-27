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
    Frame-Aligned Point Error (FAPE) Loss, как в AlphaFold2.
    """
    def __init__(self, clash_weight=0.05, physics_weight=0.3, clamp_dist=10.0):
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
            pred_data (dict): Словарь с предсказанными 'rots', 'trans', 'coords'
            true_data (dict): Словарь с истинными 'rots', 'trans', 'coords', 'atom_types'
        """
        fape_loss = self.compute_fape(
            (pred_data['rots'], pred_data['trans']),
            (true_data['rots'], true_data['trans']),
            pred_data['coords'],
            true_data['coords'],
            res_mask=pred_data.get('res_mask'),
            mask=pred_data.get('mask'),
        )

        clash_loss = self.clash_loss_fn(pred_data['coords'], true_data['atom_types'])
        physics_loss = self.physics_loss_fn(pred_data['coords'], true_data['atom_types'])

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

    def compute_fape(self, pred_frames, true_frames, pred_coords, true_coords, res_mask=None, mask=None, eps=1e-8):
        # Распаковка фреймов
        rots_pred, trans_pred = pred_frames
        rots_true, trans_true = true_frames

        # Инвертируем фреймы
        inv_rots_pred, inv_trans_pred = invert_frames(rots_pred, trans_pred)
        inv_rots_true, inv_trans_true = invert_frames(rots_true, trans_true)
        
        # Переводим глобальные координаты всех атомов в локальные системы координат КАЖДОГО остатка
        # Shape: [B, N_res, N_atoms, 3]
        local_coords_pred = apply_frames(inv_rots_pred.unsqueeze(2), inv_trans_pred.unsqueeze(2), pred_coords.unsqueeze(1))
        local_coords_true = apply_frames(inv_rots_true.unsqueeze(2), inv_trans_true.unsqueeze(2), true_coords.unsqueeze(1))
        
        # Считаем L2-расстояние
        error_dist = torch.sqrt(torch.sum((local_coords_pred - local_coords_true) ** 2, dim=-1) + eps)
        
        # Ограничиваем расстояние
        clamped_error = torch.clamp(error_dist, max=self.clamp_dist)
        
        # Применяем маски, если они есть
        if mask is not None:
             clamped_error = clamped_error * mask.unsqueeze(1)
        if res_mask is not None:
             clamped_error = clamped_error * res_mask.unsqueeze(2)
        
        # Усредняем
        loss = torch.sum(clamped_error) / (torch.sum(mask) * torch.sum(res_mask) + eps)
        
        return loss


class ClashLoss(nn.Module):
    def forward(self, coords, atom_types, vdw_radii=VDW_RADII_TENSOR, eps=1e-8):
        """Вычисляет штраф за стерические столкновения."""
        # Получаем радиусы Ван-дер-Ваальса для каждого атома
        radii = vdw_radii[atom_types]
        
        # Вычисляем все попарные расстояния
        distances = torch.cdist(coords, coords, p=2)
        
        # Вычисляем минимальные допустимые расстояния
        min_distances = radii.unsqueeze(-1) + radii.unsqueeze(-2)
        
        # Маска для столкновений
        clashes = (distances < min_distances) & (distances > eps)
        
        # Подсчитываем количество столкновений
        num_clashes = torch.sum(clashes.float(), dim=(-1, -2)) / 2.0
        
        num_atoms = coords.shape[-2]
        if num_atoms > 0:
            normalized_clashes = num_clashes / num_atoms
        else:
            normalized_clashes = torch.zeros_like(num_clashes)
        
        # Возвращаем среднее количество столкновений на пример в батче
        return torch.mean(normalized_clashes)


class PhysicsLoss(nn.Module):
    def forward(self, coords, atom_types, backbone_elements={'C', 'N', 'O'}, eps=1e-8):
        """
        Вычисляет физико-химические ограничения.
        В данном случае - штраф за нереалистичные длины пептидных связей (C-N).
        """
        c_type_idx = ATOM_TYPE_MAP['C']
        n_type_idx = ATOM_TYPE_MAP['N']
        
        is_c = (atom_types == c_type_idx)
        is_n = (atom_types == n_type_idx)
        
        # Вычисляем все расстояния между C и N атомами
        c_coords = coords.unsqueeze(-2)
        n_coords = coords.unsqueeze(-3)
        cn_distances = torch.norm(c_coords - n_coords, dim=-1)
        
        # Создаем маску для пар C-N
        c_mask = is_c.unsqueeze(-1)
        n_mask = is_n.unsqueeze(-2)
        cn_mask = c_mask & n_mask
        
        # Ожидаемая длина пептидной связи C-N ~ 1.33 Angstrom
        expected_length = 1.33
        # Допустимое отклонение
        tolerance = 0.2
        
        # Вычисляем отклонение от ожидаемой длины
        deviation = torch.abs(cn_distances - expected_length)
        
        # Штраф для пар, которые выходят за пределы допуска
        penalty = F.relu(deviation - tolerance)
        
        # Применяем маску и усредняем
        masked_penalty = penalty * cn_mask.float()
        # Суммируем по всем парам и нормализуем
        total_penalty = torch.sum(masked_penalty, dim=(-1, -2))
        # Нормализуем по количеству пар C-N в каждом примере
        num_cn_pairs = torch.sum(cn_mask.float(), dim=(-1, -2))
        num_cn_pairs = torch.clamp(num_cn_pairs, min=1.0)
        
        avg_penalty_per_sample = total_penalty / num_cn_pairs
        
        # Возвращаем средний штраф по батчу
        return torch.mean(avg_penalty_per_sample)


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
