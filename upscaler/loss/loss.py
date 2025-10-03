import torch
import torch.nn as nn
import torch.nn.functional as F

from upscaler.data.dataset import ProteinUpscalingDataset


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


class ProteinUpscalingLoss(nn.Module):
    """Многокомпонентная функция потерь для апскейлинга белковых структур."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.register_buffer('vdw_radii', VDW_RADII_TENSOR.to(device))
        
        # Веса для компонентов потерь
        self.coord_weight = 0.5
        self.lddt_weight = 2.0
        self.clash_weight = 0.05
        self.physics_weight = 0.3

    def forward(self, pred_coords, true_coords, atom_types):
        """
        Args:
            pred_coords: Tensor of shape [..., N, 3] - Предсказанные координаты.
            true_coords: Tensor of shape [..., N, 3] - Истинные координаты.
            atom_types: LongTensor of shape [..., N] - Типы атомов.
        Returns:
            total_loss: Scalar tensor - Общая потеря.
            metrics: Dict - Словарь с отдельными компонентами потерь.
        """
        # 1. Coordinate RMSD
        coord_loss = self.compute_coord_rmsd_loss(pred_coords, true_coords)
        
        # 2. Local Distance Difference Test (lDDT)
        lddt_loss = self.compute_lddt_loss(pred_coords, true_coords)
        
        # 3. Clash score (штраф за перекрывающиеся атомы)
        clash_loss = self.compute_clash_penalty(pred_coords, atom_types)
        
        # 4. Физико-химические ограничения (длины связей)
        physics_loss = self.compute_physics_constraints(pred_coords, atom_types)
        
        total_loss = (
            self.coord_weight * coord_loss +
            self.lddt_weight * lddt_loss +
            self.clash_weight * clash_loss +
            self.physics_weight * physics_loss
        )
        
        metrics = {
            'coord_rmsd': coord_loss.detach(),
            'lddt': lddt_loss.detach(),
            'clash': clash_loss.detach(),
            'physics': physics_loss.detach(),
            'total': total_loss.detach()
        }
        
        return total_loss, metrics

    def compute_coord_rmsd_loss(self, pred_coords, true_coords):
        """Вычисляет RMSD между предсказанными и истинными координатами."""
        diff = pred_coords - true_coords
        # torch.norm(diff, dim=-1) дает расстояния для каждого атома
        # torch.mean по всем атомам и батчам
        return torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1)) + 1e-8)

    def compute_lddt_loss(self, pred_coords, true_coords, cutoff=15.0, eps=1e-8):
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

    def compute_clash_penalty(self, coords, atom_types, eps=1e-8):
        """Вычисляет штраф за стерические столкновения."""
        # Получаем радиусы Ван-дер-Ваальса для каждого атома
        radii = self.vdw_radii[atom_types]
        
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

    def compute_physics_constraints(self, coords, atom_types, backbone_elements={'C', 'N', 'O'}, eps=1e-8):
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
