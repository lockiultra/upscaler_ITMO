import torch

from upscaler.data.dataset import ProteinUpscalingDataset
from upscaler.data.align import kabsch_superimpose


VDW_RADII_ANGSTROM = {
    'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'H': 1.2, 'P': 1.8,
    'SE': 1.9, 'FE': 2.0, 'ZN': 2.1, 'MG': 1.73, 'CA': 1.75, 'MN': 2.0,
    'CU': 1.89, 'NI': 1.84, 'CO': 1.8, 'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'F': 1.47
}

ATOM_TYPE_MAP = ProteinUpscalingDataset._build_map(
    ['C', 'N', 'O', 'S', 'H', 'P', 'SE',
     'FE', 'ZN', 'MG', 'CA', 'MN', 'CU', 'NI', 'CO',
     'CL', 'BR', 'I', 'F']
)


vdw_radii_list = [0.0] 
for atom_symbol, _ in sorted(ATOM_TYPE_MAP.items(), key=lambda item: item[1]):
    if atom_symbol == 'PAD':
        continue
    vdw_radii_list.append(VDW_RADII_ANGSTROM.get(atom_symbol, 1.7))

VDW_RADII_TENSOR = torch.tensor(vdw_radii_list, dtype=torch.float32)


class QualityMetrics:
    """Класс для вычисления метрик качества апскейлинга белковых структур."""
    
    def __init__(self, device='cpu'):
        """
        Args:
            device (str or torch.device): Устройство для вычислений.
        """
        self.device = device
        self.vdw_radii = VDW_RADII_TENSOR.to(device)

    @staticmethod
    def compute_rmsd(pred_coords, true_coords, eps=1e-8, mask=None):
        """Kabsch-aligned RMSD по батчу.

        Если ``mask`` не передана, неявно фильтруем строки, где обе
        координаты ровно (0,0,0) — типичный признак паддинга после
        ``pad_sequence``. Это защищает caller-ов, забывших mask, от
        систематически неверного RMSD.
        """
        if mask is None:
            pad_mask = (pred_coords.abs().sum(dim=-1) < eps) & (
                true_coords.abs().sum(dim=-1) < eps
            )
            mask = ~pad_mask

        rmsds = []
        for i in range(pred_coords.shape[0]):
            row_mask = mask[i].to(dtype=torch.bool)
            pred_i = pred_coords[i][row_mask]
            true_i = true_coords[i][row_mask]
            if len(pred_i) > 0:
                _, rmsd, _, _ = kabsch_superimpose(
                    true_i.detach().cpu().numpy(),
                    pred_i.detach().cpu().numpy(),
                )
                rmsds.append(rmsd)
        if not rmsds:
            return torch.tensor(0.0, device=pred_coords.device)
        return torch.tensor(rmsds, device=pred_coords.device).mean()

    @staticmethod
    def compute_lddt(pred_coords, true_coords, cutoff=15.0, eps=1e-8, mask=None):
        """
        Вычисляет Local Distance Difference Test (lDDT) score.
        lDDT находится в диапазоне [0, 1], где 1 - идеальное совпадение.
        
        Args:
            pred_coords (torch.Tensor): Предсказанные координаты [..., N, 3].
            true_coords (torch.Tensor): Истинные координаты [..., N, 3].
            cutoff (float): Порог расстояния для локальных взаимодействий.
            eps (float): Малое значение для избежания деления на ноль.
            
        Returns:
            torch.Tensor: Скалярный тензор со значением lDDT.
        """
        # Вычисление всех попарных расстояний
        pred_distances = torch.cdist(pred_coords, pred_coords, p=2)
        true_distances = torch.cdist(true_coords, true_coords, p=2)
        
        # Маска для локальных взаимодействий
        local_mask = (true_distances < cutoff) & (true_distances > eps)

        # Учитываем маску паддинга атомов (если задана)
        if mask is not None:
            m = mask.to(dtype=torch.bool)
            pair_mask = m.unsqueeze(-1) & m.unsqueeze(-2)
            local_mask = local_mask & pair_mask
        
        # Вычисление разницы расстояний
        diff = torch.abs(pred_distances - true_distances)
        
        # lDDT score
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
        
        # Возвращаем средний lDDT по всем примерам в батче
        return torch.mean(lddt_per_sample)

    def compute_clash_score(self, coords, atom_types, eps=1e-8, mask=None):
        """
        Подсчитывает количество стерических столкновений.
        
        Args:
            coords (torch.Tensor): Координаты атомов [..., N, 3].
            atom_types (torch.LongTensor): Типы атомов [..., N].
            eps (float): Малое значение для избежания деления на ноль.
            
        Returns:
            torch.Tensor: Скалярный тензор со средним количеством столкновений на атом.
        """

        radii = self.vdw_radii[atom_types]
        
        # Вычисляем все попарные расстояния
        distances = torch.cdist(coords, coords, p=2)
        
        # Вычисляем минимальные допустимые расстояния
        min_distances = radii.unsqueeze(-1) + radii.unsqueeze(-2)
        
        # Маска для столкновений: расстояние < минимальное расстояние
        clashes = (distances < min_distances) & (distances > eps)

        # Учитываем маску паддинга атомов (если задана)
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

        # Подсчитываем количество столкновений
        num_clashes = torch.sum(clashes.float(), dim=(-1, -2)) / 2.0 # [...]

        # Нормализуем по количеству валидных атомов
        valid_atoms = torch.clamp(valid_atoms, min=1.0)
        clash_score_per_sample = num_clashes / valid_atoms
        
        # Возвращаем средний clash score по всем примерам в батче
        return torch.mean(clash_score_per_sample)
