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
        """
        Вычисляет Root Mean Square Deviation (RMSD).
        
        Args:
            pred_coords (torch.Tensor): Предсказанные координаты [..., N, 3].
            true_coords (torch.Tensor): Истинные координаты [..., N, 3].
            eps (float): Малое значение для стабильности sqrt.
            
        Returns:
            torch.Tensor: Скалярный тензор со значением RMSD.
        """
        rmsds = []
        for i in range(pred_coords.shape[0]):
            pred_i = pred_coords[i][mask[i]] if mask is not None else pred_coords[i]
            true_i = true_coords[i][mask[i]] if mask is not None else true_coords[i]
            if len(pred_i) > 0:
                aligned, rmsd, _, _ = kabsch_superimpose(true_i.cpu().numpy(), pred_i.cpu().numpy())
                rmsds.append(rmsd)
        return torch.tensor(rmsds, device=pred_coords.device).mean() if rmsds else torch.tensor(0.0, device=pred_coords.device)
        # diff = pred_coords - true_coords
        # return torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1)) + eps)

    @staticmethod
    def compute_lddt(pred_coords, true_coords, cutoff=15.0, eps=1e-8):
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

    def compute_clash_score(self, coords, atom_types, eps=1e-8):
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
        
        # Подсчитываем количество столкновений
        num_clashes = torch.sum(clashes.float(), dim=(-1, -2)) / 2.0 # [...]
        
        # Нормализуем по количеству атомов
        num_atoms = coords.shape[-2]
        if num_atoms > 0:
            clash_score_per_sample = num_clashes / num_atoms
        else:
            clash_score_per_sample = torch.zeros_like(num_clashes)
        
        # Возвращаем средний clash score по всем примерам в батче
        return torch.mean(clash_score_per_sample)
