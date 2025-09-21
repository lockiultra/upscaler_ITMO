import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from torch_scatter import scatter_mean


ATOM_TYPE_MAP = {
    'PAD': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'H': 5, 'P': 6, 'SE': 7,
    'FE': 8, 'ZN': 9, 'MG': 10, 'CA': 11, 'MN': 12, 'CU': 13, 'NI': 14, 'CO': 15,
    'CL': 16, 'BR': 17, 'I': 18, 'F': 19
}
RESIDUE_TYPE_MAP = {
    'PAD': 0, 'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6, 'GLU': 7,
    'GLY': 8, 'HIS': 9, 'ILE': 10, 'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14,
    'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20
}

def gaussian_soft_one_hot(input_tensor, start, end, number):
    """
    Простая и надежная реализация soft one-hot с гауссовым ядром.
    """
    # Нормализуем вход в диапазон [0, 1]
    input_scaled = (input_tensor - start) / (end - start)
    # Обрезаем значения вне [0, 1] для стабильности
    input_scaled = torch.clamp(input_scaled, 0.0, 1.0)
    
    # Создаем центры гауссиан в диапазоне [0, 1]
    centers = torch.linspace(0.0, 1.0, number, device=input_tensor.device) # [number]
    
    # Вычисляем расстояния между нормализованным входом и центрами
    diff = input_scaled - centers  # Broadcasting
    
    # Применяем гауссово ядро
    sigma = 1.0 / number
    gaussian = torch.exp(- (diff ** 2) / (2 * sigma ** 2))
    
    # Нормализуем по сумме для получения распределения вероятности
    sum_gaussian = gaussian.sum(dim=-1, keepdim=True)
    sum_gaussian = torch.where(sum_gaussian > 0, sum_gaussian, torch.ones_like(sum_gaussian))
    
    normalized = gaussian / sum_gaussian
    
    return normalized


# class SE3PositionalEncoder(nn.Module):
#     """SE(3)-инвариантный энкодер позиций.
#     Использует e3nn для создания инвариантных признаков из относительных векторов.
#     """
#     def __init__(self, irreps_out="64x0e", max_radius=10.0, number_of_basis=10):
#         super().__init__()
#         self.irreps_out = o3.Irreps(irreps_out)
#         self.max_radius = max_radius
#         self.number_of_basis = number_of_basis
        
#         input_irreps = o3.Irreps("1x1o")
        
#         self.tp = FullyConnectedTensorProduct(
#             input_irreps, # r_ij
#             f"{number_of_basis}x0e",
#             self.irreps_out,
#             shared_weights=False
#         )

#     def forward(self, coords):
#         """
#         Args:
#             coords: Tensor of shape [..., N, 3]
#         Returns:
#             pos_feats: Tensor of shape [..., N, self.irreps_out.dim]
#         """
#         original_shape = coords.shape
#         coords_flat = coords.view(-1, original_shape[-2], original_shape[-1])
#         batch_size, num_nodes, _ = coords_flat.shape
        
#         # Вычисляем все пары относительных векторов
#         sender_coords = coords_flat.unsqueeze(2).expand(-1, -1, num_nodes, -1).reshape(-1, 3)
    
#         receiver_coords = coords_flat.unsqueeze(1).expand(-1, num_nodes, -1, -1).reshape(-1, 3)
        
#         r_ij = sender_coords - receiver_coords # [B*N*N, 3]
        
#         # Вычисляем расстояния
#         distances = torch.norm(r_ij, dim=-1, keepdim=True) # [B*N*N, 1]
        
#         # Нормализуем векторы
#         r_ij_normalized = r_ij / (distances + 1e-8) # [B*N*N, 3]
        
#         # Радиальные признаки с использованием нашей soft one-hot
#         cutoff_values = distances / self.max_radius
#         # Применяем soft one-hot
#         radial_basis = gaussian_soft_one_hot(
#             cutoff_values, 0.0, 1.0, self.number_of_basis
#         )
        
#         # Преобразуем векторы в e3nn Irreps
#         r_ij_irreps = r_ij_normalized
        
#         # Вычисляем инвариантные признаки
#         invariant_features = self.tp(r_ij_irreps, radial_basis)
        
#         # Агрегируем по receiver
#         invariant_features = invariant_features.view(batch_size, num_nodes, num_nodes, -1)
#         aggregated_features = torch.mean(invariant_features, dim=2)
        
#         output_shape = list(original_shape[:-2]) + [num_nodes, self.irreps_out.dim]
#         return aggregated_features.view(output_shape)


class SE3PositionalEncoder(nn.Module):
    """
    Экономная версия: строит локальный kNN-граф вместо всех пар (O(B * N * k) вместо O(B * N^2)).
    """
    def __init__(self, irreps_out="64x0e + 32x1e + 16x2e", max_radius=10.0, number_of_basis=8, k=16):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.k = k

        input_irreps = o3.Irreps("1x1o")
        self.tp = FullyConnectedTensorProduct(
            input_irreps,
            f"{number_of_basis}x0e",
            self.irreps_out,
            shared_weights=False
        )

    def forward(self, coords):
        # coords: [B, N, 3]
        B, N, _ = coords.shape
        # build knn graph по батчу (возвращает индексы по B*N)
        edge_index = build_knn_graph(coords, k=self.k)  # [2, E]
        senders, receivers = edge_index  # 1D tensors length E

        coords_flat = coords.view(-1, 3)
        sender_coords = coords_flat[senders]      # [E, 3]
        receiver_coords = coords_flat[receivers]  # [E, 3]
        r_ij = sender_coords - receiver_coords    # [E, 3]

        distances = torch.norm(r_ij, dim=-1, keepdim=True)  # [E, 1]
        r_ij_normalized = r_ij / (distances + 1e-8)         # [E, 3]

        # radial soft one-hot на расстояниях / max_radius
        cutoff = (distances / self.max_radius).clamp(0.0, 1.0)  # [E,1]
        radial_basis = gaussian_soft_one_hot(cutoff, 0.0, 1.0, self.number_of_basis)  # [E, K]

        # FullyConnectedTensorProduct ожидает (vector, scalar-basis)
        invariant_features = self.tp(r_ij_normalized, radial_basis)  # [E, out_dim]

        num_total = B * N
        aggregated = scatter_mean(invariant_features, receivers, dim=0, dim_size=num_total)

        aggregated = aggregated.view(B, N, -1)  # [B, N, out_dim]
        return aggregated



class AtomTypeEmbedder(nn.Module):
    def __init__(self, vocab_size=len(ATOM_TYPE_MAP), irreps_out="32x0e + 16x1e"):
        super().__init__()
        self.vocab_size = vocab_size
        self.irreps_out = o3.Irreps(irreps_out)
        self.embedding = nn.Embedding(vocab_size, self.irreps_out.dim, padding_idx=0)

    def forward(self, atom_types):
        """
        Args:
            atom_types: LongTensor of shape [...]
        Returns:
            atom_feats: Tensor of shape [..., self.irreps_out.dim]
        """
        return self.embedding(atom_types)

class ResidueTypeEmbedder(nn.Module):
    def __init__(self, vocab_size=len(RESIDUE_TYPE_MAP), irreps_out="32x0e + 16x1e"):
        super().__init__()
        self.vocab_size = vocab_size
        self.irreps_out = o3.Irreps(irreps_out)
        self.embedding = nn.Embedding(vocab_size, self.irreps_out.dim, padding_idx=0)

    def forward(self, residue_types):
        """
        Args:
            residue_types: LongTensor of shape [...]
        Returns:
            res_feats: Tensor of shape [..., self.irreps_out.dim]
        """
        return self.embedding(residue_types)

class ProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_encoder = SE3PositionalEncoder(irreps_out="64x0e + 32x1e + 16x2e")
        self.atom_embedder = AtomTypeEmbedder(irreps_out="64x0e + 32x1e")
        self.residue_embedder = ResidueTypeEmbedder(irreps_out="64x0e + 32x1e")
        self.d_out = self.position_encoder.irreps_out.dim + self.atom_embedder.irreps_out.dim + self.residue_embedder.irreps_out.dim
        self.irreps_out = self.position_encoder.irreps_out + self.atom_embedder.irreps_out + self.residue_embedder.irreps_out

    def forward(self, coords, atom_types, residue_types):
        """
        Args:
            coords: Tensor of shape [..., N, 3]
            atom_types: LongTensor of shape [..., N]
            residue_types: LongTensor of shape [..., N]
        Returns:
            node_features: Tensor of shape [..., N, d_out] (инвариантные признаки)
        """
        pos_feats = self.position_encoder(coords) 
        atom_feats = self.atom_embedder(atom_types)
        res_feats = self.residue_embedder(residue_types)
        node_features = torch.cat([pos_feats, atom_feats, res_feats], dim=-1)
        return node_features

def build_knn_graph(coords, k=8):
    """
    Строит граф k-ближайших соседей.
    Args:
        coords: Tensor of shape [N, 3] or [B, N, 3]
        k: int, количество соседей
    Returns:
        edge_index: LongTensor of shape [2, E]
    """
    from torch_geometric.nn import knn_graph
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    batch_size, num_nodes, _ = coords.shape
    coords_flat = coords.view(-1, 3)
    batch_indices = torch.arange(batch_size, device=coords.device).repeat_interleave(num_nodes)
    edge_index = knn_graph(coords_flat, k=k, batch=batch_indices, loop=False)
    return edge_index
