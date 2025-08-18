import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from torch_geometric.nn import MessagePassing

from upscaler.model.encoders import build_knn_graph


class SE3EquivariantLayer(MessagePassing):
    """SE(3)-эквивариантный слой на основе e3nn и MessagePassing."""
    def __init__(self, irreps_input, irreps_output, irreps_edge_attr="0e + 1o", **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        
        self.tp = FullyConnectedTensorProduct(
            self.irreps_input,
            self.irreps_edge_attr,
            self.irreps_output,
        )
        
        self.lin = Linear(self.irreps_output, self.irreps_output)
        
        # Gate для нелинейности
        # irreps_gated = o3.Irreps("32x0e")
        # self.gate = Gate(
        #     "32x0e", [torch.nn.functional.silu],
        #     "32x0o", [torch.sigmoid],
        #     "64x0o + 32x1e + 16x2e" 
        # )

    def forward(self, node_feats, edge_index, edge_attr):
        """
        Args:
            node_feats: Tensor of shape [num_nodes, irreps_input.dim]
            edge_index: LongTensor of shape [2, num_edges]
            edge_attr: Tensor of shape [num_edges, irreps_edge_attr.dim]
        Returns:
            out: Tensor of shape [num_nodes, irreps_output.dim]
        """
        message = self.propagate(edge_index, node_feats=node_feats, edge_attr=edge_attr)
        out = self.lin(message)
        # out = self.gate(out)
        return out

    def message(self, node_feats_i, node_feats_j, edge_attr):
        return self.tp(node_feats_j, edge_attr)

class LocalRefinementBlock(nn.Module):
    """Блок локальной обработки с SE(3)-эквивариантными слоями."""
    def __init__(self, d_model=128, radius=5.0, k=8):
        super().__init__()
        self.radius = radius
        self.k = k
        self.irreps_input = o3.Irreps(f"{d_model}x0e")
        self.irreps_output = self.irreps_input
        self.irreps_edge_attr = o3.Irreps("1x1o") # Вектор
        
        self.equivariant_layer = SE3EquivariantLayer(
            self.irreps_input,
            self.irreps_output,
            self.irreps_edge_attr
        )

    def forward(self, node_feats, coords):
        """
        Args:
            node_feats: Tensor of shape [B, N, d_model] (инвариантные признаки)
            coords: Tensor of shape [B, N, 3]
        Returns:
            refined_feats: Tensor of shape [B, N, d_model] (инвариантные признаки)
        """
        batch_size, num_nodes, d_model = node_feats.shape
        
        # Строим локальный граф
        edge_index = build_knn_graph(coords, k=self.k)
        
        # Преобразуем координаты в векторные атрибуты ребер
        senders, receivers = edge_index
        sender_coords = coords.view(-1, 3)[senders]
        receiver_coords = coords.view(-1, 3)[receivers]
        edge_vectors = sender_coords - receiver_coords
        edge_attr = edge_vectors
        
        # Преобразуем в формат PyG
        node_feats_flat = node_feats.view(-1, d_model)
        
        # Прямой проход через эквивариантный слой
        refined_scalar = self.equivariant_layer(node_feats_flat, edge_index, edge_attr)
        
        # Возвращаем к форме батча
        refined_feats = refined_scalar.view(batch_size, num_nodes, d_model)
        return refined_feats

class GlobalAttentionBlock(nn.Module):
    """Блок глобального внимания."""
    def __init__(self, d_model=128, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_feats):
        """
        Args:
            node_feats: Tensor of shape [B, N, d_model]
        Returns:
            attended_feats: Tensor of shape [B, N, d_model]
        """
        attn_out, _ = self.attention(node_feats, node_feats, node_feats)
        # residual и norm
        out = self.norm(node_feats + self.dropout(attn_out))
        return out

class CoordinatePredictor(nn.Module):
    """Предсказывает обновления координат."""
    def __init__(self, d_model=128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, node_feats):
        """
        Args:
            node_feats: Tensor of shape [..., d_model]
        Returns:
            coord_updates: Tensor of shape [..., 3]
        """
        return self.predictor(node_feats)
