import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from torch_geometric.nn import MessagePassing
from pytorch3d.transforms import quaternion_apply
from pytorch3d.transforms.rotation_conversions import axis_angle_to_quaternion

from upscaler.model.encoders import build_knn_graph


class SE3EquivariantLayer(MessagePassing):
    """SE(3)-эквивариантный слой на основе e3nn и MessagePassing."""
    def __init__(self, irreps_input='64x0e + 32x1e + 16x2e', irreps_output='64x0e + 32x1e + 16x2e', irreps_edge_attr="1x1o", **kwargs):
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
    def __init__(self, irreps_model, radius=5.0, k=8):
        super().__init__()
        self.radius = radius
        self.k = k
        self.irreps_input = o3.Irreps(irreps_model)
        self.irreps_output = self.irreps_input
        self.irreps_edge_attr = o3.Irreps("1x1o")
        
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
        d_model = self.irreps_input.dim
        
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
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_feats, mask=None):
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask 
        attn_out, _ = self.attention(node_feats, node_feats, node_feats, key_padding_mask=key_padding_mask)
        out = self.norm(node_feats + self.dropout(attn_out))
        return out


class CoordinatePredictor(nn.Module):
    """Предсказывает обновления координат."""
    def __init__(self, d_model=256):
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


class GeometricUpdateHead(nn.Module):
    def __init__(self, d_model: int, num_modules: int = 8, max_translation: float = 3.0, max_rotation_angle: float = torch.pi):
        super().__init__()
        self.num_modules = num_modules
        self.d_model = d_model
        self.max_translation = max_translation
        self.max_rotation_angle = max_rotation_angle
        self.param_predictor = nn.Linear(d_model, num_modules * 8)

    def forward(self, node_feats: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = coords.shape

        input_dtype = coords.dtype
        
        params = self.param_predictor(node_feats)
        params = params.view(B, N, self.num_modules, 8)
        
        anchor_logits, select_logits, rot_axis, trans_vectors = torch.split(
            params, [1, 1, 3, 3], dim=-1
        )
        
        anchor_weights = F.softmax(anchor_logits, dim=1)
        select_probs = torch.sigmoid(select_logits)

        rot_axis_float32 = rot_axis.float()
        rot_angles = torch.norm(rot_axis_float32, dim=-1, keepdim=True)
        rot_axis_norm = F.normalize(rot_axis_float32, dim=-1)
        rot_angles = torch.tanh(rot_angles) * self.max_rotation_angle
        
        quaternions = axis_angle_to_quaternion(rot_axis_norm * rot_angles)
        
        trans_vectors = torch.tanh(trans_vectors.float()) * self.max_translation

        current_coords = coords.clone()
        
        for i in range(self.num_modules):
            q_i = quaternions[:, :, i, :]
            t_i = trans_vectors[:, :, i, :]
            anchor_w_i = anchor_weights[:, :, i].float()
            select_p_i = select_probs[:, :, i].float()

            current_coords_float32 = current_coords.float()

            anchor_point = torch.sum(current_coords_float32 * anchor_w_i, dim=1, keepdim=True)
            
            sum_select_probs = select_p_i.sum(dim=1, keepdim=True).clamp(min=1e-8)
            q_group = torch.sum(q_i * select_p_i, dim=1, keepdim=True) / sum_select_probs
            t_group = torch.sum(t_i * select_p_i, dim=1, keepdim=True) / sum_select_probs
            
            q_group = F.normalize(q_group, p=2, dim=-1)

            centered_coords = current_coords_float32 - anchor_point
            rotated_coords = quaternion_apply(q_group, centered_coords)
            moved_coords = rotated_coords + anchor_point + t_group
            
            updated_coords_float32 = torch.lerp(current_coords_float32, moved_coords, select_p_i)
            
            current_coords = updated_coords_float32.to(input_dtype)

            if mask is not None:
                current_coords = current_coords * mask.unsqueeze(-1)

        return current_coords
