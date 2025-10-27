import torch


def frames_from_bb_coords(n_coords, ca_coords, c_coords, eps=1e-8):
    """
    Создает фреймы для каждого остатка.

    Args:
        n_coords (torch.Tensor): Координаты атомов N. Shape: [..., N_res, 3]
        ca_coords (torch.Tensor): Координаты атомов CA. Shape: [..., N_res, 3]
        c_coords (torch.Tensor): Координаты атомов C. Shape: [..., N_res, 3]

    Returns:
        tuple: (rots, trans), где rots - матрицы вращения [..., N_res, 3, 3], 
               trans - векторы трансляции [..., N_res, 3]
    """
    translations = ca_coords

    v1 = c_coords - ca_coords
    v2 = n_coords - ca_coords

    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    
    u2 = v2 - torch.sum(e1 * v2, dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    
    e3 = torch.cross(e1, e2, dim=-1)

    rotations = torch.stack([e1, e2, e3], dim=-1)

    return rotations, translations


def invert_frames(rots, trans):
    """Инвертирует фреймы."""
    inv_rots = rots.transpose(-1, -2)
    inv_trans = -torch.einsum('...ij,...j->...i', inv_rots, trans)
    return inv_rots, inv_trans


def apply_frames(rots, trans, coords):
    """
    Применяет преобразование фреймов к набору координат.
    """
    rotated_coords = torch.einsum('...ij,...j->...i', rots, coords)
    return rotated_coords + trans


def rots_from_vec(vec, eps=1e-8):
    """Создает матрицу вращения из вектора по формуле Родрига."""
    theta = torch.norm(vec, dim=-1, keepdim=True)
    v = vec / (theta + eps)
    
    K = torch.zeros(*vec.shape[:-1], 3, 3, device=vec.device, dtype=vec.dtype)
    K[..., 0, 1] = -v[..., 2]
    K[..., 0, 2] = v[..., 1]
    K[..., 1, 0] = v[..., 2]
    K[..., 1, 2] = -v[..., 0]
    K[..., 2, 0] = -v[..., 1]
    K[..., 2, 1] = v[..., 0]
    
    theta = theta.unsqueeze(-1)
    return torch.eye(3, device=vec.device, dtype=vec.dtype) + \
           torch.sin(theta) * K + \
           (1 - torch.cos(theta)) * torch.einsum('...ij,...jk->...ik', K, K)
