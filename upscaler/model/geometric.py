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
    """Создает матрицу вращения из вектора по формуле Родрига.

    Реализация через ``torch.stack`` — необходимо, чтобы матрица K оставалась
    дифференцируемой по ``vec``. In-place присвоения по индексам в новый
    ``torch.zeros``-тензор обрывают autograd-граф.
    """
    theta = torch.norm(vec, dim=-1, keepdim=True)
    v = vec / (theta + eps)

    zero = torch.zeros_like(v[..., 0])
    row0 = torch.stack([zero,        -v[..., 2],  v[..., 1]], dim=-1)
    row1 = torch.stack([ v[..., 2],   zero,      -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1],   v[..., 0],  zero      ], dim=-1)
    K = torch.stack([row0, row1, row2], dim=-2)  # [..., 3, 3]

    theta = theta.unsqueeze(-1)  # [..., 1, 1]
    eye = torch.eye(3, device=vec.device, dtype=vec.dtype).expand_as(K)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.einsum(
        '...ij,...jk->...ik', K, K
    )
