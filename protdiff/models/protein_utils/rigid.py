import torch
import numpy as np

# pylint: disable=bad-whitespace
QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # kr
QUAT_TO_ROT = torch.FloatTensor(QUAT_TO_ROT)


# calculate on average of valid atoms in 3GCB_A
STD_RIGID_COORD = torch.FloatTensor(
    [[-1.4589e+00, -2.0552e-07,  2.1694e-07],
     [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
     [ 5.4261e-01,  1.4237e+00,  1.0470e-07],
     [ 5.2744e-01, -7.8261e-01, -1.2036e+00],]
)


def norm_vec(v, epsilon=1.0e-8):
    return v / (epsilon + torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True)) )


def rigid_from_3_points(x, origin, xy, epsilon=1e-8):
    x0 = origin - x
    x1 = xy - origin

    e0 = norm_vec(x0)
    c = torch.sum(x1 * e0, dim=-1, keepdim=True)
    e1 = x1 - c * e0
    e1 = norm_vec(e1)

    e2 = torch.cross(e0, e1, dim=-1)
    
    rot = torch.cat(
        [e0.unsqueeze(-1), e1.unsqueeze(-1), e2.unsqueeze(-1)],
        dim=-1
    )
    trans = origin

    rigid = { 'rot': rot, 'trans': trans }
    return rigid


def invert_rigid(rigid):
    rot = rigid['rot']
    trans = rigid['trans']

    inv_rot = torch.inverse(rot)
    inv_trans = - torch.bmm(inv_rot, trans.unsqueeze(-1)).squeeze(-1)

    inv_rigid = {'rot': inv_rot, 'trans': inv_trans}
    return inv_rigid


def apply_xform(rot, trans, coord):
    assert(coord.shape[-1] == 3)
    shapes = [s for s in coord.shape]
    coord = torch.reshape(coord, (-1, 3))
    coord1 = torch.einsum('ab,nb->na', rot, coord) + trans[None]
    coord1 = torch.reshape(coord1, shapes)
    return coord1


def rot_to_quat(rot, unstack_inputs=True):
    """Convert rotation matrix to quaternion.

    Note that this function calls self_adjoint_eig which is extremely expensive on
    the GPU. If at all possible, this function should run on the CPU.

    Args:
        rot: rotation matrix (see below for format).
        unstack_inputs:  If true, rotation matrix should be shape (..., 3, 3)
        otherwise the rotation matrix should be a list of lists of tensors.

    Returns:
        Quaternion as (..., 4) tensor.
    """
    if unstack_inputs:
        rot = [torch.moveaxis(x, -1, 0) for x in torch.moveaxis(rot, -2, 0)]

    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    # pylint: disable=bad-whitespace
    k = [[ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
         [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
         [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
         [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]]
    # pylint: enable=bad-whitespace

    k = (1./3.) * torch.stack([torch.stack(x, dim=-1) for x in k],
                            dim=-2)

    # Get eigenvalues in non-decreasing order and associated.
    _, qs = torch.linalg.eigh(k)
    return qs[..., -1]


def quat_to_rot(quat):
    """Convert a normalized quaternion to a rotation matrix."""
    rot_tensor = torch.sum(
        # np.reshape(QUAT_TO_ROT.to(normalized_quat.device), (4, 4, 9)) *
        QUAT_TO_ROT.to(quat.device).view(4, 4, 9) *
        quat[..., :, None, None] *
        quat[..., None, :, None],
        dim=(-3, -2))
    
    new_shape = [s for s in quat.shape[:-1]] + [3, 3]
    rot = torch.reshape(rot_tensor, new_shape)
    return rot


def quat_to_axis_angles(quat, epsilon=1e-8):
    quat_norm = norm_vec(quat)
    # cos_theta = quat_norm[..., 0].clamp(min=epsilon - 1.0, max=1.0-epsilon)
    cos_theta = quat_norm[..., 0].clamp(min= - 1.0, max=1.0)
    theta = 2.0 * torch.acos(cos_theta)
    # v = quat_norm[..., 1:] / torch.sqrt(1.0 - cos_theta.unsqueeze(-1)**2)
    v = quat_norm[..., 1:]
    v = norm_vec(v)

    axis = v

    # phi: angle on x-y plan
    phi = torch.atan2(v[..., 1], v[..., 0])

    # phi: angle with x-y plan
    # cos_psi = v[..., 2].clamp(min=epsilon - 1.0, max=1.0-epsilon)
    cos_psi = v[..., 2].clamp(min= - 1.0, max=1.0)
    psi = torch.acos(cos_psi)

    angles = torch.stack([theta, phi, psi], dim=-1)
    return angles, axis


def axis_to_angles(v):
    # phi: angle on x-y plan
    phi = torch.atan2(v[..., 1], v[..., 0])

    # phi: angle with x-y plan
    # cos_psi = v[..., 2].clamp(min=epsilon - 1.0, max=1.0-epsilon)
    cos_psi = v[..., 2].clamp(min= - 1.0, max=1.0)
    psi = torch.acos(cos_psi)

    angles = torch.stack([phi, psi], dim=-1)
    return angles


def angles_to_axis(angles):
    theta, phi, psi = angles[..., 0], angles[..., 1], angles[..., 2]
    cos_psi = torch.cos(psi)
    abs_sin_psi = torch.abs(torch.sin(psi))
    v = torch.stack(
        [ torch.cos(phi) * abs_sin_psi, torch.sin(phi) * abs_sin_psi,  cos_psi],
        dim=-1
    )
    return v

def angles_to_quat(angles):
    theta, phi, psi = angles[..., 0], angles[..., 1], angles[..., 2]
    cos_psi = torch.cos(psi)
    abs_sin_psi = torch.abs(torch.sin(psi))
    v = torch.stack(
        [ torch.cos(phi) * abs_sin_psi, torch.sin(phi) * abs_sin_psi,  cos_psi],
        dim=-1
    )

    theta2 = theta.unsqueeze(-1) * 0.5
    sin_theta2 = torch.sin(theta2)
    cos_theta2 = torch.cos(theta2)

    quat = torch.cat(
        [cos_theta2, sin_theta2 * v], dim=-1
    )

    return quat

def axis_angle_to_pos(axis_angle, trans):
    rot = axis_angle_to_rot(axis_angle)
    pos = []
    for i in range(rot.shape[0]):
        p = apply_xform(rot[i], trans[i], STD_RIGID_COORD.to(trans.device))
        pos.append(p)
    pos = torch.stack(pos)
    # pos = apply_xform(rot, trans, STD_RIGID_COORD)
    return pos


def affine_to_pos(affine):
    quat = affine[..., :4]
    trans = affine[..., 4:]
    rot = quat_to_rot(quat)
    pos = []
    for i in range(rot.shape[0]):
        p = apply_xform(rot[i], trans[i], STD_RIGID_COORD.to(trans.device))
        pos.append(p)
    pos = torch.stack(pos)
    # pos = apply_xform(rot, trans, STD_RIGID_COORD)
    return pos


def axis_angle_to_rot(axis_angle):
    quat = angles_to_quat(axis_angle)
    rot = quat_to_rot(quat)
    return rot


def pos_to_axis_angle(pos):
    frame = rigid_from_3_points(pos[:, 0], pos[:, 1], pos[:, 2])
    quat = rot_to_quat(frame['rot'])
    angles, axis = quat_to_axis_angles(quat)

    affine = torch.cat([angles, frame['trans']],dim=-1)
    return affine, axis

def affine6_to_affine7(affine6):
    angles = affine6[..., :3]
    trans = affine6[..., 3:]
    quat = angles_to_quat(angles)
    affine7 = torch.cat([quat, trans],dim=-1)
    return affine7

def affine7_to_affine6(affine7):
    quat = affine7[..., :4]
    trans = affine7[..., 4:]
    angles, axis = quat_to_axis_angles(quat)
    affine6 = torch.cat([angles, trans],dim=-1)
    return affine6


def pos_to_affine7(pos, nc_xaxis=False):
    if nc_xaxis:
        frame = rigid_from_3_points(
            pos[..., 0, :], pos[..., 1, :], pos[..., 2, :])
    else:
        frame = rigid_from_3_points(
            pos[..., 2, :], pos[..., 1, :], pos[..., 0, :])
    quat = rot_to_quat(frame['rot'])
    affine = torch.cat([quat, frame['trans']],dim=-1)
    return affine


def slert(q0, q1, t):
    t1 = t[..., None, None]
    cos_theta = torch.sum(q0 * q1, -1, keepdim=True).clamp(0.0, 1.0)
    theta = torch.acos(cos_theta)
    qt = (q0 * torch.sin((1-t1)*theta) + q1 * torch.sin(t1*theta)) / (torch.sin(theta) + 1e-8)
    return qt


def angle_interp(a0, a1, t):
    return (1-t) * a0 + t * a1


def rand_quat(data_size):
    theta = torch.rand(data_size) * 2.0 * torch.pi
    phi = torch.rand(data_size) * 2.0 * torch.pi
    psi = torch.rand(data_size) * torch.pi

    angles = torch.stack([theta, phi, psi], dim=-1)
    quat = angles_to_quat(angles)
    return quat


def perturb_rigid_from_coord(coord, noising_scale):
    frames = rigid_from_3_points(coord[:, 0], coord[:, 1], coord[:, 2])
    quat = rot_to_quat(frames['rot'])
    axis_angle = quat_to_axis_angles(quat)

    affine_tensor = torch.cat([axis_angle, frames['trans']], dim=-1)
    noise = torch.randn_like(affine_tensor) * noising_scale
    perturbed_affine_tensor = affine_tensor + noise

    perturbed_axis_angle = perturbed_affine_tensor[:, :3]
    perturbed_ca = perturbed_affine_tensor[:, 3:]
    perturbed_coord = axis_angle_to_pos(perturbed_axis_angle, perturbed_ca)
    perturbed_rot = axis_angle_to_rot(perturbed_axis_angle)
    
    perturbed_data = {
        'noise': noise,
        'perturbed_ca': perturbed_ca,
        'perturbed_rot': perturbed_rot,
        'perturbed_coord': perturbed_coord
    }

    return perturbed_data


def noising_coord(x0, noise_scale, condition=None, sigma=0.01):
    noise = torch.randn_like(x0)
    if condition is not None:
        noise = torch.where(condition == 1, noise * sigma, noise * noise_scale)
        xt = x0 + noise
        return xt
    xt = x0 + noise_scale * noise
    return xt


def noising_quat(q0):
    batch_size = q0.shape[0]
    qT = rand_quat(q0.shape[:-1]).to(q0.device)
    alpha_t = torch.rand((batch_size, ), device=q0.device)
    qt = slert(q0, qT, alpha_t)
    return qt


def affine_to_frame12(affine):
    batch_size, res_num, _ = affine.shape
    quat = affine[..., :4]  # B, L, 4
    trans = affine[..., 4:] # B, L, 3
    rot = quat_to_rot(quat).reshape(batch_size, res_num, 9) # B, L, 3, 3
    frame12 = torch.cat([rot, trans], axis=-1)

    return frame12

