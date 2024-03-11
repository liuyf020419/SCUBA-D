import numpy as np
import torch
import torch.nn.functional as F

from protein_constant_utils import *


def dist(x1, x2, axis=2):
    d = x2-x1
    # print("DD",d.shape)
    d2 = np.sum(d*d, axis=axis)
    return np.sqrt(d2)


def dist_ch(x1, x2, axis=2):
    d = x2-x1
    d2 = torch.sum(d*d, dim=axis) # √(x2 + y2 + z2)
    return torch.sqrt(d2)


def angle(x1, x2, x3, degrees=True, axis=2):
    """
    calc_angle of point(x1), point(x2), point(x3)
    """
    ba = x1 - x2
    ba /= np.linalg.norm(ba, axis=axis, keepdims=True)
    bc = x3 - x2
    bc /= np.linalg.norm(bc, axis=axis, keepdims=True)
    cosine_angle = np.sum(ba*bc, axis=axis)
    if degrees:
        return np.degrees(np.arccos(cosine_angle))
    else:
        return np.arccos(cosine_angle)


def angle_ch(x1, x2, x3, degrees=True, axis=2):
    ba = x1 - x2
    ba /= torch.norm(ba, dim=axis, keepdim=True)
    bc = x3 - x2
    bc /= torch.norm(bc, dim=axis, keepdim=True)
    cosine_angle = torch.sum(ba*bc, dim=axis)
    if degrees:
        return np.float32(180.0 / np.pi) * torch.acos(cosine_angle)
    else:
        return torch.acos(cosine_angle)


def angle42vecs(vecs1, vecs2, degrees=True):
    vecs1 /= torch.norm(vecs1, dim=1, keepdim=True)
    vecs2 /= torch.norm(vecs2, dim=1, keepdim=True)
    cosine_angle = torch.sum(vecs1*vecs2, dim=1)
    if degrees:
        return np.float32(180.0 / np.pi) * torch.acos(cosine_angle - 1e-6)
    else:
        return torch.acos(cosine_angle - 1e-6)


def orientation4vecs(vecs1, vecs2, quaternion=True):
    """
    Assume the third vecs are orthognal to the plane consist of the first and second vecs
    Note the fist dim is not for batch_size, but for number of residue
    vecs.shape == (n_res, 3, 3) (vec_nca, vec_cca, vec_x)
    """
    assert ((len(vecs1) == 3) and (len(vecs2) == 3))
    assert (all((vecs1[:, 2, :] * vecs1[:, 1, :]).sum(-1)) and all((vecs1[:, 2, :] * vecs1[:, 0, :]).sum(-1)) and
            all((vecs2[:, 2, :] * vecs2[:, 1, :]).sum(-1)) and all((vecs2[:, 2, :] * vecs2[:, 0, :]).sum(-1)))

    orth_axis_vecs1_new = np.cross(vecs1[:, 2, :], vecs1[:, 0, :])
    orth_axis_vecs2_new = np.cross(vecs2[:, 2, :], vecs2[:, 0, :])

    Ori1_norm = F.normalize(torch.stack([vecs1[:, 0, :], orth_axis_vecs1_new, vecs1[:, 2, :]], 1), 1)
    Ori2_norm = F.normalize(torch.stack([vecs2[:, 0, :], orth_axis_vecs2_new, vecs2[:, 2, :]], 1), 1)
    # (n_res, 3, 3) * (n_res, 3, 3)
    R = torch.matmul(Ori1_norm.transpose(-1, -2), Ori2_norm)

    if quaternion:
        quats = rmts2quats(R)
        return quats
    else:
        return R


def torsion(x1, x2, x3, x4, degrees = True, axis=2):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1) + 1e-10
    b1 = x3 - x2 + 1e-10
    b2 = x4 - x3 + 1e-10
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= (np.linalg.norm(b1, axis=axis, keepdims=True) + 1e-10)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.sum(b0*b1, axis=axis, keepdims=True) * b1
    w = b2 - np.sum(b2*b1, axis=axis, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x

    x = np.sum(v*w, axis=axis)
    b1xv = np.cross(b1, v, axisa=axis, axisb=axis)
    y = np.sum(b1xv*w, axis=axis)
    if degrees:
        return np.float32(180.0 / np.pi) * np.arctan2(y, x)
    else:
        return np.arctan2(y, x)


def torsion_ch(x1, x2, x3, x4, degrees=True, axis=2):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= torch.norm(b1, dim=axis, keepdim=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0*b1, dim=axis, keepdim=True) * b1
    w = b2 - torch.sum(b2*b1, dim=axis, keepdim=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v*w, dim=axis)
    b1xv = torch.cross(b1, v)
    y = torch.sum(b1xv*w, dim=axis)
    if degrees:
        return np.float32(180.0 / np.pi) * torch.atan2(y, x)
    else:
        return torch.atan2(y, x)



def rmts2quats(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [...,3,3]
        Q [...,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,:,:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q


def rbf_ch(distance, num_rbf=16):
    """
    distance: input
    num_rbf: central bin
    """
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape(-1, 1)
    D_sigma = (D_max - D_min) / D_count
    RBF = torch.exp(-((distance - D_mu) / D_sigma) ** 2).transpose(1,0)
    return RBF


def rbf(distance, num_rbf=16):
    """
    distance: input
    num_rbf: central bin
    """
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape(-1, 1)
    D_sigma = (D_max - D_min) / D_count
    RBF = np.exp(-((distance - D_mu) / D_sigma) ** 2).transpose(1,0)
    return RBF


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rand_n_rotation_matrix_ch(deflection=1.0, randnums=None, n=1):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = torch.ones(3, n).uniform_(0, 1)

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = torch.sqrt(z)

    Vx, Vy, Vz = V = torch.stack(
        [torch.sin(phi) * r,
        torch.cos(phi) * r,
        torch.sqrt(2.0 - z)]
    )

    st = torch.sin(theta)
    ct = torch.cos(theta)

    base_vec = torch.ones(n)
    R = torch.stack(
        (torch.stack([ct, st, base_vec * 0]),
         torch.stack([-st, ct, base_vec * 0]),
         torch.stack([base_vec * 0, base_vec * 0, base_vec * 1])))
    # R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1))), rotation around z axis
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = torch.bmm((bouter(V.transpose(1, 0), V) - torch.stack([torch.eye(3) for _ in range(n)], -1)).permute(2, 0, 1),
                  R.permute(2, 0, 1))
    return M


def bouter(v1, v2):
    """
    v1.shape(N, M), v2.shape(N, M)
    return (N, N, M)
    """
    return (v1[None, :, :] * v2[:, :, None]).permute([0, 2, 1])


def isRotationMatrix(R):
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], np.float), rtol=1.e-4, atol=1.e-4)
    should_be_one = np.allclose(np.linalg.det(R), 1, rtol=1.e-4, atol=1.e-4)
    return should_be_identity and should_be_one


def trans_mtx_ch(x):
    """
    [x,y,z,1].dot(trans_mtx)
    """
    base_vec = torch.ones(x.shape[0])

    return torch.stack([torch.stack([base_vec*1, base_vec*0, base_vec*0, base_vec*0]),
                        torch.stack([base_vec*0, base_vec*1, base_vec*0, base_vec*0]),
                        torch.stack([base_vec*0, base_vec*0, base_vec*1, base_vec*0]),
                        torch.stack([x[:, 0], x[:, 1], x[:, 2], base_vec*1])]).permute(2, 0, 1)


if __name__ == "__main__":
    rd_rmts = rand_n_rotation_matrix_ch(deflection=1.0, randnums=None, n=10)
    print(rd_rmts.shape)
    rd_rmt_ = (rd_rmts[0, :, :]).numpy()
    print(isRotationMatrix((rd_rmts[0, :, :]).numpy()))
    rd_rmt = rand_rotation_matrix()
    print(isRotationMatrix(rd_rmt))