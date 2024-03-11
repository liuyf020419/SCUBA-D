# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quaternion geometry modules.

This introduces a representation of coordinate frames that is based around a
‘QuatAffine’ object. This object describes an array of coordinate frames.
It consists of vectors corresponding to the
origin of the frames as well as orientations which are stored in two
ways, as unit quaternions as well as a rotation matrices.
The rotation matrices are derived from the unit quaternions and the two are kept
in sync.
For an explanation of the relation between unit quaternions and rotations see
https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

This representation is used in the model for the backbone frames.

One important thing to note here, is that while we update both representations
the jit compiler is going to ensure that only the parts that are
actually used are executed.
"""


import functools
from typing import Tuple

import torch
from torch.nn import functional as F
from . import utils

import numpy as np

# pylint: disable=bad-whitespace
# QUAT_TO_ROT stores 9 parameter in rotation matrix within 16 position in quaternion outer product 
# for equation see https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/

# [[q0^2, q0q1, q0q2, q0q3],
#  [q1q0, q1^2, q1q2, q1q3],
#  [q2q0, q2q1, q2^2, q2q3],
#  [q3q0, q3q1, q3q2, q3^2]]

# see Supplemetary Algorithm 23
QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]  # kk

QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # kr

QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]  # ik

QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]  # jk

QUAT_TO_ROT = torch.FloatTensor(QUAT_TO_ROT)

# halmilton product see wiki
# QUAT_TO_ROT stores 4 parameter for halmilton product within 16 position in quaternion outer product 
# (a1a2 - b1b2 - c1c2 - d1d2)r + 
# (a1b2 + b1a2 + c1d2 - d1c2)i + 
# (a1c2 - d2b1 + c1a2 + d1b2)j +
# (a1d2 + b1c2 - c1b2 + d1a2)k
QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0,-1]]

QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                          [ 1, 0, 0, 0],
                          [ 0, 0, 0, 1],
                          [ 0, 0,-1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                          [ 0, 0, 0,-1],
                          [ 1, 0, 0, 0],
                          [ 0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                          [ 0, 0, 1, 0],
                          [ 0,-1, 0, 0],
                          [ 1, 0, 0, 0]]
QUAT_MULTIPLY = torch.FloatTensor(QUAT_MULTIPLY)

QUAT_MULTIPLY_BY_VEC = QUAT_MULTIPLY[:, 1:, :]
# pylint: enable=bad-whitespace


def rot_to_quat(rot, unstack_inputs=False):
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
    rot = [utils.moveaxis(x, -1, 0) for x in utils.moveaxis(rot, -2, 0)]

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
  # _, qs = torch.linalg.eigh(k)
  kk = np.array(k.detach().cpu().numpy(), dtype=np.float32)
  # import pdb; pdb.set_trace()
  _, qss = np.linalg.eigh(kk)
  # try:
  #   _, qss = np.linalg.eigh(kk)
  # except:
  #   import pdb; pdb.set_trace()
  qs = torch.from_numpy(qss).to(k.device)
  return qs[..., -1]


def rot_list_to_tensor(rot_list):
  """Convert list of lists to rotation tensor."""
  return torch.stack(
      [torch.stack(rot_list[0], dim=-1),
       torch.stack(rot_list[1], dim=-1),
       torch.stack(rot_list[2], dim=-1)],
      dim=-2)


def vec_list_to_tensor(vec_list):
  """Convert list to vector tensor."""
  return torch.stack(vec_list, dim=-1)


def quat_to_rot(normalized_quat):
  """Convert a normalized quaternion to a rotation matrix."""
  rot_tensor = torch.sum(
      # np.reshape(QUAT_TO_ROT.to(normalized_quat.device), (4, 4, 9)) *
      QUAT_TO_ROT.to(normalized_quat.device).view(4, 4, 9) *
      normalized_quat[..., :, None, None] *
      normalized_quat[..., None, :, None],
      dim=(-3, -2))
  rot = utils.moveaxis(rot_tensor, -1, 0)  # Unstack.
  return [[rot[0], rot[1], rot[2]],
          [rot[3], rot[4], rot[5]],
          [rot[6], rot[7], rot[8]]]


def quat_multiply_by_vec(quat, vec):
  """Multiply a quaternion by a pure-vector quaternion."""
  return torch.sum(
      QUAT_MULTIPLY_BY_VEC.to(quat.device) *
      quat[..., :, None, None] *
      vec[..., None, :, None],
      dim=(-3, -2))


def quat_multiply(quat1, quat2):
  """Multiply a quaternion by another quaternion."""
  return torch.sum(
      QUAT_MULTIPLY *
      quat1[..., :, None, None] *
      quat2[..., None, :, None],
      dim=(-3, -2))


def apply_rot_to_vec(rot, vec, unstack=False):
  """Multiply rotation matrix by a vector."""
  if unstack:
    x, y, z = [vec[:, i] for i in range(3)]
  else:
    x, y, z = vec
  return [rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
          rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
          rot[2][0] * x + rot[2][1] * y + rot[2][2] * z]


def apply_inverse_rot_to_vec(rot, vec):
  """Multiply the inverse of a rotation matrix by a vector."""
  # Inverse rotation is just transpose
  return [rot[0][0] * vec[0] + rot[1][0] * vec[1] + rot[2][0] * vec[2],
          rot[0][1] * vec[0] + rot[1][1] * vec[1] + rot[2][1] * vec[2],
          rot[0][2] * vec[0] + rot[1][2] * vec[1] + rot[2][2] * vec[2]]


class QuatAffine(object):
  """Affine transformation represented by quaternion and vector."""

  def __init__(self, quaternion, translation, rotation=None, normalize=True,
               unstack_inputs=False):
    """Initialize from quaternion and translation.

    Args:
      quaternion: Rotation represented by a quaternion, to be applied
        before translation.  Must be a unit quaternion unless normalize==True.
      translation: Translation represented as a vector.
      rotation: Same rotation as the quaternion, represented as a (..., 3, 3)
        tensor.  If None, rotation will be calculated from the quaternion.
      normalize: If True, l2 normalize the quaternion on input.
      unstack_inputs: If True, translation is a vector with last component 3
    """
    if quaternion is not None:
      assert quaternion.shape[-1] == 4

    if unstack_inputs:
      if rotation is not None:
        rotation = [utils.moveaxis(x, -1, 0)   # Unstack.
                    for x in utils.moveaxis(rotation, -2, 0)]  # Unstack.
      translation = utils.moveaxis(translation, -1, 0)  # Unstack.

    if normalize and quaternion is not None:
      quaternion = quaternion / quaternion.square().sum(dim=-1, keepdims=True).sqrt() # torch.linalg.norm(quaternion, dim=-1,
                                                # keepdims=True)
    # because the updated quaternion is normalized
    # it could be transformed to a rotation matrix
    if rotation is None:
      rotation = quat_to_rot(quaternion)

    self.quaternion = quaternion
    self.rotation = [list(row) for row in rotation]
    self.translation = list(translation)

    assert all(len(row) == 3 for row in self.rotation)
    assert len(self.translation) == 3

  def to_tensor(self):
    return torch.cat(
        [self.quaternion] +
        [x.unsqueeze(-1) for x in self.translation],
        dim=-1)

  def apply_tensor_fn(self, tensor_fn):
    """Return a new QuatAffine with tensor_fn applied (e.g. stop_gradient)."""
    return QuatAffine(
        tensor_fn(self.quaternion),
        [tensor_fn(x) for x in self.translation],
        rotation=[[tensor_fn(x) for x in row] for row in self.rotation],
        normalize=False)

  def apply_rotation_tensor_fn(self, tensor_fn):
    """Return a new QuatAffine with tensor_fn applied to the rotation part."""
    return QuatAffine(
        tensor_fn(self.quaternion),
        [x for x in self.translation],
        rotation=[[tensor_fn(x) for x in row] for row in self.rotation],
        normalize=False)

  def scale_translation(self, position_scale):
    """Return a new quat affine with a different scale for translation."""

    return QuatAffine(
        self.quaternion,
        [x * position_scale for x in self.translation],
        rotation=[[x for x in row] for row in self.rotation],
        normalize=False)

  @classmethod
  def from_tensor(cls, tensor, normalize=False):
    quaternion = tensor[..., :4]
    tx = tensor[..., 4]
    ty = tensor[..., 5]
    tz = tensor[..., 6]
    return cls(quaternion,
               [tx, ty, tz],
               normalize=normalize)

  def pre_compose(self, update, fix_region=None):
    """Return a new QuatAffine which applies the transformation update first.
    ??? why not QuatAffnie product and translation update
    Args:
      update: Length-6 vector. 3-vector of x, y, and z such that the quaternion
        update is (1, x, y, z) and zero for the 3-vector is the identity
        quaternion. 3-vector for translation concatenated.

    Returns:
      New QuatAffine object.
    """
    # b, c, d in Supplementary 23
    vector_quaternion_update = update[..., :3] 
    # import pdb; pdb.set_trace()
    # coordinates
    if fix_region is not None:
      # import pdb; pdb.set_trace()
      coord_update = update[..., 3:] 
      zeros_cood = torch.zeros_like(coord_update)
      coord_update = torch.where(fix_region[:, :, None] == 1, zeros_cood, coord_update)
      # import pdb; pdb.set_trace()
      x = coord_update[..., 0] 
      y = coord_update[..., 1] 
      z = coord_update[..., 2] 
    else:
      x = update[..., 3] 
      y = update[..., 4] 
      z = update[..., 5] 
    trans_update = [x, y, z]
    # only use imaginary part to generate new quaternion
    new_quaternion = (self.quaternion +
                      quat_multiply_by_vec(self.quaternion,
                                           vector_quaternion_update))
    # import pdb; pdb.set_trace()
    if fix_region is not None:
      new_quaternion = torch.where(fix_region[:, :, None] == 1, self.quaternion, new_quaternion)
    # rotate translation
    trans_update = apply_rot_to_vec(self.rotation, trans_update)

    new_translation = [
        self.translation[0] + trans_update[0],
        self.translation[1] + trans_update[1],
        self.translation[2] + trans_update[2]]

    return QuatAffine(new_quaternion, new_translation)

  def apply_to_point(self, point, extra_dims=0):
    """Apply affine to a point (3 dimension).
    first rotation; secondly translation

    Args:
      point: List of 3 tensors (xyz) to apply affine.
      extra_dims:  Number of dimensions at the end of the transformed_point
        shape that are not present in the rotation and translation.  The most
        common use is rotation N points at once with extra_dims=1 for use in a
        network.

    Returns:
      Transformed point after applying affine.
    """
    rotation = self.rotation
    translation = self.translation
    for _ in range(extra_dims):
      # rotation is in quaternion form
      # self.rotation.shape == (r, 3, 3)
      rotation = [[r.unsqueeze(-1) for r in rr] for rr in rotation]
      translation = [t.unsqueeze(-1) for t in translation]
      # rotation.shape == (r, 3, 3, 1)
      # translation.shape == (r, 3, 1)

    # first rotation; secondly translation
    # import pdb; pdb.set_trace()
    rot_point = apply_rot_to_vec(rotation, point)
    # [x, y, z]
    return [
        rot_point[0] + translation[0],
        rot_point[1] + translation[1],
        rot_point[2] + translation[2]]

  def invert_point(self, transformed_point, extra_dims=0):
    """Apply inverse of transformation to a point (not exactly 3 dimension).
    first translation; secondly rotation
    same function like invert_rigids

    Args:
      transformed_point: List of 3 tensors to apply affine
      extra_dims:  Number of dimensions at the end of the transformed_point
        shape that are not present in the rotation and translation.  The most
        common use is rotation N points at once with extra_dims=1 for use in a
        network.

    Returns:
      Transformed point after applying affine.
    """
    rotation = self.rotation
    translation = self.translation
    for _ in range(extra_dims):
      rotation = [[r.unsqueeze(-1) for r in rr] for rr in rotation]
      translation = [t.unsqueeze(-1) for t in translation]
    # first translation; secondly rotation
    rot_point = [
        transformed_point[0] - translation[0],
        transformed_point[1] - translation[1],
        transformed_point[2] - translation[2]]

    return apply_inverse_rot_to_vec(rotation, rot_point)

  def __repr__(self):
    return 'QuatAffine(%r, %r)' % (self.quaternion, self.translation)


def _multiply(a, b):
  return torch.stack([
      torch.stack([a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0],
                 a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1],
                 a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2]]),

      torch.stack([a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0],
                 a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1],
                 a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2]]),

      torch.stack([a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0],
                 a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1],
                 a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2]])])


def make_canonical_transform(
    n_xyz: torch.FloatTensor,
    ca_xyz: torch.FloatTensor,
    c_xyz: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
  """Returns translation and rotation matrices to canonicalize residue atoms.

  Note that this method does not take care of symmetries. If you provide the
  atom positions in the non-standard way, the N atom will end up not at
  [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
  need to take care of such cases in your code.

  Args:
    n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
    ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
    c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.

  Returns:
    A tuple (translation, rotation) where:
      translation is an array of shape [batch, 3] defining the translation.
      rotation is an array of shape [batch, 3, 3] defining the rotation.
    After applying the translation and rotation to all atoms in a residue:
      * All atoms will be shifted so that CA is at the origin,
      * All atoms will be rotated so that C is at the x-axis,
      * All atoms will be shifted so that N is in the xy plane.
  """
  assert len(n_xyz.shape) == 2, n_xyz.shape
  assert n_xyz.shape[-1] == 3, n_xyz.shape
  assert n_xyz.shape == ca_xyz.shape == c_xyz.shape, (
      n_xyz.shape, ca_xyz.shape, c_xyz.shape)
  # Place CA at the origin.
  translation = -ca_xyz
  n_xyz = n_xyz + translation
  c_xyz = c_xyz + translation

  # Place C on the x-axis.
  c_x, c_y, c_z = [c_xyz[:, i] for i in range(3)]
  # Rotate by angle c1 in the x-y plane (around the z-axis).
  sin_c1 = -c_y / torch.sqrt(1e-20 + c_x**2 + c_y**2)
  cos_c1 = c_x / torch.sqrt(1e-20 + c_x**2 + c_y**2)
  zeros = torch.zeros_like(sin_c1)
  ones = torch.ones_like(sin_c1)
  # pylint: disable=bad-whitespace
  c1_rot_matrix = torch.stack([torch.stack([cos_c1, -sin_c1, zeros]),
                             torch.stack([sin_c1,  cos_c1, zeros]),
                             torch.stack([zeros,    zeros,  ones])])

  # Rotate by angle c2 in the x-z plane (around the y-axis).
  sin_c2 = c_z / torch.sqrt(1e-20 + c_x**2 + c_y**2 + c_z**2)
  cos_c2 = torch.sqrt(c_x**2 + c_y**2) / torch.sqrt(
      1e-20 + c_x**2 + c_y**2 + c_z**2)
  c2_rot_matrix = torch.stack([torch.stack([cos_c2,  zeros, sin_c2]),
                             torch.stack([zeros,    ones,  zeros]),
                             torch.stack([-sin_c2, zeros, cos_c2])])

  c_rot_matrix = _multiply(c2_rot_matrix, c1_rot_matrix)
  n_xyz = torch.stack(apply_rot_to_vec(c_rot_matrix, n_xyz, unstack=True)).transpose(0, 1)

  # Place N in the x-y plane.
  _, n_y, n_z = [n_xyz[:, i] for i in range(3)]
  # Rotate by angle alpha in the y-z plane (around the x-axis).
  sin_n = -n_z / torch.sqrt(1e-20 + n_y**2 + n_z**2)
  cos_n = n_y / torch.sqrt(1e-20 + n_y**2 + n_z**2)
  n_rot_matrix = torch.stack([torch.stack([ones,  zeros,  zeros]),
                            torch.stack([zeros, cos_n, -sin_n]),
                            torch.stack([zeros, sin_n,  cos_n])])
  # pylint: enable=bad-whitespace

  return (translation, _multiply(n_rot_matrix, c_rot_matrix).permute(2, 0, 1))
          # torch.permute(_multiply(n_rot_matrix, c_rot_matrix), [2, 0, 1]))


def make_transform_from_reference(
    n_xyz: torch.FloatTensor,
    ca_xyz: torch.FloatTensor,
    c_xyz: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
  """Returns rotation and translation matrices to convert from reference.

  Note that this method does not take care of symmetries. If you provide the
  atom positions in the non-standard way, the N atom will end up not at
  [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
  need to take care of such cases in your code.

  Args:
    n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
    ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
    c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.

  Returns:
    A tuple (rotation, translation) where:
      rotation is an array of shape [batch, 3, 3] defining the rotation.
      translation is an array of shape [batch, 3] defining the translation.
    After applying the translation and rotation to the reference backbone,
    the coordinates will approximately equal to the input coordinates.

    The order of translation and rotation differs from make_canonical_transform
    because the rotation from this function should be applied before the
    translation, unlike make_canonical_transform.
  """
  translation, rotation = make_canonical_transform(n_xyz, ca_xyz, c_xyz)
  return rotation.permute(0, 2, 1), -translation
