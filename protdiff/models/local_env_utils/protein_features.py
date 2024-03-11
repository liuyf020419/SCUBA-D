import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

from .self_attention import gather_edges, gather_nodes, Normalize
from ..dense_block import TransformerPositionEncoding

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2,1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range 

    def forward(self, E_idx):
        # i-j
        N_batch = E_idx.size(0)
        N_nodes = E_idx.size(1)
        N_neighbors = E_idx.size(2)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).to(E_idx.device)
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).to(E_idx.device)

        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, features_type='full', augment_eps=0., dropout=0.1, max_len=500000):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            'coarse': (3, num_positional_embeddings + num_rbf + 7),
            'full': (6, num_positional_embeddings + num_rbf + 7),
            'dist': (6, num_positional_embeddings + num_rbf),
            'hbonds': (3, 2 * num_positional_embeddings),
        }

        # Positional encoding
        # self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.embeddings = TransformerPositionEncoding(max_len, num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization and embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF

    def rot_to_quat(self, rot, unstack_inputs=False):
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
            rot = [moveaxis(x, -1, 0) for x in moveaxis(rot, -2, 0)]

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


    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(1e-10 + torch.abs(1 + torch.stack([
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
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True)) + 1e-10) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Pair features

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        # import pdb; pdb.set_trace()
        # Bond angle calculation
        # cosA = -(u_1 * u_0).sum(-1)
        # cosA = torch.clamp(cosA, -1+eps, 1-eps)
        # A = torch.acos(cosA)
        # Angle between normals
        # cosD = (n_2 * n_1).sum(-1)
        # cosD = torch.clamp(cosD, -1+eps, 1-eps)
        # D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        # AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        # AD_features = F.pad(AD_features, (0,0,1,2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        # import pdb; pdb.set_trace()
        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        # import pdb; pdb.set_trace()
        Q = self.rot_to_quat(rot=R, unstack_inputs=True)
        # Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)

        return O_features


    def _orientations_frame(self, X, E_idx, eps=1e-6):
        # Pair features
        # Shifted slices of unit vectors
        # Pair features
        vec_0 = F.normalize(X[:, :, 0] - X[:, :, 1], -1, eps=eps)
        vec_1 = F.normalize(X[:, :, 2] - X[:, :, 1], -1, eps=eps)

        # Build relative orientations
        # import pdb; pdb.set_trace()
        X_ca = X[:, :, 1]
        O = torch.stack((vec_0, vec_1, torch.cross(vec_0, vec_1)), 2)
        # import pdb; pdb.set_trace()
        O = O.view(list(O.shape[:2]) + [9])
        # O = F.pad(O, (0,0,1,2), 'constant', 0)

        O_neighbors = gather_nodes(O, E_idx)
        X_ca_neighbors = gather_nodes(X_ca, E_idx)
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        # import pdb; pdb.set_trace()
        # Rotate into local reference frames
        dX = X_ca_neighbors - X_ca.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        # import pdb; pdb.set_trace()
        Q = self.rot_to_quat(rot=R, unstack_inputs=True)
        # Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)

        return O_features


    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def forward(self, X, L, mask, single_res_rel):
        """ Featurize coordinates as an attributed graph """
       
        assert self.features_type == 'full'
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)
        RBF = self._rbf(D_neighbors)

        # Pairwise features
        # import pdb; pdb.set_trace()
        # O_features = self._orientations_coarse(X_ca, E_idx)
        O_features = self._orientations_frame(X, E_idx)

        # Pairwise embeddings
        # E_positional = self.embeddings(E_idx)
        batch_size, res_num, knn = E_idx.shape
        E_single_res_rel = torch.gather(single_res_rel, -1, E_idx.reshape(batch_size, -1))
        E_positional = self.embeddings(E_single_res_rel.reshape(batch_size, -1), index_select=True).reshape(batch_size, res_num, knn, -1)

        # Full backbone angles
        V = self._dihedrals(X)
        E = torch.cat((E_positional, RBF, O_features), -1)
        # E = torch.cat((E_positional, RBF), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return V, E, E_idx




class ProteinMPNNFeaturesNew(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, features_type='mpnn', augment_eps=0., dropout=0.1, max_len=500000, node_angle_len=7):
        """ Extract protein features """
        super(ProteinMPNNFeaturesNew, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.node_angle_len = node_angle_len
        self.node_pad_num = self.node_angle_len//2

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            'mpnn': (6 * node_angle_len, num_rbf*16, 7)
        }

        # Positional encoding
        # self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.embeddings = TransformerPositionEncoding(max_len, node_features)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization and embedding
        node_in, edge_dist_in, edge_orient_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_dist_embedding = nn.Linear(edge_dist_in, edge_features, bias=True)
        self.edge_orient_embedding = nn.Linear(edge_orient_in, edge_features, bias=True)

        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)


    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx


    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF


    def _dihedrals(self, X, eps=1e-7):
        res_num = X.shape[1]
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        # phi, psi, omega = torch.unbind(D,-1)

        pad_D = F.pad(D, (0, 0, self.node_pad_num, self.node_pad_num), 'constant', 0)
        D_angles = torch.stack([ 
            pad_D.transpose(1, 0)[torch.arange(self.node_angle_len) + node_idx].transpose(1, 0).reshape(-1, self.node_angle_len * 3) 
            for node_idx in np.arange(res_num) ], 1)
        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D_angles), torch.sin(D_angles)), 2)
        return D_features

    
    def rot_to_quat(self, rot, unstack_inputs=False):
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
            rot = [moveaxis(x, -1, 0) for x in moveaxis(rot, -2, 0)]

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


    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(1e-10 + torch.abs(1 + torch.stack([
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
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True)) + 1e-10) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Pair features

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        # import pdb; pdb.set_trace()
        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        # import pdb; pdb.set_trace()
        Q = self.rot_to_quat(rot=R, unstack_inputs=True)
        # Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)

        return O_features

    def forward(self, X, L, mask, single_res_rel):
        """ Featurize coordinates as an attributed graph """
       
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        # O = X[:,:,3,:]

        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._dist(X_ca, mask)
        # RBF = self._rbf(D_neighbors)

        # Pairwise features
        # import pdb; pdb.set_trace()
        O_features = self._orientations_coarse(X_ca, E_idx)
        # O_features = self._orientations_frame(X, E_idx)
        
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        # RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        # RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        # RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        # RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        # RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        # RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        # RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        # RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        # RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        # Pairwise embeddings
        # E_positional = self.embeddings(E_idx)
        batch_size, res_num, knn = E_idx.shape
        E_single_res_rel = torch.gather(single_res_rel, -1, E_idx.reshape(batch_size, -1))
        E_positional = self.embeddings(E_single_res_rel.reshape(batch_size, -1), index_select=True).reshape(batch_size, res_num, knn, -1)

        # Full backbone angles
        V = self._dihedrals(X)
        # E = torch.cat((E_positional, RBF_all, O_features), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_dist_embedding(RBF_all) + self.edge_orient_embedding(O_features) + E_positional
        E = self.norm_edges(E)

        return V, E, E_idx

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B





class ProteinMPNNFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, features_type='mpnn', augment_eps=0., dropout=0.1, max_len=500000, node_angle_len=7):
        """ Extract protein features """
        super(ProteinMPNNFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            'coarse': (3, num_positional_embeddings + num_rbf + 7),
            'full': (6, num_positional_embeddings + num_rbf + 7),
            'dist': (6, num_positional_embeddings + num_rbf),
            'hbonds': (3, 2 * num_positional_embeddings),
            'mpnn': (6, num_positional_embeddings + num_rbf*16 + 7)
        }

        # Positional encoding
        # self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.embeddings = TransformerPositionEncoding(max_len, num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization and embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    # def _dist(self, X, mask, eps=1E-6):
    #     """ Pairwise euclidean distances """
    #     # Convolutional network on NCHW
    #     mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
    #     dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    #     D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

    #     # Identify k nearest neighbors (including self)
    #     D_max, _ = torch.max(D, -1, keepdim=True)
    #     D_adjust = D + (1. - mask_2D) * D_max
    #     D_neighbors, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False)
    #     mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

    #     return D_neighbors, E_idx, mask_neighbors


    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx


    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF


    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    
    def rot_to_quat(self, rot, unstack_inputs=False):
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
            rot = [moveaxis(x, -1, 0) for x in moveaxis(rot, -2, 0)]

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


    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(1e-10 + torch.abs(1 + torch.stack([
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
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True)) + 1e-10) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Pair features

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        # import pdb; pdb.set_trace()
        # Bond angle calculation
        # cosA = -(u_1 * u_0).sum(-1)
        # cosA = torch.clamp(cosA, -1+eps, 1-eps)
        # A = torch.acos(cosA)
        # Angle between normals
        # cosD = (n_2 * n_1).sum(-1)
        # cosD = torch.clamp(cosD, -1+eps, 1-eps)
        # D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        # AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        # AD_features = F.pad(AD_features, (0,0,1,2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        # import pdb; pdb.set_trace()
        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        # import pdb; pdb.set_trace()
        Q = self.rot_to_quat(rot=R, unstack_inputs=True)
        # Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)

        return O_features

    def forward(self, X, L, mask, single_res_rel):
        """ Featurize coordinates as an attributed graph """
       
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        # O = X[:,:,3,:]

        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._dist(X_ca, mask)
        # RBF = self._rbf(D_neighbors)

        # Pairwise features
        # import pdb; pdb.set_trace()
        O_features = self._orientations_coarse(X_ca, E_idx)
        # O_features = self._orientations_frame(X, E_idx)
        
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        # RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        # RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        # RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        # RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        # RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        # RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        # RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        # RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        # RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        # Pairwise embeddings
        # E_positional = self.embeddings(E_idx)
        batch_size, res_num, knn = E_idx.shape
        E_single_res_rel = torch.gather(single_res_rel, -1, E_idx.reshape(batch_size, -1))
        E_positional = self.embeddings(E_single_res_rel.reshape(batch_size, -1), index_select=True).reshape(batch_size, res_num, knn, -1)

        # Full backbone angles
        V = self._dihedrals(X)
        # E = torch.cat((E_positional, RBF, O_features), -1)
        # import pdb; pdb.set_trace()
        # E = torch.cat((E_positional, RBF_all), -1)
        # import pdb; pdb.set_trace()
        E = torch.cat((E_positional, RBF_all, O_features), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return V, E, E_idx

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B


def moveaxis(data, source, destination):
  n_dims = len(data.shape)
  dims = [i for i in range(n_dims)]
  if source < 0:
    source += n_dims
  if destination < 0:
    destination += n_dims

  if source < destination:
    dims.pop(source)
    dims.insert(destination, source)
  else:
    dims.pop(source)
    dims.insert(destination, source)

  return data.permute(*dims)
