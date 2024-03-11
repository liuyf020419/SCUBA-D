import os
import sys
import collections

import numpy as np
import torch


BB_BUILD_INFO = {"BONDLENS": {'n-ca': 1.442,
                              'ca-c': 1.498,
                              'c-n' : 1.379,
                              'c-o' : 1.229,                    # From parm10.dat
                              'c-oh': 1.364},                   # From parm10.dat, for OXT
                 # For placing oxygens
                 "BONDANGS": {'ca-c-o':  2.0944,             # Approximated to be 2pi / 3; parm10.dat says 2.0350539
                              'ca-c-oh': 2.0944,
                              'n-ca-c': 1.941,
                              'ca-c-n': 2.028,
                              'c-n-ca': 2.124},            # Equal to 'ca-c-o', for OXT
                 "BONDTORSIONS": {'n-ca-c-n': -0.785398163}    # A simple approximation, not meant to be exact.
                 }

BB_TOPO = collections.namedtuple('BB_TOPO', ['bb', 'tors'])


def nerf(a, b, c, l, theta, chi, dim=0):
    """
    Natural extension reference frame method for placing the 4th atom given
    atoms 1-3 and the relevant angle inforamation. This code was originally
    written by Rohit Bhattacharya (rohit.bhattachar@gmail.com,
    https://github.com/rbhatta8/protein-design/blob/master/nerf.py) and I
    have extended it to work with PyTorch. His original documentation is
    below:
    Nerf method of finding 4th coord (d) in cartesian space
        Params:
            a, b, c : coords of 3 points
            l : bond length between c and d
            theta : bond angle between b, c, d (in degrees)
            chi : dihedral using a, b, c, d (in degrees)
        Returns:
            d: tuple of (x, y, z) in cartesian space
    """
    # calculate unit vectors AB and BC
    assert -np.pi <= theta <= np.pi, "theta must be in radians and in [-pi, pi]. theta = " + str(theta)

    W_hat = torch.nn.functional.normalize(b - a, dim=dim)
    x_hat = torch.nn.functional.normalize(c-b, dim=dim)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = torch.nn.functional.normalize(n_unit, dim=dim)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=dim+1)
    # import pdb; pdb.set_trace()
    # calculate coord pre rotation matrix
    d = torch.stack([torch.squeeze(-l * torch.cos(theta)),
                     torch.squeeze(l * torch.sin(theta) * torch.cos(chi)),
                     torch.squeeze(l * torch.sin(theta) * torch.sin(chi))])

    # calculate with rotation as our final output
    # TODO: is the squeezing necessary?
    d = d.unsqueeze(1).to(torch.float32)
    res = c + torch.mm(M, d).squeeze()
    return res.squeeze()


def res_build_bb(prev_res = None, torsion=None):
    """ Builds backbone for residue. """
    if prev_res is None:
        bb = res_init_bb(torsion)
    else:
        assert torsion is not None

        pts = [prev_res.bb[0], prev_res.bb[1], prev_res.bb[2]]
        for j in range(4):
            if j == 0:
                # Placing N
                t = torch.tensor([BB_BUILD_INFO["BONDANGS"]["ca-c-n"]])     # thetas["ca-c-n"]
                b = torch.tensor([BB_BUILD_INFO["BONDLENS"]["c-n"]])
                dihedral = prev_res.tors[1]  # psi of previous residue
            elif j == 1:
                # Placing Ca
                t = torch.tensor([BB_BUILD_INFO["BONDANGS"]["c-n-ca"] ])         # thetas["c-n-ca"]
                b = torch.tensor([BB_BUILD_INFO["BONDLENS"]["n-ca"]])
                dihedral = prev_res.tors[2]  # omega of previous residue
            elif j == 2:
                # Placing C
                t = torch.tensor([BB_BUILD_INFO["BONDANGS"]["n-ca-c"] ])              # thetas["n-ca-c"]
                b = torch.tensor([BB_BUILD_INFO["BONDLENS"]["ca-c"]])
                dihedral = torsion[0]       # phi of current residue
            else:
                # Placing O
                t = torch.tensor([BB_BUILD_INFO["BONDANGS"]["ca-c-o"]])
                b = torch.tensor([BB_BUILD_INFO["BONDLENS"]["c-o"]])
                dihedral = torsion[1] - np.pi      # opposite to psi of current residue

            next_pt = nerf(pts[-3], pts[-2], pts[-1], b, t, dihedral)
            pts.append(next_pt)
        bb = pts[3:]

    return BB_TOPO(bb=torch.stack(bb), tors=torsion)


def res_init_bb(torsion=None):
    """ Initialize the first 3 points of the protein's backbone. Placed in an arbitrary plane (z = .001). """
    n = torch.tensor([0, 0, 0.001])
    ca = n + torch.tensor([BB_BUILD_INFO["BONDLENS"]["n-ca"], 0, 0])

    cx = np.cos(np.pi - BB_BUILD_INFO["BONDANGS"]["n-ca-c"]) * BB_BUILD_INFO["BONDLENS"]["ca-c"]
    cy = np.sin(np.pi - BB_BUILD_INFO["BONDANGS"]["n-ca-c"]) * BB_BUILD_INFO["BONDLENS"]['ca-c']
    c = ca + torch.tensor([cx, cy, 0], dtype=torch.float32)
    o = nerf(n, ca, c, torch.tensor(BB_BUILD_INFO["BONDLENS"]["c-o"]),
                       torch.tensor(BB_BUILD_INFO["BONDANGS"]["ca-c-o"]),
                       torsion[1] - np.pi) # opposite to current residue's psi
    return [n, ca, c, o]


def build_first_two_residues(torsion=None):
    """ Constructs the first two residues of the protein. """
    first_res = res_build_bb(prev_res=None, torsion=torsion[0])
    second_res = res_build_bb(prev_res=first_res, torsion=torsion[1])

    return first_res, second_res


def build_cart_from_tors(torsions=None):
    """
    Construct all of the atoms for a residue. Special care must be taken
    for the first residue in the sequence in order to place its CB, if
    present.
    """
    coords = []
    # Build the first and second residues, a special case
    first, second = build_first_two_residues(torsions[:2])
    prev_res = second
    coords.append(first.bb)
    coords.append(second.bb)

    for i, tors in enumerate(torsions[2:]):
        prev_res = res_build_bb(prev_res=prev_res, torsion=tors)
        coords.append(prev_res.bb)

    coords = torch.stack(coords)

    return coords
