import numpy as np
import torch

def nerf(a, b, c, l, theta, chi):
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

    W_hat = torch.nn.functional.normalize(b - a, dim=0)
    x_hat = torch.nn.functional.normalize(c-b, dim=0)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = torch.nn.functional.normalize(n_unit, dim=0)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=1)
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


def torsion_v0(x1, x2=None, x3=None, x4=None, degrees = False, axis=2):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    if (x2 is None) or (x3 is None) or (x4 is None):
        x1, x2, x3, x4 = x1
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1, axis=axis, keepdims=True)

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


def add_atom_O(coord3):
    CO_bond = 1.229
    CACO_angle = torch.tensor([2.0944]).float()
    assert isinstance(coord3, np.ndarray)
    assert len(coord3.shape) == 3
    seqlen, _, _ = coord3.shape

    def calc_psi_tors(coord3):
        assert len(coord3.shape) == 3
        N_atoms = coord3[:, 0]
        CA_atoms = coord3[:, 1]
        C_atoms = coord3[:, 2]

        n1_atoms = N_atoms[:-1]
        ca_atoms = CA_atoms[:-1]
        c_atoms = C_atoms[:-1]
        n2_atoms = N_atoms[1:]

        psi_tors = torsion_v0(n1_atoms, ca_atoms, c_atoms, n2_atoms, axis=1)
        return np.concatenate([psi_tors, [0]])

    psi_tors = torch.from_numpy(calc_psi_tors(coord3)).float()
    coord3 = torch.from_numpy(coord3).float()
    atomO_coord = [nerf(atom3[-3], atom3[-2], atom3[-1], CO_bond, CACO_angle, psi_tors[resabsD]-np.pi) \
                                    for resabsD, atom3 in enumerate(coord3)]
    new_coord = torch.cat([coord3.reshape(seqlen, -1), torch.stack(atomO_coord)], 1).reshape(seqlen, 4, 3)

    return new_coord.numpy()



def rebiuld_from_atom_crd(crd_list, chain="A", filename='testloop.pdb', natom=4, natom_dict=None):
    from Bio.PDB.StructureBuilder import StructureBuilder
    from Bio.PDB import PDBIO
    from Bio.PDB.Atom import Atom
    if natom_dict is None:
        natom_dict = {3: {0:'N', 1:'CA', 2: 'C'},
                      4: {0:'N', 1:'CA', 2: 'C', 3:'O'}}
    natom_num = natom_dict[natom]
    sb = StructureBuilder()
    sb.init_structure("pdb")
    sb.init_seg(" ")
    sb.init_model(0)
    chain_id = chain
    sb.init_chain(chain_id)
    for num, line in enumerate(crd_list):
        name = natom_num[num % natom]

        line = np.around(np.array(line, dtype='float'), decimals=3)
        res_num = num // natom
        # print(num//4,line)
        atom = Atom(name=name, coord=line, element=name[0:1], bfactor=1, occupancy=1, fullname=name,
                    serial_number=num,
                    altloc=' ')
        sb.init_residue("GLY", " ", res_num, " ")  # Dummy residue
        sb.structure[0][chain_id].child_list[res_num].add(atom.copy())

    structure = sb.structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)


if __name__ == "__main__":
    import os
    import sys

    sys.path.append("/home/liuyf/proteins/protein_utils")

    from protein_map_gen import FastPoteinParser

    pdbparser = FastPoteinParser(poteinfile = "/home/liuyf/database/datanew/r2/1r26.cif.gz", chain="A",
                                 datatype="gz", pseudo_gly=True, mergedsspin_=False, dsspfile=None)

    phi = pdbparser.get_torsions("phi_torsion", False)
    psi = pdbparser.get_torsions("psi_torsion", False)
    omega = pdbparser.get_torsions("bbomega_torsion", False)
    torsions = torch.from_numpy(np.concatenate([phi, psi, omega]).T.astype(np.float32))

    crd = pdbparser.chain_main_crd_array.reshape(-1, 5, 3)
    new_coord = add_atom_O(crd[:, :3])
    rebiuld_from_atom_crd(new_coord.reshape(-1, 3), natom=4, filename="test0724_3atomsAddO_321_torch1.pdb")
    import pdb; pdb.set_trace()
