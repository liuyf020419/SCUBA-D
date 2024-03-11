
import numpy as np
import torch


def compute_rmsd(true_atom_pos, pred_atom_pos, atom_mask=None, eps=1e-6):
    if len(true_atom_pos.shape) != 2:
        true_atom_pos = true_atom_pos.reshape(-1, 3)
    if len(pred_atom_pos.shape) != 2:
        pred_atom_pos = pred_atom_pos.reshape(-1, 3)
        
    if isinstance(true_atom_pos, np.ndarray):
        sq_diff = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1, keepdims=False)
        if atom_mask is not None:
            sq_diff = sq_diff[atom_mask]
        msd = np.mean(sq_diff)
        msd = np.nan_to_num(msd, nan=1e8)
        return np.sqrt(msd + eps)
    
    else:
        sq_diff = torch.square(true_atom_pos - pred_atom_pos).sum(dim=-1, keepdim=False)
        if atom_mask is not None:
            sq_diff = sq_diff[atom_mask]
        msd = torch.mean(sq_diff)
        msd = torch.nan_to_num(msd, nan=1e8)
        return torch.sqrt(msd + eps)
    
    
def compute_tmscore(true_atom_pos, pred_atom_pos, eps=1e-6, reduction='mean'):
    assert reduction in ['mean', 'ca']
    assert len(true_atom_pos.shape) == 3
    res_num = true_atom_pos.shape[0]
    d0 = 1.24 * (res_num - 15) ** (1.0 / 3.0) - 1.8
    d02 = d0 ** 2
        
    if isinstance(true_atom_pos, np.ndarray):
        sq_diff = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1, keepdims=False)
        d_i2 = np.nan_to_num(sq_diff, nan=1e8) + eps
    else:
        sq_diff = torch.square(true_atom_pos - pred_atom_pos).sum(dim=-1, keepdim=False)
        d_i2 = torch.nan_to_num(sq_diff, nan=1e8) + eps

    if reduction == 'mean':
        d_i2 = d_i2.mean(-1)
    else:
        d_i2 = d_i2[1]
        
    tm = (1 / (1 + (d_i2 / d02))).mean()
    return tm


def kabsch_rotation(P, Q, datatype):
    assert datatype in ['np', 'torch']
    assert ( (len(P.shape) == 2) and (len(Q.shape) == 2) )

    C = P.transpose(-1, -2) @ Q
    # Computation of the optimal rotation matrix
    if datatype == 'np':
        V, _, W = np.linalg.svd(C)
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    else:
        V, _, W = torch.linalg.svd(C)
        d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = V @ W
    return U


def rotrans_coords(coords, rotrans):
    rot, trans = rotrans
    if len(trans.shape) == 1:
        trans = trans[None, :]
    elif len(trans.shape) == 2:
        pass
    else:
        raise ValueError(f'shape {trans.shape} unknown')
    return coords @ rot + trans


class KabschCycleAlign(object):
    def align(self, mobile, target, cutoff: float=2.0, cycles: int=5, return_dict=False, verbose=True):
        assert type(mobile) == type(target)
        assert isinstance(mobile, np.ndarray) or isinstance(mobile, torch.Tensor)
        assert len(mobile.shape) == 3
        assert mobile.shape == target.shape
        res_num = mobile.shape[0]
        
        if isinstance(mobile, np.ndarray):
            datatype = 'np'
            res_mask = np.ones((res_num, )).astype(np.bool_)
        else:
            datatype = 'torch'
            res_mask = torch.ones((res_num, )).bool()
        
        overall_rmsd_list = []
        overall_tmscore_list = []
        overall_masked_rmsd_list = []
        overall_res_mask_list = []
        mobile_traj_list = []
        for c_ in range(cycles):
            aligned_res = sum(res_mask)
            c_rotransed_mobile = self.align_single_term(mobile, target, datatype, res_mask)
            res_mask, c_overall_rmsd = self.get_outlier_residue_mask(c_rotransed_mobile, target, cutoff, datatype)
            
            c_masked_rmsd = compute_rmsd(c_rotransed_mobile[res_mask], target[res_mask])
            c_overall_tmscore = compute_tmscore(c_rotransed_mobile, target)
            if datatype == 'torch':
                c_masked_rmsd = c_masked_rmsd.item()

            if verbose:
                print("term:%02d overall_rmsd: %02f; overall_tmscore: %02f; unmasked_overall_rmsd: %02f; aligned res_num: %2d" %
                        (c_, c_overall_rmsd, c_overall_tmscore, c_masked_rmsd, aligned_res))
                
            overall_rmsd_list.append(c_overall_rmsd)
            overall_tmscore_list.append(c_overall_tmscore)
            overall_masked_rmsd_list.append(c_masked_rmsd)
            overall_res_mask_list.append(res_mask)
            mobile_traj_list.append(c_rotransed_mobile)
            
        if datatype == 'torch':
            mobile_traj_list = torch.stack(mobile_traj_list).detach().cpu().numpy()
            overall_res_mask_list = torch.stack(overall_res_mask_list).detach().cpu().numpy()
        else:
            mobile_traj_list = np.stack(mobile_traj_list)
            overall_res_mask_list = np.stack(overall_res_mask_list)
        
        if return_dict:
            align_dict = {
                'rotransed_mobile_list': mobile_traj_list,
                'overall_rmsd_list': np.array(overall_rmsd_list),
                'overall_tmscore_list': np.array(overall_tmscore_list),
                'overall_unmasked_rmsd_list': np.array(overall_masked_rmsd_list),
                'overall_res_mask_list': overall_res_mask_list
            }
            return align_dict
        else:  
            return c_rotransed_mobile
        
    
    def align_single_term(self, mobile, target, datatype, res_mask=None):
        assert len(mobile.shape) == 3
        res_num = mobile.shape[0]

        mobile_ca = mobile.reshape(res_num, -1, 3)[:, 1]
        target_ca = target.reshape(res_num, -1, 3)[:, 1]
        
        if datatype == 'np':
            masked_mass_center_trans_mobile_ca = mobile_ca[res_mask].mean(0, keepdims=True)
            masked_mass_center_trans_target_ca = target_ca[res_mask].mean(0, keepdims=True)
        else:
            masked_mass_center_trans_mobile_ca = mobile_ca[res_mask].mean(0, keepdim=True)
            masked_mass_center_trans_target_ca = target_ca[res_mask].mean(0, keepdim=True)
        
        masked_mobile_to_target_rot = self.kabsch_rotation(
            (mobile[res_mask]-masked_mass_center_trans_mobile_ca).reshape(-1, 3),
            (target[res_mask]-masked_mass_center_trans_target_ca).reshape(-1, 3), 
            datatype)
        
        rotransed_mobile = self.rotrans_coords(
            mobile - masked_mass_center_trans_mobile_ca, 
            (masked_mobile_to_target_rot, masked_mass_center_trans_target_ca))
        
        return rotransed_mobile
    
    
    def get_outlier_residue_mask(self, coords_A, coords_B, cutoff:int, datatype: str, eps=1e-6):
        assert len(coords_A.shape) == 3
        coords_A_ca = coords_A[:, 1]
        coords_B_ca = coords_B[:, 1]
        
        if datatype == 'np':
            sq_diff = np.square(coords_A_ca - coords_B_ca).sum(axis=-1, keepdims=False)
            ca_rmsd = np.sqrt(np.nan_to_num(sq_diff, nan=1e8) + eps)
            overall_rmsd = compute_rmsd(coords_A.reshape(-1, 3), coords_B.reshape(-1, 3))
        else:
            sq_diff = torch.square(coords_A_ca - coords_B_ca).sum(dim=-1, keepdim=False)
            ca_rmsd = torch.sqrt(torch.nan_to_num(sq_diff, nan=1e8) + eps)
            overall_rmsd = compute_rmsd(coords_A.reshape(-1, 3), coords_B.reshape(-1, 3)).item()
            
        return (ca_rmsd <= cutoff, overall_rmsd)


    def kabsch_rotation(self, P, Q, datatype):
        assert datatype in ['np', 'torch']
        assert ( (len(P.shape) == 2) and (len(Q.shape) == 2) )

        C = P.transpose(-1, -2) @ Q
        # Computation of the optimal rotation matrix
        if datatype == 'np':
            V, _, W = np.linalg.svd(C)
            d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
        else:
            V, _, W = torch.linalg.svd(C)
            d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0
        if d:
            V[:, -1] = -V[:, -1]

        # Create Rotation matrix U
        U = V @ W
        return U


    def rotrans_coords(self, coords, rotrans):
        rot, trans = rotrans
        if len(trans.shape) == 1:
            trans = trans[None, :]
        elif len(trans.shape) == 2:
            pass
        else:
            raise ValueError(f'shape {trans.shape} unknown')
        return coords @ rot + trans

        
        
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    def protein_coord_parser(proteinfile: str, chain: str):
        from Bio.PDB.PDBParser import PDBParser
        from Bio.PDB.MMCIFParser import MMCIFParser

        filetype = (proteinfile.split('.')[1]).lower()
        assert filetype in ['pdb', 'cif']
        if filetype == 'pdb':
            fileparser = PDBParser()
        else:
            fileparser = MMCIFParser()
        
        structure = fileparser.get_structure('Protein', proteinfile)
        chain_ = structure[0][chain]
        
        chain_coords_dict = {}
        for residue_ in chain_:
            res_idx = str(residue_).split('resseq=')[1].split('icode')[0].strip()
            res_coords_dict = {}
            atom_dict = residue_.child_dict
            # import pdb; pdb.set_trace()
            if all(np.isin(['N', 'CA', 'C', 'O'], list(atom_dict.keys()))):
                for atom_ in (residue_.get_atoms()):
                    atomname = atom_.get_full_id()[4][0]
                    if atomname in ['N', 'CA', 'C', 'O']:
                        res_coords_dict[atomname] = list(atom_.get_coord())
            else:
                continue
                        
            chain_coords_dict[res_idx] = res_coords_dict
            
        return chain_coords_dict
            
            

    def write_from_atom_crd(coords, chain="A", filename='test.pdb', natom=4, natom_dict=None):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Bio.PDB.StructureBuilder import StructureBuilder
            from Bio.PDB import PDBIO
            from Bio.PDB.Atom import Atom
            if natom_dict is None:
                natom_dict = {3: {0:'N', 1:'CA', 2:'C'},
                            4: {0:'N', 1:'CA', 2:'C', 3:'O'},
                            5: {0:'N', 1:'CA', 2:'C', 3:'O', 4:'CB'}}
            natom_num = natom_dict[natom]
            sb = StructureBuilder()
            sb.init_structure("pdb")
            sb.init_seg(" ")
            sb.init_model(0)
            chain_id = chain
            sb.init_chain(chain_id)
            for num, line in enumerate(coords):
                name = natom_num[num % natom]

                line = np.around(np.array(line, dtype='float'), decimals=3)
                res_num = num // natom

                atom = Atom(name=name, coord=line, element=name[0:1], bfactor=1, occupancy=1, fullname=name,
                            serial_number=num,
                            altloc=' ')
                sb.init_residue("GLY", " ", res_num, " ")  # Dummy residue
                sb.structure[0][chain_id].child_list[res_num].add(atom.copy())

            structure = sb.structure
            io = PDBIO()
            io.set_structure(structure)
            io.save(filename)


    kabschalign = KabschCycleAlign()    
    A_coords_dict = protein_coord_parser('5tdg.pdb', 'A')
    A_main_coords = np.concatenate([ np.array(list(res_coord.values()) )for res_coord in A_coords_dict.values() ])
    
    B_coords_dict = protein_coord_parser('5tdg.pdb', 'B')
    B_main_coords = np.concatenate([ np.array(list(res_coord.values()) )for res_coord in B_coords_dict.values() ])
    
    res_num = A_main_coords.shape[0]//4
    
    rotransed_A_coord = kabschalign.align(
        torch.from_numpy(A_main_coords.reshape(res_num, -1, 3)), 
        torch.from_numpy(B_main_coords.reshape(res_num, -1, 3)), 
        cycles=5, return_dict=False, verbose=False)
    
    # import pdb; pdb.set_trace()
    # write_from_atom_crd(rotransed_A_coord.reshape(-1, 3), filename='A_to_B_c5.pdb')
    # write_from_atom_crd(B_main_coords.reshape(-1, 3), filename='B_c5.pdb')
    
