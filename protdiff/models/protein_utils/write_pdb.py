import numpy as np
from ..folding_af2.residue_constants import restype_1to3


def write_singlechain_from_atoms(crd_list, chain="A", filename='test.pdb', natom=4, natom_dict=None):
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



def write_multichain_from_atoms(multcoords: list, write_file: str, aatype: list=None, natom=4, chains: list=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from Bio.PDB.StructureBuilder import StructureBuilder
        from Bio.PDB import PDBIO
        from Bio.PDB.Atom import Atom
        import string
        natom_dict = {
            3: {0: "N", 1: "CA", 2: "C"},
            4: {0: "N", 1: "CA", 2: "C", 3: "O"},
            5: {0: "N", 1: "CA", 2: "C", 3: "O", 4: "CB"}
        }
        natom_num = natom_dict[natom]
        sb = StructureBuilder()
        sb.init_structure("pdb")
        sb.init_seg(" ")
        sb.init_model(0)
        atom_idx = 0
        # import pdb; pdb.set_trace()
        if chains is None:
            chainname = string.ascii_uppercase
        else:
            chainname = chains

        for chain_idx, coords in enumerate(multcoords):
            sb.init_chain(chainname[chain_idx])
            chain_coord = multcoords[chain_idx]
            if aatype is not None:
                chain_aatype = aatype[chain_idx]

            for atom_idx, atom_coord in enumerate(chain_coord):
                name = natom_num[atom_idx % natom]
                res_num = atom_idx // natom
                atom = Atom(name=name, coord=atom_coord, element=name[0:1], bfactor=1, occupancy=1,
                            fullname=name, serial_number=atom_idx, altloc=' ')
                if aatype is not None:
                    res_aatype = chain_aatype[res_num]
                else:
                    res_aatype = 'G'
                try:
                    sb.init_residue(restype_1to3[res_aatype], " ", res_num, " ")  # Dummy residue
                    sb.structure[0][chainname[chain_idx]].child_list[-1].add(atom.copy())
                except:
                    import pdb; pdb.set_trace()

        structure = sb.structure
        io = PDBIO()
        io.set_structure(structure)
        io.save(write_file)