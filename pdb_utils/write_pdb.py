import numpy as np

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}
idx_to_restype_with_x = {i: restype for i, restype in enumerate(restypes_with_x)}

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define a restype name for all unknown residues.
unk_restype = 'UNK'

resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx = {resname: i for i, resname in enumerate(resnames)}



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