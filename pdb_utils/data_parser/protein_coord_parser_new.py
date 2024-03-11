# @Date    : 2022-02-03 13:36:00
# @Author  : Liu Yufeng (lyf020419@mail.ustc.edu.cn)
# @Version : $preprocess$

import os
import time
import logging
import warnings
from copy import deepcopy

import numpy as np
import torch
import pandas as pd

import Bio.PDB as bpdb
from Bio.PDB.MMCIFParser import MMCIF2Dict

from protein_constant_utils import *
from protein_geom import *
from gzmmcif_parser import *
from dssp4protein import *

MAINCHAINATOMS = ["N", "CA", "C", "CB", "O"]
NOCBMAINCHAINATOMS = ["N", "CA", "C", "O"]
atom_types = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT"
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.
ALLATOMS = atom_types
ALLATOMS_ORDER = atom_order
ALLATOMS_NUM = len(ALLATOMS)
MAPSNAME = ["CB_dist_map", "omega_torsion_map", "theta_torsion_map", "phi_angle_map"]
MASK_DISTS = 20
DSSP_DIR = "/home/liuyf/alldata/divideddssp"


def reduce_all_chain_dicts(all_chain_dicts):
    reduced_dicts = {}
    for chain_id in all_chain_dicts.keys():
        for resname in all_chain_dicts[chain_id].keys():
            new_key = f'{chain_id}_{resname}'
            reduced_dicts[new_key] = all_chain_dicts[chain_id][resname]
    return reduced_dicts


class PoteinCoordsParser(object):
    def __init__(self, poteinfile: str, datatype: str=None, chain: str=None, pseudo_gly=True,
                 mergedsspin_=False, dsspfile=None, with_o_atoms=True, omit_mainatoms_missing=True, authchain_dssp=False, only_single_model=True):
        self.proteinname = os.path.basename(poteinfile)
        self.poteinfile = poteinfile

        if chain is None:
            self.chain_ids = None
        else:
            if isinstance(chain, str):
                self.chain_ids = [chain]
            elif isinstance(chain, list):
                self.chain_ids = chain
            else:
                raise TypeError(f'{type(chain)} unknow')

        if not with_o_atoms:
            self.natom_per_res = 4
            self.mainatoms = MAINCHAINATOMS[:-1]
        else:
            self.natom_per_res = 5
            self.mainatoms = MAINCHAINATOMS
        self.natom_all_per_res = ALLATOMS_NUM
        self.pseudo_gly = pseudo_gly
        self.mergedsspin_ = mergedsspin_
        self.omit_mainatoms_missing = omit_mainatoms_missing
        self.only_single_model = only_single_model
        self.sequence = []

        if datatype is not None:
            assert datatype in ["PDB", "pdb", "mmCIF", "mmcif", "cif", "gz", "GZ"]
        else:
            assert self.proteinname.split('.')[-1] in ["PDB", "pdb", "mmCIF", "mmcif", "cif", "gz", "GZ"]
            datatype = self.proteinname.split('.')[-1]

        if datatype == "mmCIF" or datatype == "mmcif" or datatype == "cif":
            self.chain_crd_dicts = self._parser_crd_dict4cif(self.chain_ids, gzfile=False, pseudo_gly = self.pseudo_gly)
        elif datatype == "PDB" or datatype == "pdb":
            self.chain_crd_dicts = self._parser_crd_dict4pdb(self.chain_ids, pseudo_gly = self.pseudo_gly)
        elif datatype == "gz":
            self.chain_crd_dicts = self._parser_crd_dict4cif(self.chain_ids, gzfile=True, pseudo_gly = self.pseudo_gly)
        else:
            raise ValueError(f"Data type: {datatype} invalid")

        self.reduced_chain_crd_dicts = reduce_all_chain_dicts(self.chain_crd_dicts)
        # import pdb; pdb.set_trace()

        if self.mergedsspin_:
            assert dsspfile is not None
            self._merge_dssp_in_(dsspfile, authchain_dssp)

        self.multichain_length_dict = {chain_id: len(chain_dict) for chain_id, chain_dict in self.chain_crd_dicts.items()}
        self.pdbresID = {chain_id: list(chain_dict.keys()) for chain_id, chain_dict in self.chain_crd_dicts.items()}
        # self.main_natoms = len(self.reduced_chain_crd_dicts) * self.natom_per_res
        # self.main_atoms_dim = self.main_natoms * 3
        # self.main_atom_indices = np.arange(self.main_atoms_dim).reshape(self.main_natoms, 3)

        self.chain_main_crd_array = self.get_main_crd_array()
        self.chain_crd_array = np.array(list(self.reduced_chain_crd_dicts.values()))

        self.pdbresID2absID = {rel2abs[0]:rel2abs[1] for rel2abs in
                               list( zip( self.reduced_chain_crd_dicts.keys(), np.arange(len(self.reduced_chain_crd_dicts)) ) )}
        self.absID2pdbresID = {v: k for k,v in self.pdbresID2absID.items()}

        # import pdb; pdb.set_trace()
        self.sequence =  "".join(list(map(lambda chain_idx: \
            "".join(list(map( lambda res_values: ENCODENUM2AA[res_values["AA"]], \
                self.chain_crd_dicts[chain_idx].values()))), 
            self.chain_crd_dicts.keys()))) 


    def __len__(self):
        return len(self.reduced_chain_crd_dicts)


    def __repr__(self):
        return f"PDB file: {self.proteinname}, chain: {self.chain_ids}, length: {len(self.reduced_chain_crd_dicts)}"


    def __getitem__(self, item):
        return self.reduced_chain_crd_dicts[item]


    def get_pdbresID2absID(self, chain_name=None):
        return {rel2abs[0]:rel2abs[1] for rel2abs in \
            list( zip( self.chain_crd_dicts[chain_name].keys(), np.arange(len(self.chain_crd_dicts[chain_name].keys())) )) }


    def get_atom_crd(self, atom_name: str) -> np.ndarray:
        return self.chain_crd_array[:, ALLATOMS_ORDER[atom_name]]


    def _parser_crd_dict4pdb(self, chain_ids, pseudo_gly=True):
        """
        get mainchain crd dict for pdbfile
        """
        parser = bpdb.PDBParser()
        all_chain_crd_dicts = {}
        structure = parser.get_structure(self.proteinname, self.poteinfile)
        models_list = list(structure.get_models())
        if ( (self.only_single_model) and (len(models_list) != 1) ):
            raise FileNotFoundError(
                f"Only single model PDBs are supported. Found {len(models_list)} models."
            )
        model = models_list[0]

        chain_ids_init = chain_ids
        if chain_ids_init is None:
            chain_ids = list(map(lambda x: str(x).split('=')[1][0], model.child_list) )
            chain_ids_list = []

        for chain_id in chain_ids:
            chain = model[chain_id]
            residuesf = chain.get_residues()
            chain_crd_dicts = {}
            for res in residuesf:
                resid = int(str(res).split("resseq=")[1].split("icode")[0].strip())
                resname = str(res).split("het")[0].split("Residue")[1].strip()
                res_crd_dicts = {}
                if resname in PROTEINLETTER3TO1.keys():
                    for atom in list(res.get_atoms()):
                        atomname = atom.get_full_id()[4][0]
                        if atomname in ALLATOMS:
                            res_crd_dicts[atomname] = list(atom.get_coord())
                    if (pseudo_gly and (resname == "GLY")):
                        # subsititute CB atom with CA atom
                        if res_crd_dicts.__contains__("CA"):
                            try:
                                vec_ca = np.asarray(res_crd_dicts["CA"])
                                vec_n = np.asarray(res_crd_dicts["N"])
                                vec_c = np.asarray(res_crd_dicts["C"])
                                b = vec_ca - vec_n
                                c = vec_c - vec_ca
                                a = np.cross(b, c)
                                CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + vec_ca
                                res_crd_dicts["CB"] = CB.tolist()
                            except:
                                continue
                        else:
                            continue
                    # filter mainchain missing residue
                    # import pdb; pdb.set_trace()
                    if self.omit_mainatoms_missing:
                        if (pseudo_gly) and (not all(np.isin(self.mainatoms, list(res_crd_dicts.keys())) )):
                            continue
                        elif (not pseudo_gly) and (not all(np.isin(NOCBMAINCHAINATOMS, list(res_crd_dicts.keys())) )):
                            continue
                        else:
                            res_crd_dicts = {atom: res_crd_dicts[atom] if atom in res_crd_dicts.keys() else [0., 0., 0.] for atom in ALLATOMS}
                            res_crd_dicts.update({"AA": ENCODEAA2NUM[PROTEINLETTER3TO1[resname]]})
                            self.sequence.append(PROTEINLETTER3TO1[resname])
                            chain_crd_dicts[resid] = res_crd_dicts
                    else:
                        res_crd_dicts = {atom: res_crd_dicts[atom] if atom in res_crd_dicts.keys() else [0., 0., 0.] for atom in ALLATOMS}
                        res_crd_dicts.update({"AA": ENCODEAA2NUM[PROTEINLETTER3TO1[resname]]})
                        self.sequence.append(PROTEINLETTER3TO1[resname])
                        chain_crd_dicts[resid] = res_crd_dicts
            all_chain_crd_dicts[chain_id] = chain_crd_dicts   

            if ( (len(chain_crd_dicts) > 1) and (chain_ids_init is None) ):
                chain_ids_list.append(chain_id)

        if chain_ids_init is None:
            # import pdb; pdb.set_trace()
            self.chain_ids = chain_ids_list     

        return all_chain_crd_dicts


    def _parser_crd_dict4cif(self, chain_ids, gzfile=False, pseudo_gly=True):
        """
        get mainchain crd dict for mmCIF file
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if gzfile: pdb_dict = GZMMCIF2Dict(self.poteinfile)
            else: pdb_dict = MMCIF2Dict(self.poteinfile)
            all_chain_crd_dicts = {}

            new_dict = {"ATOM": pdb_dict["_atom_site.group_PDB"], "atom_id": pdb_dict["_atom_site.auth_seq_id"],
                        "chain": pdb_dict["_atom_site.auth_asym_id"],
                        "AA_type": pdb_dict["_atom_site.label_comp_id"], "icode": pdb_dict["_atom_site.pdbx_PDB_ins_code"],
                        "Atom_type": pdb_dict['_atom_site.label_atom_id'], "X": pdb_dict["_atom_site.Cartn_x"],
                        "Y": pdb_dict["_atom_site.Cartn_y"], "Z": pdb_dict["_atom_site.Cartn_z"],
                        "altloc": pdb_dict['_atom_site.label_alt_id'], "model_num": pdb_dict['_atom_site.pdbx_PDB_model_num'],
                        "bfactor": pdb_dict['_atom_site.B_iso_or_equiv']
                        }

            df = pd.DataFrame.from_dict(new_dict)
            model_list = list(set(pdb_dict['_atom_site.pdbx_PDB_model_num']))

            if ( (self.only_single_model) and (len(model_list) != 1) ):
                raise FileNotFoundError(f"Only single model PDBs are supported. Found {len(model_list)} models.")

            df = df[df["model_num"] == model_list[0]]
            df = df[df['AA_type'].isin( list(PROTEINLETTER3TO1.keys()) )]
            df['ATOM'] = 'ATOM'
            
            chain_ids_init = chain_ids
            if chain_ids_init is None:
                chain_ids = list(set(df["chain"]))
                chain_ids_list = []
            
            for chain_id in chain_ids:
                chain_crd_dicts = {}
                altloclist = list( set(df[(df["chain"] == chain_id) & (df["ATOM"] == "ATOM")]["altloc"].tolist()) )
                if len(altloclist) == 0:
                    continue
                no_dot_altloclist = deepcopy(altloclist)
                if '.' in no_dot_altloclist:
                    no_dot_altloclist.remove('.')

                if len(no_dot_altloclist) > 0:
                    no_dot_altloclist = sorted(no_dot_altloclist)
                    first_no_dot_altloclist = no_dot_altloclist[0]
                    altloc_filter = ['.', first_no_dot_altloclist]
                else:
                    altloc_filter = ['.']
                # import pdb; pdb.set_trace()
                if "." in altloclist:
                    atomdf = df[(df["icode"] == "?") # & (df["Atom_type"].isin(ALLATOMS)) 
                                & (df["ATOM"] == "ATOM") & ( df["altloc"].isin(altloc_filter) ) & (df["chain"] == chain_id)]
                else:
                    atomdf = df[(df["icode"] == "?") #  & (df["Atom_type"].isin(ALLATOMS)) 
                                & (df["ATOM"] == "ATOM") & ( df["altloc"] == first_no_dot_altloclist ) & (df["chain"] == chain_id)]

                atomdf.loc[:, "atom_id"] = atomdf.loc[:, "atom_id"].astype(int)
                
                if atomdf.shape[0] == 0:
                    continue
                atomdf.loc[:, 'crd'] = atomdf.apply(lambda row: (float(row['X']), float(row['Y']), float(row['Z'])), axis=1)

                filterdf = pd.DataFrame({"res_id": atomdf["atom_id"], "AA_type": atomdf["AA_type"],
                                        "Atom_type": atomdf["Atom_type"], "crd": atomdf["crd"]}).set_index('res_id')

                for resid, resdf in filterdf.groupby("res_id"):
                    res_crd_dicts = resdf.iloc[:, -2:].set_index('Atom_type').transpose().to_dict()
                    resname = resdf["AA_type"].iloc[0]
                    # subsititute CB atom with CA atom
                    if (pseudo_gly and (resname == "GLY")):
                        if res_crd_dicts.__contains__("CA"):
                            # import pdb; pdb.set_trace()
                            try:
                                vec_ca = np.asarray(res_crd_dicts["CA"]["crd"])
                                vec_n = np.asarray(res_crd_dicts["N"]["crd"])
                                vec_c = np.asarray(res_crd_dicts["C"]["crd"])
                                b = vec_ca - vec_n
                                c = vec_c - vec_ca
                                a = np.cross(b, c)
                                CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + vec_ca
                                res_crd_dicts["CB"] = {"crd": tuple(CB.tolist())}
                            except KeyError:
                                continue
                        else:
                            continue
                    if self.omit_mainatoms_missing:
                        # filter mainchain missing residue
                        if (pseudo_gly) and (not all(np.isin( self.mainatoms, list(res_crd_dicts.keys()) ))):
                            continue
                        elif (not pseudo_gly) and (not all(np.isin( NOCBMAINCHAINATOMS, list(res_crd_dicts.keys()) ))):
                            continue
                        else:
                            res_crd_dicts = {atom: list(res_crd_dicts[atom]["crd"]) if atom in res_crd_dicts.keys() else [0., 0., 0.] for
                                            atom in ALLATOMS}
                            res_crd_dicts.update({"AA": ENCODEAA2NUM[PROTEINLETTER3TO1[resname]]})
                            self.sequence.append(PROTEINLETTER3TO1[resname])
                            chain_crd_dicts[resid] = res_crd_dicts
                    else:
                        res_crd_dicts = {atom: list(res_crd_dicts[atom]["crd"]) if atom in res_crd_dicts.keys() else [0., 0., 0.] for
                                        atom in ALLATOMS}
                        res_crd_dicts.update({"AA": ENCODEAA2NUM[PROTEINLETTER3TO1[resname]]})
                        self.sequence.append(PROTEINLETTER3TO1[resname])
                        chain_crd_dicts[resid] = res_crd_dicts

                all_chain_crd_dicts[chain_id] = chain_crd_dicts
                if ( (len(chain_crd_dicts) > 1) and (chain_ids_init is None) ):
                    chain_ids_list.append(chain_id)

            if chain_ids_init is None:
                self.chain_ids = chain_ids_list

            return all_chain_crd_dicts


    def get_main_crd_dict(self, reduce_dict=True, chain_name=None) -> dict:
        """return main crd dict"""
        if reduce_dict:       
            chain_main_crd_dicts = dict(
                map(lambda res_id:
                    (res_id, dict(map(
                        lambda atom: (atom, self.reduced_chain_crd_dicts[res_id][atom]),
                        self.mainatoms))
                    ), self.reduced_chain_crd_dicts.keys())
                )
        else:
            if chain_name is not None:
                chain_main_crd_dicts = dict(
                map(lambda res_id:
                        (res_id, dict(map(
                            lambda atom: (atom, self.chain_crd_dicts[chain_name][res_id][atom]),
                            self.mainatoms))
                        ), self.chain_crd_dicts[chain_name].keys()
                    )
                )
            else:
                chain_main_crd_dicts = dict(
                        map(lambda chain_id:
                            map(lambda res_id:
                                (res_id, dict(map(
                                    lambda atom: (atom, self.chain_crd_dicts[chain_id][res_id][atom]),
                                    self.mainatoms))
                                ), self.chain_crd_dicts[chain_id].keys()
                            ), self.chain_ids
                        )
                    )
        return chain_main_crd_dicts


    def get_raw_crd_dict(self) -> dict:
        return self.chain_crd_dicts


    def get_main_crd_array(self, chain_name=None) -> np.ndarray:
        """return main crd array"""
        if chain_name is None:
            main_crd_dict = self.get_main_crd_dict(True)
        else:
            main_crd_dict = self.get_main_crd_dict(False, chain_name)
        # import pdb; pdb.set_trace()
        main_coord = np.array([ 
            list(res.values()) 
            for res in main_crd_dict.values() 
            ])
        return  main_coord


    def get_sequence(self, chain_name=None):
        if chain_name is None:
            return self.sequence
        else:
            return "".join(list(map( lambda res_values: ENCODENUM2AA[res_values["AA"]], self.chain_crd_dicts[chain_name].values())))
        

    def _merge_dssp_in_(self, dsspfile: str, authchain=False):
        pseudo_dssp = {'SS3': 1, 'SS8': 7, 'RSA': 1000}
        # import pdb; pdb.set_trace()
        for chain_id in self.chain_ids:
            # try:
            dssp_dict = extract_SS_ASA_fromDSSP(dsspfile, chain_id, authchain=authchain)
            for res in list(self.chain_crd_dicts[chain_id].keys()):
                if dssp_dict.__contains__(res):
                    self.chain_crd_dicts[chain_id][res].update(dssp_dict[res])
                    self.reduced_chain_crd_dicts[f'{chain_id}_{res}'].update(dssp_dict[res])
                else:
                    self.chain_crd_dicts[chain_id][res].update(pseudo_dssp)
                    self.reduced_chain_crd_dicts[f'{chain_id}_{res}'].update(pseudo_dssp)
            # except:
            #     raise FileNotFoundError(f'failure in loading {self.proteinname}')


    def get_ss_inf(self, ss3=False, ss8=False, chain_name=None):
        assert self.mergedsspin_ == True
        # import pdb; pdb.set_trace()
        ss3_list, ss8_list = [], []
        if chain_name is None:
            chain_names = self.chain_ids
        else:
            chain_names = [chain_name]
        # try:
        for chain_id in chain_names:
            for resid in self.chain_crd_dicts[chain_id].keys():
                if ss3: ss3_list.append(self.chain_crd_dicts[chain_id][resid]["SS3"])
                if ss8: ss8_list.append(self.chain_crd_dicts[chain_id][resid]["SS8"])
        # except:
        #     import pdb; pdb.set_trace()

        if ss8 and ss3:
            return np.asarray(ss3_list), np.asarray(ss8_list)
        else:
            if ss8: return np.asarray(ss8_list)
            elif ss3: return np.asarray(ss3_list)


    def get_rsa(self) -> list:
        assert self.mergedsspin_ == True
        rsa_list = []
        for chain_id in self.chain_ids:
            for resid in self.chain_crd_dicts[chain_id].keys():
                rsa_list.append(self.chain_crd_dicts[chain_id][resid]["RSA"])
        return rsa_list


    def get_simp_SS(self, ss3=True, ss8=True):
        assert self.mergedsspin_ is True
        SSlist = self.get_ss_inf(ss3=ss3, ss8=ss8)
        if len(SSlist) == 2:
            simp_ss3 = self._calc_simp_SS(SSlist[0])
            simp_ss8 = self._calc_simp_SS(SSlist[1])
            return simp_ss3, simp_ss8
        else:
            simp_ss = self._calc_simp_SS(SSlist)
            return simp_ss


    def _calc_simp_SS(self, SS) -> np.ndarray:
        simp_SS = []
        last_ss = None
        for id, sstate in enumerate(SS):
            if id == 0:
                last_ss = sstate
                simp_SS.append([id, sstate])
            else:
                if sstate == last_ss:
                    continue
                else:
                    last_ss = sstate
                    simp_SS.append([id, sstate])
        return np.asarray(simp_SS)


    def get_kai_torsions(self, degrees = True) -> dict:
        chain_tors = {}
        padding_num = 2 * np.rad2deg(np.pi) if degrees else 2 * np.pi
        for chain_id in self.chain_ids:
            for res_id in self.chain_crd_dicts[chain_id].keys():
                # import pdb; pdb.set_trace()
                torsions_list = []
                AA = PROTEINLETTER1TO3[ENCODENUM2AA[self.chain_crd_dicts[chain_id][res_id]["AA"]] ]
                torsions_atoms_list = chi_angles_atoms[AA]
                if len(torsions_atoms_list) == 0:
                    torsions_list.extend(4*[padding_num])
                else:
                    torsions_list = list(map(
                        lambda tor_id: torsion(
                            np.asarray(self.chain_crd_dicts[chain_id][res_id][tor_id[0]], dtype=np.float32)[None, :],
                            np.asarray(self.chain_crd_dicts[chain_id][res_id][tor_id[1]], dtype=np.float32)[None, :],
                            np.asarray(self.chain_crd_dicts[chain_id][res_id][tor_id[2]], dtype=np.float32)[None, :],
                            np.asarray(self.chain_crd_dicts[chain_id][res_id][tor_id[3]], dtype=np.float32)[None, :],
                            degrees = degrees, axis=1).tolist()[0]
                        if all( np.isin(tor_id, list( self.chain_crd_dicts[chain_id][res_id].keys()) ) )
                        else padding_num, torsions_atoms_list))

                torsions_list.extend((4 - len(torsions_list)) * [padding_num])
                chain_tors.update({res_id: torsions_list})

        return chain_tors




if __name__ == "__main__":

    # pdbinfPDB = FastPoteinParser("/train14/superbrain/yfliu25/dataset/sabdab/hcdrs_cluster_pdb_gly/e7/7e7x.pdb")
    pdbfile = '/home/liuyf/alldata/Newdatafolder/datanew/g9/4g9s.cif.gz'
    pdbinfPDB = PoteinCoordsParser(pdbfile)
    import pdb; pdb.set_trace()

    # dsspfile = "/train14/superbrain/lili27/protein/ABACUS-R-pub/assembly_data/dssp_for_assembly/ih/1ihu-assembly2.dssp"
    # pdbfile = "/train14/superbrain/lili27/protein/ABACUS-R-pub/assembly_data/assembly_cif_download/ih/1ihu-assembly2.cif"
    # pdbinfPDB = PoteinCoordsParser(pdbfile, mergedsspin_=True, dsspfile=dsspfile, authchain_dssp=True)
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt

    # def plot_geom_maps(geom_maps, name, sigsize):
    #     # Distance
    #     plt.figure(figsize=sigsize)

    #     plt.subplot(141)
    #     plt.imshow(geom_maps[0]).set_cmap("hot")
    #     plt.colorbar()

    #     plt.subplot(142)
    #     plt.imshow(geom_maps[1]).set_cmap("hot")
    #     plt.colorbar()

    #     plt.subplot(143)
    #     plt.imshow(geom_maps[2]).set_cmap("hot")
    #     plt.colorbar()

    #     plt.subplot(144)
    #     plt.imshow(geom_maps[3]).set_cmap("hot")
    #     plt.colorbar()

    #     plt.savefig(name, bbox_inches='tight', dpi=600, transparent=False)


    # pdbinfPDB = FastPoteinParser("/train14/superbrain/yfliu25/dataset/sabdab/hcdrs_cluster_pdb_gly/e7/7e7x.pdb", "A",  datatype="pdb")
    # import pdb; pdb.set_trace()
    # # pdbinfPDB.get_kai_torsions()
    # # import pdb; pdb.set_trace()