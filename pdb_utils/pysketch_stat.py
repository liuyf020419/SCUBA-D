import os, sys
from tqdm import trange

import numpy as np
import torch

from sklearn.mixture._gaussian_mixture import GaussianMixture

import matplotlib.pyplot as plt

sys.path.append("pdb_utils/data_parser")
from protein_coord_parser_new import PoteinCoordsParser

import sys
sys.path.append("protdiff/models")
from protein_geom_utils import get_internal_angles, get_internal_angles3


def get_sstype(main_chain_dict):
    return torch.LongTensor(list(map(lambda residx: main_chain_dict[residx]['SS3'], main_chain_dict.keys())))


def plot_psi_psi(data, png_file, label=None):
    phi = data[:, 0]
    psi = data[:, 1]
    plt.figure()
    if label is None:
        plt.scatter(phi, psi, alpha=0.4)
    else:
        plt.scatter(phi, psi, c=label, alpha=0.4)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])
    plt.tight_layout()
    plt.savefig(png_file)


def stat_phi_psi(pdb_root, dssp_root, pdbname_list, outfile, max_num=1000000000):
    helix_phipsi, beta_phipsi, coil_phipsi = [], [], []
    pdbname_list = pdbname_list[:max_num]

    for pdb_idx in trange(len(pdbname_list)):
        try:
            pdbname = pdbname_list[pdb_idx]

            pdbname, chain = pdbname
            poteinfile = f'{pdb_root}/{pdbname[1:3]}/{pdbname}.cif'
            dsspfile = f'{dssp_root}/{pdbname[1:3]}/{pdbname}.dssp'
            protein_coord_parser = PoteinCoordsParser(poteinfile, mergedsspin_=True, dsspfile=dsspfile, chain=chain, authchain_dssp=True)
            multichain_merged_coords = protein_coord_parser.chain_main_crd_array

            ss3type = get_sstype(protein_coord_parser.chain_crd_dicts[chain])
            coords_ic = get_internal_angles3(torch.from_numpy(multichain_merged_coords[None]))[0]
            # import pdb; pdb.set_trace()

            ss_start_indexs = (torch.where((ss3type[1:] - ss3type[:-1]) != 0)[0] + 1).long()
            ss_start_indexs = torch.cat([torch.LongTensor([0]), ss_start_indexs])
            ss_end_indexs = torch.cat([ss_start_indexs[1:]-1, torch.LongTensor([len(ss3type)])])
            ss_lens = ss_start_indexs[1:] - ss_start_indexs[:-1]
            ss_lens = torch.cat([ss_lens, (len(ss3type) - ss_start_indexs[-1]).unsqueeze(0)])
            start_sstypes = torch.index_select(ss3type, 0, ss_start_indexs)

            for ss_idx, ss in enumerate(start_sstypes):
                ss_len = ss_lens[ss_idx]
                ss_start_index = ss_start_indexs[ss_idx]
                ss_end_index = ss_end_indexs[ss_idx]
                
                if ((ss_len > 2) and (ss_idx != 0) and (ss_idx != len(start_sstypes)-1)):
                    ss_ic = coords_ic[ss_start_index: ss_end_index+1]
                    if ss == 0:
                        helix_phipsi.extend(ss_ic.tolist())
                    elif ss == 1:
                        coil_phipsi.extend(ss_ic.tolist())
                    elif ss == 2:
                        beta_phipsi.extend(ss_ic.tolist())
        except FileNotFoundError as e:
            continue

    ic_dict = {
        'helix': np.stack(helix_phipsi),
        'beta': np.stack(beta_phipsi),
        'coil': np.stack(coil_phipsi)
    }
    np.save(outfile, ic_dict)


def load_gmm(params, random_state=None):
    n_components = params[0].shape[0]
    model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    model._set_parameters(params)
    return model


def gmm_fit_ic(data_x, n_components=20):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm = gmm.fit(data_x)
    label = gmm.predict(data_x)
    return label, gmm
    # new_x = gmm.sample(400, random_state=42)


def plt_test_aic(data_x, figname, n_components_range=[1, 21, 1]):
    n_components = np.arange(*n_components_range)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data_x)
            for n in n_components]
    ic_data = [[n_components[m_idx], m.aic(data_x), m.bic(data_x)]  for m_idx, m in enumerate(models)]
    ic_data = np.stack(ic_data)
    plt.figure()
    plt.plot(ic_data[:, 0], ic_data[:, 1], label='AIC')
    plt.plot(ic_data[:, 0], ic_data[:, 2], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.savefig(figname)
    return ic_data


if __name__ == '__main__':
    pdb_list = '/train14/superbrain/yfliu25/dataset/assemble/data_list/pisces_PC70_noNMR_4res.txt'
    pdb_root = '/train14/superbrain/lhchen/data/PDB/20220102/mmcif'
    dssp_root = '/train14/superbrain/yfliu25/dataset/divideddssp'
    outfile = '/train14/superbrain/yfliu25/structure_refine/monomer_joint_PriorDDPM_ESM1b_unfixCEhead_Dnet_LE_MPNN_LC_trans_newmask/pdb_utils/sketch_dat/12000_ic_3.npy'
    png_root = '/train14/superbrain/yfliu25/structure_refine/monomer_joint_PriorDDPM_ESM1b_unfixCEhead_Dnet_LE_MPNN_LC_trans_newmask/pdb_utils/sketch_dat'
    
    # pdbname_list = []
    # with open(pdb_list, 'r') as reader:
    #     for l_idx, line in enumerate(reader.readlines()):
    #         if l_idx > 0:
    #             pdbname_list.append([line.strip().split()[0][:4].lower(), line.strip().split()[0][4]])
    # # import pdb; pdb.set_trace()
    # stat_phi_psi(pdb_root, dssp_root, pdbname_list, outfile, max_num=12000)

    data_ic_dict = np.load(outfile, allow_pickle=True).item()
    # import pdb; pdb.set_trace()
    # # # helix_ic = plt_test_aic(data_ic_dict['helix'], f'{png_root}/helix3_aic.png', n_components_range=[124, 132, 1]) # min bic 24, 128(124, 132)(BIC, AIC)[128]
    # # beta_ic = plt_test_aic(data_ic_dict['beta'], f'{png_root}/beta3_aic.png', n_components_range=[100, 108, 1]) # min bic 28, 148(144, 152)(BIC), 104(100, 108)(AIC, BIC)[104]
    # coil_ic = plt_test_aic(data_ic_dict['coil'], f'{png_root}/coil3_aic.png', n_components_range=[188, 196, 1]) # min bic 37, 192(188, 196)(BIC)[195]
    # import pdb; pdb.set_trace()

    ic_params_dict = {}
    label, gmm = gmm_fit_ic(data_ic_dict['helix'], n_components=128)
    print('helix max prob center', np.rad2deg(gmm._get_parameters()[1][np.argmax(gmm._get_parameters()[0])]))
    ic_params_dict['helix'] = gmm._get_parameters()
    # plot_psi_psi(data_ic_dict['helix'][:100000], f'{png_root}/helix_debug.png')
    new_x = gmm.sample(4000)[0]
    plot_psi_psi(new_x, f'{png_root}/helix3_gen.png')

    label, gmm = gmm_fit_ic(data_ic_dict['beta'], n_components=104)
    print('beta max prob center', np.rad2deg(gmm._get_parameters()[1][np.argmax(gmm._get_parameters()[0])]))
    ic_params_dict['beta'] = gmm._get_parameters()
    # plot_psi_psi(data_ic_dict['beta'][:100000], f'{png_root}/beta_debug.png')
    new_x = gmm.sample(4000)[0]
    plot_psi_psi(new_x, f'{png_root}/beta3_gen.png')

    label, gmm = gmm_fit_ic(data_ic_dict['coil'], n_components=195)
    print('coil max prob center', np.rad2deg(gmm._get_parameters()[1][np.argmax(gmm._get_parameters()[0])]))
    ic_params_dict['coil'] = gmm._get_parameters()
    # plot_psi_psi(data_ic_dict['coil'][:100000], f'{png_root}/coil_debug.png')
    new_x = gmm.sample(4000)[0]
    plot_psi_psi(new_x, f'{png_root}/coil3_gen.png')

    import pdb; pdb.set_trace()
    np.save(f'{png_root}/gmm_ic3_params_12000.npy', ic_params_dict)

    # ic_params_dict = np.load(f'{png_root}/gmm_ic_params.npy', allow_pickle=True).item()

    # helix_ic_gmm = load_gmm(ic_params_dict['helix'])
    # beta_ic_gmm = load_gmm(ic_params_dict['beta'])
    # coil_ic_gmm = load_gmm(ic_params_dict['coil'])
    # import pdb; pdb.set_trace()


            
