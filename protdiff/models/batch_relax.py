import os
from relax import relax
import numpy as np
from pathlib import Path


def get_absl_path_pdbfile(absl_root_path: str, filter_str: str='diff_3_'):
    absl_path_pdbfile_list = []
    subdir_list = os.listdir(absl_root_path)
    for subdir in subdir_list:
        target_pdbfile = [f'{absl_root_path}/{subdir}/{f}' for f in os.listdir(f'{absl_root_path}/{subdir}') if ((filter_str in f) and ('pdb' in f)) ]
        absl_path_pdbfile_list.extend(target_pdbfile)
    return absl_path_pdbfile_list


def batch_relax(pdbfile_list: list, outdir: Path):
    amber_relax = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=20,
    )
    for pdbfile in pdbfile_list:
        # import pdb; pdb.set_trace()
        # try:
        out_file, data_type = os.path.basename(pdbfile).split('.pdb')
        batch_num = out_file.split('_')[-1]
        relaxed_pdb_str, _, _ = amber_relax.process(pdbfile=pdbfile)
        prefix = out_file.split('_diff_3')[0].split('_')[0]
        # suboutdir = outdir.joinpath(f'{subdir}')
        # os.makedirs(suboutdir, exist_ok=True)
        unrelaxed_pdb_file = 'relax-batch'.join(out_file.replace('_', '-').split('batch'))
        unrelaxed_pdb_path = outdir.joinpath(f'{unrelaxed_pdb_file}.pdb')
        unrelaxed_pdb_path.write_text(relaxed_pdb_str)
        
        
if __name__ == '__main__':
    absl_root_path = '/home/liuyf/alldata/SCUBA-D-experiment/a2B_loop/raw_pdb'
    outdir = Path('/home/liuyf/alldata/SCUBA-D-experiment/a2B_loop/relaxed_pdb')
    # absl_path_pdbfile_list= get_absl_path_pdbfile(absl_root_path, filter_str='diff_3_term_4')
    absl_path_pdbfile_list = [f'{absl_root_path}/{f}' for f in os.listdir(absl_root_path) if (('a2B_activate_loop_' in f) and ('diff_3_term_0' in f)) ]
    import pdb; pdb.set_trace()
    batch_relax(absl_path_pdbfile_list, outdir)
        