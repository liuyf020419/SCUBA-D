#This program generates (αβ)n sketch structures.
import os, sys
import numpy as np
  
def to_cartesian(polar_vector):
    length, angle = polar_vector[0], polar_vector[1]
    return np.array([length*np.cos(angle), length*np.sin(angle)]).T


def gen_alphabeta_sketch(motif_num=12, beta_dist=5, helix_to_beta_dist = 7, first_length=5, second_length=10, N_pad=3, N_pad_minus_z=4, motif_str='E_H', out_f=None): #N_pad_minus_z：The negative value of the translation of the n-terminal coordinate on the z axis.
    pseudo_motif_num = motif_num+1
    motif_angle = 360.0/pseudo_motif_num
    side0_angle = motif_angle/2
    side1_angle = (180 - motif_angle)/2
    side2_length = np.sin(np.deg2rad(side1_angle))/np.sin(np.deg2rad(side0_angle)) * beta_dist/2
    structure_rad = np.sqrt(side2_length ** 2 + (beta_dist/2) ** 2)
    ss_first, ss_second = motif_str.split('_')
    
    motif_angle = np.linspace(0, 2 * np.pi, pseudo_motif_num)
    beta_motif_len = np.ones(pseudo_motif_num) * structure_rad
    beta_xy_coords = to_cartesian([beta_motif_len, motif_angle])[:-1]
    helix_motif_len = np.ones(pseudo_motif_num) * (structure_rad + helix_to_beta_dist)
    helix_xy_coords = to_cartesian([helix_motif_len, motif_angle])[:-1]
    xycoords = np.concatenate([beta_xy_coords, helix_xy_coords], -1).reshape(-1, 2)
    xyzcoords = np.concatenate([xycoords, np.zeros(xycoords.shape[0])[:, None]], -1)
    with open(out_f, 'w') as writer:
        for motif_idx, motif in enumerate(xyzcoords):
            if ((motif_idx % 2) == 0):
                ss = ss_first
                ss_length = first_length
                if (motif_idx == 0):
                    ss_length = first_length + N_pad
                    z_coords = -N_pad_minus_z
                else:
                    z_coords = 0.0
                writer.write(f'{ss} {ss_length} {ss_length} N;{round(motif[0], 2)} {round(motif[1], 2)} {z_coords}; 0 0 1\n')
            else:
                ss = ss_second
                ss_length = second_length
            
                writer.write(f'{ss} {ss_length} {ss_length} C;{round(motif[0], 2)} {round(motif[1], 2)} {round(motif[2], 2)}; 0 0 -1\n')
    
    
if __name__ == '__main__':    
    gen_dir = './alphabetaN'
    cur_dir = os.getcwd()
    # import pdb; pdb.set_trace()
    for motif_num in np.arange(8, 11, 1): #motif num range:8,9,10
        motif_str = 'E_H' #The prefix of the output file
        beta_length = 5 #The length of βstands.
        helix_length = 10 #The length of αhelix.
        os.makedirs(f'{gen_dir}/{motif_str}{motif_num}_{beta_length}_{helix_length}', exist_ok=True)
        out_f = f'{gen_dir}/{motif_str}{motif_num}_{beta_length}_{helix_length}/{motif_str}{motif_num}_{beta_length}_{helix_length}.txt'
        sketch_par_f = f'{gen_dir}/{motif_str}{motif_num}_{beta_length}_{helix_length}/sketch.par' #Parameter file, which is automatically generated if it does not exist
        with open(sketch_par_f, 'w') as writer:
            writer.write(f'START SketchPar\n') 
            writer.write(f'SketchFile = {motif_str}{motif_num}_{beta_length}_{helix_length}.txt\n') #sketch coordinate file.
            writer.write(f'LinkLoop = 1\n') #link loop option. 0 is not connected and 1 is connected.
            writer.write(f'RandomSeed = 123\n')
            writer.write(f'OutputFile = {motif_str}{motif_num}_{beta_length}_{helix_length}_\n') #Output the file name of the pdb with this prefix.
            writer.write(f'GenerationNumber = 3\n') #Number of structures generated.
            writer.write(f'END SketchPar\n')

        gen_alphabeta_sketch(motif_num=motif_num,first_length=beta_length, second_length=helix_length, N_pad=3, motif_str=motif_str,out_f=out_f, beta_dist=6) #N_pad:Add an additional number of residues to the N terminal. beta_dist:The distance between β-strands in respective motifs.
        os.chdir(f'{gen_dir}/{motif_str}{motif_num}_{beta_length}_{helix_length}')
        os.system(f'~/pySCUBA/pySCUBA/cpp_bin/SCUBASketch sketch.par') #generate pdb from coordinate file. Change this path to your installation path.
        os.chdir(cur_dir)
    
