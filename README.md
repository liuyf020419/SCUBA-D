# SCUBA-D
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10947360.svg)](https://doi.org/10.5281/zenodo.10947360)


SCUBA-D: De novo protein design with a denoising diffusion network independent of pre-trained structure prediction models

SCUBA-D is a method based on deep learning for generating protein structure backbone. Here we published the source code and the demos for SCUBA-D(version 1.0).

To run SCUBA-D, clone this GitHub repository and install Python.

## Requirements

-  Operating System: Linux (Recommended)
-  No non-standard hardware is required.

## Install Dependencies
```
conda create -n SCUBA-D python=3.8
conda activate SCUBA-D

pip install -r ./install/requirements.txt
bash ./install/postInstall.sh
```

Optimal, the SCUBA(SCUBA-sketch) software can be downloaded and installed from zenodo: https://doi.org/10.5281/zenodo.10939749 or github: https://github.com/USTCwangsheng/pySCUBA if you wish to utilize it for generating an initial structure(sketch) as input for SCUBA-D.

## Quick start

First, you need to check whether the weights file `checkpoint_clean.pt` is in the `/savedir/priorDDPM_full_ESM_GAN/checkpoint/` path, if not, you need to download the weights file from https://biocomp.ustc.edu.cn/servers/downloads/checkpoint_clean.pt and save it under the `/savedir/priorDDPM_full_ESM_GAN/checkpoint/` path.

Input for `run.sh` is a text file that contains controls for each design.
The following options are suggested to be manually changed according to your needs:

* `CUDA_VISIBLE_DEVICES` - visible GPU idx
* `test_list` - list of designs
* `batch_size` - number of designs for each batch


### Control Templates
All controls of the design are set in a JSON file. For convenience, we have provided some templates for different usages as references:
* `demo_json/gen_from_noise/gen_from_all_noise.json` - unconditional generation from noise
* `demo_json/gen_from_noise/gen_from_all_sstype.json` - generation with provided secondary structure as prior
* `demo_json/gen_from_noise/gen_from_noise_partial_fixed.json` - unconditional generation but fix some coordinates of motifs
* `demo_json/refine_prior/structure_refine.json` - refine the provided structure.
We used the SCUBA suite to generate input files. [SCUBA](https://github.com/USTCwangsheng/pySCUBA) can be installed from https://github.com/USTCwangsheng/pySCUBA or https://doi.org/10.5281/zenodo.10947360.


Users can also create their own templates by changing the parser in `protdiff/dataset/refine_dataset_par.py`.

To provide more specific details on how to use the provided templates, see `demo_json/README.md` for more details.

## Inference

Sample protein with
```
bash run.sh
```

-----------------------------------------------------------------------------------------------------
Output example in target dir `results`:
```
gen_from_all_noise.json
gen_from_all_noise_batch_0.pdb
```

-----------------------------------------------------------------------------------------------------
It takes approximately 30 seconds for SCUBA-D to generate a protein backbone of 100 amino acids from noise on a computer equipped with an RTX 3090 graphics card. It takes about 5 minutes to run the entire demo.
