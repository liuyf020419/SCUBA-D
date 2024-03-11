# Control Templates

To provide more specific details on how to use the provided templates, the following information explains the JSON file structure:

Each JSON file should contain a `global_config` section, which controls the following parameters: `input_file`, `output_prefix`, and `parser`:

* `input_file` - The input to SCUBA-D, which should be in PDB or mmCIF format.
* `output_prefix` - Prefix of the output files.
* `parser` - Specifies which parser to use for reading the template, including `gen_from_noise`, and `refine_prior`.

## `gen_from_noise` parser

This parser is recommended for use with templates such as `gen_from_all_noise.json`, `gen_from_all_sstype.json`, and `gen_from_noise_partial_fixed.json`.

In this mode, two additional controls (`target_sstype` and `chain_config`) need to be set.

* `target_sstype` - Secondary structure type (sstype) specified as a string consisting of H (helix), E (beta), and L (loop). Use `''` if no sstype is to be provided.
* `chain_config` - Chains that need to be considered during design in the `input_file` should be listed here. Controls for each chain should be separated by the token `;`. There are two formats available: `X1` and `X2_X3`(`]`). `X1` denotes residues generated from noise with a length of `X1`, while `X2_X3`(`]`) denotes a region that needs to be fixed. In this context, `X2` is defined as the start index and `X3` as the end index, both referring to indices within the `input_file` rather than absolute indices. If the `]` token is absent, the residue with the end index will determine whether it should be fixed or not.

For example, if we want to generate residues for chain A from noise with a length of 10 and fix a region ranging from residue 5 to residue 20 in chain B, then the `chain_config` in the JSON file should be:
```
"chain_config": {
    "A": "10",
    "B": "5_20]"
}
```
## `refine_prior` parser

This parser is recommended for use with `structure_refine.json`. The JSON file format is similar to `chain_config` in the `gen_from_noise` mode, but with only one format available: `X2_X3`(`]`), which fixes the coordinates of specific residues.

We used the SCUBA software to generate input files. [SCUBA](https://github.com/USTCwangsheng/pySCUBA) can be installed from https://github.com/USTCwangsheng/pySCUBA.

we provide a scripts to genarte an initail srtucre (sketch) as used in the mannu. pdb utils py,py1 ,2.

We provide a script as mentioned to generate the initial structure(sketch) as used in the manuscript.The `gen_alphabetaN_crd.py` and `gen_beta_propellerN_crd.py` files under the `/pdb_utils/sketch/` path can be used to generate the initial structure(sketch) of the (αβ)n-barrels and (β4)n-propellers respectively under the support of SCUBA. See demo in [SCUBA](https://github.com/USTCwangsheng/pySCUBA) for more details.