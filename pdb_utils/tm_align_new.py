import os
import tempfile
import numpy as np

# from ops_pdb import write_chain
import Bio.PDB as bio
from Bio.PDB.StructureBuilder import StructureBuilder

TMalign_bin = "/home/wangsheng/tools/TMalign"
# TMalign_bin = "/train14/superbrain/lhchen/protein/Alphafold2/tools/TMalign/TMalign"


def parse_matrixfile(matrix_filename: str):
    assert os.path.exists(matrix_filename)
    lines = open(matrix_filename).readlines()

    matrix = []
    for l_idx, line in enumerate(lines):
        if (l_idx == 2 or l_idx == 3 or l_idx == 4):
            matrix_str = line.strip().split()[1:]
            matrix.append(list(map(float, matrix_str)))

    return np.asarray(matrix)


def write_chain(pdbfile: str, target_chain:str, outpdb: str):
    
    parser = bio.PDBParser()
    structure = parser.get_structure(os.path.basename(pdbfile), pdbfile)

    sb = StructureBuilder()
    sb.init_structure("pdb")
    sb.init_seg(" ")
    sb.init_model(0)
    sb.structure[0].add(structure[0][target_chain].copy())

    structure = sb.structure
    io = bio.PDBIO()
    io.set_structure(structure)
    io.save(outpdb)


def readtmfile(tmfile):
    tmscore = None
    rmsd = None
    alignment = []
    aligned_len = None
    alignmentidx = None

    with open(tmfile, "r") as reader:
        for idx, line in enumerate(reader.readlines()):
            if line.startswith('Aligned'):
                rmsd = float(line.strip().split(",")[1].split("=")[1].strip())
                aligned_len = int(line.strip().split(",")[0].split("=")[1].strip())
            if 'if normalized by length of Chain_1' in line:
                tmscore = float(line.strip().split("(")[0].split("=")[1].strip())
            if "denotes residue pairs of d < " in line:
                alignmentidx = [idx + 1, idx + 2, idx + 3]
            if ( (alignmentidx is not None) and (idx in alignmentidx) ):
                alignment.append(line.strip())

    return tmscore, rmsd, alignment, aligned_len


class TMalign(object):
    def __init__(self, bin_file=None):
        super(TMalign, self).__init__()
        if bin_file is None:
            self.bin_file = TMalign_bin
        else:
            self.bin_file = bin_file
        assert os.path.isfile(self.bin_file)
        self.cutoff_value = None

    def run(self, proteinA: str, proteinB: str, chainA=None, chainB=None, cutoff_value=None, matrix_file=None, write_file=None):
        if ((chainA == None) and (chainB == None)):
            cmd = [self.bin_file, proteinA, proteinB]
        else:
            assert ((chainA != None) and (chainB != None))
            tmp_baseroot = os.path.basename(proteinA)
            mobile_filename = tempfile.mktemp('.pdb', 'mobile')
            target_filename = tempfile.mktemp('.pdb', 'target')
            write_chain(proteinA, chainA, os.path.join(tmp_baseroot, mobile_filename))
            write_chain(proteinB, chainB, os.path.join(tmp_baseroot, target_filename))
            cmd = [self.bin_file, mobile_filename, target_filename]
            if cutoff_value is not None:
                assert isinstance(cutoff_value, int)
                self.cutoff_value = cutoff_value
                cmd += ['-d', str(self.cutoff_value)]
            if matrix_file is not None:
                assert isinstance(matrix_file, str)
                cmd += ['-m', matrix_file]

        try:
            if write_file is None:
                tmlines = os.popen(" ".join(cmd)).readlines()
                tmscore, rmsd, alignment, aligned_len = self.parser_tmlines(tmlines)
                return tmscore, rmsd, alignment, aligned_len
            else:
                assert isinstance(write_file, str)
                cmd.extend([">", write_file])
                os.system(" ".join(cmd))
        except OSError as e:
            print(str(e))
        finally:
            if not ((chainA == None) and (chainB == None)):
                os.remove(mobile_filename)
                os.remove(target_filename)

    def parser_tmlines(self, tmlines):
        tmscore = None
        rmsd = None
        aligned_len = None
        alignment = []
        alignmentidx = None

        for idx, line in enumerate(tmlines):
            if line.startswith('Aligned'):
                rmsd = float(line.strip().split(",")[1].split("=")[1].strip())
                aligned_len = int(line.strip().split(",")[0].split("=")[1].strip())
            if 'if normalized by length of Chain_1' in line:
                tmscore = float(line.strip().split("(")[0].split("=")[1].strip())
            if "denotes residue pairs of d < " in line:
                alignmentidx = [idx + 1, idx + 2, idx + 3]
            if ( (alignmentidx is not None) and (idx in alignmentidx) ):
                alignment.append(line.strip())

            # if idx == 16:
            #     rmsd = float(line.strip().split(",")[1].split("=")[1].strip())
            #     aligned_len = int(line.strip().split(",")[0].split("=")[1].strip())
            # if self.cutoff_value is not None:
            #     if idx == 19:
            #         tmscore = float(line.strip().split("(")[0].split("=")[1].strip())
            #     if (idx == 23 or idx == 24 or idx == 25):
            #         alignment.append(line.strip())
            # else:
            #     if idx == 17:
            #         tmscore = float(line.strip().split("(")[0].split("=")[1].strip())
            #     if (idx == 22 or idx == 23 or idx == 24):
            #         alignment.append(line.strip())

        return tmscore, rmsd, alignment, aligned_len


if __name__ == "__main__":
    proteinA = "/train14/superbrain/yfliu25/dataset/monomer_demo/PDBfile/5j0j.pdb"
    proteinB = "/train14/superbrain/yfliu25/dataset/monomer_demo/PDBfile/5j0j.pdb"
    # proteinB = "/home/liuyf/alldata/test_tmalign/5tdl.pdb"
    matrix_file = "/train14/superbrain/yfliu25/dataset/monomer_demo/PDBfile/testmatrix.txt"

    tmaligner = TMalign()
    # tmaligner.run(proteinA, proteinB, chainA="B", chainB="A", write_file="/home/liuyf/alldata/ABACUS-R-pub/utils/test_cut_default.tm")

    tmscore, rmsd, alignment, aligned_len = tmaligner.run(proteinA, proteinB, chainA="B", chainB="A",
                                                          matrix_file=matrix_file, cutoff_value=10)
    print(tmscore)
    print(rmsd)
    print(aligned_len)

    print(alignment[0])
    print(alignment[1])
    print(alignment[2])

    # tmscore, rmsd, alignment, aligned_len = readtmfile('/train14/superbrain/yfliu25/dataset/monomer_demo/PDBfile/tmtest.txt')
    # import pdb; pdb.set_trace()