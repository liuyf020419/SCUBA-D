import pandas as pd

from Bio.PDB.DSSP import make_dssp_dict

from protein_constant_utils import *

def parse_dssp_from_dict(dssp_file, authchain=False):

    d = make_dssp_dict(dssp_file)
    appender = []
    for k in d[1]:
        to_append = []
        y = d[0][k]
        chain = k[0]
        residue = k[1]
        het = residue[0]
        resnum = residue[1]
        icode = residue[2]
        to_append.extend([chain, resnum, icode])
        to_append.extend(y)
        appender.append(to_append)

    cols = ['chain','resnum', 'icode' ,'aa', 'ss', 'exposure_rsa', 'phi', 'psi','ggg',
            'NH_O_1_relidx', 'NH_O_1_energy', 'O_NH_1_relidx',
            'O_NH_1_energy', 'NH_O_2_relidx', 'NH_O_2_energy',
            'O_NH_2_relidx', 'O_NH_2_energy']

    df = pd.DataFrame.from_records(appender, columns=cols)

    return df

def extract_SS_ASA_fromDSSP(dssp_file: str, chain: str, encoding=True, authchain=False):

    df = parse_dssp_from_dict(dssp_file, authchain)

    df = df[(df["chain"] == chain) & (df["icode"] == " ")]
    ss8_series = df['ss']
    ss3_series = ss8_series.copy()
    rsa_series = df['exposure_rsa']

    ss3_series.loc[(ss8_series == 'T')|(ss8_series == 'S')|(ss8_series == '-')] = "L"
    ss3_series.loc[(ss8_series == 'H') | (ss8_series == 'G') | (ss8_series == 'I')] = "H"
    ss3_series.loc[(ss8_series == 'B') | (ss8_series == 'E')] = "E"

    ss3_series.replace(ENCODESS32NUM, inplace=True)
    ss8_series.replace(ENCODESS82NUM, inplace=True)

    newdf = pd.DataFrame({"resid": df["resnum"], "SS3": ss3_series, "SS8": ss8_series, "RSA": rsa_series}).set_index("resid")
    dssp_dict = newdf.transpose().to_dict()

    return dssp_dict

if __name__ == "__main__":
    dssp_dict = extract_SS_ASA_fromDSSP("testPDB/4r80.dssp", "A")
    print([v["SS8"] for v in dssp_dict.values()])
