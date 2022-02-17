""" Test against molecules from Hirshfeld's original paper,
    https://link.springer.com/content/pdf/10.1007/BF00549096.pdf

    Geometries optimized using b3lyp DFT via
    #========================================================
    #| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
    #| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
    #| http://dx.doi.org/10.1063/1.4952956                                    |#
    #========================================================
"""

import numpy as np

from pyscf import gto
from pyscf.hirshfeld import Hirshfeld

Bohr2Ang = 0.529177210903

def fmt(num):
    return f"{num:8.3f}"

def run_hirsh(mol_z, basis='3-21G'):
    tok = mol_z.split()
    mol = [(elem,0,0,z) for elem, z in zip(tok[::2], map(float, tok[1::2]))]
    H = Hirshfeld(gto.M(atom=mol, unit='B', basis=basis), functional='HF')
    q = H.charges()
    def f_dqp(r):
        pt = np.zeros((len(r),3))
        pt[:,0] = r[:,2] * Bohr2Ang
        pt[:,1] = r[:,2]**2 * Bohr2Ang**2
        pt[:,2] = r[:,0]**2 * Bohr2Ang**2
        return pt
    dqp = H.integrate(f_dqp)

    # Print a nice table like in the paper
    print(" ".join(tok[::2]))
    print(" ".join(map(fmt,q)))
    for i in range(3):
        print(" ".join(map(fmt,dqp[:,i])))

def test_HCN():
    run_hirsh("H 0 C 2.03095373 N 4.21747158")

def test_HCCCN():
    run_hirsh("H 0 C 2.02402880 C 4.31787644 C 6.91863341 N 9.12194857")

def test_HCCH():
    run_hirsh("H 0 C 2.02416234 C 4.31068640 H 6.33484874")

def test_NCCN():
    run_hirsh("N 0 C 2.19812828 C 4.81305138 N 7.01117966")

def test_OCO():
    run_hirsh("O 0 C 2.20641937 O 4.41283875")

if __name__=="__main__":
    test_HCN()
    test_HCCCN()
    test_HCCH()
    test_NCCN()
    test_OCO()
