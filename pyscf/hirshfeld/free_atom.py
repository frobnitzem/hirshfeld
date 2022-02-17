""" Use this tool to determine the `occ_pattern` to input
    into Hirshfeld.  It scans spin states and outputs the
    orbital symmetry & occupancy summary of the lowest energy state.
"""
import numpy as np

from pyscf import gto, dft, symm
from pyscf.data import elements

from pyscf.hirshfeld.hirshfeld import occ_pattern, make_symmetric

irrep_nelec = {
        'H': [ {} ],
        'C': [ {'p-1': (1,0), 'p+0': (1,0), 'p+1': (0,0)}
             , {'p-1': (0,0), 'p+0': (1,0), 'p+1': (1,0)}
             , {'p-1': (1,0), 'p+0': (0,0), 'p+1': (1,0)} ],
        'N': [ {'p-1': (1,0), 'p+0': (1,0), 'p+1': (1,0)} ],
        'P': [ {'p-1': (2,1), 'p+0': (2,1), 'p+1': (2,1)} ],
        'O': [ {'p-1': (1,1), 'p+0': (1,0), 'p+1': (1,0)}
             , {'p-1': (1,0), 'p+0': (1,1), 'p+1': (1,0)}
             , {'p-1': (1,0), 'p+0': (1,0), 'p+1': (1,1)} ],
        'S': [ {'p-1': (2,2), 'p+0': (2,1), 'p+1': (2,1)}
             , {'p-1': (2,1), 'p+0': (2,2), 'p+1': (2,1)}
             , {'p-1': (2,1), 'p+0': (2,1), 'p+1': (2,2)} ],
        'Cl':[ {'p-1': (2,2), 'p+0': (2,2), 'p+1': (2,1)}
             , {'p-1': (2,2), 'p+0': (2,1), 'p+1': (2,2)}
             , {'p-1': (2,1), 'p+0': (2,2), 'p+1': (2,2)} ],
        'F': [ {'p-1': (1,1), 'p+0': (1,1), 'p+1': (1,0)}
             , {'p-1': (1,1), 'p+0': (1,0), 'p+1': (1,1)}
             , {'p-1': (1,0), 'p+0': (1,1), 'p+1': (1,1)} ],
        }


"""
H: 1 (en = -0.4785525528599545)

C: 2 (en = -37.795298667084126)
s+0, double-occ = 2, single-occ = 0
p-1, double-occ = 0, single-occ = 1
p+0, double-occ = 0, single-occ = 1
p+1, double-occ = 0, single-occ = 0
[-10.01492367  -0.48366937  -0.16790578  -0.16764366  -0.18746127]
[2. 2. 1. 1. 0.]

N: 3 (en = -54.53241947725623)
s+0, double-occ = 2, single-occ = 0
p-1, double-occ = 0, single-occ = 1
p+0, double-occ = 0, single-occ = 1
p+1, double-occ = 0, single-occ = 1
[-14.07484175  -0.6426787   -0.22282949  -0.22282949  -0.22282949]
[2. 2. 1. 1. 1.]

O: 2 (en = -75.00625166250832)
s+0, double-occ = 2, single-occ = 0
p-1, double-occ = 0, single-occ = 1
p+0, double-occ = 1, single-occ = 0
p+1, double-occ = 0, single-occ = 1
[-18.86577303  -0.85202663  -0.30667838  -0.30637857  -0.30618884]
[2. 2. 2. 1. 1.]

S: 2 (en = -397.9433177355339)
s+0, double-occ = 3, single-occ = 0
p-1, double-occ = 2, single-occ = 0
p+0, double-occ = 1, single-occ = 1
p+1, double-occ = 1, single-occ = 1
[-88.14422591  -7.7339021   -5.75401706  -5.75399438  -5.75394368
  -0.61834799  -0.24682103  -0.24670458  -0.2465759]
[2. 2. 2. 2. 2. 2. 2. 1. 1.]

Cl: 1 (en = -459.96539663668693)
s+0, double-occ = 3, single-occ = 0
p-1, double-occ = 2, single-occ = 0
p+0, double-occ = 1, single-occ = 1
p+1, double-occ = 2, single-occ = 0
[-100.75043888   -9.22944598   -7.04866484   -7.04866313   -7.04865757
   -0.74921458   -0.31131841   -0.31129072   -0.31127481]
[2. 2. 2. 2. 2. 2. 2. 2. 1.]

P: 3 (en = -341.11050760606327)
s+0, double-occ = 3, single-occ = 0
p-1, double-occ = 1, single-occ = 1
p+0, double-occ = 1, single-occ = 1
p+1, double-occ = 1, single-occ = 1
[-76.38522486  -6.35090583  -4.56883852  -4.56883852  -4.56883852
  -0.49060992  -0.18269624  -0.18269624  -0.18269624]
[2. 2. 2. 2. 2. 2. 1. 1. 1.]

F: 1 (en = -99.66731739587644)
s+0, double-occ = 2, single-occ = 0
p-1, double-occ = 1, single-occ = 0
p+0, double-occ = 1, single-occ = 0
p+1, double-occ = 0, single-occ = 1
[-24.33526724  -1.0820701   -0.38571528  -0.38563697  -0.41263714]
[2. 2. 2. 2. 1.]
"""

def avg_irreps(elem):
    """
    Compute the electronic structure for each fixed irrep_nelec version
    and check:
      1. how well the average density matrix matches the `make_symmetric` result of one computation
      2. the frontier orbital dot-product between each irrep_nelec
    """
    Vsub = []
    dm = 0
    for ir in irrep_nelec[elem]:
        mol = gto.M(atom=[(elem,0,0,0)], spin=s, basis='cc-pvqz', symmetry='SO3')
        mf = dft.RKS(mol, xc='pbe')
        mf.irrep_nelec = ir
        en = mf.kernel()
        # mf.mo_energy, mf.mo_occ
        w = np.where(mf.mo_occ < 0.001)[0][0]+1
        print(elem, en, str(mf.mo_energy[:w]) + '\n' + str(mf.mo_occ[:w]))
        dm = mf.make_rdm1() + dm
        print(mf.mo_occ[:w])
        try:
            occ = occ_pattern[elem]
        except KeyError:
            raise KeyError(f"Hirshfeld needs a definition for frontier orbital occupancy of '{elem}'.")
        try:
            Vsub.append(make_symmetric(mf, occ))
        except ValueError:
            print("Symmetrization error!")

    dm = dm/len(irrep_nelec[elem])
    #make_symmetric(mf, occ) # already symmetric
    dm2 = mf.make_rdm1()
    print("DM avg error norm = %e / %e" % (abs(dm-dm2).max(), abs(dm).max()))

    print("cross-orbital overlap matrices:")
    S = mf.get_ovlp()
    for i,Vi in enumerate(Vsub):
        for j in range(i+1,len(Vsub)):
            print(f"    {i} {j}:")
            print(np.dot(Vi.transpose(), np.dot(S, Vsub[j])))

def run_single(elem, s : int, xc='pbe'):
    mol = gto.M(atom=[(elem,0,0,0)], spin=s, basis='cc-pvqz', symmetry='SO3')

    mf = dft.UKS(mol, xc=xc)
    en = mf.kernel()

    return s, en, mf

def myocc(mf):
    mo_occ = mf.mo_occ[0]+mf.mo_occ[1]
    w = np.where(mo_occ < 0.001)[0][0]+2

    mol = mf.mol
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff[0])
    irrep_name = dict(zip(mol.irrep_id, mol.irrep_name))

    print("symmetry occupancy energy")
    for i in range(w):
        print(f"{irrep_name[orbsym[i]]} {mf.mo_occ[0][i]},{mf.mo_occ[1][i]} {mf.mo_energy[0][i]},{mf.mo_energy[1][i]}")

def determine_spin_state(elem):
    Z = elements.NUC[elem]
    best = None
    for spin in range(Z%2, min(Z+1,6), 2):
        ans = run_single(elem, spin)
        if best is None:
            best = ans
        elif ans[1] < best[1]:
            best = ans
    print(f"Lowest energy spin state for element {elem} is spin={best[0]}:")
    print(f"  energy = {best[1]}")
    myocc(best[2])

def main(argv):
    assert len(argv) == 2, f"Usage: {argv[0]} <element>"
    elem = argv[1]
    determine_spin_state(elem)

if __name__=="__main__":
    import sys
    main(sys.argv)
