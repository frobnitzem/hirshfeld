# Carry out Hirshfeld partitioning of the charge density
# https://link.springer.com/content/pdf/10.1007/BF00549096.pdf

import numpy as np
from pyscf import gto, dft, symm
from pyscf.data import elements

# Occupancy pattern of the highest occupied few orbitals
# we would like to symmetrize.  If the length is 3, then
# the occupancies of all 3 are averaged by `make_symmetric`.
# `make_symmetric` further checks that these 3 correspond to p-orbitals
# in the solved structure.
#
# If the length is 1, it denotes an s-orbital that contains
# the unpaired electron.  No symmetrization is needed, but the
# spin 1 solution needs to be specified.
#
# If the length is 0, it's because we don't need to symmetrize
# anything, (highest orbital symmetry is full-shell).
#
# TODO: replace the spherical symmetric averaging with
# elements.NRSRHF_CONFIGURATION (used in SAD calculations)
occ_pattern = {
         'H':  [1],
         'He': [],
         'Li': [1,0,0],
         'Be': [],
         'B':  [1,0,0],
         'C':  [1,1,0],
         'N':  [1,1,1],
         'O':  [2,1,1],
         'F':  [2,2,1],
         'Ne': [],
         'Na': [1],
         'Mg': [],
         'Al': [1,0,0],
         'Si': [1,1,0],
         'P':  [1,1,1],
         'S':  [2,1,1],
         'Cl': [2,2,1],
         'Ar': [],
       }
# To get the spin for a given elementm, add up all the ones above.
#spin = dict( (k, (np.array(v)==1).sum()) for k,v in occ_pattern.items())

def _calc(mol : gto.Mole, functional : str):
    """ Return the molecule, DFT result, and total electron density matrix
        for the given set of atoms.

    Args:
        mol : gto.Mole
            Molecule to compute DFT result.
            e.g. gto.M( atom=[(C,0,0,0)], basis=basis, unit=unit, spin=spin, charge=0 )

    Kwargs:
        functional : str
            Density functional for use with dft.UKS.

    Note:
        This routine enforces a spherically symmetric solution for
        isolated atoms.  This is not necessarily the ground state.

        A better alternative would be to average the density matrices
        resulting from a "correct" calculation using spherically symmetric
        rotations.  I don't know how to do that in general.
    """
    Z = mol.atom_charges()
    if mol.spin == 0:
        mf = dft.RKS(mol, xc=functional)
    else:
        mf = dft.UKS(mol, xc=functional)
    energy = mf.kernel()
    return mf

def _construct(lm, V): # reconstruct a matrix from eigenvectors
    V1 = V[:,lm>0]
    return np.dot(V1*lm[lm>0], V1.conj().T)

def _dm_of(ks, mo_occ):
    if isinstance(mo_occ, tuple) or len(mo_occ.shape) == 2:
        return _construct(mo_occ[0], ks.mo_coeff[0]) \
             + _construct(mo_occ[1], ks.mo_coeff[1])

    return _construct(mo_occ, ks.mo_coeff)

def calc_rho(crds, mol, dm):
    """ Return the electron density at the set of points, `crds`
    """
    psi = dft.numint.eval_ao(mol, crds)
    return np.einsum('ij,ki,kj->k', dm, psi.conjugate(), psi)

# note existence of "mf.scf_summary" : {'e1': -122.53416626161197, 'coul': 46.60980082520342, 'exc': -9.294990514163601, 'nuc': 8.841020169010916}
# https://pyscf.org/_modules/pyscf/dft/rks.html

class Hirshfeld:
    """
    This class computes Hirshfeld-partitioned integrals
    over the molecular electronic density.
    """
    # Cached results of single-atom DFT calculations.
    # This is stored statically as part of the Hirshfeld class
    # so that all Hirshfeld() instances may access the same one.
    cache = {}
    # Cached results of single-atom relative volume (integral rho*r^3)
    vol_cache = {}

    def __init__(self, mol : gto.M, functional : str, ks = None):
        self.mol = mol
        self.functional = functional
        if ks is None:
            ks = _calc(self.mol, self.functional)
        dm = _dm_of(ks, ks.mo_occ)
        self.dm = dm

        #crd, wt = dft.grids.get_partition(mol)
        crd, wt = ks.grids.coords, ks.grids.weights
        self.crd = crd
        self.wt  = wt

        zsum = self.mol.charge
        # wref = atoms x integration points array of 
        #        "proatom" integration weights for each atom
        #
        # wref[i,p] = wt[p] * ref[i,p] / rho_ref[p]
        # rho_ref[p] = sum_{atom i} ref[i, p]
        self.wref = np.zeros((self.mol.natm,) + self.wt.shape)
        for i, elem in enumerate(self.mol.elements):
            ri = self.mol.atom_coord(i)
            mref, ksref, dmref = self.single_atom(elem)
            self.wref[i] = calc_rho(crd - ri, mref, dmref)
            # Scale integral to equal mol.atom_charge(i).
            ichg = np.vdot(wt, self.wref[i])
            zi = self.mol.atom_charge(i)
            if abs(ichg - zi) > 1e-4:
                print(f"Atom {i} {elem} charge = {zi} integrates to {ichg}!")
            self.wref[i] *= zi / ichg
            zsum += elements.NUC[elem]
        self.rho_ref = self.wref.sum(0)     # density units
        mzero = self.rho_ref == 0.0
        self.wref *= (1-mzero) * self.wt / (mzero + self.rho_ref) # convert to weights

        # Scale integral to equal zsum
        self.rho = calc_rho(crd, self.mol, dm)
        ichg = np.vdot(wt, self.rho)
        if abs(ichg - zsum) > 1e-4:
            print(f"Sum of atomic charges = {ichg}, differs significantly from expected = {zsum}")
        self.rho *= zsum / ichg
        self.chg = self.rho_ref - self.rho # difference charge density

    def charges(self): # Hirshfeld charge partion
        return self.mol.atom_charges() \
                - np.dot(self.wref, self.rho)

    def ratios(self):
        """
        Compute Hirshfeld volume ratios according to Phys. Rev. Lett. 102, 073005 (2009).
        [10.1103/PhysRevLett.102.073005]
        """
        elem_vols = {}
        rats = np.zeros(self.mol.natm)
        for i,elem in enumerate(self.mol.elements):
            rvol = self.single_vol(elem)
            ri = self.mol.atom_coord(i)
            #print(self.crd.shape, ((((self.crd - ri)**2).sum(1))**1.5)[:10])
            voli = np.dot(self.rho*self.wref[i], (((self.crd - ri)**2).sum(1))**1.5)
            rats[i] = voli / rvol

        return rats

    def __repr__(self):
        return f"Hirshfeld({self.mol!r}, {self.functional!r})"

    def integrate(self, fn):
        """ Return an array of integrals, 1 for each atom.
            fn should take r : array (pts, R^3) -> array (pts, S)
            Here, r are coordinates in units of Bohr.

            The return shape from `integrate` will be (atoms,) + S

            The integral done for every atom, a (at r_a), is,

               int fn(r - r_a) chg_a dr^3

            where chg_a = -1*[rho - rho_ref]*(weight function for atom a)
        """

        natm = self.mol.natm
        ans0 = fn(self.crd - self.mol.atom_coord(0))
        assert len(ans0) == len(self.crd), "Invalid return shape from fn."
        ans = np.zeros( (natm,) + ans0.shape[1:] )
        ans[0] = np.tensordot(self.chg*self.wref[0], ans0, axes=[0,0])
        for i in range(1, natm):
            ri = self.mol.atom_coord(i)
            ans[i] = np.tensordot(self.chg*self.wref[i],
                                  fn(self.crd - ri), axes=[0,0])
        return ans
    
    def single_atom(self, elem : str):
        """ Return a (gto.Mole, UKS, density_matrix) tuple for the given
            element - using a basis and functional consistent with
            self.mol
        """
        key = (elem, self.mol.basis, self.functional) # cache results
        ans = Hirshfeld.cache.get(key, None)
        if ans is not None:
            return ans # mol, ks, dm

        try:
            occ = occ_pattern[elem]
        except KeyError:
            raise KeyError(f"Hirshfeld needs a definition for frontier orbital occupancy of '{elem}'.")
        spin = int( ( np.array(occ)==1 ).sum() )

        mol = gto.M( atom=[(elem,0,0,0)], basis=self.mol.basis,
                     spin=spin, charge=0, symmetry='SO3' )
        ks = _calc(mol, functional=self.functional)
        mo_occ = make_symmetric(ks, occ)
        dm = _dm_of(ks, mo_occ)

        ans = mol, ks, dm
        Hirshfeld.cache[key] = ans
        return ans

    def ref_en(self):
        return np.sum([self.single_atom(elem)[1].e_tot \
                        for elem in self.mol.elements])

    def single_vol(self, elem):
        key = (elem, self.mol.basis, self.functional) # cache results
        ans = Hirshfeld.vol_cache.get(key, None)
        if ans is not None:
            return ans # floatt
        
        mref, ksref, dmref = self.single_atom(elem)
        crd, wt = ksref.grids.coords, ksref.grids.weights
        rho = calc_rho(crd, mref, dmref)
        #print(crd.shape, (((crd*crd).sum(1))**1.5)[:10])
        ans = np.dot( ((crd*crd).sum(1))**1.5, wt*rho )

        Hirshfeld.vol_cache[key] = ans
        return ans

from functools import reduce
def get_index(a, b):
    """
    return index of `b` where subset `a` first occurs -- like str.index()

    raises ValueError if not found
    """
    sep = '\x00'
    sa = reduce(lambda x,y: x+sep+y, map(str, a))
    sb = reduce(lambda x,y: x+sep+y, map(str, b))
    i = sb.index(sa)
    return sb[:i].count(sep)

def make_symmetric(mf, occ):
    """ Return a version of mf.mo_occ with a spherically symmetrized
        orbital occupancy.
        Only handles p-orbitals for now.
    """
    if len(occ) <= 1:
        return mf.mo_occ

    mo_occ = np.array( mf.mo_occ )
    twospin = False
    if len(mo_occ.shape) == 2:
        twospin = True
        mo_occ = mo_occ.sum(0)
    mol = mf.mol
    norb = len(occ) # number of orbitals to average
    try:
        p_start = get_index(map(float, occ), mo_occ)
    except ValueError:
        raise ValueError(f"Unable to find expected occupancy pattern in solution")

    if twospin:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff[0])
    else:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff)
    orbsym = np.array(orbsym)

    # Map orbital symmetry number to 'p-1,p+0,p+1' to double-check
    # that one each is in the list found.
    numid = dict(zip(mol.irrep_id, mol.irrep_name))
    psmap = {'p-1':0, 'p+0':1, 'p+1':2}
    ps = [0]*len(psmap)
    for i in range(p_start, p_start+norb):
        sym = numid[ orbsym[i] ] # name of orbital symmetry type
        #print(f"{i} {sym} {mo_occ[i]}")
        try:
            ps[psmap[sym]] += 1
        except KeyError:
            raise ValueError(f"Frontier orbital has unexpected symmetry {sym}")
    if tuple(ps) != (1,)*len(psmap):
        raise ValueError("Unexpected occupancies for frontier orbitals: p-1,p+0,p+1 ~ {ps}")

    rng = slice(p_start, p_start+norb)
    if not twospin:
        mo_occ[rng] = mo_occ[rng].sum() / norb
        return mo_occ

    # average spins separately
    occ1 = mf.mo_occ[0].copy()
    occ1[rng] = occ1[rng].sum() / norb
    occ2 = mf.mo_occ[1].copy()
    occ2[rng] = occ2[rng].sum() / norb
    return (occ1, occ2)
    ##return mf.mo_coeff[:,rng]

def test_hirshfeld():
    # both of the following units options work
    #L = 0.95251897494 # Ang
    #mol = gto.M( atom=[('N',0,0,0), ('N',0,0,L)], unit='A' )
    L = 0.95251897494/.5291772083 # Bohr
    mol = gto.M( atom=[('N',0,0,0), ('N',0,0,L)], unit='B' )
    print(mol.atom_coords())

    H = Hirshfeld(mol, functional='b3lyp')
    #H.kernel()
    #print(H)
    #print(H.single_atom('N'))

    # compare total charge density integrals
    assert abs( np.dot(H.rho,  H.wt) - 14 ) < 1e-12

    # pro-atom partitioned charges
    assert abs(np.dot(H.wref[0], H.rho_ref) - 7) < 1e-12
    assert abs(np.dot(H.wref[1], H.rho_ref) - 7) < 1e-12

    # Hirshfeld Charges
    assert abs(np.dot(H.rho, H.wref[0]) - 7) < 1e-12
    assert abs(np.dot(H.rho, H.wref[1]) - 7) < 1e-12

    # Hirshfeld Dipoles
    d1 = np.dot(H.chg*H.wref[0], H.crd-mol.atom_coord(0))
    d2 = np.dot(H.chg*H.wref[1], H.crd-mol.atom_coord(1))

    print(d2)
    assert abs(d1[0]) < 1e-12
    assert abs(d1[1]) < 1e-12
    assert abs(d2[0]) < 1e-12
    assert abs(d2[1]) < 1e-12
    assert abs(d1[2]+d2[2]) < 1e-12
    assert abs(d2[2] + 0.356765311) < 1e-5

    dN = H.integrate(lambda r: r)
    assert np.allclose(dN[0], d1)
    assert np.allclose(dN[1], d2)
    
    print(H.ratios())

if __name__=="__main__":
    test_hirshfeld()
