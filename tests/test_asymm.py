import numpy as np
import numpy.random as rand

from pyscf.gto import M
from pyscf.hirshfeld import Hirshfeld, calc_rho

def test_asymm(atoms = ['C', 'O']):
    mol = M(atom=[(atoms[0],0,0,0),
                  (atoms[1],0,0,1.2)],
            unit='B', basis='cc-pvdz')
    H = Hirshfeld(mol, 'pbe')

    rad = np.arange(0.015, 7, 0.03)
    NProfile = 10
    profiles = np.zeros((NProfile, len(rad)))
    for i in range(NProfile):
        th = rand.random()*2*np.pi
        z = rand.random()*2-1.0
        scale = (1.0 - z*z)**0.5
        xyz = np.array([scale*np.cos(th), scale*np.sin(th), z])

        pts = rad[:,None]*xyz[None,:]

        mref, ksref, dmref = H.single_atom(atoms[0])
        profiles[i] = calc_rho(pts, mref, dmref)

    avg = profiles.sum(0)/NProfile
    err = np.abs(profiles - avg[None,:]).max() / profiles.max()
    print(err)
    assert err < 1e-3

    err2 = ((profiles - avg[None,:])**2).sum() / (profiles**2).sum()
    err2 = err2**0.5
    assert err2 < 1e-3
    print(err2)
    return rad, avg

if __name__=="__main__":
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    rad, p = test_asymm()
    ax.plot(rad, rad*rad*p)
    #ax.set_xscale('log')
    plt.show()
