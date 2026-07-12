from fci_slow_numba import make_rdm1, make_rdm12, kernel
from pyscf.fci import fci_slow, selected_ci_slow
import numpy as np
from pyscf import gto, scf, ao2mo
import time

def test():
    mol = gto.Mole()
    mol.atom = [
        ['H', (0., 0., 0.)],
        ['H', (0., 0., 1.)], 
        ['H', (0., 0., 2.)], 
        ['H', (0., 0., 3.)], 
        # ['H', (0., 0., 4.)],
        # ['H', (0., 0., 5.)], 
        # ['H', (0., 0., 6.)],  
        # ['H', (0., 0., 7.)], 
        # ['H', (0., 0., 8.)], 
        # ['H', (0., 0., 9.)], 
        # ['H', (0., 0., 10.)],
        # ['H', (0., 0., 11.)], 
        # ['H', (0., 0., 12.)],
        # ['H', (0., 0., 13.)],  
    ]
    mol.basis = 'sto-3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = mf.get_hcore()
    int1e = np.einsum('ip,ij,jq->pq', mf.mo_coeff, h1e, mf.mo_coeff)
    eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    ecore = mf.energy_nuc()
    sta = time.time()
    e = kernel(int1e, eri, norb, nelec, ecore)
    # rdm1 = make_rdm1(civec, norb, nelec)
    # rdm2 = make_rdm12(civec, norb, nelec)
    end = time.time()
    print("Energy: ", e)    
    print("Time taken: ", end - sta)
    sta = time.time()
    e_fci_slow = fci_slow.kernel(int1e, eri, norb, nelec, ecore)
    end = time.time()
    print("Time taken (fci_slow): ", end - sta)
    print("Energy (fci_slow): ", e_fci_slow)

    return

if __name__ == "__main__":
    test()


