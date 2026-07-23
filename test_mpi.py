from fci_plugin.fci_slow_mpi import kernel as kernel_mpi
from pyscf import gto, scf, ao2mo
import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # 自分のプロセス番号
size = comm.Get_size() 

def test():
    # int1e = None
    # eri = None
    # norb = None
    # nelec = None
    # ecore = None
    # if rank == 0:
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
    # int1e = comm.bcast(int1e, root=0)
    # eri = comm.bcast(eri, root=0)
    # norb = comm.bcast(norb, root=0)
    # nelec = comm.bcast(nelec, root=0)
    # ecore = comm.bcast(ecore, root=0)

    sta = time.time()
    e = kernel_mpi(int1e, eri, norb, nelec, ecore)
    # rdm1 = make_rdm1(civec, norb, nelec)
    # rdm2 = make_rdm12(civec, norb, nelec)
    if rank == 0:
        end = time.time()
        print("Energy: ", e)    
        print("Time taken: ", end - sta)

if __name__ == "__main__":
    test()