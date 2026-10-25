from fci_slow_mpi import kernel as kernel_mpi
from pyscf import gto, scf, ao2mo
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # 自分のプロセス番号
size = comm.Get_size() 


def test():
    if rank == 0:
        mol = gto.Mole()
        mol.atom = [
            ['H', (0., 0., 0.)],
            ['H', (0., 0., 1.)],
            ['H', (0., 0., 2.)],
            ['H', (0., 0., 3.)],
            ['H', (0., 0., 4.)],
            ['H', (0., 0., 5.)],
            ['H', (0., 0., 6.)],
            ['H', (0., 0., 7.)],
            ['H', (0., 0., 8.)],
            ['H', (0., 0., 9.)],
            ['H', (0., 0., 10.)],
            ['H', (0., 0., 11.)],
        ]
        mol.basis = 'sto-3g'
        mol.build()
        mf = scf.RHF(mol)
        mf.kernel()
        norb = mf.mo_coeff.shape[1]
        nelec = mol.nelectron
        hcore_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False)
        eri = eri.reshape(norb, norb, norb, norb)
        ecore = mol.energy_nuc()
        data = (hcore_mo, eri, norb, nelec, ecore)
    else:
        data = None

    int1e, eri, norb, nelec, ecore = comm.bcast(data, root=0)
    comm.Barrier()
    start_time = MPI.Wtime()
    e = kernel_mpi(int1e, eri, norb, nelec, ecore)
    elapsed = comm.allreduce(MPI.Wtime() - start_time, op=MPI.MAX)
    if rank == 0:
        print("Energy:", e)
        print("MPI ranks:", size)
        print("FCI time (seconds):", elapsed)

if __name__ == "__main__":
    test()
