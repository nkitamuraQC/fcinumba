#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

# Modifications Copyright 2026 Naoki Kitamura.
# Modified from the original PySCF implementation for Numba
# optimization and MPI parallelization.

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    '''Compute E_{pq}E_{rs}|CI>'''
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na, nb)

    t1 = numpy.zeros((norb, norb, na, nb), dtype=fcivec.dtype)
    start_a = len(link_indexa) * rank // size
    end_a = len(link_indexa) * (rank + 1) // size
    start_b = len(link_indexb) * rank // size
    end_b = len(link_indexb) * (rank + 1) // size

    # --- t1の構築(自分の担当分だけ) ---
    for str0 in range(start_a, end_a):
        tab = link_indexa[str0]
        for a, i, str1, sign in tab:
            t1[a, i, str1] += sign * ci0[str0]

    for str0 in range(start_b, end_b):
        tab = link_indexb[str0]          # ← link_indexa から link_indexb に修正
        for a, i, str1, sign in tab:
            t1[a, i, :, str1] += sign * ci0[:, str0]

    # Sum the partial intermediates in place to avoid a second full-size array.
    comm.Allreduce(MPI.IN_PLACE, t1, op=MPI.SUM)

    # Distribute the expensive ERI matrix multiplication by output orbital
    # pair. Previously every rank performed the complete matrix product.
    eri2 = eri.reshape(norb * norb, norb * norb)
    t12 = t1.reshape(norb * norb, na * nb)
    row_start = norb * norb * rank // size
    row_end = norb * norb * (rank + 1) // size
    local_t1 = (eri2[row_start:row_end] @ t12).reshape(
        row_end - row_start, na, nb
    )
    del t1, t12

    fcinew = numpy.zeros_like(ci0, dtype=fcivec.dtype)

    # Each rank contracts only the output orbital pairs from its matrix block.
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            row = a * norb + i
            if row_start <= row < row_end:
                fcinew[str1] += sign * local_t1[row - row_start, str0]

    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            row = a * norb + i
            if row_start <= row < row_end:
                fcinew[:, str1] += sign * local_t1[row - row_start, :, str0]

    # Combine each rank's partial CI vector in place.
    comm.Allreduce(MPI.IN_PLACE, fcinew, op=MPI.SUM)
    return fcinew.reshape(fcivec.shape)


def contract_1e(f1e, fcivec, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na, nb)
    t1 = numpy.zeros((norb, norb, na, nb), dtype=fcivec.dtype)
    start_a = len(link_indexa) * rank // size
    end_a = len(link_indexa) * (rank + 1) // size
    start_b = len(link_indexb) * rank // size
    end_b = len(link_indexb) * (rank + 1) // size
    for str0 in range(start_a, end_a):
        tab = link_indexa[str0]
        for a, i, str1, sign in tab:
            t1[a, i, str1] += sign * ci0[str0]
    for str0 in range(start_b, end_b):
        tab = link_indexb[str0]
        for a, i, str1, sign in tab:
            t1[a, i, :, str1] += sign * ci0[:, str0]

    comm.Allreduce(MPI.IN_PLACE, t1, op=MPI.SUM)
    t12 = t1.reshape(norb * norb, na * nb)
    f1e_flat = f1e.reshape(-1)
    row_start = norb * norb * rank // size
    row_end = norb * norb * (rank + 1) // size
    fcinew = f1e_flat[row_start:row_end] @ t12[row_start:row_end]
    comm.Allreduce(MPI.IN_PLACE, fcinew, op=MPI.SUM)
    return fcinew.reshape(fcivec.shape)



def contract_2e_hubbard(u, fcivec, norb, nelec, opt=None):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    u_aa, u_ab, u_bb = u

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na, nb)
    occ_a = numpy.asarray(
        [[(int(string) >> orbital) & 1 for orbital in range(norb)]
         for string in strsa],
        dtype=numpy.int8,
    )
    occ_b = numpy.asarray(
        [[(int(string) >> orbital) & 1 for orbital in range(norb)]
         for string in strsb],
        dtype=numpy.int8,
    )
    nalpha = occ_a.sum(axis=1)
    nbeta = occ_b.sum(axis=1)
    fcinew = numpy.zeros_like(ci0)

    # Each rank owns disjoint alpha-string rows of the output CI matrix.
    start_a = na * rank // size
    end_a = na * (rank + 1) // size
    for addr_a in range(start_a, end_a):
        overlap = occ_b @ occ_a[addr_a]
        coeff = u_aa * nalpha[addr_a] + u_bb * nbeta + 2 * u_ab * overlap
        fcinew[addr_a] = coeff * ci0[addr_a]

    comm.Allreduce(MPI.IN_PLACE, fcinew, op=MPI.SUM)
    return fcinew


def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    h2e = ao2mo.restore(1, eri.copy(), norb).astype(h1e.dtype, copy=False)
    f1e = h1e - numpy.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e * fac


def make_hdiag(h1e, eri, norb, nelec, opt=None):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    occslista = cistring.gen_occslst(range(norb), neleca)
    occslistb = cistring.gen_occslst(range(norb), nelecb)
    eri = ao2mo.restore(1, eri, norb)
    diagj = numpy.einsum('iijj->ij', eri)
    diagk = numpy.einsum('ijji->ij', eri)
    na = len(occslista)
    nb = len(occslistb)
    hdiag = numpy.zeros(na * nb, dtype=h1e.dtype)
    start = na * nb * rank // size
    end = na * nb * (rank + 1) // size
    for index in range(start, end):
        ia, ib = divmod(index, nb)
        aocc = occslista[ia]
        bocc = occslistb[ib]
        e1 = h1e[aocc, aocc].sum() + h1e[bocc, bocc].sum()
        e2 = diagj[aocc][:, aocc].sum() + diagj[aocc][:, bocc].sum() \
            + diagj[bocc][:, aocc].sum() + diagj[bocc][:, bocc].sum() \
            - diagk[aocc][:, aocc].sum() - diagk[bocc][:, bocc].sum()
        hdiag[index] = e1 + e2 * .5
    comm.Allreduce(MPI.IN_PLACE, hdiag, op=MPI.SUM)
    return hdiag

def kernel(h1e, eri, norb, nelec, ecore=0):
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)

    na = cistring.num_strings(norb, nelec//2)
    ci0 = numpy.zeros((na,na))
    ci0[0,0] = 1

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, eri, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond)
    return e+ecore


# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec//2)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)
    rdm1 = numpy.zeros((norb,norb))
    start = len(link_index) * rank // size
    end = len(link_index) * (rank + 1) // size
    for str0 in range(start, end):
        tab = link_index[str0]
        for a, i, str1, sign in tab:
            rdm1[a, i] += sign * numpy.dot(fcivec[str1], fcivec[str0])
    for str0 in range(start, end):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a, i] += sign * numpy.dot(fcivec[:, str1], fcivec[:, str0])
    comm.Allreduce(MPI.IN_PLACE, rdm1, op=MPI.SUM)
    return rdm1

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec//2)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)

    rdm1 = numpy.zeros((norb,norb))
    rdm2 = numpy.zeros((norb,norb,norb,norb))
    start = len(link_index) * rank // size
    end = len(link_index) * (rank + 1) // size
    for str0 in range(start, end):
        tab = link_index[str0]
        t1 = numpy.zeros((na,norb,norb))
        for a, i, str1, sign in link_index[str0]:
            t1[:,i,a] += sign * fcivec[str1,:]

        for k, tab in enumerate(link_index):
            for a, i, str1, sign in tab:
                t1[k,i,a] += sign * fcivec[str0,str1]

        rdm1 += numpy.einsum('m,mij->ij', fcivec[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        rdm2 += numpy.einsum('mij,mkl->jikl', t1, t1)
    comm.Allreduce(MPI.IN_PLACE, rdm1, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, rdm2, op=MPI.SUM)
    return reorder_rdm(rdm1, rdm2)


def reorder_rdm(rdm1, rdm2, inplace=True):
    '''reorder from rdm2(pq,rs) = <E^p_q E^r_s> to rdm2(pq,rs) = <e^{pr}_{qs}>.
    Although the "reoredered rdm2" is still in Mulliken order (rdm2[e1,e1,e2,e2]),
    it is the true 2e DM (dotting it with int2e gives the energy of 2e parts)
    '''
    nmo = rdm1.shape[0]
    if inplace:
        rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo)
    else:
        rdm2 = rdm2.copy().reshape(nmo,nmo,nmo,nmo)
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    return rdm1, rdm2


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1 = kernel(h1e, eri, norb, nelec)
    print(e1, e1 - -7.9766331504361414)
