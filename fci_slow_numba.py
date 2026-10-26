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
from numba import njit


@njit(cache=True)
def _contract_1e_numba(f1e, fcivec, link_indexa, link_indexb, norb, na, nb):
    ci0 = fcivec.reshape(na, nb)
    t1 = numpy.zeros((norb, norb, na, nb), dtype=fcivec.dtype)
    for str0 in range(link_indexa.shape[0]):
        tab = link_indexa[str0]
        for k in range(tab.shape[0]):
            a, i, str1, sign = tab[k]
            for ib in range(nb):
                t1[a, i, str1, ib] += sign * ci0[str0, ib]
    for str0 in range(link_indexb.shape[0]):
        tab = link_indexb[str0]
        for k in range(tab.shape[0]):
            a, i, str1, sign = tab[k]
            for ia in range(na):
                t1[a, i, ia, str1] += sign * ci0[ia, str0]
    result = f1e.reshape(norb * norb) @ t1.reshape(norb * norb, na * nb)
    return result.reshape(na, nb)


def contract_1e(f1e, fcivec, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    return _contract_1e_numba(
        f1e, fcivec, link_indexa, link_indexb, norb, na, nb
    ).reshape(fcivec.shape)


@njit(cache=True)
def _contract_2e_hubbard_numba(u_aa, u_ab, u_bb, fcivec, strsa, strsb,
                               norb, na, nb):
    t1a = numpy.zeros((norb, na, nb))
    t1b = numpy.zeros((norb, na, nb))
    fcinew = numpy.zeros_like(fcivec)
    for addr in range(strsa.shape[0]):
        s = strsa[addr]
        for i in range(norb):
            if s & (1 << i):
                for ib in range(nb):
                    t1a[i, addr, ib] += fcivec[addr, ib]
    for addr in range(strsb.shape[0]):
        s = strsb[addr]
        for i in range(norb):
            if s & (1 << i):
                for ia in range(na):
                    t1b[i, ia, addr] += fcivec[ia, addr]
    if u_aa != 0:
        for addr in range(strsa.shape[0]):
            s = strsa[addr]
            for i in range(norb):
                if s & (1 << i):
                    for ib in range(nb):
                        fcinew[addr, ib] += t1a[i, addr, ib] * u_aa
    if u_ab != 0:
        for addr in range(strsa.shape[0]):
            s = strsa[addr]
            for i in range(norb):
                if s & (1 << i):
                    for ib in range(nb):
                        fcinew[addr, ib] += t1b[i, addr, ib] * u_ab
        for addr in range(strsb.shape[0]):
            s = strsb[addr]
            for i in range(norb):
                if s & (1 << i):
                    for ia in range(na):
                        fcinew[ia, addr] += t1a[i, ia, addr] * u_ab
    if u_bb != 0:
        for addr in range(strsb.shape[0]):
            s = strsb[addr]
            for i in range(norb):
                if s & (1 << i):
                    for ia in range(na):
                        fcinew[ia, addr] += t1b[i, ia, addr] * u_bb
    return fcinew


def contract_2e_hubbard(u, fcivec, norb, nelec, opt=None):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    return _contract_2e_hubbard_numba(
        u[0], u[1], u[2], fcivec.reshape(na, nb), strsa, strsb,
        norb, na, nb
    )


@njit(cache=True)
def contract_2e(eri, fcivec, norb, nelec, link_indexa, link_indexb, na, nb):
    '''Compute E_{pq}E_{rs}|CI>'''
    ci0 = fcivec.reshape(na,nb)
    t1 = numpy.zeros((norb,norb,na,nb), dtype=fcivec.dtype)

    # Build the alpha and beta excitation intermediates. These updates can
    # target the same element, so keep them serial rather than using prange.
    for str0 in range(link_indexa.shape[0]):
        tab = link_indexa[str0]
        for k in range(tab.shape[0]):
            a = tab[k, 0]
            i = tab[k, 1]
            str1 = tab[k, 2]
            sign = tab[k, 3]
            for ib in range(nb):
                t1[a,i,str1,ib] += sign * ci0[str0,ib]

    for str0 in range(link_indexb.shape[0]):
        tab = link_indexb[str0]
        for k in range(tab.shape[0]):
            a = tab[k, 0]
            i = tab[k, 1]
            str1 = tab[k, 2]
            sign = tab[k, 3]
            for ia in range(na):
                t1[a,i,ia,str1] += sign * ci0[ia,str0]

    # Let NumPy's optimized matrix multiplication handle the dense ERI step.
    eri2 = eri.reshape(norb*norb, norb*norb)
    t12 = t1.reshape(norb*norb, na*nb)
    t1 = (eri2 @ t12).reshape(norb, norb, na, nb)

    fcinew = numpy.zeros_like(ci0, dtype=fcivec.dtype)
    for str0 in range(link_indexa.shape[0]):
        tab = link_indexa[str0]
        for k in range(tab.shape[0]):
            a = tab[k, 0]
            i = tab[k, 1]
            str1 = tab[k, 2]
            sign = tab[k, 3]
            for ib in range(nb):
                fcinew[str1,ib] += sign * t1[a,i,str0,ib]

    for str0 in range(link_indexb.shape[0]):
        tab = link_indexb[str0]
        for k in range(tab.shape[0]):
            a = tab[k, 0]
            i = tab[k, 1]
            str1 = tab[k, 2]
            sign = tab[k, 3]
            for ia in range(na):
                fcinew[ia,str1] += sign * t1[a,i,ia,str0]
    return fcinew.reshape(-1)


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


@njit(cache=True)
def _make_hdiag_numba(h1e, diagj, diagk, occslista, occslistb):
    na = occslista.shape[0]
    nb = occslistb.shape[0]
    hdiag = numpy.empty(na * nb, dtype=h1e.dtype)
    for ia in range(na):
        for ib in range(nb):
            e1 = 0.0
            e2 = 0.0
            for p in range(occslista.shape[1]):
                pa = occslista[ia, p]
                for q in range(occslista.shape[1]):
                    qa = occslista[ia, q]
                    e2 += diagj[pa, qa] - diagk[pa, qa]
                for q in range(occslistb.shape[1]):
                    qb = occslistb[ib, q]
                    e2 += diagj[pa, qb]
                e1 += h1e[pa, pa]
            for p in range(occslistb.shape[1]):
                pb = occslistb[ib, p]
                for q in range(occslista.shape[1]):
                    qa = occslista[ia, q]
                    e2 += diagj[pb, qa]
                for q in range(occslistb.shape[1]):
                    qb = occslistb[ib, q]
                    e2 += diagj[pb, qb] - diagk[pb, qb]
                e1 += h1e[pb, pb]
            hdiag[ia * nb + ib] = e1 + e2 * 0.5
    return hdiag


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
    return _make_hdiag_numba(h1e, diagj, diagk, occslista, occslistb)

def kernel(h1e, eri, norb, nelec, ecore=0):
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    nelecb = nelec//2
    neleca = nelec - nelecb
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    na = cistring.num_strings(norb, nelec//2)
    ci0 = numpy.zeros((na,na))
    ci0[0,0] = 1

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec, link_indexa, link_indexb, na, na)
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, eri, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond)
    return e+ecore


@njit(cache=True)
def _make_rdm1_numba(fcivec, link_index, norb, na):
    rdm1 = numpy.zeros((norb, norb))
    for str0 in range(link_index.shape[0]):
        tab = link_index[str0]
        for k in range(tab.shape[0]):
            a, i, str1, sign = tab[k]
            value = 0.0
            for ib in range(na):
                value += fcivec[str1, ib] * fcivec[str0, ib]
            rdm1[a, i] += sign * value
    for str0 in range(link_index.shape[0]):
        tab = link_index[str0]
        for k in range(tab.shape[0]):
            a, i, str1, sign = tab[k]
            value = 0.0
            for ia in range(na):
                value += fcivec[ia, str1] * fcivec[ia, str0]
            rdm1[a, i] += sign * value
    return rdm1


# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec // 2)
    na = cistring.num_strings(norb, nelec // 2)
    return _make_rdm1_numba(fcivec.reshape(na, na), link_index, norb, na)

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec//2)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)

    rdm1 = numpy.zeros((norb,norb))
    rdm2 = numpy.zeros((norb,norb,norb,norb))
    for str0, tab in enumerate(link_index):
        t1 = numpy.zeros((na,norb,norb))
        for k in range(tab.shape[0]):
            a = tab[k, 0]
            i = tab[k, 1]
            str1 = tab[k, 2]
            sign = tab[k, 3]
            t1[:,i,a] += sign * fcivec[str1,:]

        for k, tab in enumerate(link_index):
            for l in range(tab.shape[0]):
                a = tab[l, 0]
                i = tab[l, 1]
                str1 = tab[l, 2]
                sign = tab[l, 3]
                t1[k,i,a] += sign * fcivec[str0,str1]

        rdm1 += numpy.einsum('m,mij->ij', fcivec[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        rdm2 += numpy.einsum('mij,mkl->jikl', t1, t1)
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
