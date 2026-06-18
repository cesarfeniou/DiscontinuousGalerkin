#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:52:32 2026

@author: cesar
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.special import factorial


@dataclass
class ModelParams:
    R: float = 2.0
    ZL: float = 2.0
    ZR: float = 1.0
    potential: str = "soft"
    delta: float = 0.2
    alpha: float = 1.5
    N_MAX: int = 10
    EPS_TOL: float = 1e-8
    alpha_pen: float = 10.0
    L_grid: float = 12.0
    N_grid: int = 701
    M_grid: int = 30
    x_int_max: float = 10.0
    N_int: int = 2500
    x_plot_max: float = 5.0
    N_plot: int = 1200
    sparsity_tol: float = 1e-10

    @property
    def RL(self):
        return -0.5 * self.R

    @property
    def RR(self):
        return 0.5 * self.R

    @property
    def ns(self):
        return range(self.N_MAX + 1)

    @property
    def Enuc(self):
        if self.potential == "delta":
            return self.ZL * self.ZR / self.R
        return self.ZL * self.ZR / np.sqrt(self.R**2 + self.delta**2)


class HermiteBasis:
    def __init__(self, p: ModelParams):
        self.p = p

    def hermite(self, n, u):
        u = np.asarray(u, dtype=float)
        if n < 0:
            return np.zeros_like(u)
        if n == 0:
            return np.ones_like(u)
        if n == 1:
            return 2 * u

        h0, h1 = np.ones_like(u), 2 * u
        for k in range(1, n):
            h0, h1 = h1, 2 * u * h1 - 2 * k * h0
        return h1

    def norm(self, n):
        a = self.p.alpha
        return (np.sqrt(2*a) / (2**n * float(factorial(n)) * np.sqrt(np.pi)))**0.5

    def gto(self, x, c, n):
        a = self.p.alpha
        x = np.asarray(x, dtype=float)
        u = np.sqrt(a) * (x - c)
        return self.norm(n) * self.hermite(n, u) * np.exp(-a * (x - c) ** 2)

    def dgto(self, x, c, n):
        a = self.p.alpha
        x = np.asarray(x, dtype=float)
        u = np.sqrt(a) * (x - c)
        h = self.hermite(n, u)
        hm = self.hermite(n - 1, u)
        return self.norm(n) * (2 * n * np.sqrt(a) * hm - 2 * a * (x - c) * h) * np.exp(-a * (x - c) ** 2)

    def cut_gto(self, x, c, n, side):
        y = self.gto(x, c, n)
        return np.where(x <= 0, y, 0.0) if side == "L" else np.where(x >= 0, y, 0.0)

    def dcut_gto(self, x, c, n, side):
        y = self.dgto(x, c, n)
        return np.where(x <= 0, y, 0.0) if side == "L" else np.where(x >= 0, y, 0.0)

    def full_raw(self):
        return [(n, c) for c in (self.p.RL, self.p.RR) for n in self.p.ns]

    def eval_full_raw(self, x):
        raw = self.full_raw()
        phi = np.array([self.gto(x, c, n) for n, c in raw])
        dphi = np.array([self.dgto(x, c, n) for n, c in raw])
        return raw, phi, dphi

    def eval_cut_raw(self, x, c, side):
        phi = np.array([self.cut_gto(x, c, n, side) for n in self.p.ns])
        dphi = np.array([self.dcut_gto(x, c, n, side) for n in self.p.ns])
        return phi, dphi


def lowdin_basis(S, tol=1e-8):
    vals, vecs = eigh(S)
    keep = vals > tol
    return vecs[:, keep] / np.sqrt(vals[keep]), vals, keep


class SingletCI:
    @staticmethod
    def csf_terms(p, q):
        if p == q:
            return [(p, q, 1.0)]
        a = 1 / np.sqrt(2)
        return [(p, q, a), (q, p, a)]

    @staticmethod
    def eri(phi, x, p: ModelParams):
        w = np.gradient(x)
        if p.potential == "delta":
            return np.einsum("ax,bx,cx,dx,x->abcd", phi, phi, phi, phi, w, optimize=True)
        if p.potential == "soft":
            W = 1 / np.sqrt((x[:, None] - x[None, :]) ** 2 + p.delta**2)
            return np.einsum("ax,by,xy,cx,dy,x,y->abcd", phi, phi, W, phi, phi, w, w, optimize=True)
        raise ValueError("potential must be 'delta' or 'soft'")

    @staticmethod
    def sparsity(J, tol=1e-10):
        nz = np.count_nonzero(np.abs(J) > tol)
        return {"nonzero": nz, "total": J.size, "density": nz / J.size}

    @classmethod
    def from_mos(cls, eps, phi, x, p: ModelParams, max_orbitals=None):
        M = len(eps) if max_orbitals is None else min(max_orbitals, len(eps))
        eps, phi = eps[:M], phi[:M]
        J = cls.eri(phi, x, p)
        pairs, H = _pair_basis(M), None
        H = np.zeros((len(pairs), len(pairs)))

        for A, (p1, q1) in enumerate(pairs):
            for B, (r1, s1) in enumerate(pairs):
                val = 0.0
                for a, b, ca in cls.csf_terms(p1, q1):
                    for c, d, cb in cls.csf_terms(r1, s1):
                        one = (eps[a] + eps[b]) * (a == c) * (b == d)
                        val += ca * cb * (one + J[a, b, c, d])
                H[A, B] = val

        E, C = eigh(H)
        return _ci_result(E, C, pairs, H, J, phi, x, p, eps=eps)

    @classmethod
    def from_one_body(cls, H1, phi, x, p: ModelParams):
        M = H1.shape[0]
        J = cls.eri(phi, x, p)
        pairs = _pair_basis(M)
        H = np.zeros((len(pairs), len(pairs)))

        for A, (p1, q1) in enumerate(pairs):
            for B, (r1, s1) in enumerate(pairs):
                val = 0.0
                for a, b, ca in cls.csf_terms(p1, q1):
                    for c, d, cb in cls.csf_terms(r1, s1):
                        one = H1[a, c] * (b == d) + H1[b, d] * (a == c)
                        val += ca * cb * (one + J[a, b, c, d])
                H[A, B] = val

        E, C = eigh(H)
        return _ci_result(E, C, pairs, H, J, phi, x, p)

    @staticmethod
    def density(ci, state=0):
        phi, pairs = ci["Phi"], ci["pairs"]
        A = np.zeros((len(phi), len(phi)))

        for c, (p, q) in zip(ci["C_all"][:, state], pairs):
            if p == q:
                A[p, q] += c
            else:
                A[p, q] += c / np.sqrt(2)
                A[q, p] += c / np.sqrt(2)

        gamma = 2 * A @ A.T
        return np.einsum("px,qx,pq->x", phi, phi, gamma, optimize=True)


def _pair_basis(M):
    return [(i, j) for i in range(M) for j in range(i, M)]


def _ci_result(E, C, pairs, H, J, phi, x, p, eps=None):
    out = dict(E_elec=E[0], E_tot=E[0] + p.Enuc, E_all=E, C_all=C, C=C[:, 0],
               pairs=pairs, H2=H, J=J, Phi=phi, x=x)
    if eps is not None:
        out["eps"] = eps
    return out


def ionic_covalent_weights(ci, basis_info, state=0):
    W = {"LL": 0.0, "LR": 0.0, "RR": 0.0}
    for coeff, (i, j) in zip(ci["C_all"][:, state], ci["pairs"]):
        pi, pj = basis_info[i]["p"], basis_info[j]["p"]
        key = "LL" if pi == pj == "L" else "RR" if pi == pj == "R" else "LR"
        W[key] += coeff**2
    s = sum(W.values())
    return {k: v / s for k, v in W.items()}


def print_ionic_covalent(ci, basis_info, label=""):
    W = ionic_covalent_weights(ci, basis_info)
    print("\n" + "═" * 70)
    print(f"Ionic / covalent weights {label}")
    print("═" * 70)
    print(f"LL  ionic left  : {W['LL']:.8f}")
    print(f"LR  covalent    : {W['LR']:.8f}")
    print(f"RR  ionic right : {W['RR']:.8f}")
    print(f"sum             : {sum(W.values()):.8f}")
    return W


def split_density(ci, state=0):
    rho = SingletCI.density(ci, state)
    x, w = ci["x"], np.gradient(ci["x"])
    nL = np.sum(rho[x <= 0] * w[x <= 0])
    nR = np.sum(rho[x >= 0] * w[x >= 0])
    return {"N_L": nL, "N_R": nR, "N_tot": nL + nR}


class BaseModel:
    name = "Base"

    def __init__(self, p: ModelParams):
        self.p = p
        self.x = np.linspace(-p.x_int_max, p.x_int_max, p.N_int)
        self.x_plot = np.linspace(-p.x_plot_max, p.x_plot_max, p.N_plot)
        self.basis = HermiteBasis(p)

    def v_ne(self, x):
        p = self.p
        if p.potential != "soft":
            raise ValueError("delta nuclei are not represented by a smooth v_ne(x)")
        return -p.ZL / np.sqrt((x - p.RL) ** 2 + p.delta**2) - p.ZR / np.sqrt((x - p.RR) ** 2 + p.delta**2)

    def at(self, y, x0):
        return float(np.interp(x0, self.x, y))

    def run(self):
        self.build_basis()
        self.solve_one_body()
        self.eri_sparsity()
        self.solve_ci()
        return self

    def solve_ci(self):
        self.ci = SingletCI.from_mos(self.eps, self.Phi_mo, self.x, self.p)
        return self.ci

    def eri_sparsity(self):
        phi = getattr(self, "Phi_basis", self.Phi_mo)
        self.J_basis = SingletCI.eri(phi, self.x, self.p)
        self.sparsity = SingletCI.sparsity(self.J_basis, self.p.sparsity_tol)
        return self.sparsity

    def summary(self):
        s = self.sparsity
        print(f"{self.name:<12} | E1={self.eps[0]: .8f} | E2={self.ci['E_tot']: .8f} | "
              f"dim={len(self.ci['pairs']):4d} | ERI density={s['density']:.3e}")

    @staticmethod
    def fix_signs(phi, x, x_ref=None):
        phi = phi.copy()
        for k, y in enumerate(phi):
            i = np.argmax(np.abs(y)) if x_ref is None else np.argmin(np.abs(x - x_ref))
            if y[i] < 0:
                phi[k] *= -1
        return phi


class GridReference(BaseModel):
    name = "Reference"

    def __init__(self, p):
        super().__init__(p)
        self.x = np.linspace(-p.L_grid, p.L_grid, p.N_grid)
        self.x_plot = np.linspace(-p.x_plot_max, p.x_plot_max, p.N_plot)

    def build_basis(self):
        p, x = self.p, self.x
        dx, N = x[1] - x[0], len(x)
        T = np.diag(np.ones(N) / dx**2)
        T += np.diag(-0.5 * np.ones(N - 1) / dx**2, 1)
        T += np.diag(-0.5 * np.ones(N - 1) / dx**2, -1)

        if p.potential == "delta":
            V = np.zeros(N)
            V[np.argmin(abs(x - p.RL))] += -p.ZL / dx
            V[np.argmin(abs(x - p.RR))] += -p.ZR / dx
        else:
            V = self.v_ne(x)

        self.H1_grid = T + np.diag(V)
        return self

    def solve_one_body(self):
        eps, C = eigh(self.H1_grid)
        M, dx = min(self.p.M_grid, len(eps)), self.x[1] - self.x[0]
        self.eps = eps[:M]
        self.Phi_mo = (C[:, :M] / np.sqrt(dx)).T
        self.Phi_mo = self.fix_signs(self.Phi_mo, self.x, self.p.RL)
        self.Phi_basis = self.Phi_mo
        return self.eps, self.Phi_mo


class HG(BaseModel):
    name = "HG"

    def build_basis(self):
        p, x = self.p, self.x
        self.raw, phi_raw, dphi_raw = self.basis.eval_full_raw(x)
        w = np.gradient(x)

        S = np.einsum("mx,nx,x->mn", phi_raw, phi_raw, w)
        T = 0.5 * np.einsum("mx,nx,x->mn", dphi_raw, dphi_raw, w)

        if p.potential == "delta":
            V = np.array([[-p.ZL * self.at(pm, p.RL) * self.at(pn, p.RL)
                           -p.ZR * self.at(pm, p.RR) * self.at(pn, p.RR)
                           for pn in phi_raw] for pm in phi_raw])
        else:
            V = np.einsum("mx,nx,x,x->mn", phi_raw, phi_raw, self.v_ne(x), w)

        self.X, self.S_eval, self.keep = lowdin_basis(S, p.EPS_TOL)
        self.Phi_basis = self.X.T @ phi_raw
        self.H1 = self.X.T @ (T + V) @ self.X
        return self

    def solve_one_body(self):
        self.eps, self.C1 = eigh(self.H1)
        self.Phi_mo = self.C1.T @ self.Phi_basis
        self.Phi_mo = self.fix_signs(self.Phi_mo, self.x, self.p.RL)
        return self.eps, self.Phi_mo


class DGHG(BaseModel):
    name = "DG-HG"

    def domain_basis(self, c, side):
        phi, dphi = self.basis.eval_cut_raw(self.x, c, side)
        S = np.einsum("mx,nx,x->mn", phi, phi, np.gradient(self.x))
        X, vals, keep = lowdin_basis(S, self.p.EPS_TOL)
        return X, phi, dphi, vals, keep

    def build_basis(self):
        p = self.p
        UL, rawL, drawL, self.SL, self.keepL = self.domain_basis(p.RL, "L")
        UR, rawR, drawR, self.SR, self.keepR = self.domain_basis(p.RR, "R")

        phiL, dphiL = UL.T @ rawL, UL.T @ drawL
        phiR, dphiR = UR.T @ rawR, UR.T @ drawR
        self.nL, self.nR = len(phiL), len(phiR)
        self.Phi_basis = np.vstack([phiL, phiR])
        self.basis_info = ([{"p": "L", "k": k} for k in range(self.nL)] +
                           [{"p": "R", "k": k} for k in range(self.nR)])

        self.H1 = self._kinetic(phiL, dphiL, phiR, dphiR, UL, UR)
        self.H1 += self._potential()
        return self

    def _kinetic(self, phiL, dphiL, phiR, dphiR, UL, UR):
        p, x = self.p, self.x
        N, w = self.nL + self.nR, np.gradient(x)
        T = np.zeros((N, N))
        T[:self.nL, :self.nL] = 0.5 * np.einsum("mx,nx,x->mn", dphiL, dphiL, w)
        T[self.nL:, self.nL:] = 0.5 * np.einsum("mx,nx,x->mn", dphiR, dphiR, w)

        valL, derL = self.traces_at_zero(p.RL, UL)
        valR, derR = self.traces_at_zero(p.RR, UR)

        def jump(i):
            b = self.basis_info[i]
            return valL[b["k"]] if b["p"] == "L" else -valR[b["k"]]

        def avg_grad(i):
            b = self.basis_info[i]
            return 0.5 * (derL[b["k"]] if b["p"] == "L" else derR[b["k"]])

        C = np.zeros_like(T)
        P = np.zeros_like(T)
        for i in range(N):
            for j in range(N):
                C[i, j] = -0.5 * (jump(i) * avg_grad(j) + avg_grad(i) * jump(j))
                P[i, j] = p.alpha_pen * jump(i) * jump(j)
        return T + C + P

    def _potential(self):
        p, x = self.p, self.x
        if p.potential == "delta":
            return np.array([[-p.ZL * self.at(pm, p.RL) * self.at(pn, p.RL)
                              -p.ZR * self.at(pm, p.RR) * self.at(pn, p.RR)
                              for pn in self.Phi_basis] for pm in self.Phi_basis])
        return np.einsum("mx,nx,x,x->mn", self.Phi_basis, self.Phi_basis, self.v_ne(x), np.gradient(x))

    def traces_at_zero(self, c, X):
        f0 = np.array([self.basis.gto(0.0, c, n) for n in self.p.ns])
        df0 = np.array([self.basis.dgto(0.0, c, n) for n in self.p.ns])
        return X.T @ f0, X.T @ df0

    def solve_one_body(self):
        self.eps, self.C1 = eigh(self.H1)
        self.Phi_mo = self.C1.T @ self.Phi_basis
        self.Phi_mo = self.fix_signs(self.Phi_mo, self.x, self.p.RL)
        return self.eps, self.Phi_mo

    def solve_local_ci(self):
        self.ci_local = SingletCI.from_one_body(self.H1, self.Phi_basis, self.x, self.p)
        self.weights = ionic_covalent_weights(self.ci_local, self.basis_info)
        return self.ci_local, self.weights


def run_models(models):
    print("\n" + "═" * 80)
    print("Running models")
    print("═" * 80)

    for m in models:
        print(f"\n--- {m.name} / {m.p.potential} / ZL={m.p.ZL}, ZR={m.p.ZR} ---")
        m.run()
        m.summary()

        if isinstance(m, DGHG):
            ci, W = m.solve_local_ci()
            print(f"DG local CI singlet energy : {ci['E_tot']:.10f}")
            print(f"LL={W['LL']:.6f}  LR={W['LR']:.6f}  RR={W['RR']:.6f}")
    return models


def print_energy_table(models):
    print("\n" + "═" * 95)
    print("Energy comparison")
    print("═" * 95)
    print(f"{'Model':<14} {'pot':<8} {'Norb':>6} {'FCI dim':>8} {'FCI tot':>16}")
    print("-" * 95)

    for m in models:
        print(f"{m.name:<14} {m.p.potential:<8} {len(m.eps):>6d} "
              f"{len(m.ci['pairs']):>8d} {m.ci['E_tot']:>16.10f}")
        if isinstance(m, DGHG) and hasattr(m, "ci_local"):
            print(f"{'DG local':<14} {m.p.potential:<8} {len(m.Phi_basis):>6d} "
                  f"{len(m.ci_local['pairs']):>8d} {m.ci_local['E_tot']:>16.10f}")