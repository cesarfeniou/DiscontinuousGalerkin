#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jun 17 15:52:32 2026

@author: cesar
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from core import SingletCI

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 1.6,
})

C_REF = "#2F2F2F"
C_FULL = "#1B9E77"
C_DG = "#D95F02"
C_FULL_RHO = "#2E86DE"
C_DG_RHO = "#C05CBF"


def _finish(ax, p, name=None, dpi=200):
    ax.tick_params(labelsize=14, width=1.4)
    for s in ax.spines.values():
        s.set_linewidth(1.6)
    if name:
        plt.savefig(name, dpi=dpi)
    plt.show()
    


def plot_orbital(models, k=0, save=True):
    styles = {
        "Reference": (C_REF, "-", 2.9),
        "HG": (C_FULL, "--", 2.4),
        "DG-HG": (C_DG, "-.", 2.4),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        if k >= len(m.eps):
            continue
        y = np.interp(m.x_plot, m.x, m.Phi_mo[k])
        c, ls, lw = styles.get(m.name, ("k", "-", 2))
        ax.plot(m.x_plot, y, c=c, ls=ls, lw=lw, label=m.name)
        print(rf"{m.name}, $\epsilon_{k}={m.eps[k]:.6f}$")

    p = models[0].p
    ax.axvline(p.RL, c="0.2", ls=":", lw=2, label="Nuclei")
    ax.axvline(p.RR, c="0.2", ls=":", lw=2)
    ax.axhline(0, c="k", lw=0.5)
    ax.set(xlabel=r"$x$", ylabel=rf"$\psi_{k+1}(x)$",
           xlim=(np.floor(-p.x_plot_max), np.ceil(p.x_plot_max)))
    ax.legend(frameon=False, fontsize=12)

    name = f"orbital_{k+1}_{p.potential}_ZL{p.ZL}_ZR{p.ZR}.png" if save else None
    _finish(ax, p, name)


def plot_density(models, state=0, save=True):
    styles = {
        "Reference": (C_REF, "-", 2.9),
        "HG": (C_FULL_RHO, "--", 2.5),
        "DG-HG": (C_DG_RHO, "-.", 2.5),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        c, ls, lw = styles.get(m.name, ("k", "-", 2))
        rho = SingletCI.density(m.ci, state)
        mask = np.abs(m.ci["x"]) <= m.p.x_plot_max
        ax.plot(m.ci["x"][mask], rho[mask], c=c, ls=ls, lw=lw, label=m.name)

    p = models[0].p
    ax.axvline(p.RL, c="0.2", ls=":", lw=2, label="Nuclei")
    ax.axvline(p.RR, c="0.2", ls=":", lw=2)
    ax.set(xlabel=r"$x$", ylabel=rf"$\rho_{state}(x)$",
           xlim=(np.floor(-p.x_plot_max), np.ceil(p.x_plot_max)))
    ax.legend(frameon=False, fontsize=12)

    name = f"density_state{state}_{p.potential}_ZL{p.ZL:g}.png" if save else None
    _finish(ax, p, name)


def plot_orbital_convergence(
    rows,
    e_ref,
    keys=("E_fci",),
    labels=None,
    filenames=None,
):
    labels = labels or {
        "E_fci": r"$|E_{\rm CI}-E_{\rm ref}|$",
        "E_mp2": r"$|E_{\rm MP2}-E_{\rm ref}|$",
    }

    filenames = filenames or {
        "E_fci": "conv_fci.png",
        "E_mp2": "conv_mp2.png",
    }

    styles = {
        "HG": ("o", "-"),
        "DG-HG": ("s", "-"),
    }

    for key in keys:
        fig, ax = plt.subplots(figsize=(7, 5))
        all_x = []

        colors = {"HG": "#1b9e77", "DG-HG": "#d95f02"}

        for method in ["HG", "DG-HG"]:
            data = sorted(
                [r for r in rows if r["method"] == method and key in r],
                key=lambda r: r["Norb"],
            )

            if not data:
                continue

            x = [r["Norb"] for r in data]
            y = [abs(r[key] - e_ref) for r in data]
            marker, ls = styles[method]

            ax.plot(
                x, y,
                marker=marker, ls=ls,
                color=colors[method],
                lw=2.8, ms=6.5,
                label=method
            )

            all_x.extend(x)

        ax.set_yscale("log")
        ax.set_xlabel(r"Number of orbitals $m$", fontsize=16)
        ax.set_ylabel(labels.get(key, rf"$|{key}-E_{{\rm ref}}|$"), fontsize=18)

        if all_x:
            ax.set_xlim(np.floor(min(all_x)), np.ceil(max(all_x)))

        ax.set_ylim(top=1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=13)
        ax.tick_params(labelsize=13, width=1.4)

        for s in ax.spines.values():
            s.set_linewidth(1.6)

        plt.tight_layout()
        # plt.savefig(filenames.get(key, f"conv_{key}.png"), dpi=300)
        plt.show()


def plot_charge_scan(rows, save=True):
    ZL = np.array([r["ZL"] for r in rows])
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(ZL, [r["LL"] for r in rows], "o-", color="tab:red", lw=2.8, ms=7, label=r"$W_{LL}$")
    ax1.plot(ZL, [r["LR"] for r in rows], "s-", color="tab:gray", lw=2.8, ms=7, label=r"$W_{LR}$")
    ax1.plot(ZL, [r["RR"] for r in rows], "^-", color="tab:blue", lw=2.8, ms=7, label=r"$W_{RR}$")

    ax2.plot(ZL, [r["N_L"] for r in rows], "--o", color="tab:red", lw=2.5, ms=7, mfc="white", label=r"$N_L$")
    ax2.plot(ZL, [r["N_R"] for r in rows], "--^", color="tab:blue", lw=2.5, ms=7, mfc="white", label=r"$N_R$")

    ax1.set(xlabel=r"$Z_L$", ylabel=r"dominant-pair weight $W$", ylim=(-0.01, 1.01))
    ax2.set(ylabel=r"Electron population $N_p$", ylim=(-0.02, 2.02))
    ax2.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=False, fontsize=12)
    ax1.grid(alpha=0.4)

    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim(xmin, xmax * 1.02)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=13, width=1.4)
        for s in ax.spines.values():
            s.set_linewidth(1.6)

    plt.tight_layout()
    if save:
        plt.savefig("ionic_weights_vs_ZL.png", dpi=300)
    plt.show()


def plot_orbitals_vs_ZL(rows, ks=(0, 1), save=True):
    cmap = LinearSegmentedColormap.from_list("blue_gray_red", ["blue", "0.5", "red"])
    colors = cmap(np.linspace(0.05, 0.95, len(rows)))

    for k in ks:
        fig, ax = plt.subplots(figsize=(7, 5))
        for r, c in zip(rows, colors):
            dg = r["dg"]
            y = np.interp(dg.x_plot, dg.x, dg.Phi_mo[k])
            ax.plot(dg.x_plot, y, c=c, lw=2.6, label=rf"$Z_L={r['ZL']:g}$")

        p = rows[0]["dg"].p
        ax.axvline(p.RL, c="0.2", ls=":", lw=2, label="Nuclei")
        ax.axvline(p.RR, c="0.2", ls=":", lw=2)
        ax.axhline(0, c="k", lw=0.5)
        ax.set(xlabel=r"$x$", ylabel=rf"$\psi_{k}(x)$",
               xlim=(np.floor(-p.x_plot_max), np.ceil(p.x_plot_max)))
        ax.legend(frameon=False, fontsize=12)
        plt.tight_layout()
        _finish(ax, p, f"orbital{k}_vs_ZL.png" if save else None, dpi=300)


def plot_density_vs_ZL(rows, save=True):
    cmap = LinearSegmentedColormap.from_list("blue_gray_red", ["k", "red"])
    colors = cmap(np.linspace(0.05, 0.95, len(rows)))
    fig, ax = plt.subplots(figsize=(7, 5))

    for r, c in zip(rows, colors):
        rho = SingletCI.density(r["dg"].ci_local)
        ax.plot(r["dg"].x, rho, c=c, lw=2.6, label=rf"$Z_L={r['ZL']:g}$")

    p = rows[0]["dg"].p
    ax.axvline(p.RL, c="0.2", ls=":", lw=2, label="Nuclei")
    ax.axvline(p.RR, c="0.2", ls=":", lw=2)
    ax.set(xlabel=r"$x$", ylabel=r"$\rho_0(x)$", xlim=(-5, 5))
    ax.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    _finish(ax, p, "density_vs_ZL.png" if save else None, dpi=300)


def plot_penalty_scan(scan, selected=(5, 7, 15), Nplot=10, save=True):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import NullLocator

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(scan)))

    for (NMAX, d), c in zip(sorted(scan.items()), colors):
        ax.plot(d["alphas"], d["err"], c=c, marker="o", lw=2.6, ms=5.5, label=rf"$n_{{\max}}={NMAX}$")

    ax.axhline(0, color="0.3", lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(-600, 60)
    ax.set(xlabel=r"$\alpha_{\rm pen}$", ylabel=r"$\varepsilon_1-\varepsilon_1^{\rm ref}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=13, ncol=1, loc="lower right", bbox_to_anchor=(0.98, 0.02))

    axins = inset_axes(ax, width="90%", height="100%", bbox_to_anchor=(0.55, 0.5, 0.42, 0.34), bbox_transform=ax.transAxes)
    for (NMAX, d), c in zip(sorted(scan.items()), colors):
        mask = d["err"] > 0
        axins.plot(d["alphas"][mask], d["err"][mask], c=c, marker="o", lw=2.0, ms=4)

    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.set_xlim(5, 1200)
    axins.set_ylim(1e-3, 1e-2)
    axins.set_yticks([1e-3, 1e-2])
    axins.yaxis.set_minor_locator(NullLocator())
    axins.grid(which="major", alpha=0.8)
    axins.tick_params(labelsize=11, direction="in")

    if save:
        fig.savefig("alpha_pen_scan_basis_delta_error.png", dpi=300, bbox_inches="tight")
    plt.show()

    d = scan[Nplot]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = ["red", "green", "k"]

    for i, a_sel in enumerate(selected):
        idx = np.argmin(np.abs(d["alphas"] - a_sel))
        m = d["models"][idx]
        y = np.interp(m.x_plot, m.x, m.Phi_mo[0])
        ax.plot(m.x_plot, y, color=colors[i % len(colors)], ls="--" if i == len(selected) - 1 else "-", lw=1.0 if i == len(selected) - 1 else 2.5, label=rf"$\alpha_{{\rm pen}}={d['alphas'][idx]:g}$")

    p = d["models"][0].p
    ax.axvline(p.RL, c="0.2", ls=":", lw=1, label="Nuclei")
    ax.axvline(p.RR, c="0.2", ls=":", lw=1)
    ax.axhline(0, c="k", lw=0.5)
    ax.set(xlabel=r"$x$", ylabel=r"$\psi_1(x)$", xlim=(np.floor(-p.x_plot_max), np.ceil(p.x_plot_max)))
    ax.legend(frameon=False, fontsize=12)

    if save:
        fig.savefig("alpha_pen_scan_delta_orbitals.png", dpi=300, bbox_inches="tight")
    plt.show()