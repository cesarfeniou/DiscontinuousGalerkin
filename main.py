#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import replace

import numpy as np

from core import (
    ModelParams,
    GridReference,
    HG,
    DGHG,
    run_models,
    print_energy_table,
    split_density,
)
from plots import (
    plot_orbital,
    plot_density,
    plot_orbital_convergence,
    plot_charge_scan,
    plot_orbitals_vs_ZL,
    plot_density_vs_ZL,
    plot_penalty_scan,
)


METHODS = {
    "ref": GridReference,
    "hg": HG,
    "dg": DGHG,
}


def run(params, methods=("ref", "hg", "dg"), table=False):
    models = run_models([METHODS[name](params) for name in methods])

    if table:
        print_energy_table(models)

    return dict(zip(methods, models))


def default_params():
    return ModelParams(
        potential="soft",
        R=2.0,
        ZL=1.0,
        ZR=1.0,
        delta=0.2,
        alpha=1.5,
        alpha_pen=15.0,
        N_MAX=50,
        M_grid=55,
        N_grid=801,
        N_int=2500,
    )


def standard_run(params):
    out = run(params, table=True)
    models = list(out.values())

    plot_orbital(models, k=0)
    plot_orbital(models, k=1)
    plot_density(models, state=0)

    print(split_density(out["dg"].ci_local))

    return out


def basis_convergence(params, e_ref):
    rows = []

    for nmax in range(2, 29, 2):
        out = run(replace(params, N_MAX=nmax), methods=("hg", "dg"))

        for key, label in (("hg", "HG"), ("dg", "DG-HG")):
            model = out[key]
            rows.append({
                "method": label,
                "NMAX": nmax,
                "Norb": len(model.eps),
                "E_fci": model.ci["E_tot"],
                "run": model,
            })

    plot_orbital_convergence(rows, e_ref)


def charge_asymmetry(params):
    rows = []

    for ZL in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
        dg = run(
            replace(params, ZL=ZL, N_MAX=20),
            methods=("dg",),
        )["dg"]

        rows.append({
            "ZL": ZL,
            "ZR": params.ZR,
            "E": dg.ci_local["E_tot"],
            "dg": dg,
            **dg.weights,
            **split_density(dg.ci_local),
        })

    plot_charge_scan(rows)
    plot_orbitals_vs_ZL(rows)
    plot_density_vs_ZL(rows)


def penalty_convergence(params, e0_ref):
    alphas = np.array([1, 2, 4, 5, 7, 10, 25, 50, 100, 1000], dtype=float)
    scan = {}

    for nmax in (10, 14, 15, 20):
        models = [
            run(
                replace(params, N_MAX=nmax, alpha_pen=alpha_pen),
                methods=("dg",),
            )["dg"]
            for alpha_pen in alphas
        ]

        energies = np.array([model.eps[0] for model in models])

        scan[nmax] = {
            "alphas": alphas,
            "E0": energies,
            "err": energies - e0_ref,
            "models": models,
        }

    plot_penalty_scan(scan)


def main():
    params = default_params()

    out = standard_run(params)

    e_ref = out["ref"].ci["E_tot"]
    e0_ref = out["ref"].eps[0]

    basis_convergence(params, e_ref)
    charge_asymmetry(params)
    penalty_convergence(params, e0_ref)


if __name__ == "__main__":
    main()
