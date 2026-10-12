"""
Check a fitted .uf3 potential inside LAMMPS: recompute energies and forces of
the fitted frames with "pair_style uf3" and compare them with the DFT values
and with the UF3 python model (.json).

Needs the LAMMPS python module with the ML-UF3 package
(e.g. "pip install lammps" on Linux/macOS).

    python tests/check_uf3_in_lammps.py AlN_uf3_potential
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def lammps_energy_forces(lmp, atoms, specorder):
    """Energy and forces of one ase.Atoms with an already-defined pair."""
    from ase.calculators.lammps import Prism
    prism = Prism(atoms.cell)
    xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism()
    lmp.command("clear")
    lmp.commands_list(["units metal", "atom_style atomic",
                       "boundary p p p", "atom_modify map array sort 0 0"])
    lmp.command(f"region box prism 0 {xhi} 0 {yhi} 0 {zhi} {xy} {xz} {yz} "
                "units box")
    lmp.command(f"create_box {len(specorder)} box")
    pos = prism.vector_to_lammps(atoms.positions, wrap=True)
    types = [specorder.index(s) + 1 for s in atoms.get_chemical_symbols()]
    lmp.create_atoms(len(atoms), list(range(1, len(atoms) + 1)), types,
                     pos.flatten().tolist())
    for k in range(len(specorder)):
        lmp.command(f"mass {k + 1} 1.0")
    return prism


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("result_dir", help="output folder of fit_AlN_uf3.py")
    parser.add_argument("--name", default="AlN")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    from lammps import lammps
    from uf3.forcefield import calculator
    from uf3.regression import least_squares

    folder = os.path.abspath(args.result_dir)
    pot = os.path.join(folder, f"{args.name}.uf3")
    df = pd.read_pickle(os.path.join(folder, "dataset.pkl"))
    model = least_squares.WeightedLinearModel.from_json(
        os.path.join(folder, f"{args.name}.json"))
    calc = calculator.UFCalculator(model)
    solutions = least_squares.arrange_coefficients(model.coefficients,
                                                   model.bspline_config)
    one_body = {el: float(np.ravel(solutions[el])[0])
                for el in model.bspline_config.element_list}
    degree = model.bspline_config.degree
    specorder = ["Al", "N"]

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    idx = np.linspace(0, len(df) - 1, min(args.frames, len(df))).astype(int)
    de_dft, de_py, df_dft, df_py = [], [], [], []
    for i in idx:
        atoms = df["geometry"].iloc[i].copy()
        n = len(atoms)
        prism = lammps_energy_forces(lmp, atoms, specorder)
        lmp.commands_list([f"pair_style uf3 {degree}",
                           f'pair_coeff * * "{pot}" {" ".join(specorder)}',
                           "run 0"])
        e_lmp = lmp.get_thermo("pe")
        f_lmp = np.ctypeslib.as_array(
            lmp.gather_atoms("f", 1, 3)).reshape(-1, 3)
        f_lmp = prism.vector_to_ase(f_lmp)
        atoms.calc = calc
        # the .uf3 file has no 1-body (per-atom constant) terms
        e_1b = sum(one_body[s] for s in atoms.get_chemical_symbols())
        e_py = atoms.get_potential_energy() - e_1b
        f_py = atoms.get_forces(apply_constraint=False)
        f_dft = np.stack([df[c].iloc[i] for c in ("fx", "fy", "fz")], 1)
        de_dft.append((e_lmp + e_1b - df["energy"].iloc[i]) / n)
        de_py.append((e_lmp - e_py) / n)
        df_dft.append(f_lmp - f_dft)
        df_py.append(f_lmp - f_py)
    de_dft, de_py = np.array(de_dft), np.array(de_py)
    df_dft, df_py = np.concatenate(df_dft), np.concatenate(df_py)
    print(f"frames checked: {len(idx)}")
    print(f"1-body energies (not in .uf3, constant per atom): {one_body}")
    print(f"LAMMPS vs python UF3 model: max |dE| {np.abs(de_py).max():.2e} "
          f"eV/atom, max |dF| {np.abs(df_py).max():.2e} eV/A")
    print(f"LAMMPS (+1-body) vs DFT: energy RMSE "
          f"{np.sqrt(np.mean(de_dft**2)) * 1000:.2f} meV/atom, force RMSE "
          f"{np.sqrt(np.mean(df_dft**2)):.4f} eV/A")
    ok = np.abs(df_py).max() < 1e-4 and np.abs(de_py).max() < 1e-5
    print("RESULT:", "OK - LAMMPS reproduces the UF3 model" if ok
          else "MISMATCH between LAMMPS and the python model")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
