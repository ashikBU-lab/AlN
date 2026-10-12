"""
Write a small *synthetic* VASP OUTCAR for wurtzite AlN, to check that the
installation and the fitting pipeline work before using real DFT data.

Energies/forces come from a simple Al-N Morse pair model, NOT from DFT,
so the potential fitted to this file is meaningless for real simulations.

    python tests/make_test_outcar.py test_data/OUTCAR
"""
import argparse
import os

import numpy as np
from ase.build import bulk
from ase.neighborlist import neighbor_list

# Morse pair model (D [eV], alpha [1/A], r0 [A]) with a smooth cutoff
MORSE = {("Al", "N"): (2.0, 1.6, 1.89),
         ("Al", "Al"): (0.30, 1.2, 2.95),
         ("N", "N"): (0.15, 1.5, 3.05)}
E_ATOM = {"Al": -0.10, "N": -0.70}  # per-atom offsets (~ -7.4 eV/atom total)
R_CUT, R_ON = 5.5, 4.5


def _switch(r):
    x = np.clip((r - R_ON) / (R_CUT - R_ON), 0.0, 1.0)
    s = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
    ds = (-30 * x**2 + 60 * x**3 - 30 * x**4) / (R_CUT - R_ON)
    return s, ds


def energy_forces(atoms):
    i, j, d, D = neighbor_list("ijdD", atoms, R_CUT)
    sym = np.array(atoms.get_chemical_symbols())
    energy = sum(E_ATOM[s] for s in sym)
    forces = np.zeros((len(atoms), 3))
    for pair, (de, a, r0) in MORSE.items():
        mask = ((sym[i] == pair[0]) & (sym[j] == pair[1])) | \
               ((sym[i] == pair[1]) & (sym[j] == pair[0]))
        r, vec = d[mask], D[mask]
        ex = np.exp(-a * (r - r0))
        e = de * (ex**2 - 2 * ex)
        de_dr = de * (-2 * a * ex**2 + 2 * a * ex)
        s, ds = _switch(r)
        energy += 0.5 * np.sum(e * s)  # each pair is listed twice
        dE = (de_dr * s + e * ds)[:, None] * vec / r[:, None]
        np.add.at(forces, i[mask], dE)  # F_i = -dE/dr_i = +dE/dr * r_ij_hat
    return energy, forces


def make_frames(n_frames, seed):
    rng = np.random.default_rng(seed)
    base = bulk("AlN", "wurtzite", a=3.11, c=4.98, u=0.382) * (3, 3, 2)
    frames = []
    for k in range(n_frames):
        atoms = base.copy()
        strain = np.eye(3) + rng.uniform(-0.04, 0.04, (3, 3)) * \
            np.array([[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]])
        atoms.set_cell(atoms.cell @ strain, scale_atoms=True)
        atoms.rattle(stdev=0.02 + 0.10 * k / n_frames,
                     seed=int(rng.integers(1e6)))
        frames.append(atoms)
    return frames


def write_outcar(path, frames, nelm=60, unconverged_steps=()):
    atoms0 = frames[0]
    # VASP order: grouped by species, as in the POSCAR
    order = np.argsort([0 if s == "Al" else 1
                        for s in atoms0.get_chemical_symbols()], kind="stable")
    n_al = sum(1 for s in atoms0.get_chemical_symbols() if s == "Al")
    n_n = len(atoms0) - n_al
    lines = [
        " vasp.6.4.2 (synthetic test file written by make_test_outcar.py)",
        " POTCAR:    PAW_PBE Al 04Jan2001",
        " POTCAR:    PAW_PBE N 08Apr2002",
        " POTCAR:    PAW_PBE Al 04Jan2001",
        " POTCAR:    PAW_PBE N 08Apr2002",
        f"   ions per type =              {n_al:d}  {n_n:d}",
        f"   NELM   =     {nelm:d};   NELMIN=  2; NELMDL= -5     # of ELM steps",
        "   ISPIN  =      1    spin polarized calculation?",
        "   k-points           NKPTS =      1   k-points in BZ     "
        "NKDIM =      1   number of bands    NBANDS=    256",
        " k-points in reciprocal lattice and weights: synthetic",
        "   0.00000000  0.00000000  0.00000000       1.000",
        "",
    ]
    for step, atoms in enumerate(frames, start=1):
        atoms = atoms[order]
        energy, forces = energy_forces(atoms)
        n_scf = nelm if step in unconverged_steps else 12
        for it in range(1, n_scf + 1):
            lines.append(f"----------------------------------------- "
                         f"Iteration {step:6d}({it:4d})  "
                         f"---------------------------------------")
        lines += ["", "  FORCE on cell =-STRESS in cart. coord.  units (eV):",
                  "  in kB       0.00000     0.00000     0.00000     "
                  "0.00000     0.00000     0.00000", "",
                  "      direct lattice vectors                 "
                  "reciprocal lattice vectors"]
        rec = atoms.cell.reciprocal()
        for v, w in zip(atoms.cell, rec):
            lines.append("  " + "".join(f"{x:13.9f}" for x in v) + "  "
                         + "".join(f"{x:13.9f}" for x in w))
        lines += ["", " POSITION                                       "
                  "TOTAL-FORCE (eV/Angst)",
                  " " + "-" * 83]
        for p, f in zip(atoms.positions, forces):
            lines.append("  " + "".join(f"{x:13.5f}" for x in p) + "    "
                         + "".join(f"{x:14.6f}" for x in f))
        lines += [" " + "-" * 83,
                  f"    total drift:                                "
                  f"{0.0:14.6f}{0.0:14.6f}{0.0:14.6f}", "", "",
                  "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
                  "  ---------------------------------------------------",
                  f"  free  energy   TOTEN  = {energy:18.8f} eV", "",
                  f"  energy  without entropy= {energy:18.8f}  "
                  f"energy(sigma->0) = {energy:18.8f}", "", ""]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("output", nargs="?", default="test_data/OUTCAR")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    frames = make_frames(args.frames, args.seed)
    # ionic step 5 is written as if its SCF hit NELM (should be dropped)
    write_outcar(args.output, frames, unconverged_steps=(5,))
    print(f"Wrote {len(frames)} synthetic AlN frames "
          f"({len(frames[0])} atoms each) to {args.output}")


if __name__ == "__main__":
    main()
