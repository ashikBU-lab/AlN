#!/usr/bin/env python
"""
Fit a UF3 interatomic potential for AlN from VASP OUTCAR file(s) with
UF3Tools (https://github.com/uf-chenlab/UF3Tools) and write the LAMMPS
potential file (AlN.uf3) for "pair_style uf3".

Example (Anaconda Prompt on Windows):

    conda activate uf3_aln
    python fit_AlN_uf3.py ^
        --outcar   "C:\\Users\\Mdashikur_Rahman1\\Desktop\\conda files\\UF3TOOL\\code" ^
        --uf3tools "C:\\Users\\Mdashikur_Rahman1\\Desktop\\conda files\\UF3TOOL\\code"

--outcar takes OUTCAR files and/or folders (folders are searched, including
sub-folders, for files whose name starts with "OUTCAR").

Pipeline (UF3Tools classes from UF3Tools/uf3_handler.py):
  OUTCAR(s) -> uf3 DataCoordinator -> StructureParser / DataFilter
  -> Featurizer (B-spline features, HDF5) -> ModelTrainer (split + fit)
  -> ModelAnalyzer (repulsion fix, RMSE, parity plots, export .json/.uf3)
"""
import argparse
import os
import re
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # write plots to files, never open windows
import matplotlib.pyplot as plt
import numpy as np

# uf3 calls DataFrame.to_hdf() positionally; pandas 2.x warns about 3.0
warnings.filterwarnings("ignore", message=".*to_hdf.*", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Folder used when --outcar / --uf3tools are not given.
DEFAULT_FOLDER = r"C:\Users\Mdashikur_Rahman1\Desktop\conda files\UF3TOOL\code"

ELEMENTS = ["Al", "N"]


def parse_args(argv=None):
    default_dir = DEFAULT_FOLDER if os.path.isdir(DEFAULT_FOLDER) else "."
    p = argparse.ArgumentParser(
        description="Fit a UF3 potential for AlN with UF3Tools and write a "
                    "LAMMPS .uf3 potential file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outcar", nargs="+", default=[default_dir],
                   help="OUTCAR file(s) and/or folder(s) containing them")
    p.add_argument("--uf3tools", default=None,
                   help="folder of the UF3Tools code (the downloaded "
                        "repository or any folder above it). Default: "
                        "search " + DEFAULT_FOLDER + ", the folder above "
                        "this script's folder and the current folder")
    p.add_argument("--output-dir", default="AlN_uf3_potential",
                   help="folder for all results")
    p.add_argument("--name", default="AlN",
                   help="base name of the potential files (AlN.uf3/.json)")
    p.add_argument("--author", default="UF3Tools",
                   help="author written in the .uf3 header (no spaces)")

    g = p.add_argument_group("data selection")
    g.add_argument("--every", type=int, default=1,
                   help="use every N-th ionic step of each OUTCAR "
                        "(e.g. 5 for long, strongly correlated AIMD runs)")
    g.add_argument("--keep-unconverged", action="store_true",
                   help="keep ionic steps whose electronic (SCF) loop hit "
                        "NELM; by default they are removed")
    g.add_argument("--max-force", type=float, default=20.0,
                   help="remove frames with any force component above this "
                        "(eV/A)")
    g.add_argument("--test-fraction", type=float, default=0.1,
                   help="fraction of data held out for testing")

    g = p.add_argument_group("UF3 model")
    g.add_argument("--degree", type=int, choices=(2, 3), default=3,
                   help="2 = pair (2-body) only, 3 = 2-body + 3-body")
    g.add_argument("--r-min-2b", type=float, default=0.001,
                   help="2-body inner cutoff (A)")
    g.add_argument("--r-max-2b", type=float, default=5.5,
                   help="2-body outer cutoff (A)")
    g.add_argument("--resolution-2b", type=int, default=20,
                   help="number of 2-body knot intervals")
    g.add_argument("--r-min-3b", type=float, default=1.5,
                   help="3-body inner cutoff (A)")
    g.add_argument("--r-max-3b", type=float, default=3.5,
                   help="3-body cutoff for r_ij and r_ik (A); r_jk uses 2x")
    g.add_argument("--resolution-3b", type=int, default=6,
                   help="3-body knot intervals for r_ij and r_ik; r_jk uses "
                        "2x")
    g.add_argument("--weight", type=float, default=0.3,
                   help="energy vs force weight (0..1, higher = energies "
                        "matter more); UF3Tools default 0.3")
    g.add_argument("--ridge-1b", type=float, default=1e-8)
    g.add_argument("--ridge-2b", type=float, default=1e-8)
    g.add_argument("--ridge-3b", type=float, default=1e-8)
    g.add_argument("--curvature-2b", type=float, default=1e-8)
    g.add_argument("--curvature-3b", type=float, default=1e-7)
    g.add_argument("--no-repulsion-fix", action="store_true",
                   help="do not make the 2-body potential repulsive below "
                        "the shortest distance in the data")
    g.add_argument("--repulsion-curvature", type=float, default=10.0,
                   help="minimum curvature (eV/A^2) of the 2-body wall that "
                        "replaces the potential below the shortest distance "
                        "in the data; larger = harder wall")

    g = p.add_argument_group("performance")
    g.add_argument("--cores", type=int,
                   default=max(1, min(4, os.cpu_count() or 1)),
                   help="processes used for featurization")
    g.add_argument("--batch-size", type=int, default=500,
                   help="frames per featurization batch")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# OUTCAR reading
# --------------------------------------------------------------------------
def find_outcars(paths):
    files = []
    for path in paths:
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                                 and d not in ("__pycache__", "site-packages"))
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.startswith("OUTCAR")
                          and not n.endswith((".py", ".ipynb"))]
        else:
            raise FileNotFoundError(f"--outcar path does not exist: {path}")
    return sorted(set(files))


_ITER_RE = re.compile(r"Iteration\s*(\d+)\s*\(\s*(\d+)\s*\)")
_NELM_RE = re.compile(r"\bNELM\s*=\s*(\d+)")


def _open_text(path):
    import bz2
    import gzip
    import lzma
    opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(
        os.path.splitext(path)[1], open)
    return opener(path, "rt", errors="replace")


def scan_outcar(outcar):
    """Return (NELM, [SCF iterations of each ionic step], is_vasp_mlff)."""
    nelm, n_scf, mlff = None, {}, False
    with _open_text(outcar) as fp:
        for line in fp:
            if "ML FREE ENERGIE" in line:
                mlff = True
            if nelm is None and "NELM" in line:
                m = _NELM_RE.search(line)
                if m:
                    nelm = int(m.group(1))
            if "Iteration" in line:
                m = _ITER_RE.search(line)
                if m:
                    ionic, elec = int(m.group(1)), int(m.group(2))
                    n_scf[ionic] = max(n_scf.get(ionic, 0), elec)
    return nelm, [n_scf[k] for k in sorted(n_scf)], mlff


def read_outcar(outcar, every=1, keep_unconverged=False):
    from ase.io import read
    nelm, n_scf, mlff = scan_outcar(outcar)
    if mlff:
        # ase would also read the steps predicted by VASP's own machine-
        # learned force field (ML_MODE = train/run), which are not DFT data
        raise ValueError("OUTCAR of a VASP machine-learning (ML_MODE) run; "
                         "its ML-predicted steps are not DFT data")
    images = read(outcar, index=":", format="vasp-out")
    if not isinstance(images, list):
        images = [images]
    keep = np.ones(len(images), dtype=bool)
    n_unconverged = 0
    if not keep_unconverged:
        if nelm is not None and len(n_scf) >= len(images):
            unconverged = np.array(n_scf[:len(images)]) >= nelm
            n_unconverged = int(unconverged.sum())
            keep &= ~unconverged
    idx = np.flatnonzero(keep)[::max(1, every)]
    return [images[i] for i in idx], len(images), n_unconverged


def load_dataframe(outcars, handler, args):
    from uf3.data import io as uf3_io
    data_coordinator = uf3_io.DataCoordinator()
    n_used = 0
    for k, outcar in enumerate(outcars):
        try:
            frames, n_total, n_bad = read_outcar(outcar, args.every,
                                                 args.keep_unconverged)
        except Exception as exc:  # not a readable OUTCAR
            print(f"  ! skipped {outcar}: {type(exc).__name__}: {exc}")
            continue
        symbols = set()
        for atoms in frames:
            symbols.update(atoms.get_chemical_symbols())
        if not symbols or not symbols <= set(ELEMENTS):
            print(f"  ! skipped {outcar}: elements {sorted(symbols)} "
                  f"are not a subset of {ELEMENTS}")
            continue
        print(f"  [{k}] {outcar}\n      {n_total} ionic steps, "
              f"{n_bad} SCF-unconverged removed, {len(frames)} used "
              f"({frames[0].get_chemical_formula()}, {len(frames[0])} atoms)")
        # Energies/forces are passed explicitly: dataframe_from_lists copies
        # the Atoms objects, and ase's Atoms.copy() drops the calculator.
        # UF3Tools index convention "ind_<n>_run_<m>__<step>": frames of the
        # same OUTCAR share the "ind_<n>" group used for the train/test split
        data_coordinator.dataframe_from_lists(
            frames, prefix=f"ind_{k}_run_0_",
            energies=[a.get_potential_energy() for a in frames],
            forces=[a.get_forces(apply_constraint=False) for a in frames])
        n_used += 1
    if n_used == 0:
        raise RuntimeError("No usable OUTCAR data found.")
    df = data_coordinator.consolidate()

    sp = handler.StructureParser(df)
    sp.append_energy_per_atom()
    data_filter = handler.DataFilter(sp.get_data())
    data_filter.filter_force(args.max_force, args.max_force, args.max_force)
    df = data_filter.get_data()
    if len(df) == 0:
        raise RuntimeError("All frames were removed by the force filter.")
    return df, n_used


def describe_data(df):
    n_atoms = int(df["size"].sum())
    forces = np.concatenate([np.concatenate(df[c].values)
                             for c in ("fx", "fy", "fz")])
    print(f"  frames: {len(df)}   atoms: {n_atoms}   "
          f"energies: {len(df)}   force components: {forces.size}")
    print(f"  energy/atom: min {df['energy_per_atom'].min():.4f}  "
          f"max {df['energy_per_atom'].max():.4f} eV/atom")
    print(f"  |force| component: mean {np.abs(forces).mean():.3f}  "
          f"max {np.abs(forces).max():.3f} eV/A")
    return n_atoms


# --------------------------------------------------------------------------
# UF3 setup / outputs
# --------------------------------------------------------------------------
def make_bspline_config(args):
    from uf3.data import composition
    from uf3.representation import bspline
    chemical_system = composition.ChemicalSystem(element_list=ELEMENTS,
                                                 degree=args.degree)
    r_min_map, r_max_map, resolution_map = {}, {}, {}
    for pair in chemical_system.interactions_map[2]:
        r_min_map[pair] = args.r_min_2b
        r_max_map[pair] = args.r_max_2b
        resolution_map[pair] = args.resolution_2b
    for trio in chemical_system.interactions_map.get(3, []):
        r_min_map[trio] = [args.r_min_3b] * 3
        r_max_map[trio] = [args.r_max_3b, args.r_max_3b, 2 * args.r_max_3b]
        resolution_map[trio] = [args.resolution_3b, args.resolution_3b,
                                2 * args.resolution_3b]
    bspline_config = bspline.BSplineBasis(
        chemical_system, r_min_map=r_min_map, r_max_map=r_max_map,
        resolution_map=resolution_map,
        leading_trim={2: 0, 3: 3}, trailing_trim={2: 3, 3: 3})
    return chemical_system, bspline_config


def shortest_pair_distances(df, chemical_system, r_cut):
    """Shortest distance per element pair over all frames of df.
    (uf3's DataAnalyzer reports ~0 A for like pairs because it counts an
    atom's distance to its own periodic image.)"""
    from ase.neighborlist import neighbor_list
    shortest = {pair: None for pair in chemical_system.interactions_map[2]}
    for atoms in df["geometry"]:
        i, j, d = neighbor_list("ijd", atoms, r_cut)
        symbols = np.array(atoms.get_chemical_symbols())
        for pair in shortest:
            mask = (symbols[i] == pair[0]) & (symbols[j] == pair[1])
            if mask.any():
                r = float(d[mask].min())
                if shortest[pair] is None or r < shortest[pair]:
                    shortest[pair] = r
    return shortest


def knot_spacing_type(bspline_config):
    """'uk' (uniform knots, faster in LAMMPS) if every knot sequence is
    uniformly spaced, else 'nk'."""
    for knots in bspline_config.knots_map.values():
        seqs = knots if isinstance(knots, (list, tuple)) else [knots]
        for seq in seqs:
            inner = np.unique(np.asarray(seq))
            steps = np.diff(inner)
            if len(steps) and not np.allclose(steps, steps[0], rtol=1e-8):
                return "nk"
    return "uk"


def save_2b_plots(model, bspline_config, chemical_system, lower_bounds):
    from uf3.regression import least_squares
    from uf3.util import plotting
    solutions = least_squares.arrange_coefficients(model.coefficients,
                                                   bspline_config)
    files = []
    for pair in chemical_system.interactions_map[2]:
        fig, ax = plotting.visualize_splines(solutions[pair],
                                             bspline_config.knots_map[pair])
        r_lo = lower_bounds.get(pair)
        if r_lo:
            ax.axvline(r_lo, color="orange", ls="--",
                       label=f"shortest {pair[0]}-{pair[1]} in data")
            ax.legend()
        ax.set_ylim(-3, 3)
        ax.set_xlim(0, bspline_config.r_max_map[pair])
        ax.set_title(f"2-body {pair[0]}-{pair[1]}")
        ax.set_xlabel("r (A)")
        ax.set_ylabel("pair energy (eV)")
        fname = f"2b_{pair[0]}-{pair[1]}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(fname)
    return files


def save_parity_plot(analyzer, label):
    analyzer.generate_parity_plots()  # UF3Tools; plt.show() is a no-op (Agg)
    fname = f"parity_{label}.png"
    plt.gcf().savefig(fname, dpi=150, bbox_inches="tight")
    plt.close("all")
    return fname


def make_lammps_compatible(path):
    """LAMMPS' pair_style uf3 (ML-UF3) only accepts "leading_trim = 0" in the
    block headers. UF3Tools writes the 3-body default leading_trim = 3, but
    the coefficient grid it writes is complete (trimmed basis functions are
    stored as explicit zeros), so header "0 3" describes the identical
    potential. Rewrite the header only when that is verified."""
    with open(path) as fp:
        lines = fp.read().split("\n")
    changed = 0
    for n, line in enumerate(lines):
        tok = line.split()
        if not tok or tok[0] not in ("2B", "3B"):
            continue
        k = 3 if tok[0] == "2B" else 4  # position of leading_trim
        lead = int(tok[k])
        if lead == 0:
            continue
        sizes = lines[n + 1].split()
        if tok[0] == "2B":
            n_basis_full = [int(sizes[1]) - 4]
            n_coeff = [int(lines[n + 3].split()[0])]
            coeffs = np.array(lines[n + 4].split(), dtype=float)[:lead]
        else:
            # knot counts are listed as (jk, ik, ij); the grid is (ij, ik, jk)
            n_basis_full = [int(x) - 4 for x in sizes[3:6]][::-1]
            n_coeff = [int(x) for x in lines[n + 5].split()]
            grid = np.array(" ".join(lines[n + 6:n + 6 + n_coeff[0]
                                           * n_coeff[1]]).split(),
                            dtype=float).reshape(n_coeff)
            coeffs = np.concatenate([grid[:lead].ravel(),
                                     grid[:, :lead].ravel(),
                                     grid[:, :, :lead].ravel()])
        if n_coeff != n_basis_full or np.any(coeffs != 0):
            raise RuntimeError(f"Cannot make '{line}' LAMMPS compatible: "
                               "coefficient grid is not complete")
        tok[k] = "0"
        lines[n] = " ".join(tok)
        changed += 1
    if changed:
        with open(path, "w") as fp:
            fp.write("\n".join(lines))
    return changed


def check_uf3_file(path, chemical_system):
    """Basic structural check of the written LAMMPS potential file."""
    with open(path) as fp:
        text = fp.read()
    n_2b = len(re.findall(r"^2B ", text, flags=re.M))
    n_3b = len(re.findall(r"^3B ", text, flags=re.M))
    exp_2b = len(chemical_system.interactions_map[2])
    exp_3b = len(chemical_system.interactions_map.get(3, []))
    if (n_2b, n_3b) != (exp_2b, exp_3b):
        raise RuntimeError(f"{path}: found {n_2b} 2B / {n_3b} 3B blocks, "
                           f"expected {exp_2b} / {exp_3b}")
    if re.search(r"\bnan\b|\binf\b", text, flags=re.I):
        raise RuntimeError(f"{path} contains NaN/inf coefficients")
    return n_2b, n_3b


def one_body_energies(model, bspline_config):
    """Per-element constant energies (eV/atom) of the UF3 model. They are not
    part of the .uf3 file, so LAMMPS energies = DFT-like energies - these."""
    from uf3.regression import least_squares
    solutions = least_squares.arrange_coefficients(model.coefficients,
                                                   bspline_config)
    return {el: float(np.ravel(solutions[el])[0])
            for el in bspline_config.element_list}


def write_lammps_example(name, degree, df, one_body):
    """Small LAMMPS test: relax bulk wurtzite AlN, then 1 ps NVT at 300 K."""
    from ase.build import bulk
    from ase.io import write
    reps = (4, 2, 3)  # orthorhombic 8-atom cell -> 192 atoms
    cell = bulk("AlN", "wurtzite", a=3.11, c=4.98, u=0.382,
                orthorhombic=True) * reps
    e1b = np.mean([one_body[s] for s in cell.get_chemical_symbols()])
    os.makedirs("lammps_test", exist_ok=True)
    write(os.path.join("lammps_test", "AlN_wurtzite.data"), cell,
          format="lammps-data", specorder=["Al", "N"], masses=True,
          atom_style="atomic")
    pair_style = f"pair_style uf3 {degree}"
    text = f"""# LAMMPS test of the fitted UF3 potential (needs LAMMPS with ML-UF3)
#   lmp -in in.AlN_uf3      (run inside the lammps_test folder)
units           metal
atom_style      atomic
boundary        p p p
read_data       AlN_wurtzite.data      # wurtzite AlN, type 1 = Al, type 2 = N

{pair_style}
pair_coeff      * * ../{name}.uf3 Al N

neighbor        1.0 bin
thermo_style    custom step pe press lx ly lz vol
thermo          10

# 1) relax atoms + cell
fix             relax all box/relax aniso 0.0 vmax 0.001
minimize        1.0e-10 1.0e-10 5000 50000
unfix           relax
variable        a equal lx/{reps[0]}
variable        a2 equal ly/({reps[1]}*sqrt(3.0))
variable        c equal lz/{reps[2]}
# LAMMPS energies exclude the constant 1-body energies of the UF3 model;
# add them back ({e1b:.6f} eV/atom for AlN) to compare with DFT energies
variable        epa equal pe/atoms+({e1b:.8f})
print           "UF3: a = ${{a}} (${{a2}}) A   c = ${{c}} A   E = ${{epa}} eV/atom"
print           "DFT training data: lowest energy = {df['energy_per_atom'].min():.4f} eV/atom"

# 2) short NVT run at 300 K
reset_timestep  0
velocity        all create 300.0 4928459 mom yes rot yes
fix             nvt all nvt temp 300.0 300.0 0.1
timestep        0.001
thermo_style    custom step temp pe etotal press
thermo          100
run             1000
"""
    with open(os.path.join("lammps_test", "in.AlN_uf3"), "w") as fp:
        fp.write(text)
    return pair_style, f"pair_coeff * * {name}.uf3 Al N"


def load_uf3tools(uf3tools_arg):
    """Find and import UF3Tools; exit with a readable message on failure."""
    import uf3tools_loader
    if getattr(uf3tools_loader, "LOADER_VERSION", 0) < 3:
        sys.exit(f"ERROR: {uf3tools_loader.__file__} is an old version. "
                 "Update fit_AlN_uf3.py and uf3tools_loader.py together "
                 "(run 'git pull' in this folder).")

    def norm(path):
        return os.path.normcase(os.path.abspath(path))

    # Where to look: --uf3tools, else DEFAULT_FOLDER, the folder above this
    # repository, this folder and the current folder. The home folder, the
    # folder above it and a drive root are too large to search: there only
    # their UF3Tools* sub-folders are used. An installed "pip install -e"
    # UF3Tools is the last resort.
    roots, advice_dir = [], None

    def add(d):
        if norm(d) not in [norm(r) for r in roots]:
            roots.append(d)

    if uf3tools_arg:
        path = os.path.abspath(os.path.expanduser(uf3tools_arg))
        if not os.path.exists(path):
            sys.exit(f"ERROR: --uf3tools folder does not exist: {path}\n"
                     "Check the spelling, put the path in double quotes and "
                     "do not end it with a backslash.")
        add(path)
    else:
        home = norm(os.path.expanduser("~"))
        skip = {home, os.path.dirname(home)}
        for d in (DEFAULT_FOLDER, os.path.dirname(HERE), HERE, os.getcwd()):
            if not os.path.isdir(d):
                continue
            too_high = (norm(d) == os.path.dirname(home)
                        or os.path.dirname(norm(d)) == norm(d))
            if advice_dir is None and not too_high:
                advice_dir = d  # a download placed here will be found
            if norm(d) in skip or os.path.dirname(norm(d)) == norm(d):
                try:
                    names = sorted(os.listdir(d))
                except OSError:
                    names = []
                for name in names:
                    sub = os.path.join(d, name)
                    if name.lower().startswith("uf3tools") and os.path.isdir(sub):
                        add(sub)
            else:
                add(d)
    try:
        return uf3tools_loader.load_uf3_handler(roots, advice_dir=advice_dir)
    except ModuleNotFoundError as exc:  # a missing third-party package
        sys.exit(f"ERROR: {exc}\nA Python package is missing. In the uf3_aln "
                 "environment run:\n  pip install -r requirements.txt\n"
                 f"(or: pip install {exc.name})")
    except FileNotFoundError as exc:
        hint = ""
        if not uf3tools_arg:
            if os.name == "nt":
                import ntpath
                example = (
                    f"  {ntpath.join(ntpath.dirname(DEFAULT_FOLDER), 'AlN')}"
                    "   (this script)\n"
                    f"  {ntpath.join(DEFAULT_FOLDER, 'UF3Tools')}"
                    "   (the UF3Tools download)")
            else:
                example = ("  ~/UF3TOOL/AlN             (this script)\n"
                           "  ~/UF3TOOL/code/UF3Tools   (the UF3Tools download)")
            hint = ("\nTip: keep this AlN folder next to the UF3Tools "
                    "download (README, section 2), e.g.\n" + example)
        sys.exit(f"ERROR: {exc}{hint}")
    except ImportError as exc:
        sys.exit(f"ERROR: {exc}")


# --------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    outcar_paths = [os.path.abspath(p) for p in args.outcar]
    out_dir = os.path.abspath(args.output_dir)

    print("== 1. Loading UF3Tools")
    handler = load_uf3tools(args.uf3tools)

    print("\n== 2. Reading OUTCAR files")
    outcars = find_outcars(outcar_paths)
    if not outcars:
        raise FileNotFoundError(f"No OUTCAR files found in {outcar_paths}")
    df, n_outcars = load_dataframe(outcars, handler, args)
    n_atoms = describe_data(df)

    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)  # UF3Tools writes some helper files to the cwd
    df.to_pickle("dataset.pkl")

    print("\n== 3. B-spline basis")
    chemical_system, bspline_config = make_bspline_config(args)
    n_coeff = sum(bspline_config.get_interaction_partitions()[0].values())
    print(f"  elements: {chemical_system.element_list}   degree: {args.degree}")
    print(f"  2-body pairs: {chemical_system.interactions_map[2]}")
    if args.degree == 3:
        print(f"  3-body trios: {chemical_system.interactions_map[3]}")
    print(f"  number of coefficients: {n_coeff}")
    n_obs = len(df) + 3 * n_atoms
    if n_obs < 5 * n_coeff:
        print(f"  ! WARNING: only {n_obs} energies+force components for "
              f"{n_coeff} coefficients. Add more DFT data (MD snapshots, "
              "strained cells, defects/surfaces) or use --degree 2.")

    print("\n== 4. Train/test split (UF3Tools ModelTrainer)")
    regularizer = dict(ridge_1b=args.ridge_1b, ridge_2b=args.ridge_2b,
                       ridge_3b=args.ridge_3b, curvature_2b=args.curvature_2b,
                       curvature_3b=args.curvature_3b)
    trainer = handler.ModelTrainer(df, chemical_system, bspline_config,
                                   regularizer)
    # Keep each OUTCAR entirely in train or test when there are enough of
    # them (>= 1 whole OUTCAR held out); otherwise split frames randomly.
    by_outcar = int(args.test_fraction * n_outcars) >= 1
    training_set, testing_set = trainer.split_data(args.test_fraction,
                                                   filter_index=by_outcar)
    if len(training_set) == 0:
        raise RuntimeError("Training set is empty.")
    print(f"  split: {'by OUTCAR' if by_outcar else 'random frames'}")

    print("\n== 5. Featurization (UF3Tools Featurizer)")
    feature_file = os.path.join(out_dir, "features.h5")
    if os.path.exists(feature_file):
        os.remove(feature_file)  # UF3Tools would silently reuse an old file
    featurizer = handler.Featurizer(df, chemical_system, bspline_config)
    featurizer.featurize_data(args.cores, feature_file, args.batch_size)

    print("\n== 6. Fitting")
    trainer.train_model(feature_file, weight=args.weight,
                        batch_size=args.batch_size)
    model = trainer.get_model()
    if not np.all(np.isfinite(model.coefficients)):
        raise RuntimeError("Fit produced NaN/inf coefficients; increase the "
                           "regularization (--ridge-*/--curvature-*).")

    print("\n== 7. Analysis (UF3Tools ModelAnalyzer)")
    analyzer = handler.ModelAnalyzer(chemical_system, model, bspline_config,
                                     training_set, testing_set,
                                     [feature_file], out_dir)
    lower_bounds = shortest_pair_distances(training_set, chemical_system,
                                           args.r_max_2b)
    for pair, r in lower_bounds.items():
        print(f"  shortest {pair[0]}-{pair[1]} distance in training data: "
              + (f"{r:.3f} A" if r else "none within cutoff"))
    if not args.no_repulsion_fix:
        # make the pair potential repulsive below the shortest distance
        # seen in the data (no DFT information there)
        for pair, r in lower_bounds.items():
            if r:
                analyzer.fix_repulsion_2b(
                    pair, r_target=r, min_curvature=args.repulsion_curvature)

    results = {}
    plots = save_2b_plots(model, bspline_config, chemical_system,
                          lower_bounds)
    for label, training in (("train", True), ("test", False)):
        subset = training_set if training else testing_set
        if len(subset) == 0:
            continue
        print(f"  -- {label} set ({len(subset)} frames)")
        pred = analyzer.test_model_prediction(
            training=training, use_elements=chemical_system.element_list)
        results[label] = (pred["rmse_e"], pred["rmse_f"])
        plots.append(save_parity_plot(analyzer, label))

    print("\n== 8. Export (UF3Tools UF3LammpsPotentialGenerator)")
    json_name, uf3_name = f"{args.name}.json", f"{args.name}.uf3"
    analyzer.export_model_to_json(out_dir, json_name)
    kst = knot_spacing_type(bspline_config)
    analyzer.export_model_to_lammps(out_dir, uf3_name, overwrite=True,
                                    author=args.author.replace(" ", "_"),
                                    units="metal", knots_spacing_type=kst)
    n_fixed = make_lammps_compatible(uf3_name)
    if n_fixed:
        print(f"  {n_fixed} block header(s) set to leading_trim=0 for "
              "LAMMPS (trimmed coefficients are stored as zeros)")
    n_2b, n_3b = check_uf3_file(uf3_name, chemical_system)
    one_body = one_body_energies(model, bspline_config)
    pair_style, pair_coeff = write_lammps_example(args.name, args.degree, df,
                                                  one_body)

    lines = [
        "UF3 potential for AlN fitted with UF3Tools",
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"OUTCAR files used: {n_outcars}",
        *[f"  {o}" for o in outcars],
        f"frames: {len(df)} (train {len(training_set)}, "
        f"test {len(testing_set)})",
        f"degree: {args.degree}   2B: r_min {args.r_min_2b} r_max "
        f"{args.r_max_2b} res {args.resolution_2b}"
        + (f"   3B: r_min {args.r_min_3b} r_max {args.r_max_3b}/"
           f"{2 * args.r_max_3b} res {args.resolution_3b}/"
           f"{2 * args.resolution_3b}" if args.degree == 3 else ""),
        f"weight: {args.weight}   regularizer: {regularizer}",
        f"repulsion fix below shortest distance: {not args.no_repulsion_fix}"
        f" {lower_bounds}",
        f"knot spacing in .uf3: {kst}",
        f"1-body energies (eV/atom, not in .uf3; LAMMPS energy + these = "
        f"UF3/DFT energy): {one_body}",
        *[f"RMSE {k}: energy {e:.5f} eV/atom, force {f:.5f} eV/A"
          for k, (e, f) in results.items()],
        "",
        "LAMMPS:",
        f"  {pair_style}",
        f"  {pair_coeff}",
    ]
    with open("fit_summary.txt", "w") as fp:
        fp.write("\n".join(lines) + "\n")

    print(f"\nDone in {time.time() - t_start:.0f} s. Results in {out_dir}:")
    print(f"  {uf3_name:<22} LAMMPS potential ({n_2b} 2-body, {n_3b} "
          f"3-body blocks)")
    print(f"  {json_name:<22} UF3 model (python / UF3Tools)")
    print(f"  {'fit_summary.txt':<22} settings and errors")
    print(f"  {'lammps_test/':<22} example LAMMPS input")
    for f in plots:
        print(f"  {f}")
    for k, (e, f) in results.items():
        print(f"  RMSE {k:<5}: energy {1000 * e:8.2f} meV/atom   "
              f"force {f:.4f} eV/A")
    print("\nIn your LAMMPS input:")
    print(f"  {pair_style}\n  {pair_coeff}")


if __name__ == "__main__":  # required on Windows (multiprocessing)
    main()
