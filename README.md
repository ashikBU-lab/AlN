# UF3 potential for AlN from VASP OUTCAR files (with UF3Tools)

`fit_AlN_uf3.py` turns your DFT data (VASP `OUTCAR` files) into a UF3
interatomic potential for AlN, using the classes of
[UF3Tools](https://github.com/uf-chenlab/UF3Tools) (`StructureParser`,
`DataFilter`, `Featurizer`, `ModelTrainer`, `ModelAnalyzer`,
`UF3LammpsPotentialGenerator`). It writes:

| file | what it is |
|---|---|
| `AlN.uf3` | **LAMMPS potential file** for `pair_style uf3` |
| `AlN.json` | UF3 model (Python / ASE calculator / UF3Tools) |
| `fit_summary.txt` | settings, RMSE errors, LAMMPS lines |
| `parity_train.png`, `parity_test.png` | predicted vs DFT energies and forces |
| `2b_Al-Al.png`, `2b_N-Al.png`, `2b_N-N.png` | fitted pair potentials |
| `lammps_test/` | ready-to-run LAMMPS test (relax wurtzite AlN + 300 K MD) |
| `dataset.pkl`, `features.h5` | parsed data and UF3 features (intermediate) |

## 1. One-time installation (Windows, Anaconda Prompt)

```bat
conda create -n uf3_aln python=3.10 -y
conda activate uf3_aln
conda install git -y
cd "C:\Users\Mdashikur_Rahman1\Desktop\conda files\UF3TOOL"
git clone https://github.com/ashikBU-lab/AlN.git
cd AlN
pip install -r requirements.txt
```

`requirements.txt` installs the **uf-chenlab fork of uf3** (the `develop`
branch, pinned to a tested commit). UF3Tools does **not** work with the
normal `pip install uf3`; it needs functions that only exist in the fork.
If you already installed the PyPI `uf3`, run `pip uninstall -y uf3` first.

You do **not** need to install UF3Tools' extra packages (phonopy, phono3py,
phonolammps, mp-api, dask, optuna, pymongo ...). They are only used by its
validation/HPC tools. `uf3tools_loader.py` imports just the fitting part of
your UF3Tools copy.

## 2. Put the files in place

```
C:\Users\Mdashikur_Rahman1\Desktop\conda files\UF3TOOL\
├── code\                     <- your folder
│   ├── UF3Tools\  (or UF3Tools-main\)   the downloaded UF3Tools repository
│   ├── OUTCAR                            your DFT file(s); sub-folders are fine,
│   └── run2\OUTCAR ...                   e.g. one folder per calculation
└── AlN\                      <- this repository
```

Every file whose name starts with `OUTCAR` (`OUTCAR`, `OUTCAR_1`,
`OUTCAR.gz`, ...) inside `--outcar` folders is used. Files that contain
elements other than Al/N are skipped.

## 3. Fit the potential

```bat
conda activate uf3_aln
cd "C:\Users\Mdashikur_Rahman1\Desktop\conda files\UF3TOOL\AlN"
python fit_AlN_uf3.py --outcar "C:\Users\Mdashikur_Rahman1\Desktop\conda files\UF3TOOL\code" --uf3tools "C:\Users\Mdashikur_Rahman1\Desktop\conda files\UF3TOOL\code"
```

Because those two paths are the script's defaults, `python fit_AlN_uf3.py`
alone does the same. Results go to `AlN_uf3_potential\`. Run
`python fit_AlN_uf3.py --help` for all options.

What the script does:

1. reads every ionic step of each OUTCAR (ASE), removes steps whose SCF loop
   hit `NELM` (not converged) and frames with force components > 20 eV/Å;
2. builds the UF3 B-spline basis: 2-body Al-Al, Al-N, N-N (0–5.5 Å, 20 knot
   intervals) + 3-body terms (1.5–3.5 Å, 6 intervals; `--degree 2` for pair
   only);
3. splits train/test (10 %; whole OUTCARs are held out when there are ≥ 10,
   otherwise random frames), featurizes and fits with UF3Tools;
4. makes each pair potential smoothly repulsive below the shortest distance
   found in your data, where DFT gives no information. Without this, atoms
   can fall onto each other in MD;
5. reports RMSEs, saves plots, and writes `AlN.json` + `AlN.uf3` (checked to
   be readable by LAMMPS).

Useful options:

| option | default | when to change |
|---|---|---|
| `--every N` | 1 | long AIMD runs: use every N-th step (neighbouring MD steps are nearly identical) |
| `--degree 2` | 3 | very little data, or a quick pair-potential test |
| `--r-max-2b`, `--r-max-3b` | 5.5, 3.5 Å | cutoffs |
| `--weight` | 0.3 | energy vs force weighting (UF3Tools default) |
| `--ridge-*`, `--curvature-*` | 1e-8 … 1e-7 | regularisation; increase if the potential is noisy/overfits |
| `--repulsion-curvature` | 10 eV/Å² | hardness of the short-range wall |
| `--cores` | 4 | processes used for featurization |

## 4. Use it in LAMMPS

Needs LAMMPS built with the **ML-UF3** package (stable release 29Aug2024
or newer; `lmp -h` must list `uf3` among the pair styles).

```
pair_style uf3 3                 # 3 = 2-body + 3-body (use 2 if fitted with --degree 2)
pair_coeff * * AlN.uf3 Al N      # LAMMPS atom type 1 = Al, type 2 = N
```

If the path to `AlN.uf3` contains spaces, put it in quotes. Quick test:

```bat
cd AlN_uf3_potential\lammps_test
lmp -in in.AlN_uf3
```

It relaxes bulk wurtzite AlN (prints `a`, `c`, energy per atom) and runs
1 ps of MD at 300 K.

Note: the `.uf3` file contains only the 2- and 3-body terms. UF3's constant
per-atom (1-body) energies are listed in `fit_summary.txt`, so LAMMPS
energies are shifted by a constant per atom relative to DFT. Forces,
stresses, energy differences and structures are not affected.

On Linux/macOS (e.g. a cluster) with `pip install lammps`, this compares
LAMMPS with the Python model and the DFT data:

```
python tests/check_uf3_in_lammps.py AlN_uf3_potential
```

## 5. How good will the potential be?

A UF3 potential can only describe configurations that resemble its
training data, so a potential fitted to one relaxation OUTCAR will be poor.
For a general-purpose AlN potential, the training set usually contains:

* AIMD of bulk wurtzite AlN at several temperatures (e.g. 300–1500 K) and
  volumes (compressed/expanded cells), every few steps;
* strained cells (equation of state, elastic deformations);
* if needed: vacancies/interstitials, surfaces, zinc-blende/rock-salt phases,
  melt.

Check `parity_test.png` and the test RMSE (as a rough guide, UF3 potentials
usually reach a few meV/atom and ~0.1–0.2 eV/Å), then compare lattice
constants, elastic constants and phonons with DFT before production runs.

## Check the installation without your data

`tests/make_test_outcar.py` writes a small synthetic AlN OUTCAR (energies
from a toy Morse model, **not DFT**) to check that everything runs:

```bat
python tests\make_test_outcar.py test_data\OUTCAR
python fit_AlN_uf3.py --outcar test_data --output-dir test_potential
```

The toy model itself is not stable in the wurtzite structure, so don't
read anything physical into `test_potential\lammps_test`.

## Notes on UF3Tools / uf3 issues handled by the script

* UF3Tools' `DataParser` path requires an `OSZICAR` next to every OUTCAR.
  The script reads OUTCARs with ASE and gets the SCF-convergence
  information from the OUTCAR itself.
* `uf3.data.io.dataframe_from_lists` loses energies/forces (ASE's
  `Atoms.copy()` drops the calculator), so they are passed explicitly.
* uf3's `DataAnalyzer` reports ~0 Å as the shortest Al-Al/N-N distance (it
  counts each atom's distance to its own periodic image), so the shortest
  distances are computed directly.
* `uf3_handler.py` calls `datetime.now()` on the `datetime` module; the
  loader patches it so the LAMMPS export works.
* LAMMPS' `pair_style uf3` only accepts `leading_trim = 0` in block
  headers. UF3Tools writes the 3-body default `3`. The coefficient grid
  already stores the trimmed basis functions as zeros, so the header is
  changed to `0` (same potential, verified numerically in LAMMPS).
* uf3 needs numpy ≤ 1.26.4 and pandas < 3.

Tested with Python 3.10, uf3 (uf-chenlab `develop` @ `0c227af`), UF3Tools
@ `ea0b51f`, and LAMMPS 22Jul2025 (ML-UF3). Windows was not tested directly,
but every pinned dependency has a Windows wheel and the featurization runs
with Windows' process start method.
