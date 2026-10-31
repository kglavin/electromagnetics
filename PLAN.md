# Electromagnetics Study Patch: Plan

Reference text: Malthe-Sørenssen, *Elementary Electromagnetism Using Python* (2026), as used in UiO FYS1120.
Web resources: elektromagnetisme.no, malthent.github.io/electromagnetism/book/bookexercises.html.

## 1. Purpose and ground rules

G1. Build a set of Jupyter notebooks and Manim animations that a learner works through alongside the textbook. Work is grouped into six modules that each bunch related chapters, so the repo follows the physics, not the book's table of contents. Each notebook carries a pointer to the chapters and sections it pairs with.
G2. No code is copied from the book. Every algorithm is written from the physics and the book's Method Box descriptions, in our own structure and naming.
G3. No exercise solutions. Notebooks reference the book's exercises by number and section but never work them. The official solutions stay out of this repo.
G4. Every numerical result is checked against a closed-form case before it is used on a case without one.
G5. Each notebook is built for retrieval practice. It opens with blank-sheet recall prompts and closes with a "rebuild this from scratch" task, matching the six-step protocol in the course guide.

## 2. Repository layout

```
electromagnetics/
  PLAN.md
  README.md                 how to set up and run
  pyproject.toml            uv project, Python 3.12
  emlib/                    small shared library, tested
    show.py                 embed rendered scenes in notebooks
    grids.py                meshgrid(indexing='ij') helpers, slices, flattening
    fields.py               point-charge E and V, discrete grad/div/curl, flux through boxes
    relax.py                Jacobi, Gauss-Seidel, sparse direct Laplace/Poisson
    biot_savart.py          parametric wire -> segments -> B on grid
    percolation.py          random resistor lattice, sparse KCL solve
    fdtd.py                 1D/2D Yee solver, CFL check, Poynting
    dielectrics.py          bound charge, D field, div(eps grad V) solver
    integrators.py          Boris pusher, RK4 for particles and circuits
  tests/                    pytest, closed-form comparisons for every emlib function
  notebooks/
    m1_fields/              one folder per roadmap module, not per book chapter
    m2_electrostatics/      notebooks numbered by topic, e.g. 02_superposition.ipynb
    m3_dielectrics/         each notebook header carries a book pointer (chapter and sections)
    m4_magnetostatics/
    m5_circuits/
    m6_waves/
  animations/
    m1_fields/ ... m6_waves/  one package per module, scenes.py + render notes
  study/
    recall_sheets/          one blank-sheet prompt file per module (md), no answers
    schedule.md             spaced-retrieval tracker per module (immediate, 24h, 7d, 30d)
    vector_calculus_ref.md  the coordinate-system and identity reference sheet
  media/<module>/           rendered mp4/png per module, gitignored
```

The book is reference only.

## 3. Environment

E1. `uv` project pinned to Python 3.12. Reason: Numba and Manim Community have no 3.14 wheels, and the platform is aarch64 Linux.
E2. Dependencies: numpy, scipy, matplotlib, sympy, numba, plotly, jupyterlab, ipywidgets, pytest, manim.
E3. Manim system deps: ffmpeg is present. Cairo and Pango dev libraries need installing in the devcontainer. LaTeX is optional and we start without it, using `Text` instead of `MathTex` until a TeX install is justified.
E4. A `make` or `uv run` task per target: `test`, `lab`, `render MODULE`.

## 4. Notebook template

Every module notebook follows the same five blocks so the pattern becomes automatic.

1. Recall. Three to five prompts to answer on paper before running any cell. Example for Ch 4: "State Gauss' law in both forms and show how the divergence theorem connects them."
2. Principle to algorithm. Short prose that maps the physical law to a discrete procedure. Written in our words, pointing to the book's section and Method Box by number.
3. Build. Code cells that implement the procedure using `emlib`, with the key step left as a short exercise cell the learner fills in. A hidden reference implementation lives in `emlib`, not in the notebook.
4. Verify. Compare to a closed-form case (point charge, infinite wire, parallel plates, solenoid). Print relative error. Plot.
5. Explore and rebuild. One open question in the computational-essay style, and the instruction to close everything and rewrite the core function from memory.

## 5. Module roadmap

Six build modules. Each lists the book chapters it serves, the `emlib` code, the notebooks, the Manim scenes, and the verification case.

### M1. Vector calculus and field foundations (Ch 1, 2, 3)

- emlib: `grids.py`, `fields.py` (point-charge E and V, superposition over arrays, `np.gradient` based E from V, discrete div and curl).
- Notebooks: a four-part vector calculus primer following chapter 1 (00a vectors and fields, 00b integration, 00c differentiation, 00d coordinates and the integral theorems) and a physics primer for chapters 2 and 3 (00e charge, Coulomb's law, field and potential). Every primer is a self-contained lesson: a full prose exposition of the ideas, each operator or law introduced as a measurement, hand-worked examples beside the code, a key-results sheet and warm-up answers, so the module can be followed without the book chapters; grids and `indexing='ij'` pitfalls; dipole and ring-of-charge E by direct summation; V on a grid and E = -grad V; line integral path independence check.
- SymPy: symbolic V, analytical E, div, curl, lambdified to NumPy, compared to the numerical versions.
- Manim: 2D vector field morphing from (-y, x) to (x, y) with a paddlewheel marker that spins only in the curl region.
- Verify: numerical E from gridded V of a point charge against Coulomb's law. Curl of a gradient field is zero to grid precision.

### M2. Electrostatic relaxation and Gauss flux (Ch 4, 6, 7, 8)

- emlib: `relax.py` (Jacobi and Gauss-Seidel with Numba, Dirichlet masks, sparse direct solve via `scipy.sparse.linalg.spsolve`), box flux in `fields.py`.
- Notebooks: three self-contained primers (00a Gauss' law by symmetry with the sphere, shell, line, cylinder, plane, slab and conductor cases worked by hand and checked against brute-force superposition; 00b from Poisson's equation to a linear system, with the 1D tridiagonal matrix and Jacobi sweeps done by hand, the 2D stencil, masks, sparse assembly, uniqueness, Neumann conditions and recovering E and charge from V; 00c conductors in equilibrium, the method of images, capacitance and field energy for chapters 7 and 8); flux through a cube around a point charge vs enclosed charge; 2D Laplace between plates with an embedded conductor; mirror charge vs relaxed solution; capacitance from field energy for a non-standard geometry.
- Manim: 2D potential surface relaxing from flat to steady state; 3D point charge with a Gaussian sphere whose surface arrows recolor by E dot dS. Rendered scenes are embedded in the matching notebook via `emlib.show.animation`.
- Verify: parallel-plate capacitance against epsilon_0 A / d; relaxed potential of a grounded plane plus point charge against the image-charge formula.

### M3. Dielectrics and bound charge (Ch 5, with 3D Plotly) 

- emlib: `dielectrics.py` (bound volume and surface charge, D = epsilon_0 E + P, dipole-pair sources, sparse solver for div(eps grad V) = -rho_f in 2D and 3D with harmonic-mean face permittivity).
- Notebooks: a self-contained primer for chapter 5 (00a dipoles, polarization, bound charge, D, linear dielectrics and boundary conditions, with the slab capacitor and the dielectric cylinder checked against the solver); non-uniform P(x,y,z), rho_b = -div P, sigma_b at a slab boundary; Plotly `Isosurface` and `Volume` for V shells and |D| contours.
- Manim: two-panel scene, free charge on plates beside dipoles aligning in the dielectric, then bound surface charge forming and the internal E shrinking.
- Verify: uniformly polarized sphere from bound surface charge gives E = -P/3eps0 inside; slab capacitor matches the series formula with D continuous; dielectric sphere in a uniform field gives E_in = 3/(eps_r+2) E0.

### M4. Magnetostatics and Biot-Savart (Ch 11, 12, 13)

- emlib: `biot_savart.py` (parametric curve to segments, vectorized sum over segments and grid points), `integrators.py` Boris pusher.
- Notebooks: B of a circular loop and a Helmholtz pair; B of a finite solenoid vs mu_0 n I; charged particle in crossed E and B with Boris vs RK4 energy drift; curl of M for a non-uniform magnetization.
- Manim: 3D Helmholtz coil with field lines traced as moving particles following B.
- Verify: on-axis loop field against the closed form; long straight wire against mu_0 I / 2 pi r.

### M5. Circuits and disordered media (Ch 9, 10, 16)

- emlib: `percolation.py` (random bond lattice, KCL sparse assembly, effective conductivity), nodal analysis solver, RLC ODE helpers in `integrators.py`.
- Notebooks: 2D current density in a sheet with varying sigma(x,y); nodal analysis of an arbitrary network; percolation sweep of sigma_eff vs p across p_c = 0.5 for bonds; RC and RLC transients; Bode sweep of a multi-stage filter with complex impedance.
- Manim: lattice current flow as p rises through p_c, backbone highlighted at the threshold.
- Verify: nodal solver against a hand-solvable bridge circuit; series RLC resonance frequency against 1 / sqrt(LC).

### M6. Induction, Maxwell, waves (Ch 14, 15, 17)

- emlib: `fdtd.py` (1D Yee grid, then 2D TE/TM, CFL enforced, Poynting on the grid), Neumann mutual inductance in `biot_savart.py`.
- Notebooks: flux through a moving loop and numerical emf; mutual inductance of two non-coaxial loops; 1D FDTD pulse with reflection at a dielectric step; 2D FDTD point source with S(x,y,t).
- Manim: 3D plane wave with E along y, B along z, Poynting arrow along x and an energy-flux counter.
- Verify: pulse speed equals c in the 1D FDTD; Neumann formula for coaxial loops against the known series result.

## 6. Build order

The order follows the course so the repo is useful from week one.

1. Environment, `emlib` skeleton, tests scaffold, README, `.gitignore`. (A1, A2)
2. M1, then M2. These carry the most reusable code.
3. M3 (short) and M4.
4. M5 and M6.
5. Study folder: one recall sheet per module and the schedule tracker, written alongside each module, not at the end.

Each module is done when: emlib tests pass, notebooks run clean top to bottom via `jupyter nbconvert --execute`, at least one Manim scene renders to `media/`, and the recall sheet exists.

## 7. Risks

R1. Python version. 3.14 is default and lacks Numba and Manim support. Mitigation: uv pin to 3.12 in `pyproject.toml`.
R2. Manim on aarch64 in a container. Cairo and Pango builds can fail. Mitigation: install distro packages first, fall back to rendering with `-ql` and no LaTeX.
R3. Book content leaking into the repo. Mitigation: gitignore the epub, cite sections by number only, write all code from the physics.
R4. 3D FDTD and large relaxation grids are slow in pure NumPy. Mitigation: Numba for relaxation, keep FDTD at 1D/2D as the outline specifies.

## 8. Immediate actions

A1. (done) Add `.gitignore` covering `*.epub`, `media/`, `.ipynb_checkpoints/`, `__pycache__/`, `.venv/`.
A2. (done) `uv init` with Python 3.12 and the E2 dependency list, install Cairo and Pango dev packages, confirm `manim` renders a one-line test scene.
A3. (done) Create `emlib/` with grids and fields modules and 12 closed-form tests.
A4. (done) Write the M1 notebooks, M1 Manim scene, M1 recall sheet.
A5. (done) Build M2: box flux, relaxation and sparse solvers, field energy, four notebooks, two scenes, recall sheet.
A6. (done) Build M3: dielectrics module with varying-permittivity solver, eleven tests, two notebooks, one scene, recall sheet.
A7. Build M4.

## 9. Open questions

Q1. (decided) Notebooks are bunched by module, not by chapter. A module has as many topic notebooks as it needs, each kept under about 30 cells. Primers are exempt: they carry the teaching material and run to 25 to 38 cells and 5000 to 7000 words.
Q2. Manim quality target: quick 480p renders for learning, or 1080p for sharing? Default is `-ql` during development.
