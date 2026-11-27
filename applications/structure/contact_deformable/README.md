# Contact between deformable surfaces with a persistent Verlet list

First increment (serial, 2d, `FE_Q(1)` with a linear mapping, frictionless penalty method).
The measured results are in the section [Results](#results); everything that is not measured is
marked as expected/proposed.

## 1. Starting point: contact path of the branch before this increment

Application `applications/structure/contact_rigid_obstacle` → `Structure::Operator`
(`spatial_discretization/operator.cpp`) → `NonLinearOperator`
(`spatial_discretization/operators/nonlinear_operator.cpp`):

| step | function | what happens |
|---|---|---|
| residual | `evaluate_nonlinear()` → `boundary_face_loop_nonlinear()` → `do_boundary_integral_contact_nonlinear()` | per face batch and quadrature point: `evaluate_contact_state()` |
| search | — | none: the obstacle is an analytic signed distance `phi = contact_bc[id]` (`RigidPlane`, `RigidBall`, `MovingBall`), evaluated at `x = X + d` by `evaluate_rigid_obstacle()` (`boundary_conditions.h`) |
| kinematics | `evaluate_contact_state()` | `g = phi(x)`, `n = grad phi(x)`, curvature `grad grad phi(x)`; no projection, no opposite face |
| activation | `evaluate_contact_state()` | penalty: `A = -eps g`; Nitsche: `A = n.(P N) - gamma g`, i.e. the **normal traction enters the activation**; active iff `A >= 0`, `lambda = [A]_+` |
| linearization | `set_solution_linearization()` | stores `displacement_lin` only; the contact state is recomputed in every vmult |
| vmult | `OperatorBase::apply()` → `boundary_face_loop_hom_operator()` → `do_boundary_integral_contact()` | `evaluate_contact_state()` at `displacement_lin` per quadrature point **in every Jacobian-vector product** (phi, gradient, Hessian, for Nitsche also `S`, `delta S`), plus the variation of `n` |

Surfaces: boundary faces with `BoundaryType::Contact`; the querying entities are the face
quadrature points (vectorized face batches). Supported and tested before this increment: analytic
rigid obstacles only (patch test, Hertz 2d/3d, sliding block, Cattaneo, friction). No pairs of
prescribed surfaces, no search between deformable surfaces, no self-contact. Hence a Verlet list
had nothing to accelerate, and this increment first adds the smallest two-deformable-surface case
with a real search.

## 2. Formulation of the deformable contact (new)

`DeformableContact<dim, Number>` (`include/exadg/structure/spatial_discretization/operators/deformable_contact.{h,cpp}`):
Gauss-point-to-segment, one pass, penalty, frictionless. A slave Gauss point `x_s` and a master
face `F_m = [x_1, x_2]` form an admissible pair if
(i) `F_m` is not the face of `x_s` (only incidence excluded), (ii) `xi in [-tau, 1 + tau]`,
(iii) `n_s . n_m < 0`, (iv) `|g| <= r_c`, `g = (x_s - x_1) . n_m`. The admissible pair with the
smallest `|g|` (ties: smallest face index) is selected; active iff `g <= 0`;
`lambda = eps <-g>_+`. Residual and tangent are the first and second derivative of
`eps/2 sum_s w_s <-g_s>_+^2` (variation of `n_m` and `xi` included, symmetric).

Why a detection radius `r_c`: the rigid code has no cutoff because `phi` is globally defined.
For discrete surfaces, "closest admissible projection without cutoff" pairs a point with the
opposite surface of the same body (self-contact: the outer face of the same arm is facing and has
`g = -thickness`) and with far back faces. `r_c` is therefore part of the formulation (both
search strategies use it), must be larger than any penetration that has to be detected and
smaller than the thickness. In exhaustive mode the queries where an active pair exists only
beyond `r_c` are counted (`cutoff hits`); with `r_c = 0.05` and `eps = 1000` they occurred in
Newton trials (6703 of 174960 queries), with `r_c = 0.2` never, with identical Newton histories.

Nitsche: not implemented for deformable surfaces in this increment. The topology keeps the
adjacent cell and local face number of every contact face (`Face::cell`, `Face::face_no`), which
the slave traction `P N` of a Nitsche variant needs; the search interface (`ContactProjection`:
face, `xi`, `g`, `n`, `t`, `l`) is formulation independent. Since Nitsche activates with
`lambda(d) - gamma g >= 0`, a point can be active with `g > 0`; with a radius it would be inactive
if `g > r_c`. The certificate needed there is `lambda(d) <= gamma r_c` at points without an
admissible pair, otherwise fallback to a search without radius (proposed, not implemented).

## 3. Verlet list and the three states

| state | content | update |
|---|---|---|
| topology (`setup()`) | contact faces, vertex DoF indices and Dirichlet flags, orientation of the outward normal, adjacent cell, slave Gauss points (`eta`, reference weight), pairs | setup (mesh change: call `setup()` again) |
| Verlet list (`build_list()`, `ensure_valid_list()`) | CSR candidates per slave face, vertex displacements of the build configuration `u_*` | when `2 D_max(u, u_*) > delta` |
| linearization (`set_linearization()`) | projection, `n`, `t`, `l`, `g`, active set at `u_k` | every `set_solution_linearization(u_k)`; all `J(u_k) v` use it |

Guarantee: the list contains all face pairs with `dist(F_s(u_*), F_m(u_*)) <= R_m`,
`R_m = r_c + delta + tau (l_m(u_*) + delta)` (bounding boxes, conservative). For straight faces,
every point moves at most `D_max = max_a |d_a(u) - d_a(u_*)|` over all contact vertices, so
`dist(F_s(u), F_m(u)) >= dist(F_s(u_*), F_m(u_*)) - 2 D_max` and `l_m(u) <= l_m(u_*) + 2 D_max`.
For `2 D_max <= delta`, the list contains every admissible pair (distance `<= |g| + tau l <=
r_c + tau l`). The certificate is checked before every query (residual, linearization) against
the configuration of the **last build**, whatever state built it (accepted or rejected trial).
`D_max` is computed from displacement differences (exact for binary fractions), so the limit
`2 D_max = delta` is tested exactly. Higher order elements or mappings are rejected in `setup()`
(the vertex motion would not bound the motion of curved faces). Because the Verlet and the
exhaustive search use the same projection code and selection, the results are bitwise identical
whenever the list is valid; there is no uncertain case and no fallback.

Newton/Krylov: residual evaluations (also every line search trial) query the search;
`set_solution_linearization(u_k)` reuses the result of the residual evaluation if the positions
are bitwise identical (always the case for the Newton solver, which linearizes at the last
accepted trial); `vmult` (`NonLinearOperator::vmult()`, which hides `OperatorBase::vmult()` for the
`KrylovSolver<NonLinearOperator>`) never searches. Constrained DoFs: rows are skipped and
constrained entries of `v` are treated as zero, as in `OperatorBase::apply()`.

Limits: serial (global DoF indices, `AssertThrow`), 2d, `FE_Q(1)`, linear mapping, no hanging
nodes, the contact is not in the preconditioner (AMG/multigrid of the elasticity operator only).
Continuous collision detection between states is not done: the guarantee holds at the queried
states only.

## 4. Commands

```bash
cd build
make -j8 exadg
cd applications/structure/contact_deformable && make -j8
./search_tests                                   # geometric/mechanical tests (exit code)
DEAL_II_NUM_THREADS=1 ./solver ../../../../applications/structure/contact_deformable/input.json
ctest -R contact_deformable                      # search_tests + two_blocks_{exhaustive,verlet}
../../../../applications/structure/contact_deformable/run_benchmark.sh ./solver <dir> "3 4 5 6" "Faces Boundary"
python3 ../../../../applications/structure/contact_deformable/collect_benchmark.py <dir>
```

`DEAL_II_NUM_THREADS=1`: without it, the runs used many threads (sys time 79 min for 9.6 min of
wall time) and the wall times of identical runs differed by 50 %.

## Results

Measured on 2026-11-27, serial, one thread, runs executed one after the other; `TwoBlocks`,
`eps = 100`, `r_c = 0.05`, `delta = 0.05`, `tau = 0.01`, load increment 0.05, Newton abs. tol.
1e-8, CG rel. tol. 1e-6 with AMG (updated every Newton iteration). Same input for both strategies.

Correctness (measured): in all 8 pairs, Newton iterations, Krylov iterations, failed load steps and
the reaction file (16 digits) are identical for the exhaustive and the Verlet search;
`search_tests` passes (Verlet == exhaustive at every queried state incl. the limit
`2 D_max = delta` and the trial sequence, U strip, adjacent fold, residual/J v bitwise,
finite differences of the tangent down to 1e-10); the 10 rigid-obstacle tests reproduce their
reference output exactly.

| case | DoFs | slave pts / master faces | Newton (steps, failed) | Krylov | solve E / V [s] | search E -> V [s] (V: queries + builds + checks) | V: builds, t [s] | V: checks, t [s] | V: candidates / query |
|---|---|---|---|---|---|---|---|---|---|
| Faces L3 | 3604 | 80 / 64 | 73 (22, 1) | 2692 | 1.86 / 1.93 | 0.0023 -> 0.00084 | 12, 0.00014 | 171, 3.6e-5 | 8.3 |
| Faces L4 | 13860 | 160 / 128 | 65 (22, 1) | 2798 | 7.28 / 7.29 | 0.0080 -> 0.0022 | 12, 0.00042 | 176, 7.1e-5 | 14.7 |
| Faces L5 | 54340 | 320 / 256 | 81 (23, 4) | 4728 | 59.5 / 59.0 | 0.076 -> 0.014 | 11, 0.0012 | 473, 2.5e-4 | 27.6 |
| Faces L6 | 215172 | 640 / 512 | 91 (26, 12) | 6549 | 425 / 438 | 0.64 -> 0.094 | 14, 0.0058 | 1038, 0.0013 | 53.2 |
| Boundary L3 | 3604 | 336 / 168 | 95 (22, 2) | 4429 | 2.78 / 2.70 | 0.023 -> 0.0079 | 109, 0.0045 | 316, 9.2e-5 | 4.3 |
| Boundary L4 | 13860 | 672 / 336 | 85 (22, 2) | 5129 | 11.9 / 11.7 | 0.056 -> 0.014 | 53, 0.0078 | 219, 1.2e-4 | 7.9 |
| Boundary L5 | 54340 | 1344 / 672 | 105 (24, 5) | 7439 | 97.0 / 94.9 | 0.53 -> 0.13 | 157, 0.084 | 580, 4.5e-4 | 14.7 |
| Boundary L6 | 215172 | 2688 / 1344 | 145 (30, 23) | 13557 | 1430 / 1430 | 6.99 -> 1.73 | 616, 1.30 | 2009, 0.0032 | 27.5 |

Fallback: none exists and none is needed (identical admissibility in both strategies); certificate
violations = builds - 1. `cutoff hits` = 0 in all runs; ties = 0.

Breakdown of the solve (instrumented rerun, Faces L5 Verlet, 65.2 s): AMG updates 21.6 s (33 %),
linear solves 43.0 s (66 %), of which matrix-free Jacobian-vector products 10.4 s (16 %) and
AMG application + Krylov vector operations ~32.6 s (50 %); residual evaluations 0.47 s (0.7 %),
search 0.014 s (0.02 %). Boundary L5 Verlet (99.4 s): AMG updates 25.5 s, linear solves 73.1 s
(vmult 17.8 s), residuals 0.67 s, search 0.13 s.

Interpretation (measured): the Verlet list reduces the search time by 3-7x and its share of the
residual evaluation from 15 % (Faces) / 50 % (Boundary) to 2.5-20 %, but even the exhaustive search is
at most 0.55 % of the solve (Boundary L5: 0.53 s of 97 s; Faces L6: 0.15 %), so the total solve
time does not change measurably (run-to-run noise ~3 %). Within the Verlet search, the exhaustive
O(N_s N_m) list build dominates for Boundary L6 (1.30 s of 1.73 s, 616 builds).

### Comparison with the analytic penalty contact (rigid plane)

`LowerBlock = Clamped` (deformable contact, master = clamped top face of the lower block, found by
the search) vs. `LowerBlock = AnalyticPlane` (existing `contact_bc` penalty with `phi(x) = y`), same
mesh, `eps`, load steps and tolerances (measured):

| case | Newton (Clamped / Analytic) | Krylov (Clamped / Analytic) | F_y Clamped | F_y Analytic | min gap Clamped | min gap Analytic |
|---|---|---|---|---|---|---|
| L3, no preconditioner | 15 / 15 | 2042 / 2042 | 4.2378568690e-02 | 4.2378568690e-02 | -4.5105417683e-04 | -4.5105417683e-04 |
| L3, AMG | 15 / 15 | 300 / 87 | 4.2378568679e-02 | 4.2378568680e-02 | -4.5105417830e-04 | -4.5105417643e-04 |
| L5, AMG | 21 / 21 | 462 / 149 | 4.2343397902e-02 | 4.2343397902e-02 | -4.5068780712e-04 | -4.5068780708e-04 |

Without preconditioner, the iteration counts and all printed digits coincide, i.e. residual and
Jacobian-vector product of the two implementations agree for a flat obstacle. With AMG, the
results agree within the solver tolerances (relative differences ~1e-9 in the pressure), but the
deformable contact needs ~3x more Krylov iterations, because the analytic contact is part of the
assembled AMG matrix and the deformable contact is not.

## Step A: contact stiffness in the AMG matrix (measured 2026-11-27)

`ContactInPreconditioner = true` adds the penalty part `eps w (dg/du) (x) (dg/du)` (rank one,
positive semi-definite per active point, geometric part omitted) at the linearization state to the
matrix from which AMG is built (`NonLinearOperator::calculate_system_matrix()`, which hides the
`OperatorBase` version for `PreconditionerAMG<NonLinearOperator>`; the matrix of the matrix-based
operator is not affected). A1 (`PreconditionerCoupling = false`): only the slave-slave and
master-master blocks (entries already in the pattern). A2 (`PreconditionerCoupling = true`,
default): full block including the slave-master coupling; the sparsity pattern is rebuilt when the
active pairs need couplings that are not yet contained (monotonically growing set of vertex pairs).

Sequential runs, one thread, Faces, eps = 100, same input except the preconditioner option:

| case | Krylov: none / A1 / A2 (analytic) | solve [s]: none / A1 / A2 (analytic) |
|---|---|---|
| Clamped L5 | 462 / 149 / 149 (149) | 10.3 / 8.3 / 7.7 (7.1) |
| Deformable L4 | 2798 / 3079 / 606 | 7.47 / 8.83 / 4.82 |
| Deformable L5 | 4728 / 4288 / 1838 | 70.6 / 62.7 / 39.3 |

Newton iterations and reaction are unchanged. A1 suffices for a rigid master but not for two
deformable bodies, since the contact energy penalizes the relative motion `n . (u_s - u_m)`.
Cost of A2 at L5: 62 pattern extensions (6602 vertex pairs), 1.24 s; matrix contributions 0.12 s.

## Step B1: diagnosis of the Newton stagnation (measured)

`LogContactState = true` prints, per linearization, the changes of the contact state and the
active points with two admissible master faces, per residual evaluation the master switches with
respect to the linearization state, and the Newton residuals. Deformable L5, load step 0.75
(failed): |r| stagnates at 1.97e-6 with damping factors down to 1.5e-5; between linearizations the
contact state does not change (0 activations, 0 switches), 5-9 active points lie in the overlap of
two master segments (normal angle ~4.5e-3 rad, relative |g| difference ~1.2e-4), and every line
search trial, even with damping ~1e-5, switches the master face of one point (0 status changes).
The residual jumps by ~ p w dtheta (0.045 * 0.0031 * 0.0045 ~ 6e-7 per point) at the switching
locus of the C0 master surface. With a clamped (flat) master, the angles are zero and no load step
fails. Conclusion: the stagnation is caused by the discontinuous master normal at master vertices,
not by active-set changes or the search; next step B2: continuous master normal.

## Step B2: smoothed (G1) master surface (measured 2026-11-27)

`SmoothNormals = true`: Hermite cubic on each master face with tangents perpendicular to the
averaged nodal normals (no averaging at corners of the reference configuration or folds > 60
degrees), closest point projection onto the curve, gap and residual/tangent by automatic
differentiation (hyper-dual numbers; Newton in double precision followed by two hyper-dual Newton
steps, which gives exact first and second derivatives). The Verlet certificate includes the
deviation of the curve from the chord (<= 0.16 l). A first variant with averaged normals but
straight faces made the gap continuous but not its gradient: the force direction still jumped at
the vertex (test 6e showed the same jump of 5e-4 as the segment formulation), so it was replaced.

Checks (`search_tests`, test 6 and 7, all passing): straight surface: identical projection, gap
and slave forces as the segment formulation; patch test with uniform pressure on non-matching
meshes: master nodal forces deviate by 2.7e-5 from p h (segment: 2.55e-4); Verlet == exhaustive
bitwise; tangent vs. directional finite differences (two blocks with curved master, U strip
self-contact): O(h^2) down to ~1e-10; total force and moment of the contact residual ~1e-16;
continuity across a convex master vertex: the residual change over a sliding step ds is
proportional to ds (2.25e-5 for 1e-4, 2.25e-6 for 1e-5), while the segment formulation jumps by
5e-4 independently of ds.

Mechanics (sequential, one thread, coupled AMG preconditioner, Newton abs. tol. 1e-10):

| case | Newton (load steps; 21 = no failure) | Krylov | solve [s] | F_y |
|---|---|---|---|---|
| rigid master L4, smoothed / analytic penalty | 18 (21) / 18 (21) | 125 / 125 | 1.96 / 1.72 | 4.2353415683e-02 / 4.2353415683e-02 |
| deformable L3, segment / smoothed | 79 (22) / 65 (21) | 790 / 620 | 1.48 / 1.32 | 2.43869e-02 / 2.43689e-02 |
| deformable L4, segment / smoothed | 73 (22) / 67 (21) | 657 / 799 | 5.19 / 4.93 | 2.43584e-02 / 2.43500e-02 |
| deformable L5, segment / smoothed | 90 (23) / 69 (21) | 1988 / 1690 | 41.2 / 25.4 | 2.43488e-02 / 2.43463e-02 |
| deformable L6, segment / smoothed | 105 (26) / 89 (22) | 2631 / 2541 | 281 / 143 | 2.434554e-02 / 2.434423e-02 |

The difference of the reaction between the formulations decreases with refinement (1.8e-5,
8.4e-6, 2.5e-6, 1.3e-6), i.e. both converge to the same solution. Smoothed L4 with
`VerifySearch`: 0 mismatches in 14080 queries. The remaining failed load step of the smoothed run
at L6 (load factor 0.65) is not caused by the contact discretization: the non-monotone rule of
Newton's method (factor 1e6, from the rigid obstacle application) accepts a full step that
increases the residual from 5e-3 to 0.36, after which the line search fails (next candidate:
a smaller non-monotone factor).

Also fixed: `PreconditionerML::update()` now clears the old AMG hierarchy before the matrix is
re-calculated; otherwise the extension of the sparsity pattern (A2) re-created the matrix while the
old hierarchy referred to it (segmentation fault in `ML_Ifpack_Destroy`, observed at L3).

## Newton: non-monotone factor (measured 2026-11-27)

Deformable blocks, smoothed surface, L6: factor 1e6 (default of the rigid obstacle application):
1 failed load step, 89 Newton, 143 s; factor 10: 0 failed, 89, 142 s; factor 2: 0 failed, 71,
119 s; non-monotone steps disabled: identical to factor 2. With factor 2: no failed load step at
L3-L6 (Newton 56 / 58 / 61 / 71), reaction unchanged (relative 1e-9). Default of this
application: `NonmonotoneFactor = 2`.

## Speed compared with the classical penalty and Nitsche contact (measured 2026-11-27)

Same problem for all methods: rigid plane y = 0 (lower block clamped), upper block approach,
compression and frictionless sliding, AMG (updated every Newton iteration), Newton abs. tol.
1e-10, non-monotone factor 2, sequential runs, one thread. Ours = `LowerBlock = Clamped`
(master faces found by the Verlet search, contact in the AMG matrix); classical = `LowerBlock =
AnalyticPlane` with `contact_bc` (penalty eps = 100, Nitsche theta = 0, gamma_0 E = 10, GMRES).

| L (DoFs) | ours smoothed | ours segment | classical penalty | classical Nitsche |
|---|---|---|---|---|
| 4 (13860) | 1.95 s (18 / 125) | 1.82 s (18 / 125) | 1.70 s (18 / 125) | 1.72 s (18 / 116) |
| 5 (54340) | 7.93 s (21 / 144) | 7.56 s (21 / 144) | 7.04 s (21 / 144) | 7.09 s (20 / 135) |
| 6 (215172) | 39.5 s (27 / 204) | 37.7 s (27 / 204) | 35.6 s (27 / 204) | 34.1 s (26 / 180) |

(solve time, Newton / Krylov iterations). Both penalty variants give the same reaction
(4.2339380242e-02 at L6, penetration 4.5e-4); Nitsche enforces the constraint (4.3195e-02, 2 %
larger reaction, i.e. penalty with eps = 100 is 2 % too soft for this compression). Overhead of
ours (smoothed) against the classical penalty at L6: 3.9 s = extensions of the AMG sparsity
pattern 1.97 s (25 extensions, partly couplings to Dirichlet-constrained vertices), projections
onto the smoothed curve in the search 1.08 s (segment: 0.007 s), Hessians at the linearization
0.50 s. AMG setup dominates all methods (30-32 s of 34-40 s at L6).

## Optimizations of the overhead (measured 2026-11-27)

(1) Couplings to vertices whose components are all constrained are not added to the sparsity
pattern of the AMG matrix (they have no entries); the elasticity pattern is computed once and
extended in place (the set of couplings only grows). (2) Exact pre-filter for the smoothed
surface: an admissible pair has a distance of x_s to the chord of at most r_c + (0.16 + tau) l, so
other pairs are rejected without the projection onto the curve (57 % of the projections at L6).
All reaction files are bitwise identical to the runs before the optimizations; all tests pass.

| case (L6) | solve before | solve after | pattern extensions | smoothed search |
|---|---|---|---|---|
| ours smoothed, rigid plane | 39.5 s | 35.9 s | 25 (1.97 s) -> 1 (0.08 s) | 1.08 s -> 0.29 s |
| ours segment, rigid plane | 37.7 s | 35.4 s | 25 (1.97 s) -> 1 (0.08 s) | - |
| classical penalty | 35.6 s | 34.9 s (rerun) | - | - |
| classical Nitsche | 34.1 s | 34.3 s (rerun) | - | - |
| deformable blocks, smoothed | 119 s | 115 s | 44 (3.54 s) -> 44 (1.97 s) | 3.44 s -> 0.96 s |

On the rigid plane, the deformable contact is now within 3 % (smoothed) and 1.4 % (segment) of
the classical penalty contact, which is the order of the run-to-run variation (~3 %).

## Matrix-free multigrid with the contact on all levels (measured 2026-11-27)

`Preconditioner = Multigrid` (h-multigrid with separate coarse triangulations, Chebyshev(5) with
point Jacobi, AMG on the coarsest level): on every level, a `DeformableContact` is set up on the
mesh of the level with the data of the fine level (re-discretization: own contact faces, slave
points, Verlet list); its linearization state is set by `set_solution_linearization()` with the
displacement interpolated to the level. The level operators contain the contact in `vmult`,
`vmult_add`, `vmult_interface_down` / `vmult_add_interface_up` (used by the multigrid algorithm for
the residual before the restriction; without them the V-cycle was not symmetric once contact was
active: b.Pa = -18.87 vs. a.Pb = -19.23, and CG failed), in the inverse diagonal of the smoother
(`DeformableContact::add_diagonal()`, equal to the diagonal of J e_i column by column, tested) and
in the coarse AMG matrix. On the levels, only the positive semi-definite part
eps w (dg/du) (x) (dg/du) is used (the geometric part makes the level operators indefinite for
large penetrations, e.g. at the first contact; Chebyshev then needs 910 instead of 250
iterations at L5), and no Hessians are computed there. The fine operator is unchanged (Newton
iterations and reactions are identical).

Two deformable blocks, smoothed surface (sequential, one thread):

| L | AMG: Krylov / solve | MG without contact on the levels | MG with contact on the levels |
|---|---|---|---|
| 3 | 517 / 1.17 s | 2121 / 2.21 s | 328 / 0.97 s |
| 4 | 709 / 4.56 s | 2821 / 9.94 s | 524 / 3.36 s |
| 5 | 1531 / 23.0 s | 3178 / 43.4 s | 612 / 12.4 s |
| 6 | 2136 / 115 s | - | 878 / 60.8 s |

Most load steps need 4-5 Krylov iterations per Newton iteration independently of L; the first
Newton iteration after the first contact (penetration 0.006, all points active) is the most
expensive one for all preconditioners (L5: MG 250, AMG 717).

Rigid plane with multigrid (the classical penalty and Nitsche contact support multigrid levels):

| method | L5 | L6 | L6 with AMG |
|---|---|---|---|
| ours (smoothed) | 3.21 s (108 Krylov) | 13.9 s (147) | 35.9 s |
| classical penalty | 2.53 s (108) | 12.5 s (147) | 34.9 s |
| classical Nitsche | 2.62 s (103) | 13.5 s (135) | 34.3 s |

Multigrid is 2.5x faster than AMG for all methods; the deformable contact on the levels needs the
same Krylov iterations as the analytic contact. Remaining overhead of ours at L6 (+1.4 s):
Hessians at the fine linearization (~0.5 s), gradients and search in the residual (~0.4 s),
contact linearization on the levels (~0.4 s).

## Cheaper derivatives of the smoothed surface (measured 2026-11-27)

Gradient and Hessian of the gap are computed in one forward pass with all directions at once
(`Dual<N>`: value and gradient; `Taylor2<N>`: value, gradient and packed Hessian) instead of one
hyper-dual evaluation per pair of directions (78 for 12 DoFs, each repeating the Newton
projection), and only with respect to the unconstrained local DoFs (the entries of constrained DoFs
are never used; postprocessing and the verification use all directions). The hyper-dual
implementation is kept as reference: `verify_smooth_derivatives()` (search_tests) gives relative
differences of ~1e-14. Reactions unchanged (rigid: same digits; deformable L6: 3e-15 relative).

Rigid plane with multigrid (sequential, one thread, same session):

| | ours (smoothed) | classical penalty | classical Nitsche |
|---|---|---|---|
| L5 | 2.57 s | 2.50 s | 2.60 s |
| L6 | 13.2 s | 12.4 s | 12.9 s |

(before: 3.21 s / 13.9 s). Fine linearization at L6: 0.51 s -> 0.13 s (penalty 0.074 s). The
remaining difference at L6 (~0.8 s, 6.5 %) is the search (projections onto the curves, ~0.3 s) and
the contact on the multigrid levels (~0.3 s). Deformable blocks L6 with multigrid: 59.9 s (no
constrained DoFs at the contact, unchanged).

## Nitsche's method between deformable bodies (measured 2026-11-27)

`DeformableContactMethod = Nitsche` (theta = 0, frictionless): lambda = [A]_+ with
A = n_m . (P_s N_s) - gamma g, where P_s N_s is the traction of the slave body evaluated in the
slave cell (St. Venant-Kirchhoff with constant coefficients, as in the rigid obstacle code) and
gamma = gamma_0 E (k+1)^2 |dK| / |K| of the slave cell (`PenaltyParameter` = gamma_0 E). Residual
R = - w lambda dg/du; tangent J v = - w (H(A) delta A dg/du + lambda d^2g/du^2 v) with
delta A = delta n . t + n . (delta P N_s) - gamma dg . v (non-symmetric, GMRES); delta n in closed
form (segment) or by one-pass automatic differentiation (smoothed surface). The preconditioner
(AMG matrix, multigrid levels) uses gamma w H(A) dg (x) dg. Points without an admissible master
face within r_c are inactive; points where the slave normal traction exceeds gamma r_c are counted
as cutoff hits (0 in all runs).

Checks: tangent vs. finite differences O(h^2) down to ~1e-11 (two blocks segment/smoothed, U strip
self-contact), diagonal == columns of J exactly; rigid master vs. the analytic Nitsche contact
(gamma_0 E = 10): reaction and maximum pressure identical in all 10 printed digits at L4 and L5,
Newton 18/18 and 20/20 iterations, solve +2.6 % at L5 (AMG).

Two deformable blocks, L4, AMG (penalty converges to Nitsche for eps -> infinity):

| method | Newton | Krylov | solve | F_y | min gap |
|---|---|---|---|---|---|
| penalty eps = 100 | 58 | 720 | 4.35 s | 2.43500e-02 (-1.2 %) | -4e-4 |
| penalty eps = 1000 | 124 | 3580 | 10.6 s | 2.46176e-02 (-0.13 %) | -6.1e-5 |
| penalty eps = 10000 | 354 | 17007 | 75.2 s | 2.46457e-02 (-0.012 %) | -6.6e-6 |
| Nitsche gamma_0 E = 10 | 310 | 18635 | 69.2 s | 2.46487e-02 | -4.5e-6 |
| Nitsche gamma_0 E = 1 | 112 | 1428 | 7.83 s | 2.46470e-02 | -1.7e-5 |
| Nitsche gamma_0 E = 0.3 | 97 | 928 | 6.45 s | 2.46451e-02 | -4.4e-5 |

The dependence on gamma_0 halves with refinement (F_y difference gamma_0 E = 0.3 vs. 10: 7.2e-6
at L3, 3.6e-6 at L4). Multigrid with the contact on the levels: Nitsche gamma_0 E = 1, L4 6.05 s
(110 Newton), L5 183 s (189 Newton; AMG 220 s), same reaction.

Open issue: Newton's method needs many more iterations than the penalty method with eps = 100,
increasing with refinement (Nitsche gamma_0 E = 1: L4 110, L5 189; penalty eps = 100: 58, 61).
The log shows that full Newton steps flip the status of most active points (the activation
A = lambda_N - gamma g is decided by gaps of order lambda / gamma ~ 1e-5 on the deformable master),
and the line search reduces the steps to 1e-3 - 1e-1 until the active set has been rebuilt. The
penalty method with a comparable accuracy (eps = 1e4) shows the same behaviour, so this is a
property of the stiff constraint enforcement, not of the Nitsche implementation. Tried without
success: non-monotone full steps (factor 1e6 or 100: worse), no extrapolation of the load steps
(worse), keeping the previous active set in the first Jacobian of a step (no gain); these
experiments were removed again.

