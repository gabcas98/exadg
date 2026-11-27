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

