/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_PARAMETERS_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

// ExaDG
#include <exadg/utilities/enum_patterns.h>

namespace ExaDG
{
namespace FSI
{
namespace OptimalControl
{
/*
 * Optimisation methods for the flux (traction) control formulation
 *
 *   min_g J(g) = 1/2 ||m(g)||^2_M + gamma/2 ||g||^2_M ,  m(g) = Pi (u_fluid(g) - v_structure(g)),
 *
 * where g is the interface traction (structure side), M the interface mass matrix and Pi the
 * L2(Gamma) projection onto the interface trace space. All methods use the Riesz gradient
 * r = S^* m + gamma g, S = dm/dg, and the M inner product.
 */
enum class Method
{
  SteepestDescent,    // d = -r, Armijo (Cauchy step as initial step length)
  LBFGS,              // two-loop recursion in the M inner product, Armijo
  GaussNewton,        // inexact GN: CG on (S^*S + gamma I) d = -r, Armijo
  LevenbergMarquardt, // CG on (S^*S + (gamma + lambda) I) d = -r, gain ratio
  TrustRegion,        // GN with Steihaug-Toint CG, ratio test, reuse of the CG path
  InterfaceNewton,    // Newton-Krylov (GMRES in M) on m(g) = 0, Armijo on 1/2||m||^2
  // least-squares secant (Anderson type II / IQN-ILS) on m(g) = 0 with reuse over time steps:
  // g+ = g - omega m - (dG - omega dM) c, c = argmin ||m - dM c||_M. No tangent or adjoint solves.
  // With Formulation::NativeTraction and omega = 1 this is IQN-ILS in the traction variable.
  InterfaceQuasiNewton
};

/*
 * Initial (inverse) Jacobian H_0 of InterfaceQuasiNewton, g+ = g - H_0 m - (dG - H_0 dM) c:
 *   Relaxation: H_0 = omega I (Anderson type II / IQN-ILS),
 *   PreviousTimeStep: H_0 = multi-secant inverse Jacobian of the previous time step (IQN-IMVJ),
 *   FrozenJacobian: H_0 = S_0^{-1} with the assembled Jacobian (requires Preconditioner =
 *   FrozenJacobian and derivatives).
 */
enum class QNInitialJacobian
{
  Relaxation,
  PreviousTimeStep,
  FrozenJacobian
};

enum class TangentDifference
{
  Analytic, // linearised boundary terms and one linear fluid solve, consistent with the adjoint
  Central,  // exact for the (at most quadratic) fluid step, two fluid solves
  Forward   // one fluid solve, first-order error in the (small) quadratic part
};

/*
 * Preconditioner of the interface Newton method (right preconditioning of GMRES on S d = -m).
 *   StructureDirichletToNeumann: P^{-1} = S_s^{-1} = -(1/c_v) M^{-1} Sigma_s, where Sigma_s is the
 *   Steklov-Poincare (Dirichlet-to-Neumann) operator of the linearized structure and S_s the
 *   structure part of S. Exact if the fluid part of S is negligible, i.e. it addresses the
 *   added-mass regime (light structure), in which Dirichlet-Neumann iterations deteriorate.
 *   FrozenJacobian: see below.
 */
enum class Preconditioner
{
  None,
  StructureDirichletToNeumann,
  // P^{-1} = S_0^{-1}, with S_0 the Jacobian S assembled densely (one tangent solve per interface
  // dof) and LU-factorized, frozen over PreconditionerUpdateInterval time steps. Practical for
  // interfaces with up to a few hundred dofs.
  FrozenJacobian
};

/*
 * Inner products on the interface (section 6.2 of the report): L2(Gamma), H^{1/2}(Gamma) for the
 * trace mismatch, H^{-1/2}(Gamma) for the traction control. The Sobolev inner products are defined
 * by spectral interpolation of (l^2 K_Gamma + M) v = mu M v, K_Gamma the Laplace-Beltrami matrix.
 */
enum class InterfaceMetric
{
  L2,
  H12,
  Hm12
};

/*
 * Space of the control (Lagrange multiplier) on the interface. Mortar theory requires that the
 * multiplier space is controlled by the trace space of the side on which the kinematic condition
 * acts: with the trace of the structure, traction modes that the (coarser) fluid trace does not
 * see make S singular and the coupled scheme unstable. The fluid trace (DG velocity space on the
 * interface faces) avoids this; the kinematic condition then acts like a Dirichlet condition for
 * the fluid, as in Dirichlet-Neumann schemes.
 */
enum class ControlSpace
{
  Structure,
  Fluid
};

/*
 * Interface conditions of the state.
 *   VelocityMismatch: traction g imposed on both fields (fluid NeumannCached), kinematic mismatch
 *     m = Pi (u_f - v_s), see InterfaceProblem.
 *   NativeTraction: the fluid keeps the native DirichletCached condition with the structure
 *     velocity, the mismatch is the traction residual m = g - Pi T_f(u_f, p_f) with the native
 *     traction of ExaDG (interpolate_stress_bc). m = 0 is exactly the fixed point of the native
 *     Dirichlet-Neumann scheme. Tangent and adjoint are obtained from the linearised boundary
 *     terms of the DG fluid (viscous SIPG, divergence, continuity penalty, explicit convective
 *     term) and the transpose of the nodal stress interpolation; finite-difference tangents are
 *     available as alternatives (NativeTangentDifference).
 */
enum class Formulation
{
  VelocityMismatch,
  NativeTraction
};

struct Parameters
{
  Parameters()
    : method(Method::GaussNewton),
      regularization(0.0),
      max_iterations(50),
      mismatch_tolerance(1.e-10),
      mismatch_tolerance_relative(0.0),
      stagnation_iterations(0),
      gradient_tolerance_relative(1.e-8),
      armijo_c1(1.e-4),
      line_search_max_iterations(20),
      krylov_max_iterations(100),
      forcing_term_max(1.e-1),
      lbfgs_memory(10),
      lm_lambda_initial(1.e-2),
      lm_lambda_min(1.e-12),
      lm_lambda_max(1.e8),
      lm_max_tries(10),
      tr_radius_initial(1.0),
      tr_radius_max(1.e6),
      tr_eta(1.e-4),
      geometry_iterations(1),
      geometry_tolerance(1.e-10),
      use_area_scaling(true),
      extrapolate_control(false),
      verify_derivatives(false),
      formulation(Formulation::VelocityMismatch),
      tangent_difference(TangentDifference::Analytic),
      geometry_in_state(false),
      qn_reuse(8),
      qn_initial_relaxation(0.3),
      qn_mixing(1.0),
      qn_filter(1.e-4),
      qn_restart_factor(2.0),
      qn_initial_jacobian(QNInitialJacobian::Relaxation),
      warm_start(false),
      adaptive_inner_tolerance(false),
      inner_tolerance_safety(1.e-2),
      control_space(ControlSpace::Fluid),
      control_degree_reduction(0),
      preconditioner(Preconditioner::None),
      preconditioner_tolerance(1.e-8),
      preconditioner_update_interval(10),
      mismatch_metric(InterfaceMetric::L2),
      control_metric(InterfaceMetric::L2),
      sobolev_length_scale(1.0),
      verbose(true)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm,
                 std::string const &        subsection_name = "OptimalControl")
  {
    prm.enter_subsection(subsection_name);
    {
      prm.add_parameter("Method", method, "Optimisation method.", Patterns::Enum<Method>(), false);
      prm.add_parameter("Regularization",
                        regularization,
                        "Tikhonov parameter gamma of the control term gamma/2 ||g||^2.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("MaxIterations",
                        max_iterations,
                        "Maximum number of outer iterations per time step.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter("MismatchTolerance",
                        mismatch_tolerance,
                        "Stop if ||m||_M <= MismatchTolerance (absolute).",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("MismatchToleranceRelative",
                        mismatch_tolerance_relative,
                        "InterfaceQuasiNewton: stop if ||m||_M <= max(MismatchTolerance, "
                        "MismatchToleranceRelative * ||g||_M), i.e. relative to the size of the "
                        "control (0: absolute tolerance only).",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("StagnationIterations",
                        stagnation_iterations,
                        "InterfaceQuasiNewton: stop if the smallest ||m|| has not decreased by 10% "
                        "within this number of iterations (noise floor of the state map); the "
                        "best iterate is restored (0: off).",
                        dealii::Patterns::Integer(0),
                        false);
      prm.add_parameter("GradientToleranceRelative",
                        gradient_tolerance_relative,
                        "Stop if ||r||_M <= tol * ||r_0||_M (not used by InterfaceNewton).",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("ArmijoC1",
                        armijo_c1,
                        "Sufficient-decrease constant.",
                        dealii::Patterns::Double(0.0, 1.0),
                        false);
      prm.add_parameter("LineSearchMaxIterations",
                        line_search_max_iterations,
                        "Maximum number of backtracking steps.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter("KrylovMaxIterations",
                        krylov_max_iterations,
                        "Maximum number of inner CG/GMRES iterations.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter(
        "ForcingTermMax",
        forcing_term_max,
        "Inner relative tolerance eta_k = min(ForcingTermMax, sqrt(||r_k||/||r_0||)).",
        dealii::Patterns::Double(0.0, 1.0),
        false);
      prm.add_parameter("LBFGSMemory",
                        lbfgs_memory,
                        "Number of stored L-BFGS pairs.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter("LMLambdaInitial",
                        lm_lambda_initial,
                        "Initial Levenberg-Marquardt parameter (L2(Gamma) units).",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("LMMaxTries",
                        lm_max_tries,
                        "Maximum number of rejected LM steps per iteration.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter("TRRadiusInitial",
                        tr_radius_initial,
                        "Initial trust-region radius in the M norm.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("TRRadiusMax",
                        tr_radius_max,
                        "Maximum trust-region radius.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter(
        "GeometryIterations",
        geometry_iterations,
        "Outer fixed-point iterations on the fluid geometry per time step. "
        "1 = geometrically explicit coupling (mesh from extrapolated displacement).",
        dealii::Patterns::Integer(1),
        false);
      prm.add_parameter("GeometryTolerance",
                        geometry_tolerance,
                        "Relative tolerance on the interface displacement change of the outer "
                        "geometry iteration.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("UseAreaScaling",
                        use_area_scaling,
                        "Convert the spatial traction (fluid side) to a nominal traction "
                        "(structure side) with the area ratio da/dA of the frozen geometry.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("ExtrapolateControl",
                        extrapolate_control,
                        "Initial control 2 g_{n-1} - g_{n-2} instead of g_{n-1}.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("VerifyDerivatives",
                        verify_derivatives,
                        "Run tangent (finite difference), adjoint (dot product) and Taylor tests "
                        "in the first time step.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("Formulation",
                        formulation,
                        "Interface conditions of the state (VelocityMismatch or NativeTraction).",
                        Patterns::Enum<Formulation>(),
                        false);
      prm.add_parameter(
        "NativeTangentDifference",
        tangent_difference,
        "Fluid tangent for Formulation::NativeTraction: Analytic (consistent with the adjoint), "
        "Central or Forward finite differences of the fluid step.",
        Patterns::Enum<TangentDifference>(),
        false);
      prm.add_parameter("GeometryInState",
                        geometry_in_state,
                        "Move the fluid mesh with the current structure displacement inside every "
                        "state evaluation (instead of the outer GeometryIterations loop; the "
                        "derivatives then neglect the shape sensitivity).",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter(
        "QNReuseTimeSteps",
        qn_reuse,
        "InterfaceQuasiNewton: number of previous solves whose secant pairs are reused.",
        dealii::Patterns::Integer(0),
        false);
      prm.add_parameter("QNInitialRelaxation",
                        qn_initial_relaxation,
                        "InterfaceQuasiNewton: relaxation g+ = g - omega0 m without secant pairs.",
                        dealii::Patterns::Double(),
                        false);
      prm.add_parameter("QNMixing",
                        qn_mixing,
                        "InterfaceQuasiNewton: mixing parameter omega (1 = IQN-ILS).",
                        dealii::Patterns::Double(),
                        false);
      prm.add_parameter("QNFilter",
                        qn_filter,
                        "InterfaceQuasiNewton: relative tolerance for dropping linearly dependent "
                        "secant pairs in the QR decomposition.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("QNRestartFactor",
                        qn_restart_factor,
                        "InterfaceQuasiNewton: discard all secant pairs if ||m|| grows by more "
                        "than this factor in one iteration.",
                        dealii::Patterns::Double(1.0),
                        false);
      prm.add_parameter("QNInitialJacobian",
                        qn_initial_jacobian,
                        "InterfaceQuasiNewton: initial inverse Jacobian (Relaxation, "
                        "PreviousTimeStep = IQN-IMVJ, FrozenJacobian).",
                        Patterns::Enum<QNInitialJacobian>(),
                        false);
      prm.add_parameter("WarmStart",
                        warm_start,
                        "InterfaceQuasiNewton: start the fluid solve of a state evaluation from "
                        "the previous evaluation of the same time step (instead of the "
                        "extrapolation in time). The state map then shows the round-off noise "
                        "of the fluid solver, so MismatchTolerance has to lie above it (about "
                        "1e-8 relative for the coupled solver with tolerances 1e-14/1e-10); not "
                        "suited for solvers with loose tolerances (e.g. dual splitting).",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("AdaptiveInnerTolerance",
                        adaptive_inner_tolerance,
                        "InterfaceQuasiNewton: solve the fluid with the relative tolerance scaled "
                        "by min(1e8, max(1, safety * ||m|| / MismatchTolerance)) (capped at "
                        "1e-2, never tighter than the original one). Pays off together with "
                        "WarmStart.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("InnerToleranceSafety",
                        inner_tolerance_safety,
                        "Safety factor of AdaptiveInnerTolerance.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("ControlSpace",
                        control_space,
                        "Interface space of the traction control (Fluid or Structure trace).",
                        Patterns::Enum<ControlSpace>(),
                        false);
      prm.add_parameter("ControlDegreeReduction",
                        control_degree_reduction,
                        "Common interface space: the control lives in discontinuous polynomials of "
                        "degree k_f - r on the fluid interface faces (L2 projection face by face). "
                        "r = 0: full fluid trace. Only for ControlSpace = Fluid.",
                        dealii::Patterns::Integer(0),
                        false);
      prm.add_parameter("Preconditioner",
                        preconditioner,
                        "Preconditioner of InterfaceNewton.",
                        Patterns::Enum<Preconditioner>(),
                        false);
      prm.add_parameter("PreconditionerTolerance",
                        preconditioner_tolerance,
                        "Relative tolerance of the inner solves of the preconditioner.",
                        dealii::Patterns::Double(0.0, 1.0),
                        false);
      prm.add_parameter("PreconditionerUpdateInterval",
                        preconditioner_update_interval,
                        "FrozenJacobian: number of time steps between re-assemblies.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter("MismatchMetric",
                        mismatch_metric,
                        "Inner product of the mismatch (L2 or H12).",
                        Patterns::Enum<InterfaceMetric>(),
                        false);
      prm.add_parameter("ControlMetric",
                        control_metric,
                        "Inner product of the control (L2 or Hm12); defines the Riesz gradient.",
                        Patterns::Enum<InterfaceMetric>(),
                        false);
      prm.add_parameter("SobolevLengthScale",
                        sobolev_length_scale,
                        "Length scale l in (l^2 K_Gamma + M) defining the Sobolev metrics.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter(
        "Verbose", verbose, "Print one line per iteration.", dealii::Patterns::Bool(), false);
    }
    prm.leave_subsection();
  }

  Method       method;
  double       regularization;
  unsigned int max_iterations;
  double       mismatch_tolerance;
  double       mismatch_tolerance_relative;
  unsigned int stagnation_iterations;
  double       gradient_tolerance_relative;
  double       armijo_c1;
  unsigned int line_search_max_iterations;
  unsigned int krylov_max_iterations;
  double       forcing_term_max;
  unsigned int lbfgs_memory;
  double       lm_lambda_initial;
  double       lm_lambda_min;
  double       lm_lambda_max;
  unsigned int lm_max_tries;
  double       tr_radius_initial;
  double       tr_radius_max;
  double       tr_eta;
  unsigned int geometry_iterations;
  double       geometry_tolerance;
  bool         use_area_scaling;
  bool         extrapolate_control;
  bool         verify_derivatives;

  Formulation       formulation;
  TangentDifference tangent_difference;
  bool              geometry_in_state;
  unsigned int      qn_reuse;
  double            qn_initial_relaxation;
  double            qn_mixing;
  double            qn_filter;
  double            qn_restart_factor;
  QNInitialJacobian qn_initial_jacobian;
  bool              warm_start;
  bool              adaptive_inner_tolerance;
  double            inner_tolerance_safety;
  ControlSpace      control_space;
  unsigned int      control_degree_reduction;
  Preconditioner    preconditioner;
  double            preconditioner_tolerance;
  unsigned int      preconditioner_update_interval;
  InterfaceMetric   mismatch_metric;
  InterfaceMetric   control_metric;
  double            sobolev_length_scale;
  bool              verbose;
};

} // namespace OptimalControl
} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_PARAMETERS_H_ */
