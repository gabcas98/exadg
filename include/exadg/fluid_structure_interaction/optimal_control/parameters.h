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
      mismatch_reduction(0.0),
      linearized_tolerance_factor(1.0),
      linearized_tolerance_target(0.0),
      recycled_preconditioner(false),
      linearized_initial_guess(false),
      tangent_state_prediction(false),
      affine_state_prediction(true),
      affine_prediction_max_pairs(64),
      warm_start_absolute_tolerance(false),
      interface_equation(false),
      linearized_history(30),
      recycle_time_steps(8),
      recycle_max_pairs(60),
      method_schedule(""),
      stagnation_iterations(0),
      gradient_tolerance_relative(1.e-8),
      armijo_c1(1.e-4),
      line_search_max_iterations(20),
      krylov_max_iterations(100),
      forcing_term_max(1.e-1),
      forcing_term_target(0.0),
      forcing_term_target_max(0.0),
      model_newton_steps(true),
      model_step_min_success(0.5),
      model_step_rate_weight(0.1),
      model_step_probe_interval(10),
      native_tangent_step(1.e-2),
      area_ratio_from_state(true),
      surrogate_diagnostics(false),
      cheap_tangent_diagnostics(false),
      tangent_initial_guess(false),
      tangent_initial_guess_pairs(48),
      tangent_initial_guess_accuracy(1.0),
      fluid_krylov_recycling(0),
      tangent_difference_tolerance_factor(1.e4),
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
      control_extrapolation_order(1),
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
      preconditioner_update_dt_change(0.0),
      preconditioner_update_krylov(0.0),
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
      prm.add_parameter("MismatchReduction",
                        mismatch_reduction,
                        "Also stop if ||m||_M <= MismatchReduction * ||m_0||_M, m_0 the mismatch "
                        "of the extrapolated control (an estimate of the local time error); 0: "
                        "off.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("LinearizedToleranceFactor",
                        linearized_tolerance_factor,
                        "Factor applied to the relative tolerance of the linear fluid solves in "
                        "the tangent and adjoint operators S, S* (1: as for the state; the "
                        "relative tolerance is capped at 1e-2).",
                        dealii::Patterns::Double(1.0),
                        false);
      prm.add_parameter(
        "LinearizedToleranceTarget",
        linearized_tolerance_target,
        "Adaptive relative tolerance of the linear fluid solves in S, S*: kappa * tolerance / "
        "||m_k|| (kappa = LinearizedToleranceTarget, 0 = off), at least as accurate as "
        "LinearizedToleranceFactor and at most 1e-2: the derivatives need only the accuracy that "
        "lets the step reach the mismatch tolerance.",
        dealii::Patterns::Double(0.0),
        false);
      prm.add_parameter(
        "LinearizedInitialGuess",
        linearized_initial_guess,
        "Linear fluid solves of S, S*: initial guess by least-squares projection "
        "onto the previous right-hand sides and solutions, true residual, and a "
        "tolerance relative to the right-hand side (solve skipped if the projection "
        "is accurate enough).",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter(
        "WarmStartAbsoluteTolerance",
        warm_start_absolute_tolerance,
        "WarmStart: the state solves that start from the previous iterate stop at the absolute "
        "accuracy of the solve from the extrapolation (relative tolerance times its initial "
        "residual) instead of reducing their own initial residual.",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter(
        "AffineStatePrediction",
        affine_state_prediction,
        "All methods: within a time step the fluid is affine in its Dirichlet data w (the "
        "structure velocity on the interface). The pairs (w_i, u_i) of the state solves and "
        "(dw_j, du_j) of the tangent solves of the time step predict the fluid state of a new w "
        "by least squares, u = u_k + sum a_l (u_l - u_k) + sum b_j du_j; the state solve starts "
        "from this prediction and stops at the absolute accuracy of the solve from the "
        "extrapolation (relative tolerance times its initial residual). Dual splitting (with "
        "WarmStart and WarmStartAbsoluteTolerance, otherwise ignored): prediction of the initial "
        "guesses of the pressure Poisson, viscous and penalty steps from the states and the "
        "difference-quotient tangents of the time step.",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter("AffinePredictionMaxPairs",
                        affine_prediction_max_pairs,
                        "AffineStatePrediction: maximum number of tangent pairs per time step "
                        "(newest kept).",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter(
        "TangentStatePrediction",
        tangent_state_prediction,
        "InterfaceNewton and the InterfaceEquation variants: the fluid is affine in "
        "the interface data, so the fluid responses of the tangent solves of the Krylov method "
        "predict the fluid state of the Newton step, u(g + a d) ~ u(g) + a sum_j c_j du_j. The "
        "state solve starts from this prediction and stops at the absolute accuracy of the solve "
        "from the extrapolation (relative tolerance times its initial residual).",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter("LinearizedHistory",
                        linearized_history,
                        "LinearizedInitialGuess: number of stored right-hand sides/solutions.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter(
        "InterfaceEquation",
        interface_equation,
        "GaussNewton, LevenbergMarquardt, TrustRegion without regularization: work with the "
        "(preconditioned) GMRES subspace of the interface equation S d = -m instead of the normal "
        "equations (no adjoint solves). For a square, invertible S, the full step is the "
        "Gauss-Newton step; LevenbergMarquardt and TrustRegion solve their subproblems in the "
        "Krylov subspace, d = Z y, with the Arnoldi model ||m + S Z y|| = ||Hbar y - beta e_1|| "
        "(hybrid Krylov methods, Brown and Saad 1990).",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter("RecycledPreconditioner",
                        recycled_preconditioner,
                        "Matrix-free preconditioner built from the exact tangent pairs (v, S v) of "
                        "the current and previous solves: H = I + (V - Y) R^{-1} (Q, .)_M with "
                        "Y = Q R, so that H S v = v on the pairs. Used by InterfaceNewton (H), "
                        "GaussNewton and LevenbergMarquardt (H H^*), LBFGS (initial inverse "
                        "Hessian H H^*) and SteepestDescent (direction -H H^* r).",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("RecycleTimeSteps",
                        recycle_time_steps,
                        "RecycledPreconditioner: number of previous solves whose pairs are kept. The tangent map "
                        "changes from step to step; on the limit cycle of FSI2 pairs older than "
                        "about 8 steps make the recycled inverse worse.",
                        dealii::Patterns::Integer(0),
                        false);
      prm.add_parameter("RecycleMaxPairs",
                        recycle_max_pairs,
                        "RecycledPreconditioner: maximum number of pairs (newest first).",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter("MethodSchedule",
                        method_schedule,
                        "Change the method in time: comma-separated list of time:Method, e.g. "
                        "\"0:InterfaceQuasiNewton,12:InterfaceNewton\" (empty: Method for all "
                        "time steps). Statistics are reported per window.",
                        dealii::Patterns::Anything(),
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
      prm.add_parameter(
        "ForcingTermTarget",
        forcing_term_target,
        "Interface equation (InterfaceNewton, InterfaceEquation variants of GN, LM, TR): "
        "solve the linear model to theta * tolerance / ||m_k||, so that a full step reaches the "
        "mismatch tolerance (theta = ForcingTermTarget, 0 = off).",
        dealii::Patterns::Double(0.0),
        false);
      prm.add_parameter(
        "ForcingTermTargetMax",
        forcing_term_target_max,
        "With ForcingTermTarget: forcing term min(ForcingTermTargetMax, theta * tolerance / "
        "||m_k||), i.e. ForcingTermMax no longer bounds it close to the tolerance (0 = off).",
        dealii::Patterns::Double(0.0, 1.0),
        false);
      prm.add_parameter(
        "ModelNewtonSteps",
        model_newton_steps,
        "All derivative-based methods: try the step d = -H m of the recycled inverse H (exact on "
        "the tangent pairs of previous steps and on the secant pairs of the current solve) before "
        "any tangent or adjoint solve. The state of the trial point verifies the step and gives an "
        "exact secant pair (S d for the affine fluid step); the method continues with its own "
        "iteration (from the better point) only if the model step does not reach the mismatch "
        "tolerance (derivatives on demand).",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter(
        "ModelStepMinSuccess",
        model_step_min_success,
        "ModelNewtonSteps: the model step is tried while the moving average of its success "
        "(tolerance reached without tangents) is at least this value; otherwise only every "
        "ModelStepProbeInterval solves. A failed model step costs a state, a successful one saves "
        "the tangents of the solve.",
        dealii::Patterns::Double(0.0, 1.0),
        false);
      prm.add_parameter("ModelStepRateWeight",
                        model_step_rate_weight,
                        "ModelNewtonSteps: weight of the last solve in the moving average.",
                        dealii::Patterns::Double(0.0, 1.0),
                        false);
      prm.add_parameter("ModelStepProbeInterval",
                        model_step_probe_interval,
                        "ModelNewtonSteps: probe interval (solves) while the success rate is low.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter(
        "TangentInitialGuess",
        tangent_initial_guess,
        "Analytic tangent (coupled fluid solver): initial guess of the fluid solve from the "
        "tangent responses of this and previous time steps (least squares in the interface data), "
        "solved to the same absolute accuracy.",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter("TangentInitialGuessPairs",
                        tangent_initial_guess_pairs,
                        "Maximal number of tangent responses kept for TangentInitialGuess.",
                        dealii::Patterns::Integer(1),
                        false);
      prm.add_parameter(
        "FluidKrylovRecycling",
        fluid_krylov_recycling,
        "Coupled fluid solver: number of previous solutions of the tangent (and of the adjoint) "
        "fluid solves onto which the right-hand side is projected with the current matrix "
        "(minimal residual initial guess; 0 = off). The solve then continues to the absolute "
        "target times TangentInitialGuessAccuracy.",
        dealii::Patterns::Integer(0),
        false);
      prm.add_parameter("TangentInitialGuessAccuracy",
                        tangent_initial_guess_accuracy,
                        "TangentInitialGuess: factor on the absolute target of the solve.",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("CheapTangentDiagnostics",
                        cheap_tangent_diagnostics,
                        "Diagnostics: contraction of a Newton step with a tangent whose linear "
                        "fluid solve is truncated at relative accuracies 1e-1 ... 1e-3 "
                        "(coupled solver; first iteration of each time step, generic methods).",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter(
        "SurrogateDiagnostics",
        surrogate_diagnostics,
        "Diagnostics of a structured tangent model S~ = I - F~ V (exact structure part, "
        "least-squares fluid model from pairs of previous time steps): contraction of the model "
        "Newton step at the first iteration of each time step (generic methods only).",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter(
        "AreaRatioFromState",
        area_ratio_from_state,
        "GeometryInState with UseAreaScaling: area ratio j = da/dA of the structure load j g from "
        "the displacement of the latest state (true; the load of an evaluation then depends on "
        "the previous one) or from the geometry of the time step (false; consistent state map).",
        dealii::Patterns::Bool(),
        false);
      prm.add_parameter(
        "NativeTangentStep",
        native_tangent_step,
        "NativeTangentDifference Forward/Central: size of the perturbation of the fluid Dirichlet "
        "data relative to the data of the state. The dual-splitting and the coupled step (frozen "
        "geometry) are affine resp. quadratic in the data, so that large steps are exact and "
        "less sensitive to the tolerance of the fluid solver.",
        dealii::Patterns::Double(0.0),
        false);
      prm.add_parameter(
        "TangentDifferenceToleranceFactor",
        tangent_difference_tolerance_factor,
        "NativeTangentDifference Forward/Central: factor on the relative tolerances of the fluid "
        "solvers in the fluid steps of the difference quotient. The quotient needs the relative "
        "accuracy of a tangent (about 1e-2), not the one of a state: with the warm start from the "
        "state at the distance NativeTangentStep, the default 1e4 gives a relative accuracy of about "
        "1e4 * (fluid tolerance) / NativeTangentStep.",
        dealii::Patterns::Double(1.0),
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
      prm.add_parameter("ControlExtrapolationOrder",
                        control_extrapolation_order,
                        "ExtrapolateControl: polynomial order of the extrapolation of the "
                        "converged controls. 1: 2 g_{n-1} - g_{n-2} (constant time step); "
                        ">= 2: Lagrange extrapolation in the actual times of the last order + 1 "
                        "time steps (variable time step).",
                        dealii::Patterns::Integer(1, 4),
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
      prm.add_parameter(
        "PreconditionerUpdateTimeStepChange",
        preconditioner_update_dt_change,
        "FrozenJacobian: re-assemble if the time step size differs by more than this "
        "relative amount from the one of the last assembly (0: off).",
        dealii::Patterns::Double(0.0),
        false);
      prm.add_parameter(
        "PreconditionerUpdateKrylov",
        preconditioner_update_krylov,
        "FrozenJacobian: re-assemble for the next time step if the Krylov iterations "
        "per nonlinear iteration exceed this number (0: off).",
        dealii::Patterns::Double(0.0),
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
  double       mismatch_reduction;
  double       linearized_tolerance_factor;
  double       linearized_tolerance_target;
  bool         recycled_preconditioner;
  bool         linearized_initial_guess;
  bool         tangent_state_prediction;
  bool         affine_state_prediction;
  unsigned int affine_prediction_max_pairs;
  bool         warm_start_absolute_tolerance;
  bool         interface_equation;
  unsigned int linearized_history;
  unsigned int recycle_time_steps;
  unsigned int recycle_max_pairs;
  std::string  method_schedule;
  unsigned int stagnation_iterations;
  double       gradient_tolerance_relative;
  double       armijo_c1;
  unsigned int line_search_max_iterations;
  unsigned int krylov_max_iterations;
  double       forcing_term_max;
  double       forcing_term_target;
  double       forcing_term_target_max;
  bool         model_newton_steps;
  double       model_step_min_success;
  double       model_step_rate_weight;
  unsigned int model_step_probe_interval;
  double       native_tangent_step;
  bool         area_ratio_from_state;
  bool         surrogate_diagnostics;
  bool         cheap_tangent_diagnostics;
  bool         tangent_initial_guess;
  unsigned int tangent_initial_guess_pairs;
  double       tangent_initial_guess_accuracy;
  unsigned int fluid_krylov_recycling;
  double       tangent_difference_tolerance_factor;
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
  unsigned int control_extrapolation_order;
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
  double            preconditioner_update_dt_change;
  double            preconditioner_update_krylov;
  InterfaceMetric   mismatch_metric;
  InterfaceMetric   control_metric;
  double            sobolev_length_scale;
  bool              verbose;
};

} // namespace OptimalControl
} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_PARAMETERS_H_ */
