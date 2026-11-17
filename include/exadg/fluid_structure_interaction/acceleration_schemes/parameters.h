/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

// ExaDG
#include <exadg/utilities/enum_patterns.h>

namespace ExaDG
{
namespace FSI
{
enum class AccelerationMethod
{
  Undefined,
  Aitken,
  IQN_ILS,
  IQN_IMVLS,
  // Not a fixed-point acceleration: the interface traction is the control variable of an
  // optimisation problem (see subsection "OptimalControl" and optimal_control/parameters.h).
  OptimalControl
};

// The initial guess used in the FSI coupling loop is computed using an extrapolation of suitable
// order (defined within the single-field time integrators) or the last iterate from the previous
// time step.
/*
 * Traction transferred from the fluid to the structure on the interface.
 *
 * Nodal: Cauchy traction rho (nu (grad u + grad u^T) n - p n) of the fluid solution interpolated at
 * the nodes of the interface faces (one-sided gradient); the original behaviour.
 *
 * Flux (default): in addition the boundary terms of the numerical fluxes through which the weakly
 * imposed interface velocity acts on the fluid momentum (interior penalty term of the viscous flux,
 * continuity penalty with boundary data). This is the force that the discrete momentum equation of
 * the fluid exchanges with the interface; it differs from Nodal by the penalised slip
 * nu tau (u - v_s) and converges to the same limit. With the nodal traction the coupled solution
 * depends on the fluid solver on under-resolved boundary layers (FSI2: 8.5% in the amplitude
 * between the coupled and the dual-splitting solver, 0.05% with Flux).
 */
enum class TractionEvaluation
{
  Nodal,
  Flux
};

/*
 * Load of a generalized-alpha structure (alpha_f != 0, i.e. GenAlpha and HHT; no effect for Bossak
 * and Newmark): the structure equation is evaluated at t_{n+1-alpha_f}, so the load has to be
 * (1 - alpha_f) t_{n+1} + alpha_f t_n (MidPoint). EndOfStep applies t_{n+1} at the mid point (the
 * original behaviour), which makes the coupled time step inconsistent with the integrator.
 */
enum class GeneralizedAlphaLoad
{
  EndOfStep,
  MidPoint
};

enum class InitialGuessCouplingScheme
{
  SolutionExtrapolatedToEndOfTimeStep,
  ConvergedSolutionOfPreviousTimeStep
};

struct Parameters
{
  Parameters()
    : acceleration_method(AccelerationMethod::Undefined),
      abs_tol(1.e-12),
      rel_tol(1.e-3),
      omega_init(0.1),
      initial_guess_coupling_scheme(
        InitialGuessCouplingScheme::SolutionExtrapolatedToEndOfTimeStep),
      reused_time_steps(0),
      iqn_restart_factor(2.0),
      penalty_velocity_from_extrapolation(false),
      partitioned_iter_max(100),
      geometric_tolerance(1.e-10),
      update_ale(true),
      update_fluid_velocity(true),
      update_fluid_pressure(true),
      monitor_point(""),
      traction_evaluation(TractionEvaluation::Flux),
      generalized_alpha_load(GeneralizedAlphaLoad::MidPoint)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & subsection_name = "FSI")
  {
    prm.enter_subsection(subsection_name);
    {
      prm.add_parameter("AccelerationMethod",
                        acceleration_method,
                        "Acceleration method.",
                        Patterns::Enum<AccelerationMethod>(),
                        true);
      prm.add_parameter(
        "AbsTol", abs_tol, "Absolute solver tolerance.", dealii::Patterns::Double(0.0, 1.0), true);
      prm.add_parameter(
        "RelTol", rel_tol, "Relative solver tolerance.", dealii::Patterns::Double(0.0, 1.0), true);
      prm.add_parameter("OmegaInit",
                        omega_init,
                        "Initial relaxation parameter.",
                        dealii::Patterns::Double(0.0, 1.0),
                        true);
      prm.add_parameter("InitialGuessCouplingScheme",
                        initial_guess_coupling_scheme,
                        "Scheme for initial guess for the FSI coupling loop at every time step.",
                        Patterns::Enum<InitialGuessCouplingScheme>(),
                        false);
      prm.add_parameter("ReusedTimeSteps",
                        reused_time_steps,
                        "Number of time steps reused for acceleration.",
                        dealii::Patterns::Integer(0, 100),
                        false);
      prm.add_parameter("IQNRestartFactor",
                        iqn_restart_factor,
                        "IQN-ILS: discard the secant pairs of previous time steps if the residual "
                        "grows by more than this factor within a partitioned iteration (0: never).",
                        dealii::Patterns::Double(0.0),
                        false);
      prm.add_parameter("PenaltyVelocityFromExtrapolation",
                        penalty_velocity_from_extrapolation,
                        "Partitioned iterations: compute the penalty parameters of the fluid from "
                        "the extrapolated velocity in every iteration (default: from the last "
                        "iterate, which makes the discrete problem depend on the iteration count).",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("PartitionedIterMax",
                        partitioned_iter_max,
                        "Maximum number of fixed-point iterations.",
                        dealii::Patterns::Integer(1, 1000),
                        true);
      prm.add_parameter("GeometricTolerance",
                        geometric_tolerance,
                        "Tolerance used to locate points at FSI interface.",
                        dealii::Patterns::Double(0.0, 1.0),
                        false);
      prm.add_parameter("UpdateALE",
                        update_ale,
                        "Include ALE update in the strong coupling.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("UpdateFluidVelocity",
                        update_fluid_velocity,
                        "Include fluid velocity subproblem in the strong coupling.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("UpdateFluidPressure",
                        update_fluid_pressure,
                        "Include fluid pressure subproblem in the strong coupling.",
                        dealii::Patterns::Bool(),
                        false);
      prm.add_parameter("MonitorPoint",
                        monitor_point,
                        "Comma-separated coordinates of a structure point whose displacement and "
                        "velocity are printed after every time step (empty: disabled).",
                        dealii::Patterns::Anything(),
                        false);
      prm.add_parameter("TractionEvaluation",
                        traction_evaluation,
                        "Traction of the fluid on the interface: Flux (default; including the "
                        "penalty terms of the numerical fluxes, i.e. the force of the discrete "
                        "fluid momentum equation) or Nodal (Cauchy stress of the fluid solution at "
                        "the face nodes, the original behaviour).",
                        Patterns::Enum<TractionEvaluation>(),
                        false);
      prm.add_parameter("GeneralizedAlphaLoad",
                        generalized_alpha_load,
                        "Interface load of a generalized-alpha structure with alpha_f != 0: "
                        "MidPoint ((1 - alpha_f) t_{n+1} + alpha_f t_n, consistent) or EndOfStep "
                        "(t_{n+1} at the mid point).",
                        Patterns::Enum<GeneralizedAlphaLoad>(),
                        false);
    }
    prm.leave_subsection();
  }

  AccelerationMethod         acceleration_method;
  double                     abs_tol;
  double                     rel_tol;
  double                     omega_init;
  InitialGuessCouplingScheme initial_guess_coupling_scheme;
  unsigned int               reused_time_steps;
  double                     iqn_restart_factor;
  bool                       penalty_velocity_from_extrapolation;
  unsigned int               partitioned_iter_max;

  // tolerance used to locate points at the fluid-structure interface
  double geometric_tolerance;

  // Semi-implicit coupling variants iteratively update only a subset of all
  // fields after the first coupling step. Stability properties are altered,
  // but temporal order of accuracy is preserved starting from suitable
  // extrapolations in time controlled via `InitialGuessCouplingScheme`.
  bool update_ale;
  bool update_fluid_velocity;
  bool update_fluid_pressure;

  // point of the structure monitored after every time step (e.g. the tip of a flag)
  std::string monitor_point;

  TractionEvaluation   traction_evaluation;
  GeneralizedAlphaLoad generalized_alpha_load;
};

} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_ACCELERATION_SCHEMES_PARAMETERS_H_ */
