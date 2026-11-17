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

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_

// ExaDG
#include <exadg/fluid_structure_interaction/acceleration_schemes/parameters.h>
#include <exadg/fluid_structure_interaction/acceleration_schemes/partitioned_solver.h>
#include <exadg/fluid_structure_interaction/optimal_control/interface_problem.h>
#include <exadg/fluid_structure_interaction/optimal_control/optimizers.h>
#include <exadg/fluid_structure_interaction/optimal_control/parameters.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/fluid.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/structure.h>
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
class Driver
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  Driver(std::string const &                           input_file,
         MPI_Comm const &                              comm,
         std::shared_ptr<ApplicationBase<dim, Number>> application,
         bool const                                    is_test);

  void
  setup();

  void
  solve() const;

  void
  print_performance_results(double const total_time) const;

private:
  void
  setup_interface_coupling();

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  void
  coupling_structure_to_ale(VectorType const & displacement_structure) const;

  void
  coupling_structure_to_fluid(unsigned int const iteration) const;

  void
  coupling_fluid_to_structure(bool const end_of_time_step) const;

  void
  solve_subproblem_ale(VectorType const & displacement_structure) const;

  void
  solve_subproblem_fluid(unsigned int const iteration,
                         bool const         update_velocity,
                         bool const         update_pressure) const;

  void
  solve_subproblem_structure(unsigned int const iteration) const;

  void
  apply_dirichlet_neumann_scheme(VectorType &       displacement_structure_updated,
                                 VectorType const & displacement_structure,
                                 unsigned int       iteration) const;

  /*
   * Optimal-control based coupling (AccelerationMethod::OptimalControl): the interface traction
   * is the control of an optimisation problem solved in every time step.
   */
  bool
  use_optimal_control() const
  {
    return parameters.acceleration_method == AccelerationMethod::OptimalControl;
  }

  // one line per time step with norms of the structural solution (for comparisons of schemes)
  void
  print_time_step_summary() const;

  // diagnostics: pressure and viscous forces of the fluid on the interface (and, for the optimal
  // control coupling, the force of the control)
  void
  print_interface_forces() const;

  void
  check_optimal_control_requirements() const;

  void
  setup_optimal_control();

  void
  solve_optimal_control() const;

  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  std::shared_ptr<SolverStructure<dim, Number>> structure;

  std::shared_ptr<SolverFluid<dim, Number>> fluid;

  // interface coupling
  std::shared_ptr<InterfaceCoupling<1, dim, Number>> structure_to_fluid;
  std::shared_ptr<InterfaceCoupling<1, dim, Number>> structure_to_ale;
  std::shared_ptr<InterfaceCoupling<1, dim, Number>> fluid_to_structure;

  // Parameters for partitioned FSI schemes
  Parameters parameters;

  // Computation time
  mutable TimerTree timer_tree;

  // Partitioned FSI solver
  std::shared_ptr<PartitionedSolver<dim, Number>> partitioned_solver;

  // Optimal-control based coupling
  OptimalControl::Parameters                                     oc_parameters;
  std::shared_ptr<OptimalControl::InterfaceProblem<dim, Number>> oc_problem;
  std::shared_ptr<OptimalControl::Optimizer<OptimalControl::InterfaceProblem<dim, Number>>>
                                     oc_optimizer;
  mutable OptimalControl::Statistics oc_statistics;
  mutable unsigned int               oc_time_steps          = 0;
  mutable unsigned int               oc_geometry_iterations = 0;
  mutable unsigned int               oc_not_converged       = 0;
  mutable VectorType                 oc_last_control;

  // GeneralizedAlphaLoad::MidPoint: alpha_f of the structure (0 if not applicable) and the
  // (unblended) interface loads at t_n and t_{n+1} (Dirichlet-Neumann schemes)
  double
  midpoint_alpha_f() const;

  mutable std::shared_ptr<ContainerInterfaceData<1, dim, double>> structure_load_n,
    structure_load_np;

  // method schedule in time (OptimalControl/MethodSchedule) and statistics per window
  std::vector<std::pair<double, OptimalControl::Method>> oc_schedule;
  struct OCWindow
  {
    OptimalControl::Method     method;
    double                     t_start = 0.0, t_end = 0.0;
    unsigned int               steps = 0, not_converged = 0;
    double                     wall_time    = 0.0;
    unsigned long long         fluid_krylov = 0;
    OptimalControl::Statistics stats;
  };
  mutable std::vector<OCWindow> oc_windows;
};

} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_ */
