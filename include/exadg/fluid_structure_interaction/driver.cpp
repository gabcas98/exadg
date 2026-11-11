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

// ExaDG
#include <exadg/fluid_structure_interaction/driver.h>
#include <exadg/grid/marked_vertices.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(std::string const &                           input_file,
                            MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
    is_test(is_test),
    application(app)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);

  dealii::ParameterHandler prm;
  parameters.add_parameters(prm);
  oc_parameters.add_parameters(prm);
  prm.parse_input(input_file, "", true, true);

  structure = std::make_shared<SolverStructure<dim, Number>>();
  fluid     = std::make_shared<SolverFluid<dim, Number>>();

  partitioned_solver = std::make_shared<PartitionedSolver<dim, Number>>(parameters, mpi_comm);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up fluid-structure interaction solver:" << std::endl;

  // setup structure
  {
    dealii::Timer timer_local;

    structure->setup(application->structure, mpi_comm, is_test);

    timer_tree.insert({"FSI", "Setup", "Structure"}, timer_local.wall_time());
  }

  // setup fluid
  {
    dealii::Timer timer_local;

    fluid->setup(application->fluid, mpi_comm, is_test);

    timer_tree.insert({"FSI", "Setup", "Fluid"}, timer_local.wall_time());
  }

  setup_interface_coupling();

  partitioned_solver->setup(fluid, structure);

  if(use_optimal_control())
    setup_optimal_control();
  else
    AssertThrow(application->structure->get_parameters().pull_back_traction == true,
                dealii::ExcMessage("Invalid parameter in context of fluid-structure interaction: "
                                   "Dirichlet-Neumann schemes require pull_back_traction = true."));

  timer_tree.insert({"FSI", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_interface_coupling()
{
  // structure to ALE
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> ALE ..." << std::endl;

    auto const & tria       = structure->pde_operator->get_dof_handler().get_triangulation();
    auto const boundary_ids = application->structure->get_boundary_descriptor()->neumann_cached_bc;
    auto const marked_vertices_structure = get_marked_vertices_via_boundary_ids(tria, boundary_ids);

    if(application->fluid->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
    {
      structure_to_ale = std::make_shared<InterfaceCoupling<1, dim, Number>>();

      std::shared_ptr<Poisson::DeformedMapping<dim, Number>> poisson_grid_motion =
        std::dynamic_pointer_cast<Poisson::DeformedMapping<dim, Number>>(fluid->ale_mapping);
      structure_to_ale->setup(
        poisson_grid_motion->get_pde_operator()->get_container_interface_data(),
        structure->pde_operator->get_dof_handler(),
        *structure->mapping,
        marked_vertices_structure,
        parameters.geometric_tolerance);
    }
    else if(application->fluid->get_parameters().mesh_movement_type ==
            IncNS::MeshMovementType::Elasticity)
    {
      structure_to_ale = std::make_shared<InterfaceCoupling<1, dim, Number>>();

      std::shared_ptr<Structure::DeformedMapping<dim, Number>> elasticity_grid_motion =
        std::dynamic_pointer_cast<Structure::DeformedMapping<dim, Number>>(fluid->ale_mapping);
      structure_to_ale->setup(
        elasticity_grid_motion->get_pde_operator()->get_container_interface_data_dirichlet(),
        structure->pde_operator->get_dof_handler(),
        *structure->mapping,
        marked_vertices_structure,
        parameters.geometric_tolerance);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> ALE"}, timer_local.wall_time());
  }

  // structure to fluid (not needed for traction-controlled coupling, where the fluid has a
  // NeumannCached boundary condition on the interface)
  if(not use_optimal_control())
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling structure -> fluid ..." << std::endl;

    auto const & tria       = structure->pde_operator->get_dof_handler().get_triangulation();
    auto const boundary_ids = application->structure->get_boundary_descriptor()->neumann_cached_bc;
    auto const marked_vertices_structure = get_marked_vertices_via_boundary_ids(tria, boundary_ids);

    structure_to_fluid = std::make_shared<InterfaceCoupling<1, dim, Number>>();
    structure_to_fluid->setup(fluid->pde_operator->get_container_interface_data(),
                              structure->pde_operator->get_dof_handler(),
                              *structure->mapping,
                              marked_vertices_structure,
                              parameters.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling structure -> fluid"}, timer_local.wall_time());
  }

  // fluid to structure
  {
    dealii::Timer timer_local;
    timer_local.restart();

    pcout << std::endl << "Setup interface coupling fluid -> structure ..." << std::endl;

    auto const & tria = fluid->pde_operator->get_dof_handler_u().get_triangulation();
    auto         boundary_ids =
      application->fluid->get_boundary_descriptor()->velocity->dirichlet_cached_bc;
    for(auto const id : application->fluid->get_boundary_descriptor()->velocity->neumann_cached_bc)
      boundary_ids.insert(id);
    auto const marked_vertices_fluid = get_marked_vertices_via_boundary_ids(tria, boundary_ids);

    fluid_to_structure = std::make_shared<InterfaceCoupling<1, dim, Number>>();
    fluid_to_structure->setup(structure->pde_operator->get_container_interface_data_neumann(),
                              fluid->pde_operator->get_dof_handler_u(),
                              *fluid->mapping,
                              marked_vertices_fluid,
                              parameters.geometric_tolerance);

    pcout << std::endl << "... done!" << std::endl;

    timer_tree.insert({"FSI", "Setup", "Coupling fluid -> structure"}, timer_local.wall_time());
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  // The fluid domain is the master that dictates the start time
  structure->time_integrator->reset_time(fluid->time_integrator->get_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  // The fluid domain is the master that dictates the time step size
  structure->time_integrator->set_current_time_step_size(
    fluid->time_integrator->get_time_step_size());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_ale(VectorType const & displacement_structure) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  structure_to_ale->update_data(displacement_structure);
  timer_tree.insert({"FSI", "Coupling structure -> ALE"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_structure_to_fluid(unsigned int iteration) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  VectorType velocity_structure;
  structure->pde_operator->initialize_dof_vector(velocity_structure);
  partitioned_solver->get_structure_velocity(velocity_structure, iteration);
  structure_to_fluid->update_data(velocity_structure);

  timer_tree.insert({"FSI", "Coupling structure -> fluid"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::coupling_fluid_to_structure(bool const end_of_time_step) const
{
  dealii::Timer sub_timer;
  sub_timer.restart();

  VectorType stress_fluid;
  fluid->pde_operator->initialize_vector_velocity(stress_fluid);
  // calculate fluid stress at fluid-structure interface
  if(end_of_time_step)
  {
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity_np(),
                                               fluid->time_integrator->get_pressure_np());
  }
  else
  {
    fluid->pde_operator->interpolate_stress_bc(stress_fluid,
                                               fluid->time_integrator->get_velocity(),
                                               fluid->time_integrator->get_pressure());
  }

  stress_fluid *= -1.0;
  fluid_to_structure->update_data(stress_fluid);

  timer_tree.insert({"FSI", "Coupling fluid -> structure"}, sub_timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve_subproblem_ale(VectorType const & displacement_structure) const
{
  // update displacement boundary data at interface
  coupling_structure_to_ale(displacement_structure);

  // move the fluid mesh and update dependent data structures
  fluid->solve_ale();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve_subproblem_fluid(unsigned int const iteration,
                                            bool const         update_velocity,
                                            bool const         update_pressure) const
{
  // update velocity boundary condition for fluid
  coupling_structure_to_fluid(iteration);

  // solve fluid problem
  fluid->time_integrator->advance_one_timestep_partitioned_solve(
    iteration == 0 /* use_extrapolation */, update_velocity, update_pressure);
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve_subproblem_structure(unsigned int const iteration) const
{
  // update stress boundary condition for solid
  coupling_fluid_to_structure(true /* end_of_time_step */);

  // solve structural problem
  structure->time_integrator->advance_one_timestep_partitioned_solve(iteration ==
                                                                     0 /* use_extrapolation */);
}

template<int dim, typename Number>
void
Driver<dim, Number>::apply_dirichlet_neumann_scheme(VectorType & displacement_structure_updated,
                                                    VectorType const & displacement_structure,
                                                    unsigned int       iteration) const
{
  if(iteration == 0 or parameters.update_ale)
  {
    solve_subproblem_ale(displacement_structure);
  }

  solve_subproblem_fluid(iteration,
                         iteration == 0 or parameters.update_fluid_velocity,
                         iteration == 0 or parameters.update_fluid_pressure);

  solve_subproblem_structure(iteration);

  displacement_structure_updated = structure->time_integrator->get_displacement_np();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  if(use_optimal_control())
  {
    solve_optimal_control();
    return;
  }

  set_start_time();

  synchronize_time_step_size();

  // compute initial acceleration for structural problem
  {
    // update stress boundary condition for solid at time t_n (not t_{n+1})
    coupling_fluid_to_structure(/* end_of_time_step = */ false);
    structure->time_integrator->compute_initial_acceleration(
      application->structure->get_parameters().restarted_simulation);
  }

  // The fluid domain is the master that dictates when the time loop is finished
  while(not fluid->time_integrator->finished())
  {
    // pre-solve
    fluid->time_integrator->advance_one_timestep_pre_solve(true);
    structure->time_integrator->advance_one_timestep_pre_solve(false);

    // solve (using strongly-coupled partitioned scheme)
    auto const lambda_dirichlet_neumann =
      [&](VectorType & d_tilde, VectorType const & d, unsigned int k) {
        apply_dirichlet_neumann_scheme(d_tilde, d, k);
      };
    partitioned_solver->solve(lambda_dirichlet_neumann);

    if(not parameters.monitor_point.empty())
      print_interface_forces();

    // post-solve
    fluid->time_integrator->advance_one_timestep_post_solve();
    structure->time_integrator->advance_one_timestep_post_solve();

    print_time_step_summary();

    if(application->fluid->get_parameters().adaptive_time_stepping)
      synchronize_time_step_size();
  }
}



template<int dim, typename Number>
void
Driver<dim, Number>::print_time_step_summary() const
{
  auto & ti = *structure->time_integrator;
  pcout << "FSI step summary: t = " << std::scientific << std::setprecision(6)
        << fluid->time_integrator->get_time() << ", |d_s| = " << std::setprecision(12)
        << ti.get_displacement_n().l2_norm() << ", |v_s| = " << ti.get_velocity_n().l2_norm()
        << std::defaultfloat << std::endl;

  if(not parameters.monitor_point.empty())
  {
    auto const coordinates = dealii::Utilities::string_to_double(
      dealii::Utilities::split_string_list(parameters.monitor_point));
    AssertThrow(coordinates.size() == dim,
                dealii::ExcMessage("MonitorPoint needs dim coordinates."));
    dealii::Point<dim> point;
    for(unsigned int d = 0; d < dim; ++d)
      point[d] = coordinates[d];

    dealii::Utilities::MPI::RemotePointEvaluation<dim> evaluator;
    evaluator.reinit({point},
                     structure->pde_operator->get_dof_handler().get_triangulation(),
                     *structure->mapping);
    AssertThrow(evaluator.all_points_found(),
                dealii::ExcMessage("MonitorPoint not found in the structure domain."));

    ti.get_displacement_n().update_ghost_values();
    ti.get_velocity_n().update_ghost_values();
    auto const d =
      dealii::VectorTools::point_values<dim>(evaluator,
                                             structure->pde_operator->get_dof_handler(),
                                             ti.get_displacement_n());
    auto const v =
      dealii::VectorTools::point_values<dim>(evaluator,
                                             structure->pde_operator->get_dof_handler(),
                                             ti.get_velocity_n());
    ti.get_displacement_n().zero_out_ghost_values();
    ti.get_velocity_n().zero_out_ghost_values();

    pcout << "FSI monitor point: t = " << std::scientific << std::setprecision(10)
          << fluid->time_integrator->get_time() << ", d = " << d[0] << ", v = " << v[0]
          << std::defaultfloat << std::endl;
  }
}


template<int dim, typename Number>
void
Driver<dim, Number>::print_interface_forces() const
{
  auto const & matrix_free = fluid->pde_operator->get_matrix_free();
  auto const & param       = application->fluid->get_parameters();
  auto const   bd          = application->fluid->get_boundary_descriptor()->velocity;

  std::set<dealii::types::boundary_id> ids = bd->dirichlet_cached_bc;
  for(auto const id : bd->neumann_cached_bc)
    ids.insert(id);

  VectorType const & velocity = fluid->time_integrator->get_velocity_np();
  VectorType const & pressure = fluid->time_integrator->get_pressure_np();
  velocity.update_ghost_values();
  pressure.update_ghost_values();

  FaceIntegrator<dim, dim, Number> phi_u(matrix_free,
                                         true,
                                         fluid->pde_operator->get_dof_index_velocity(),
                                         fluid->pde_operator->get_quad_index_velocity_standard());
  FaceIntegrator<dim, 1, Number>   phi_p(matrix_free,
                                       true,
                                       fluid->pde_operator->get_dof_index_pressure(),
                                       fluid->pde_operator->get_quad_index_velocity_standard());

  dealii::Tensor<1, dim, double> force_p, force_v;
  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
      ++face)
  {
    if(ids.find(matrix_free.get_boundary_id(face)) == ids.end())
      continue;
    phi_u.reinit(face);
    phi_p.reinit(face);
    phi_u.gather_evaluate(velocity, dealii::EvaluationFlags::gradients);
    phi_p.gather_evaluate(pressure, dealii::EvaluationFlags::values);
    unsigned int const n_lanes = matrix_free.n_active_entries_per_face_batch(face);
    for(unsigned int q = 0; q < phi_u.n_q_points; ++q)
    {
      auto const n  = phi_u.normal_vector(q);
      auto const G  = phi_u.get_gradient(q);
      auto const fv = param.density * param.viscosity * ((G + transpose(G)) * n) * phi_u.JxW(q);
      auto const fp = -param.density * phi_p.get_value(q) * n * phi_u.JxW(q);
      for(unsigned int v = 0; v < n_lanes; ++v)
        for(unsigned int d = 0; d < dim; ++d)
        {
          force_v[d] += fv[d][v];
          force_p[d] += fp[d][v];
        }
    }
  }
  velocity.zero_out_ghost_values();
  pressure.zero_out_ghost_values();
  force_p = dealii::Utilities::MPI::sum(force_p, mpi_comm);
  force_v = dealii::Utilities::MPI::sum(force_v, mpi_comm);

  // forces of the fluid on the fluid boundary; the force on the structure is the negative
  pcout << "FSI interface forces: t = " << std::scientific << std::setprecision(8)
        << fluid->time_integrator->get_time() << ", F_pressure = " << force_p
        << ", F_viscous = " << force_v;
  if(use_optimal_control() and oc_last_control.size() > 0)
    pcout << ", F_control = " << oc_problem->integrate_on_interface(oc_last_control);
  pcout << std::defaultfloat << std::endl;
}

template<int dim, typename Number>
void
Driver<dim, Number>::check_optimal_control_requirements() const
{
  auto const & fluid_param     = application->fluid->get_parameters();
  auto const & structure_param = application->structure->get_parameters();

  auto const check = [](bool const condition, std::string const & message) {
    AssertThrow(condition, dealii::ExcMessage("AccelerationMethod::OptimalControl: " + message));
  };

  // fluid: the step has to be affine in the interface traction, with a symmetric system matrix
  check(fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution,
        "the fluid has to use TemporalDiscretization::BDFCoupledSolution.");
  check(not fluid_param.convective_problem() or
          fluid_param.treatment_of_convective_term == IncNS::TreatmentOfConvectiveTerm::Explicit,
        "the convective term has to be treated explicitly.");
  check(fluid_param.formulation_viscous_term ==
          IncNS::FormulationViscousTerm::DivergenceFormulation,
        "the viscous term needs the divergence formulation (natural boundary condition = Cauchy "
        "traction).");
  check(not fluid_param.viscosity_is_variable(), "variable viscosity is not supported.");
  check(not fluid_param.use_scaling_continuity,
        "scaling of the continuity equation is not supported.");
  check(not fluid_param.apply_penalty_terms_in_postprocessing_step or
          (not fluid_param.use_divergence_penalty and not fluid_param.use_continuity_penalty),
        "penalty terms have to be part of the monolithic system "
        "(apply_penalty_terms_in_postprocessing_step = false).");
  if(oc_parameters.formulation == OptimalControl::Formulation::NativeTraction)
  {
    check(not application->fluid->get_boundary_descriptor()->velocity->dirichlet_cached_bc.empty(),
          "Formulation::NativeTraction needs a DirichletCached velocity boundary condition of the "
          "fluid on the interface.");
    check(oc_parameters.control_space == OptimalControl::ControlSpace::Structure,
          "Formulation::NativeTraction requires ControlSpace = Structure.");
    check(fluid_param.treatment_of_convective_term == IncNS::TreatmentOfConvectiveTerm::Explicit and
            fluid_param.formulation_convective_term ==
              IncNS::FormulationConvectiveTerm::ConvectiveFormulation and
            fluid_param.type_dirichlet_bc_convective == IncNS::TypeDirichletBCs::Mirror,
          "Formulation::NativeTraction: the linearised convective boundary term assumes the "
          "convective formulation with the mirror principle.");
    check(fluid_param.divu_formulation == IncNS::FormulationVelocityDivergenceTerm::Weak,
          "Formulation::NativeTraction: weak formulation of the velocity divergence required.");
  }
  else
  {
    check(not application->fluid->get_boundary_descriptor()->velocity->neumann_cached_bc.empty(),
          "the fluid needs a NeumannCached velocity boundary condition on the interface.");
    check(application->fluid->get_boundary_descriptor()->velocity->dirichlet_cached_bc.empty(),
          "the fluid must not have DirichletCached velocity boundary conditions.");
  }

  // structure: dead load with symmetric tangent stiffness
  check(not oc_parameters.geometry_in_state or oc_parameters.geometry_iterations == 1,
        "GeometryInState replaces the outer geometry loop: set GeometryIterations = 1.");
  check(not structure_param.large_deformation or not structure_param.pull_back_traction,
        "the structure has to use pull_back_traction = false (the area ratio of the frozen "
        "geometry is applied by the coupling scheme).");
  check(not application->structure->get_boundary_descriptor()->neumann_cached_bc.empty(),
        "the structure needs a NeumannCached boundary condition on the interface.");
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_optimal_control()
{
  dealii::Timer timer_local;
  timer_local.restart();

  pcout << std::endl << "Setup optimal-control interface coupling ..." << std::endl;

  check_optimal_control_requirements();

  auto const & fluid_param = application->fluid->get_parameters();

  std::set<dealii::types::boundary_id> fluid_interface_ids =
    application->fluid->get_boundary_descriptor()->velocity->neumann_cached_bc;
  for(auto const id : application->fluid->get_boundary_descriptor()->velocity->dirichlet_cached_bc)
    fluid_interface_ids.insert(id);

  oc_problem = std::make_shared<OptimalControl::InterfaceProblem<dim, Number>>(
    fluid,
    structure,
    application->structure->get_boundary_descriptor()->neumann_cached_bc,
    fluid_interface_ids,
    fluid_param.viscosity,
    fluid_param.density,
    oc_parameters.use_area_scaling and application->structure->get_parameters().large_deformation,
    oc_parameters.control_space,
    oc_parameters.formulation,
    parameters.geometric_tolerance,
    mpi_comm);

  oc_problem->configure(oc_parameters);
  oc_problem->set_mesh_update(
    [this](VectorType const & displacement) { solve_subproblem_ale(displacement); });

  oc_optimizer =
    std::make_shared<OptimalControl::Optimizer<OptimalControl::InterfaceProblem<dim, Number>>>(
      *oc_problem, oc_parameters, pcout);

  pcout << "  method:            " << OptimalControl::method_to_string(oc_parameters.method)
        << std::endl
        << "  regularization:    " << oc_parameters.regularization << std::endl
        << "  geometry its.:     " << oc_parameters.geometry_iterations << std::endl
        << "  control space:     "
        << (oc_parameters.control_space == OptimalControl::ControlSpace::Fluid ? "Fluid" :
                                                                                 "Structure")
        << std::endl
        << "  interface dofs:    " << oc_problem->n_interface_dofs << std::endl
        << "  control degree reduction: " << oc_parameters.control_degree_reduction << std::endl
        << "  preconditioner:    "
        << (oc_parameters.preconditioner == OptimalControl::Preconditioner::None ?
              "None" :
              (oc_parameters.preconditioner == OptimalControl::Preconditioner::FrozenJacobian ?
                 "FrozenJacobian" :
                 "StructureDirichletToNeumann"))
        << std::endl
        << "  metric eigenvalue range (l^2 K + M, M): " << oc_problem->metric_eigenvalue_range.first
        << " ... " << oc_problem->metric_eigenvalue_range.second << std::endl;

  pcout << std::endl << "... done!" << std::endl;

  timer_tree.insert({"FSI", "Setup", "Optimal control"}, timer_local.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve_optimal_control() const
{
  set_start_time();

  synchronize_time_step_size();

  // compute initial acceleration for structural problem with the fluid traction at t_0
  {
    coupling_fluid_to_structure(/* end_of_time_step = */ false);
    structure->time_integrator->compute_initial_acceleration(
      application->structure->get_parameters().restarted_simulation);
  }

  // initial control: traction of the fluid at t_0
  VectorType control, control_old, displacement_geometry, displacement_previous;
  oc_problem->compute_fluid_traction(control, /* end_of_time_step = */ false);
  control_old = control;
  structure->pde_operator->initialize_dof_vector(displacement_geometry);
  structure->pde_operator->initialize_dof_vector(displacement_previous);

  bool verify = oc_parameters.verify_derivatives;

  while(not fluid->time_integrator->finished())
  {
    fluid->time_integrator->advance_one_timestep_pre_solve(true);
    structure->time_integrator->advance_one_timestep_pre_solve(false);

    dealii::Timer timer;
    timer.restart();

    // initial guess for the control
    if(oc_parameters.extrapolate_control and oc_time_steps > 0)
    {
      VectorType extrapolated(control);
      extrapolated.sadd(2.0, -1.0, control_old);
      control_old = control;
      control     = extrapolated;
    }
    else
    {
      control_old = control;
    }

    if(oc_parameters.preconditioner == OptimalControl::Preconditioner::FrozenJacobian and
       oc_time_steps % oc_parameters.preconditioner_update_interval == 0)
      oc_problem->request_preconditioner_update();

    // outer fixed-point iteration on the geometry of the fluid domain
    structure->time_integrator->extrapolate_displacement_to_np(displacement_geometry);

    for(unsigned int k = 0; k < oc_parameters.geometry_iterations; ++k)
    {
      solve_subproblem_ale(displacement_geometry);
      oc_problem->set_geometry(displacement_geometry);

      if(verify)
      {
        oc_optimizer->verify_derivatives(control);
        oc_problem->print_diagnostics(pcout);
        verify = false;
      }

      OptimalControl::Statistics const stats = oc_optimizer->solve(control);
      oc_statistics.add(stats);
      ++oc_geometry_iterations;

      displacement_previous = displacement_geometry;
      displacement_geometry = structure->time_integrator->get_displacement_np();

      if(oc_parameters.geometry_iterations > 1)
      {
        VectorType difference(displacement_geometry);
        difference.add(-1.0, displacement_previous);
        double const change =
          double(difference.l2_norm()) / std::max<double>(displacement_geometry.l2_norm(), 1.e-20);
        if(oc_parameters.verbose)
          pcout << "  geometry iteration " << k
                << ": relative displacement change = " << std::scientific << std::setprecision(3)
                << change << std::defaultfloat << std::endl;
        if(change <= oc_parameters.geometry_tolerance)
          break;
      }
    }

    // consistency check: control vs. traction computed from the fluid solution
    if(oc_parameters.verbose)
    {
      VectorType traction_fluid, difference;
      oc_problem->compute_fluid_traction(traction_fluid, /* end_of_time_step = */ true);
      difference = control;
      difference.add(-1.0, traction_fluid);
      pcout << "  traction check: |g| = " << std::scientific << std::setprecision(3)
            << oc_problem->interface_norm(control)
            << ", |t_fluid| = " << oc_problem->interface_norm(traction_fluid)
            << ", |g - t_fluid| = " << oc_problem->interface_norm(difference) << std::defaultfloat
            << std::endl;
    }

    if(not parameters.monitor_point.empty())
    {
      oc_last_control = control;
      print_interface_forces();
    }

    ++oc_time_steps;
    timer_tree.insert({"FSI", "Optimal control"}, timer.wall_time());

    fluid->time_integrator->advance_one_timestep_post_solve();
    structure->time_integrator->advance_one_timestep_post_solve();

    print_time_step_summary();

    if(application->fluid->get_parameters().adaptive_time_stepping)
      synchronize_time_step_size();
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

  // iterations
  pcout << std::endl << "Average number of iterations:" << std::endl;

  pcout << std::endl << "FSI:" << std::endl;
  if(use_optimal_control())
  {
    double const n = std::max(1.0, double(oc_time_steps));
    print_list_of_iterations(pcout,
                             {"Native fluid solves (tangent)",
                              "Preconditioner setup solves (S)",
                              "Geometry iterations",
                              "Optimisation iterations",
                              "State evaluations",
                              "Tangent solves (S)",
                              "Adjoint solves (S*)",
                              "Krylov iterations"},
                             {oc_problem->native_fluid_solves / n,
                              oc_problem->preconditioner_setup_solves / n,
                              oc_geometry_iterations / n,
                              oc_statistics.iterations / n,
                              oc_statistics.state_evaluations / n,
                              oc_statistics.tangent_solves / n,
                              oc_statistics.adjoint_solves / n,
                              oc_statistics.krylov_iterations / n});
  }
  else
  {
    partitioned_solver->print_iterations(pcout);
  }

  pcout << std::endl << "Fluid:" << std::endl;
  fluid->time_integrator->print_iterations();

  pcout << std::endl << "ALE:" << std::endl;
  fluid->ale_mapping->print_iterations();

  pcout << std::endl << "Structure:" << std::endl;
  structure->time_integrator->print_iterations();

  // wall times
  pcout << std::endl << "Wall times:" << std::endl;

  timer_tree.insert({"FSI"}, total_time);

  timer_tree.insert({"FSI"}, fluid->time_integrator->get_timings(), "Fluid");
  timer_tree.insert({"FSI"}, fluid->get_timings_ale());
  timer_tree.insert({"FSI"}, structure->time_integrator->get_timings(), "Structure");
  timer_tree.insert({"FSI"}, partitioned_solver->get_timings());

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  dealii::types::global_dof_index DoFs =
    fluid->pde_operator->get_number_of_dofs() + structure->pde_operator->get_number_of_dofs();

  if(application->fluid->get_parameters().mesh_movement_type == IncNS::MeshMovementType::Poisson)
  {
    std::shared_ptr<Poisson::DeformedMapping<dim, Number>> poisson_ale_mapping =
      std::dynamic_pointer_cast<Poisson::DeformedMapping<dim, Number>>(fluid->ale_mapping);

    DoFs += poisson_ale_mapping->get_pde_operator()->get_number_of_dofs();
  }
  else if(application->fluid->get_parameters().mesh_movement_type ==
          IncNS::MeshMovementType::Elasticity)
  {
    std::shared_ptr<Structure::DeformedMapping<dim, Number>> structure_ale_mapping =
      std::dynamic_pointer_cast<Structure::DeformedMapping<dim, Number>>(fluid->ale_mapping);

    DoFs += structure_ale_mapping->get_pde_operator()->get_number_of_dofs();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;

  unsigned int N_time_steps = fluid->time_integrator->get_number_of_time_steps();

  print_throughput_unsteady(pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace FSI
} // namespace ExaDG
