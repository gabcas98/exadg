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

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_INTERFACE_PROBLEM_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_INTERFACE_PROBLEM_H_

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

// C/C++
#include <functional>
#include <iomanip>
#include <random>
#include <set>

// ExaDG
#include <exadg/fluid_structure_interaction/optimal_control/parameters.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/fluid.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/structure.h>
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/grid/marked_vertices.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace FSI
{
namespace OptimalControl
{
/*
 * Flux (traction) control formulation of one FSI time step (section 4 of the report, applied to
 * fluid-structure interaction).
 *
 * Unknown: the interface traction g (traction acting on the structure, spatial), an element of
 * the control space V: the trace space of the structure finite element space on Gamma or the
 * trace space of the (DG) fluid velocity space on Gamma (see ControlSpace). M is the mass matrix
 * of V on Gamma.
 *
 * State map for a given g (geometry of the fluid domain frozen):
 *   structure: nominal traction j * g on Gamma (dead load, j = da/dA of the frozen geometry),
 *              one generalized-alpha step  ->  v_s,
 *   fluid:     traction -g on Gamma, i.e. h = F(u)n = -g/(rho_f nu) with p = 0 on Gamma (the
 *              kinematic divergence formulation has the natural boundary condition
 *              rho_f (nu F(u) n - p n) = Cauchy traction), one BDF step of the coupled solver
 *              with explicit convection  ->  u_f,
 *   mismatch:  m = Pi (u_f - v_s), Pi = L2(Gamma) projection onto V.
 *
 * With the operators (primal -> dual)
 *   L_s: V -> structure,  structure load j g,       L_f: V -> fluid, fluid load -g/rho_f,
 *   R_s: structure -> V', int v_s . psi,             R_f: fluid -> V', int u_f . psi,
 * the mismatch is m = M^{-1}(R_f u_f - R_s v_s) and the derivatives are (exact for the discrete
 * problem with frozen geometry)
 *   S v       = M^{-1} [ R_f A^{-1} L_f v - c_v R_s K^{-1} L_s v ],
 *   S^*_M w   = M^{-1} [ L_f^T A^{-1} R_f^T w - c_v L_s^T K^{-1} R_s^T w ],
 * with A the (symmetric saddle point) system matrix of the coupled fluid solver, K the tangent
 * stiffness of the structure (symmetric for hyperelasticity and dead loads), and
 * c_v = d v_{n+1} / d d_{n+1-alpha_f}.
 */
template<int dim, typename Number>
class InterfaceProblem
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

private:
  typedef FaceIntegrator<dim, dim, Number>                        Integrator;
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;
  typedef ContainerInterfaceData<1, dim, double>                  Container;

  /*
   * Trace space of one side on the interface: face quadrature of that side, mask of the dofs
   * with support on Gamma (and not constrained), inverse diagonal of the interface mass matrix.
   */
  struct TraceSpace
  {
    dealii::MatrixFree<dim, Number> const * matrix_free = nullptr;
    unsigned int                            dof_index   = 0;
    unsigned int                            quad_index  = 0;
    std::set<dealii::types::boundary_id>    ids;
    VectorType                              mask, inverse_diagonal;

    bool
    contains(unsigned int const face) const
    {
      return ids.find(matrix_free->get_boundary_id(face)) != ids.end();
    }

    unsigned int
    faces_begin() const
    {
      return matrix_free->n_inner_face_batches();
    }

    unsigned int
    faces_end() const
    {
      return matrix_free->n_inner_face_batches() + matrix_free->n_boundary_face_batches();
    }
  };

public:
  InterfaceProblem(std::shared_ptr<SolverFluid<dim, Number>>     fluid_in,
                   std::shared_ptr<SolverStructure<dim, Number>> structure_in,
                   std::set<dealii::types::boundary_id> const &  structure_interface_ids,
                   std::set<dealii::types::boundary_id> const &  fluid_interface_ids,
                   double const                                  viscosity_in,
                   double const                                  fluid_density_in,
                   bool const                                    use_area_scaling_in,
                   ControlSpace const                            control_space_in,
                   Formulation const                             formulation_in,
                   double const                                  geometric_tolerance,
                   MPI_Comm const &                              mpi_comm_in)
    : fluid(fluid_in),
      structure(structure_in),
      viscosity(viscosity_in),
      fluid_density(fluid_density_in),
      use_area_scaling(use_area_scaling_in),
      control_space(control_space_in),
      formulation(formulation_in),
      mpi_comm(mpi_comm_in)
  {
    operator_coupled =
      std::dynamic_pointer_cast<IncNS::OperatorCoupled<dim, Number>>(fluid->pde_operator);
    // derivatives (tangent, adjoint) require the coupled fluid solver; derivative-free methods
    // with Formulation::NativeTraction work with any fluid solver
    AssertThrow(operator_coupled.get() or formulation_in == Formulation::NativeTraction,
                dealii::ExcMessage("Formulation::VelocityMismatch requires the coupled fluid "
                                   "solver (TemporalDiscretization::BDFCoupledSolution)."));

    structure_space.matrix_free = structure->pde_operator->get_matrix_free().get();
    structure_space.dof_index   = structure->pde_operator->get_dof_index();
    structure_space.quad_index  = structure->pde_operator->get_quad_index_public();
    structure_space.ids         = structure_interface_ids;

    fluid_space.matrix_free  = &fluid->pde_operator->get_matrix_free();
    fluid_space.dof_index    = fluid->pde_operator->get_dof_index_velocity();
    fluid_space.quad_index   = fluid->pde_operator->get_quad_index_velocity_standard();
    fluid_space.ids          = fluid_interface_ids;
    fluid_pressure_dof_index = fluid->pde_operator->get_dof_index_pressure();
    fluid_degree             = fluid->pde_operator->get_dof_handler_u().get_fe().degree;

    // fluid interface data: Neumann (traction) data for VelocityMismatch, Dirichlet (velocity) data
    // for NativeTraction
    if(formulation == Formulation::NativeTraction)
    {
      AssertThrow(control_space == ControlSpace::Structure,
                  dealii::ExcMessage(
                    "Formulation::NativeTraction requires ControlSpace::Structure."));
      fluid_neumann = fluid->pde_operator->get_container_interface_data();
      AssertThrow(fluid_neumann.get(),
                  dealii::ExcMessage("Formulation::NativeTraction needs a DirichletCached velocity "
                                     "boundary condition of the fluid on the FSI interface."));
    }
    else
    {
      fluid_neumann = fluid->pde_operator->get_container_interface_data_neumann();
      AssertThrow(fluid_neumann.get(),
                  dealii::ExcMessage("The fluid needs a NeumannCached velocity boundary condition "
                                     "on the FSI interface."));
    }
    structure_load = structure->pde_operator->get_container_interface_data_neumann();
    AssertThrow(structure_load.get(),
                dealii::ExcMessage("The structure needs a NeumannCached boundary condition on the "
                                   "FSI interface."));

    // structure dof vectors -> fluid quadrature points (E_f)
    {
      auto const & tria = structure->pde_operator->get_dof_handler().get_triangulation();
      auto const   marked_vertices =
        get_marked_vertices_via_boundary_ids(tria, structure_interface_ids);

      structure_to_fluid = std::make_shared<InterfaceCoupling<1, dim, Number>>();
      structure_to_fluid->setup(fluid_neumann,
                                structure->pde_operator->get_dof_handler(),
                                *structure->mapping,
                                marked_vertices,
                                geometric_tolerance);
    }

    // fluid dof vectors -> structure quadrature points (P_f)
    {
      fluid_at_structure = std::make_shared<Container>();
      fluid_at_structure->setup(*structure_space.matrix_free,
                                structure_space.dof_index,
                                std::vector<unsigned int>{structure_space.quad_index},
                                structure_interface_ids);

      auto const & tria          = fluid->pde_operator->get_dof_handler_u().get_triangulation();
      auto const marked_vertices = get_marked_vertices_via_boundary_ids(tria, fluid_interface_ids);

      fluid_to_structure = std::make_shared<InterfaceCoupling<1, dim, Number>>();
      fluid_to_structure->setup(fluid_at_structure,
                                fluid->pde_operator->get_dof_handler_u(),
                                *fluid->mapping,
                                marked_vertices,
                                geometric_tolerance);
    }

    setup_trace_space(structure_space);
    setup_trace_space(fluid_space);

    interior_mask.reinit(structure_space.mask);
    for(unsigned int i = 0; i < interior_mask.locally_owned_size(); ++i)
      interior_mask.local_element(i) = 1.0 - structure_space.mask.local_element(i);
    structure->pde_operator->get_affine_constraints().set_zero(interior_mask);

    structure->pde_operator->initialize_dof_vector(geometry_displacement);
    structure->pde_operator->initialize_dof_vector(linearization);

    n_interface_dofs = (unsigned int)(control().mask.l1_norm() + 0.5);
  }

  /*
   * Interface metrics and preconditioner (see parameters.h)
   */
  void
  configure(Parameters const & param)
  {
    geometry_in_state = param.geometry_in_state;
    warm_start        = param.warm_start and param.method == Method::InterfaceQuasiNewton;
    // warm start: the step has to remain a function of the boundary data only
    fluid->time_integrator->set_penalty_velocity_from_extrapolation(warm_start);
    qn_initial_relaxation    = param.qn_initial_relaxation;
    tangent_difference       = param.tangent_difference;
    preconditioner           = param.preconditioner;
    preconditioner_tolerance = param.preconditioner_tolerance;

    AssertThrow(param.control_degree_reduction == 0 or control_space == ControlSpace::Fluid,
                dealii::ExcMessage(
                  "ControlDegreeReduction is implemented for ControlSpace::Fluid."));
    AssertThrow(param.control_degree_reduction <= fluid_degree,
                dealii::ExcMessage("ControlDegreeReduction must not exceed the fluid degree."));
    control_degree = fluid_degree - param.control_degree_reduction;
    use_projection = (control_space == ControlSpace::Fluid and param.control_degree_reduction > 0);
    if(use_projection)
      n_interface_dofs = count_projected_dimension();

    auto const exponent = [](InterfaceMetric const metric) {
      if(metric == InterfaceMetric::H12)
        return 0.5;
      else if(metric == InterfaceMetric::Hm12)
        return -0.5;
      return 0.0;
    };
    double const s_m = exponent(param.mismatch_metric);
    double const s_c = exponent(param.control_metric);

    use_dense_metrics = (s_m != 0.0 or s_c != 0.0);

    AssertThrow(control_space == ControlSpace::Structure or
                  (not use_dense_metrics and
                   preconditioner != Preconditioner::StructureDirichletToNeumann),
                dealii::ExcMessage("Sobolev metrics and the structure Dirichlet-to-Neumann "
                                   "preconditioner are implemented for ControlSpace::Structure."));

    if(use_dense_metrics or preconditioner == Preconditioner::FrozenJacobian or
       param.qn_initial_jacobian != QNInitialJacobian::Relaxation)
      setup_interface_numbering();

    AssertThrow(param.qn_initial_jacobian != QNInitialJacobian::FrozenJacobian or
                  preconditioner == Preconditioner::FrozenJacobian,
                dealii::ExcMessage("QNInitialJacobian = FrozenJacobian requires Preconditioner = "
                                   "FrozenJacobian."));

    if(use_dense_metrics)
      setup_dense_metrics(s_m, s_c, param.sobolev_length_scale);
  }

  /*
   * Interface space V and inner products: (a,b)_G for the control, (a,b)_A for the mismatch.
   */
  void
  initialize_vector(VectorType & v) const
  {
    if(control_space == ControlSpace::Structure)
      structure->pde_operator->initialize_dof_vector(v);
    else
      fluid->pde_operator->initialize_vector_velocity(v);
  }

  double
  inner_product(VectorType const & a, VectorType const & b) const
  {
    if(use_dense_metrics)
      return dense_inner_product(G_dense, a, b);
    return l2_inner_product(a, b);
  }

  double
  inner_product_mismatch(VectorType const & a, VectorType const & b) const
  {
    if(use_dense_metrics)
      return dense_inner_product(A_dense, a, b);
    return l2_inner_product(a, b);
  }

  double
  l2_inner_product(VectorType const & a, VectorType const & b) const
  {
    VectorType Mb;
    initialize_vector(Mb);
    apply_mass(control(), Mb, b);
    return a * Mb;
  }

  // int_Gamma g (traction acting on the structure), integrated on the side of the control space
  dealii::Tensor<1, dim, double>
  integrate_on_interface(VectorType const & g) const
  {
    TraceSpace const & space       = control();
    auto const &       matrix_free = *space.matrix_free;

    dealii::Tensor<1, dim, double> force;
    g.update_ghost_values();
    Integrator phi(matrix_free, true, space.dof_index, space.quad_index);
    for(unsigned int face = space.faces_begin(); face < space.faces_end(); ++face)
    {
      if(not space.contains(face))
        continue;
      phi.reinit(face);
      phi.gather_evaluate(g, dealii::EvaluationFlags::values);
      unsigned int const n_lanes = matrix_free.n_active_entries_per_face_batch(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        vector const value = phi.JxW(q) * phi.get_value(q);
        for(unsigned int v = 0; v < n_lanes; ++v)
          for(unsigned int d = 0; d < dim; ++d)
            force[d] += value[d][v];
      }
    }
    g.zero_out_ghost_values();
    return dealii::Utilities::MPI::sum(force, mpi_comm);
  }

  double
  interface_norm(VectorType const & v) const
  {
    return std::sqrt(std::max(0.0, l2_inner_product(v, v)));
  }

  void
  fill_random(VectorType & v) const
  {
    initialize_vector(v);
    std::mt19937                           generator(4711 + v.locally_owned_size());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for(unsigned int i = 0; i < v.locally_owned_size(); ++i)
      v.local_element(i) = control().mask.local_element(i) * distribution(generator);
    project_control(v);
  }

  /*
   * Frozen geometry: displacement that defines the fluid mesh (already moved by the caller) and
   * the area ratio j = da/dA. The fluid trace mass depends on the current fluid geometry.
   */
  void
  set_geometry(VectorType const & displacement)
  {
    geometry_displacement = displacement;
    setup_trace_space(fluid_space);
  }

  /*
   * Traction acting on the structure computed from the current fluid solution, -rho_f sigma_f n_f,
   * represented in the control space.
   */
  void
  compute_fluid_traction(VectorType & dst, bool const end_of_time_step) const
  {
    VectorType stress;
    fluid->pde_operator->initialize_vector_velocity(stress);
    if(end_of_time_step)
      fluid->pde_operator->interpolate_stress_bc(stress,
                                                 fluid->time_integrator->get_velocity_np(),
                                                 fluid->time_integrator->get_pressure_np());
    else
      fluid->pde_operator->interpolate_stress_bc(stress,
                                                 fluid->time_integrator->get_velocity(),
                                                 fluid->time_integrator->get_pressure());
    stress *= -1.0;

    initialize_vector(dst);
    if(control_space == ControlSpace::Fluid)
    {
      dst = stress;
      dst.scale(fluid_space.mask);
      project_control(dst);
    }
    else
    {
      fluid_to_structure->update_data(stress);
      VectorType rhs;
      initialize_vector(rhs);
      integrate_container(structure_space, rhs, *fluid_at_structure, false);
      solve_mass(structure_space, dst, rhs);
    }
  }

  /*
   * State evaluation: solve the structure and the fluid for the control g and compute m(g).
   */
  void
  evaluate(VectorType & m, VectorType const & g)
  {
    if(formulation == Formulation::NativeTraction)
    {
      evaluate_native(m, g);
      return;
    }

    bool const extrapolate = not(warm_start and evaluations_in_step > 0);
    ++evaluations_in_step;

    // structure: nominal traction j * g
    fill_structure_load(g);
    // the structure always starts from the extrapolation: a warm start lets the Newton solver
    // stop without iteration for small changes of g (absolute tolerance), which makes m(g) noisy
    structure->time_integrator->advance_one_timestep_partitioned_solve(true);

    update_geometry_from_state();

    // fluid: kinematic formulation, nu F(u) n - p n = -g / rho_f  ->  h = -g / (rho_f nu)
    fill_fluid_neumann_data(g, -1.0 / (fluid_density * viscosity));
    fluid->pde_operator->set_solver_tolerance_factor(state_tolerance_factor);
    fluid->time_integrator->advance_one_timestep_partitioned_solve(extrapolate, true, true);
    fluid->pde_operator->set_solver_tolerance_factor(1.0);

    // linearization point of the structure: mid-point displacement
    auto & ti = *structure->time_integrator;
    linearization.equ(1.0 - ti.get_alpha_f(), ti.get_displacement_np());
    linearization.add(ti.get_alpha_f(), ti.get_displacement_n());

    // M m = R_f u_f - R_s v_s
    VectorType rhs, tmp;
    initialize_vector(rhs);
    initialize_vector(tmp);
    apply_R_f(rhs, fluid->time_integrator->get_velocity_np());
    apply_R_s(tmp, ti.get_velocity_np());
    rhs.add(-1.0, tmp);
    solve_mass(control(), m, rhs);
    project_control(m);
  }

  VectorType const &
  get_linearization() const
  {
    return linearization;
  }

  void
  set_linearization(VectorType const & data)
  {
    linearization = data;
  }

  /*
   * Tangent: dst = S src = M^{-1} [ R_f A^{-1} L_f src - c_v R_s K^{-1} L_s src ]
   * (diagnostics: component 1 = fluid part, component 2 = structure part)
   */
  void
  apply_S(VectorType & dst, VectorType const & src) const
  {
    if(formulation == Formulation::NativeTraction)
    {
      apply_S_native(dst, src);
      return;
    }

    VectorType rhs, tmp;
    initialize_vector(rhs);
    initialize_vector(tmp);

    if(component != 2)
    {
      BlockVectorType rhs_fluid, sol_fluid;
      operator_coupled->initialize_block_vector_velocity_pressure(rhs_fluid);
      operator_coupled->initialize_block_vector_velocity_pressure(sol_fluid);
      apply_L_f(rhs_fluid.block(0), src);
      solve_fluid(sol_fluid, rhs_fluid);
      apply_R_f(rhs, sol_fluid.block(0));
    }

    if(component != 1)
    {
      VectorType load, y;
      structure->pde_operator->initialize_dof_vector(load);
      structure->pde_operator->initialize_dof_vector(y);
      apply_L_s(load, src);
      solve_structure(y, load);
      apply_R_s(tmp, y);
      rhs.add(-velocity_factor(), tmp);
    }

    solve_mass(control(), dst, rhs);
    project_control(dst);
  }

  /*
   * Adjoint with respect to the metrics: (S v, w)_A = (v, S^* w)_G, i.e.
   * S^* = G^{-1} S^T A = (G^{-1} M) S^*_M (M^{-1} A), with the M-adjoint S^*_M = M^{-1} S^T M.
   */
  void
  apply_S_adjoint(VectorType & dst, VectorType const & src) const
  {
    if(not use_dense_metrics)
    {
      apply_S_adjoint_l2(dst, src);
      return;
    }

    VectorType tmp, tmp2;
    initialize_vector(tmp);
    initialize_vector(tmp2);
    dense_apply(tmp, pre_dense, src);
    apply_S_adjoint_l2(tmp2, tmp);
    dense_apply(dst, post_dense, tmp2);
  }

  /*
   * M-adjoint: dst = M^{-1} [ L_f^T A^{-1} R_f^T src - c_v L_s^T K^{-1} R_s^T src ]
   */
  void
  apply_S_adjoint_l2(VectorType & dst, VectorType const & src) const
  {
    if(formulation == Formulation::NativeTraction)
    {
      apply_S_adjoint_native(dst, src);
      return;
    }

    VectorType rhs, tmp;
    initialize_vector(rhs);
    initialize_vector(tmp);

    if(component != 2)
    {
      BlockVectorType rhs_fluid, sol_fluid;
      operator_coupled->initialize_block_vector_velocity_pressure(rhs_fluid);
      operator_coupled->initialize_block_vector_velocity_pressure(sol_fluid);
      apply_R_f_transpose(rhs_fluid.block(0), src);
      solve_fluid(sol_fluid, rhs_fluid); // A is symmetric
      apply_L_f_transpose(rhs, sol_fluid.block(0));
    }

    if(component != 1)
    {
      VectorType load, y;
      structure->pde_operator->initialize_dof_vector(load);
      structure->pde_operator->initialize_dof_vector(y);
      apply_R_s_transpose(load, src);
      solve_structure(y, load); // K is symmetric
      apply_L_s_transpose(tmp, y);
      rhs.add(-velocity_factor(), tmp);
    }

    rhs.scale(control().mask);
    solve_mass(control(), dst, rhs);
    project_control(dst);
  }

  bool
  has_preconditioner() const
  {
    return preconditioner != Preconditioner::None;
  }

  /*
   * dst = P^{-1} src ~ S^{-1} src (structure control space only):
   * S_s ~ -c_v Sigma_s^{-1} M  ->  S_s^{-1} = -(1/c_v) M^{-1} Sigma_s
   */
  void
  apply_preconditioner(VectorType & dst, VectorType const & src) const
  {
    if(preconditioner == Preconditioner::FrozenJacobian)
    {
      if(jacobian_needs_update)
        assemble_frozen_jacobian();
      dealii::Vector<double> x = gather(src);
      jacobian_lu.solve(x);
      scatter(dst, x);
      return;
    }

    AssertThrow(preconditioner == Preconditioner::StructureDirichletToNeumann and
                  control_space == ControlSpace::Structure,
                dealii::ExcNotImplemented());

    VectorType sigma;
    initialize_vector(sigma);
    apply_structure_steklov_poincare(sigma, src);
    solve_mass(structure_space, dst, sigma);
    dst *= -1.0 / velocity_factor();
  }

  /*
   * Diagnostics for the adjoint: symmetry of the fluid system matrix and of the structure tangent,
   * and adjoint tests of the fluid and structure parts of S separately.
   */
  void
  print_diagnostics(dealii::ConditionalOStream const & pcout)
  {
    std::mt19937                           generator(17);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    auto const randomize = [&](VectorType & v) {
      for(unsigned int i = 0; i < v.locally_owned_size(); ++i)
        v.local_element(i) = distribution(generator);
    };

    pcout << std::scientific << std::setprecision(6);

    if(formulation == Formulation::NativeTraction)
      print_native_diagnostics(pcout, randomize);

    {
      BlockVectorType x, y, Ax, Ay;
      operator_coupled->initialize_block_vector_velocity_pressure(x);
      operator_coupled->initialize_block_vector_velocity_pressure(y);
      operator_coupled->initialize_block_vector_velocity_pressure(Ax);
      operator_coupled->initialize_block_vector_velocity_pressure(Ay);
      for(unsigned int b = 0; b < 2; ++b)
      {
        randomize(x.block(b));
        randomize(y.block(b));
      }
      operator_coupled->apply_linearized_problem(Ax, x);
      operator_coupled->apply_linearized_problem(Ay, y);
      double const a = Ax * y, c = x * Ay;
      pcout << "  symmetry of fluid matrix:      rel. diff = " << std::abs(a - c) / std::abs(a)
            << std::endl;
    }

    {
      VectorType x, y, Kx, Ky;
      structure->pde_operator->initialize_dof_vector(x);
      structure->pde_operator->initialize_dof_vector(y);
      structure->pde_operator->initialize_dof_vector(Kx);
      structure->pde_operator->initialize_dof_vector(Ky);
      randomize(x);
      randomize(y);
      structure->pde_operator->get_affine_constraints().set_zero(x);
      structure->pde_operator->get_affine_constraints().set_zero(y);
      solve_structure(Kx, x);
      solve_structure(Ky, y);
      double const a = Kx * y, c = x * Ky;
      pcout << "  symmetry of structure K^{-1}:  rel. diff = " << std::abs(a - c) / std::abs(a)
            << std::endl;
    }

    {
      VectorType v, w, Sv, Sw;
      fill_random(v);
      fill_random(w);
      initialize_vector(Sv);
      initialize_vector(Sw);
      for(int c = 1; c <= 2; ++c)
      {
        component = c;
        apply_S(Sv, v);
        apply_S_adjoint(Sw, w);
        double const a = inner_product_mismatch(Sv, w), b = inner_product(v, Sw);
        pcout << "  adjoint test " << (c == 1 ? "fluid part:    " : "structure part:")
              << " (Sv,w) = " << a << ", (v,S*w) = " << b
              << ", rel. diff = " << std::abs(a - b) / std::abs(a) << std::endl;
      }
      component = 0;
    }

    pcout << std::defaultfloat;
  }

  void
  set_mesh_update(std::function<void(VectorType const &)> const & function)
  {
    mesh_update = function;
  }

  /*
   * Control of the state solves (used by InterfaceQuasiNewton)
   */
  void
  begin_time_step()
  {
    evaluations_in_step = 0;
  }

  void
  set_state_tolerance_factor(double const factor)
  {
    state_tolerance_factor = factor;
  }

  /*
   * Initial inverse Jacobian H_0 of InterfaceQuasiNewton
   */
  bool
  has_initial_inverse_jacobian(QNInitialJacobian const type) const
  {
    if(type == QNInitialJacobian::PreviousTimeStep)
      return H_previous.m() > 0;
    if(type == QNInitialJacobian::FrozenJacobian)
      return preconditioner == Preconditioner::FrozenJacobian;
    return false;
  }

  void
  apply_initial_inverse_jacobian(VectorType &            dst,
                                 VectorType const &      src,
                                 QNInitialJacobian const type) const
  {
    if(type == QNInitialJacobian::FrozenJacobian)
    {
      apply_preconditioner(dst, src);
      return;
    }
    dense_apply(dst, H_previous, src);
  }

  /*
   * IQN-IMVJ: H <- H_0 + (dG - H_0 dM) (dM^T M dM)^{-1} (M dM)^T, with H_0 the current inverse
   * Jacobian (omega I before the first update).
   */
  void
  update_initial_inverse_jacobian(std::vector<VectorType> const & dG,
                                  std::vector<VectorType> const & dM,
                                  double const                    omega)
  {
    unsigned int const n = n_dense;
    unsigned int const k = dM.size();
    if(H_previous.m() == 0)
    {
      H_previous.reinit(n, n);
      for(unsigned int i = 0; i < n; ++i)
        H_previous(i, i) = omega;
    }
    if(k == 0)
      return;

    dealii::FullMatrix<double> X(n, k), MDM(n, k), Z(k, k);
    VectorType                 tmp;
    initialize_vector(tmp);
    for(unsigned int j = 0; j < k; ++j)
    {
      dealii::Vector<double> const g_col = gather(dG[j]), m_col = gather(dM[j]);
      dealii::Vector<double>       Hm(n);
      H_previous.vmult(Hm, m_col);
      apply_mass(control(), tmp, dM[j]);
      dealii::Vector<double> const Mm = gather(tmp);
      for(unsigned int i = 0; i < n; ++i)
      {
        X(i, j)   = g_col[i] - Hm[i];
        MDM(i, j) = Mm[i];
      }
      for(unsigned int l = 0; l < k; ++l)
        Z(l, j) = gather(dM[l]) * Mm;
    }
    Z.gauss_jordan();
    dealii::FullMatrix<double> T(k, n), update(n, n);
    Z.mTmult(T, MDM); // T = Z^{-1} MDM^T
    X.mmult(update, T);
    H_previous.add(1.0, update);
  }

  void
  reset_initial_inverse_jacobian()
  {
    if(H_previous.m() > 0)
    {
      H_previous = 0.0;
      for(unsigned int i = 0; i < H_previous.m(); ++i)
        H_previous(i, i) = qn_initial_relaxation;
    }
  }

  // FrozenJacobian: re-assemble at the next application of the preconditioner
  void
  request_preconditioner_update()
  {
    jacobian_needs_update = true;
  }

  template<typename Randomize>
  void
  print_native_diagnostics(dealii::ConditionalOStream const & pcout, Randomize & randomize)
  {
    auto const container_dot = [&](Container const & a, Container const & b) {
      double sum = 0.0;
      for(auto const qi : a.get_quad_indices())
      {
        auto const & va = a.get_array_solution(qi);
        auto const & vb = b.get_array_solution(qi);
        for(unsigned int i = 0; i < va.size(); ++i)
          sum += va[i] * vb[i];
      }
      return dealii::Utilities::MPI::sum(sum, mpi_comm);
    };

    double const time = fluid->time_integrator->get_next_time();

    // data perturbation dc = E_f y for a random structure velocity y
    VectorType y;
    structure->pde_operator->initialize_dof_vector(y);
    randomize(y);
    y.scale(structure_space.mask);
    y *= std::max<double>(native_velocity.l2_norm(), 1.e-4) / std::max<double>(y.l2_norm(), 1.e-30);
    structure_to_fluid->update_data(y);
    Container const dc = *fluid_neumann;
    pcout << "  native diagnostics: |v_s| = " << native_velocity.l2_norm()
          << ", |y| = " << y.l2_norm() << ", |dc| = " << std::sqrt(container_dot(dc, dc))
          << std::endl;

    // (a) D against the fluid operator as a black box
    {
      BlockVectorType mine, r0, r1, tmp;
      operator_coupled->initialize_block_vector_velocity_pressure(mine);
      operator_coupled->initialize_block_vector_velocity_pressure(r0);
      operator_coupled->initialize_block_vector_velocity_pressure(r1);
      operator_coupled->initialize_block_vector_velocity_pressure(tmp);
      native_boundary_terms(mine, dc);

      VectorType dummy;
      *fluid_neumann = native_base_data;
      operator_coupled->rhs_linear_problem(r0, dummy, time);
      // c0 + dc
      {
        Container c = native_base_data;
        for(auto const qi : c.get_quad_indices())
          for(unsigned int i = 0; i < c.get_array_solution(qi).size(); ++i)
            const_cast<typename Container::data_type &>(c.get_array_solution(qi)[i]) +=
              dc.get_array_solution(qi)[i];
        *fluid_neumann = c;
      }
      operator_coupled->rhs_linear_problem(r1, dummy, time);
      r1.add(-1.0, r0);

      // explicit convective term: rhs -= C(u_extrapolated, c), central difference in c
      double const eps = 1.e-3;
      VectorType   c_plus, c_minus;
      fluid->pde_operator->initialize_vector_velocity(c_plus);
      fluid->pde_operator->initialize_vector_velocity(c_minus);
      for(int sign : {1, -1})
      {
        Container c = native_base_data;
        for(auto const qi : c.get_quad_indices())
          for(unsigned int i = 0; i < c.get_array_solution(qi).size(); ++i)
            const_cast<typename Container::data_type &>(c.get_array_solution(qi)[i]) +=
              (sign * eps) * dc.get_array_solution(qi)[i];
        *fluid_neumann = c;
        fluid->pde_operator->evaluate_convective_term(sign > 0 ? c_plus : c_minus,
                                                      native_extrapolated_velocity,
                                                      time);
      }
      *fluid_neumann = native_base_data;
      c_plus.add(-1.0, c_minus);
      r1.block(0).add(-0.5 / eps, c_plus);

      tmp = mine;
      tmp.add(-1.0, r1);
      pcout << "  native D vs. fluid operator:   rel. diff = " << tmp.l2_norm() / r1.l2_norm()
            << " (velocity block " << tmp.block(0).l2_norm() / r1.block(0).l2_norm()
            << ", pressure block "
            << tmp.block(1).l2_norm() / std::max<double>(r1.block(1).l2_norm(), 1.e-300) << ")"
            << std::endl;
    }

    // (b) transpose of D
    {
      BlockVectorType z, Ddc;
      operator_coupled->initialize_block_vector_velocity_pressure(z);
      operator_coupled->initialize_block_vector_velocity_pressure(Ddc);
      randomize(z.block(0));
      randomize(z.block(1));
      native_boundary_terms(Ddc, dc);
      Container DTz = native_base_data;
      native_boundary_terms_transpose(DTz, z);
      double const a = Ddc * z, b = container_dot(DTz, dc);
      pcout << "  transpose test D:              " << a << ", " << b
            << ", rel. diff = " << std::abs(a - b) / std::abs(a) << std::endl;
    }

    // (c) transpose of the nodal stress interpolation
    {
      BlockVectorType up, ITw;
      operator_coupled->initialize_block_vector_velocity_pressure(up);
      operator_coupled->initialize_block_vector_velocity_pressure(ITw);
      randomize(up.block(0));
      randomize(up.block(1));
      VectorType w, stress;
      fluid->pde_operator->initialize_vector_velocity(w);
      fluid->pde_operator->initialize_vector_velocity(stress);
      // dual vectors as produced by the coupling: w = P_f^T (weights at structure points)
      VectorType r;
      fill_random(r);
      fill_weighted_trace(structure_space, *fluid_at_structure, r, false);
      fluid_to_structure->add_transpose(w);
      fluid->pde_operator->interpolate_stress_bc(stress, up.block(0), up.block(1));
      stress_interpolation_transpose(ITw, w);
      double const a = stress * w, b = up * ITw;
      pcout << "  transpose test stress interp.: " << a << ", " << b
            << ", rel. diff = " << std::abs(a - b) / std::abs(a) << std::endl;
    }

    *fluid_neumann = native_base_data;
  }

public:
  unsigned int               n_interface_dofs                = 0;
  mutable unsigned long long native_fluid_solves             = 0;
  mutable unsigned long long preconditioner_setup_solves     = 0;
  mutable unsigned int       preconditioner_assemblies       = 0;
  mutable unsigned long long preconditioner_inner_iterations = 0;
  std::pair<double, double>  metric_eigenvalue_range         = {1.0, 1.0};

private:
  TraceSpace const &
  control() const
  {
    return control_space == ControlSpace::Structure ? structure_space : fluid_space;
  }

  double
  velocity_factor() const
  {
    return structure->time_integrator->get_velocity_derivative_wrt_mid_displacement();
  }

  /*
   * ---------------------------------------------------------------------------------------------
   * The four coupling operators and their transposes
   * ---------------------------------------------------------------------------------------------
   */

  // structure load (dual structure vector) L_s src = int j src . psi_s
  void
  apply_L_s(VectorType & dst, VectorType const & src) const
  {
    if(control_space == ControlSpace::Structure)
    {
      integrate_structure_trace(dst, src, true);
    }
    else
    {
      fluid_to_structure->update_data(src);
      integrate_container(structure_space, dst, *fluid_at_structure, true);
    }
  }

  // dst (control dual) = L_s^T y, y a structure primal vector
  void
  apply_L_s_transpose(VectorType & dst, VectorType const & y) const
  {
    if(control_space == ControlSpace::Structure)
    {
      integrate_structure_trace(dst, y, true);
    }
    else
    {
      fill_weighted_trace(structure_space, *fluid_at_structure, y, true);
      dst = 0.0;
      fluid_to_structure->add_transpose(dst);
    }
  }

  // fluid momentum load (dual fluid vector) L_f src = int (-src / rho_f) . phi_f
  void
  apply_L_f(VectorType & dst, VectorType const & src) const
  {
    if(control_space == ControlSpace::Structure)
    {
      structure_to_fluid->update_data(src);
      integrate_container(fluid_space, dst, *fluid_neumann, false);
    }
    else
    {
      apply_mass(fluid_space, dst, src);
    }
    dst *= -1.0 / fluid_density;
  }

  // dst (control dual) = L_f^T z, z a fluid primal vector
  void
  apply_L_f_transpose(VectorType & dst, VectorType const & z) const
  {
    if(control_space == ControlSpace::Structure)
    {
      fill_weighted_trace(fluid_space, *fluid_neumann, z, false);
      dst = 0.0;
      structure_to_fluid->add_transpose(dst);
    }
    else
    {
      apply_mass(fluid_space, dst, z);
    }
    dst *= -1.0 / fluid_density;
  }

  // dst (control dual) = R_f u = int u . psi
  void
  apply_R_f(VectorType & dst, VectorType const & u) const
  {
    if(control_space == ControlSpace::Structure)
    {
      fluid_to_structure->update_data(u);
      integrate_container(structure_space, dst, *fluid_at_structure, false);
    }
    else
    {
      apply_mass(fluid_space, dst, u);
    }
  }

  // dst (fluid dual) = R_f^T w
  void
  apply_R_f_transpose(VectorType & dst, VectorType const & w) const
  {
    if(control_space == ControlSpace::Structure)
    {
      fill_weighted_trace(structure_space, *fluid_at_structure, w, false);
      dst = 0.0;
      fluid_to_structure->add_transpose(dst);
    }
    else
    {
      apply_mass(fluid_space, dst, w);
    }
  }

  // dst (control dual) = R_s v = int v . psi
  void
  apply_R_s(VectorType & dst, VectorType const & v) const
  {
    if(control_space == ControlSpace::Structure)
    {
      apply_mass(structure_space, dst, v);
    }
    else
    {
      structure_to_fluid->update_data(v);
      integrate_container(fluid_space, dst, *fluid_neumann, false);
    }
  }

  // dst (structure dual) = R_s^T w
  void
  apply_R_s_transpose(VectorType & dst, VectorType const & w) const
  {
    if(control_space == ControlSpace::Structure)
    {
      apply_mass(structure_space, dst, w);
    }
    else
    {
      fill_weighted_trace(fluid_space, *fluid_neumann, w, false);
      dst = 0.0;
      structure_to_fluid->add_transpose(dst);
    }
  }

  /*
   * Common interface space: L2(Gamma) projection of a fluid trace onto discontinuous polynomials
   * of tensor degree control_degree on each fluid interface face (face-local Legendre basis on the
   * Gauss points of the face quadrature, local mass matrix with the actual JxW), followed by the
   * representation in the fluid trace space (M_f^{-1}). The projection is self-adjoint in M_f.
   */
  void
  project_control(VectorType & v) const
  {
    if(not use_projection)
      return;

    auto const & matrix_free = *fluid_space.matrix_free;

    VectorType rhs;
    initialize_vector(rhs);
    v.update_ghost_values();

    Integrator phi(matrix_free, true, fluid_space.dof_index, fluid_space.quad_index);

    unsigned int const n_q  = phi.n_q_points;
    unsigned int const n_1d = (dim == 2) ? n_q : (unsigned int)(std::round(std::sqrt(double(n_q))));
    AssertThrow(dealii::Utilities::pow(n_1d, dim - 1) == n_q,
                dealii::ExcMessage("Tensor-product face quadrature expected."));

    // Legendre polynomials (on [-1,1]) at the 1D Gauss points of the face quadrature
    dealii::QGauss<1> const          quad_1d(n_1d);
    unsigned int const               n_basis_1d = control_degree + 1;
    std::vector<std::vector<double>> legendre(n_basis_1d, std::vector<double>(n_1d));
    for(unsigned int i = 0; i < n_1d; ++i)
    {
      double const x      = 2.0 * quad_1d.point(i)[0] - 1.0;
      double       p_prev = 1.0, p = x;
      legendre[0][i] = 1.0;
      if(n_basis_1d > 1)
        legendre[1][i] = x;
      for(unsigned int k = 2; k < n_basis_1d; ++k)
      {
        double const p_next = ((2.0 * k - 1.0) * x * p - (k - 1.0) * p_prev) / double(k);
        p_prev              = p;
        p                   = p_next;
        legendre[k][i]      = p;
      }
    }
    unsigned int const n_basis = dealii::Utilities::pow(n_basis_1d, dim - 1);
    // basis function b at face quadrature point q (lexicographic tensor product)
    auto const basis = [&](unsigned int const b, unsigned int const q) {
      double       value = 1.0;
      unsigned int bb = b, qq = q;
      for(unsigned int d = 0; d < dim - 1; ++d)
      {
        value *= legendre[bb % n_basis_1d][qq % n_1d];
        bb /= n_basis_1d;
        qq /= n_1d;
      }
      return value;
    };

    dealii::FullMatrix<double> local_mass(n_basis, n_basis);
    dealii::Vector<double>     local_rhs(n_basis), coefficients(n_basis);

    for(unsigned int face = fluid_space.faces_begin(); face < fluid_space.faces_end(); ++face)
    {
      if(not fluid_space.contains(face))
        continue;

      phi.reinit(face);
      phi.gather_evaluate(v, dealii::EvaluationFlags::values);

      std::vector<vector> projected(n_q);
      unsigned int const  n_lanes = matrix_free.n_active_entries_per_face_batch(face);
      for(unsigned int lane = 0; lane < n_lanes; ++lane)
      {
        local_mass = 0.0;
        for(unsigned int q = 0; q < n_q; ++q)
          for(unsigned int i = 0; i < n_basis; ++i)
            for(unsigned int j = 0; j < n_basis; ++j)
              local_mass(i, j) += phi.JxW(q)[lane] * basis(i, q) * basis(j, q);
        local_mass.gauss_jordan();

        for(unsigned int c = 0; c < dim; ++c)
        {
          local_rhs = 0.0;
          for(unsigned int q = 0; q < n_q; ++q)
            for(unsigned int i = 0; i < n_basis; ++i)
              local_rhs(i) += phi.JxW(q)[lane] * phi.get_value(q)[c][lane] * basis(i, q);
          local_mass.vmult(coefficients, local_rhs);
          for(unsigned int q = 0; q < n_q; ++q)
          {
            double value = 0.0;
            for(unsigned int i = 0; i < n_basis; ++i)
              value += coefficients(i) * basis(i, q);
            projected[q][c][lane] = value;
          }
        }
      }

      for(unsigned int q = 0; q < n_q; ++q)
        phi.submit_value(projected[q], q);
      phi.integrate_scatter(dealii::EvaluationFlags::values, rhs);
    }
    rhs.compress(dealii::VectorOperation::add);
    v.zero_out_ghost_values();

    solve_mass(fluid_space, v, rhs);
  }

  // dimension of the common interface space (number of interface faces times local dimension)
  unsigned int
  count_projected_dimension() const
  {
    auto const & matrix_free = *fluid_space.matrix_free;

    unsigned int n_faces = 0;
    for(unsigned int face = fluid_space.faces_begin(); face < fluid_space.faces_end(); ++face)
      if(fluid_space.contains(face))
        n_faces += matrix_free.n_active_entries_per_face_batch(face);
    n_faces = dealii::Utilities::MPI::sum(n_faces, mpi_comm);

    return n_faces * dim * dealii::Utilities::pow(control_degree + 1, dim - 1);
  }

  // GeometryInState: move the fluid mesh with the current structure displacement
  void
  update_geometry_from_state()
  {
    if(not geometry_in_state)
      return;
    AssertThrow(mesh_update, dealii::ExcMessage("No mesh update function set."));
    VectorType const & d = structure->time_integrator->get_displacement_np();
    mesh_update(d);
    set_geometry(d);
  }

  /*
   * ---------------------------------------------------------------------------------------------
   * Formulation::NativeTraction
   * ---------------------------------------------------------------------------------------------
   */

  // traction acting on the structure computed natively from the current fluid solution,
  // represented in the structure trace space: dst = Pi_s (-rho sigma(u_np, p_np) n_f)
  void
  native_fluid_traction(VectorType & dst) const
  {
    VectorType stress;
    fluid->pde_operator->initialize_vector_velocity(stress);
    fluid->pde_operator->interpolate_stress_bc(stress,
                                               fluid->time_integrator->get_velocity_np(),
                                               fluid->time_integrator->get_pressure_np());
    stress *= -1.0;
    fluid_to_structure->update_data(stress);
    VectorType rhs;
    initialize_vector(rhs);
    integrate_container(structure_space, rhs, *fluid_at_structure, false);
    solve_mass(structure_space, dst, rhs);
  }

  // fluid step with the native Dirichlet data v (structure velocity) and its native traction
  void
  native_fluid_response(VectorType &       traction,
                        VectorType const & v,
                        bool const         extrapolate = true) const
  {
    structure_to_fluid->update_data(v);
    fluid->time_integrator->advance_one_timestep_partitioned_solve(extrapolate, true, true);
    native_fluid_traction(traction);
  }

  void
  evaluate_native(VectorType & m, VectorType const & g)
  {
    bool const extrapolate = not(warm_start and evaluations_in_step > 0);
    ++evaluations_in_step;

    fill_structure_load(g);
    // the structure always starts from the extrapolation: a warm start lets the Newton solver
    // stop without iteration for small changes of g (absolute tolerance), which makes m(g) noisy
    structure->time_integrator->advance_one_timestep_partitioned_solve(true);

    update_geometry_from_state();

    auto & ti = *structure->time_integrator;
    linearization.equ(1.0 - ti.get_alpha_f(), ti.get_displacement_np());
    linearization.add(ti.get_alpha_f(), ti.get_displacement_n());
    native_velocity = ti.get_velocity_np();

    VectorType traction;
    initialize_vector(traction);
    fluid->pde_operator->set_solver_tolerance_factor(state_tolerance_factor);
    native_fluid_response(traction, native_velocity, extrapolate);
    fluid->pde_operator->set_solver_tolerance_factor(1.0);
    native_traction = traction;

    // data of the linearisation of the fluid boundary terms: Dirichlet data of the state and the
    // extrapolated velocity of the explicit convective term
    native_base_data = *fluid_neumann;
    if(operator_coupled)
    {
      std::vector<VectorType const *> velocities;
      std::vector<double>             times;
      fluid->time_integrator->get_velocities_and_times(velocities, times);
      fluid->pde_operator->initialize_vector_velocity(native_extrapolated_velocity);
      for(unsigned int i = 0; i < velocities.size(); ++i)
        native_extrapolated_velocity.add(fluid->time_integrator->get_extrapolation_coefficient(i),
                                         *velocities[i]);
    }

    m = g;
    m.add(-1.0, traction);
    m.scale(structure_space.mask);
  }

  /*
   * S v = v - T'(v_s) dv_s,  dv_s = c_v K^{-1} L_s v; the fluid part T' by central differences,
   * exact since the fluid step is (at most) quadratic in the Dirichlet data.
   */
  void
  apply_S_native(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(operator_coupled.get() or tangent_difference != TangentDifference::Analytic,
                dealii::ExcMessage("The analytic tangent requires the coupled fluid solver."));
    VectorType load, y;
    structure->pde_operator->initialize_dof_vector(load);
    structure->pde_operator->initialize_dof_vector(y);
    apply_L_s(load, src);
    solve_structure(y, load);
    y *= velocity_factor();

    dst                 = src;
    double const y_norm = y.l2_norm();
    if(y_norm > 0.0 and tangent_difference == TangentDifference::Analytic)
    {
      // dT = Pi_s P_f (-I_sigma(A^{-1} D E_f y))
      structure_to_fluid->update_data(y);
      BlockVectorType rhs_fluid, sol_fluid;
      operator_coupled->initialize_block_vector_velocity_pressure(rhs_fluid);
      operator_coupled->initialize_block_vector_velocity_pressure(sol_fluid);
      native_boundary_terms(rhs_fluid, *fluid_neumann);
      *fluid_neumann = native_base_data;
      solve_fluid(sol_fluid, rhs_fluid);

      VectorType traction;
      initialize_vector(traction);
      native_traction_of(traction, sol_fluid.block(0), sol_fluid.block(1));
      dst.add(-1.0, traction);
    }
    else if(y_norm > 0.0)
    {
      double const eps = 1.e-2 * std::max<double>(native_velocity.l2_norm(), 1.e-12) / y_norm;

      VectorType v_plus(native_velocity), t_plus;
      v_plus.add(eps, y);
      initialize_vector(t_plus);
      native_fluid_response(t_plus, v_plus);
      if(tangent_difference == TangentDifference::Central)
      {
        VectorType v_minus(native_velocity), t_minus;
        v_minus.add(-eps, y);
        initialize_vector(t_minus);
        native_fluid_response(t_minus, v_minus);
        t_plus.add(-1.0, t_minus);
        dst.add(-0.5 / eps, t_plus);
        native_fluid_solves += 2;
      }
      else
      {
        t_plus.add(-1.0, native_traction);
        dst.add(-1.0 / eps, t_plus);
        native_fluid_solves += 1;
      }
    }
    dst.scale(structure_space.mask);
  }

  /*
   * Adjoint of the native formulation: S = I - M^{-1} G, G = R_T A^{-1} D E_f c_v K^{-1} L_s,
   * S^*_M w = w - M^{-1} G^T w,  G^T = c_v L_s^T K^{-1} E_f^T D^T A^{-1} R_T^T,
   * R_T = (integral against psi) P_f (-I_sigma).
   */
  void
  apply_S_adjoint_native(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(operator_coupled.get(),
                dealii::ExcMessage("The adjoint requires the coupled fluid solver."));
    // R_T^T src: structure weights -> P_f^T -> (-I_sigma)^T
    fill_weighted_trace(structure_space, *fluid_at_structure, src, false);
    VectorType stress_dual;
    fluid->pde_operator->initialize_vector_velocity(stress_dual);
    fluid_to_structure->add_transpose(stress_dual);
    stress_dual *= -1.0;

    BlockVectorType rhs_fluid, z;
    operator_coupled->initialize_block_vector_velocity_pressure(rhs_fluid);
    operator_coupled->initialize_block_vector_velocity_pressure(z);
    stress_interpolation_transpose(rhs_fluid, stress_dual);

    // A is symmetric
    solve_fluid(z, rhs_fluid);

    // D^T z at the fluid quadrature points, then E_f^T
    native_boundary_terms_transpose(*fluid_neumann, z);
    VectorType load, y, tmp;
    structure->pde_operator->initialize_dof_vector(load);
    structure->pde_operator->initialize_dof_vector(y);
    initialize_vector(tmp);
    structure_to_fluid->add_transpose(load);
    *fluid_neumann = native_base_data;

    solve_structure(y, load);
    apply_L_s_transpose(tmp, y);
    tmp *= velocity_factor();
    tmp.scale(structure_space.mask);

    VectorType Minv_tmp;
    initialize_vector(Minv_tmp);
    solve_mass(structure_space, Minv_tmp, tmp);
    dst = src;
    dst.add(-1.0, Minv_tmp);
    dst.scale(structure_space.mask);
  }

  // dst = Pi_s P_f (-I_sigma(u, p)) for a fluid (perturbation) state
  void
  native_traction_of(VectorType & dst, VectorType const & u, VectorType const & p) const
  {
    VectorType stress;
    fluid->pde_operator->initialize_vector_velocity(stress);
    fluid->pde_operator->interpolate_stress_bc(stress, u, p);
    stress *= -1.0;
    fluid_to_structure->update_data(stress);
    VectorType rhs;
    initialize_vector(rhs);
    integrate_container(structure_space, rhs, *fluid_at_structure, false);
    solve_mass(structure_space, dst, rhs);
  }

  /*
   * Linearised dependence of the right-hand side of the coupled fluid system on the Dirichlet data
   * of the interface (data given in the container at all quadrature indices):
   *   viscous term (SIPG, mirror principle u+ = -u- + 2g), velocity divergence ({u} = g),
   *   continuity penalty with boundary data, explicit convective term linearised at the state
   *   (extrapolated velocity, Dirichlet data of the state).
   * Each contribution is local to the quadrature point; the coefficients are obtained from the
   * kernels of the fluid operator, so that forward and transpose use identical formulas.
   */
  struct NativeLocalTerms
  {
    vector value;    // tested with the velocity
    tensor gradient; // tested with the velocity gradient
    scalar pressure; // tested with the pressure
  };

  // standard quadrature: viscous, divergence and continuity penalty terms for the data g
  NativeLocalTerms
  native_local_terms_standard(Integrator const & phi, unsigned int const q, vector const & g) const
  {
    auto const & param  = fluid->pde_operator->get_parameters();
    auto const   kernel = fluid->pde_operator->get_viscous_kernel();

    vector const n = phi.normal_vector(q);
    vector       zero;
    vector const value_p = 2.0 * g; // mirror principle, inhomogeneous part (u- = 0)

    NativeLocalTerms terms;
    terms.pressure = scalar(0.0);

    scalar const viscosity = kernel->get_viscosity_boundary_face(phi.get_current_cell_index(), q);
    tensor const gradient_flux = kernel->calculate_gradient_flux(zero, value_p, n, viscosity);
    vector const value_flux = kernel->calculate_value_flux(zero, zero, zero, value_p, n, viscosity);
    // operator: submit_gradient(gradient_flux), submit_value(-value_flux); rhs = -operator
    terms.gradient = -gradient_flux;
    terms.value    = value_flux;

    if(param.use_continuity_penalty and not param.apply_penalty_terms_in_postprocessing_step and
       param.continuity_penalty_use_boundary_data)
      terms.value -=
        fluid->pde_operator->get_continuity_penalty_kernel()->calculate_flux(zero, value_p, n);

    if(param.divu_use_boundary_data)
    {
      // pressure block of the rhs: -(-div rhs) with {u}.n = g.n
      terms.pressure = g * n;
    }
    return terms;
  }

  // overintegration quadrature: linearised explicit convective term, d(-C)/dg [dg]
  vector
  native_local_convective(Integrator const & phi_u,
                          Integrator const & phi_w,
                          unsigned int const q,
                          vector const &     g0,
                          vector const &     dg) const
  {
    auto const kernel = fluid->pde_operator->get_convective_kernel();

    vector const n      = phi_u.normal_vector(q);
    vector const u_m    = phi_u.get_value(q); // extrapolated velocity
    vector const u_grid = phi_w.get_value(q);

    double scale = 1.e-30;
    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      for(unsigned int d = 0; d < dim; ++d)
        scale = std::max<double>(scale,
                                 std::abs(g0[d][v]) + std::abs(u_m[d][v]) + std::abs(u_grid[d][v]));
    Number const h = 1.e-4 * scale;

    auto const flux = [&](vector const & g) {
      vector const u_p = -u_m + 2.0 * g; // mirror principle
      return kernel->calculate_flux_nonlinear_boundary(
        u_m, u_p, n, u_grid, IncNS::BoundaryTypeU::Dirichlet);
    };
    // exact for the (piecewise quadratic) flux unless the kink of |u.n| is crossed
    vector const flux_derivative = (flux(g0 + h * dg) - flux(g0 - h * dg)) / (Number(2.0) * h);
    return -flux_derivative; // rhs -= C
  }

  // dst = D c (fluid block vector)
  void
  native_boundary_terms(BlockVectorType & dst, Container const & data) const
  {
    auto const & matrix_free = *fluid_space.matrix_free;
    auto const & param       = fluid->pde_operator->get_parameters();

    dst                          = 0.0;
    unsigned int const quad_std  = fluid->pde_operator->get_quad_index_velocity_standard();
    unsigned int const quad_over = fluid->pde_operator->get_quad_index_velocity_overintegration();

    Integrator                     phi(matrix_free, true, fluid_space.dof_index, quad_std);
    FaceIntegrator<dim, 1, Number> phi_p(matrix_free, true, fluid_pressure_dof_index, quad_std);
    Integrator                     phi_c(matrix_free, true, fluid_space.dof_index, quad_over);
    Integrator                     phi_u(matrix_free, true, fluid_space.dof_index, quad_over);
    Integrator                     phi_w(matrix_free, true, fluid_space.dof_index, quad_over);

    bool const convective = param.convective_problem();
    if(convective)
    {
      native_extrapolated_velocity.update_ghost_values();
      fluid->pde_operator->get_convective_kernel()->get_grid_velocity().update_ghost_values();
    }

    for(unsigned int face = fluid_space.faces_begin(); face < fluid_space.faces_end(); ++face)
    {
      if(not fluid_space.contains(face))
        continue;

      phi.reinit(face);
      phi_p.reinit(face);
      fluid->pde_operator->get_viscous_kernel()->reinit_boundary_face(phi, fluid_space.dof_index);
      if(fluid->pde_operator->get_continuity_penalty_kernel())
        fluid->pde_operator->get_continuity_penalty_kernel()->reinit_boundary_face(phi);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        vector const g = FunctionEvaluator<1, dim, Number>::value(data, face, q, quad_std);
        NativeLocalTerms const terms = native_local_terms_standard(phi, q, g);
        phi.submit_value(terms.value, q);
        phi.submit_gradient(terms.gradient, q);
        phi_p.submit_value(terms.pressure, q);
      }
      phi.integrate_scatter(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients,
                            dst.block(0));
      phi_p.integrate_scatter(dealii::EvaluationFlags::values, dst.block(1));

      if(convective)
      {
        phi_c.reinit(face);
        phi_u.reinit(face);
        phi_w.reinit(face);
        phi_u.gather_evaluate(native_extrapolated_velocity, dealii::EvaluationFlags::values);
        phi_w.gather_evaluate(fluid->pde_operator->get_convective_kernel()->get_grid_velocity(),
                              dealii::EvaluationFlags::values);
        for(unsigned int q = 0; q < phi_c.n_q_points; ++q)
        {
          vector const g0 =
            FunctionEvaluator<1, dim, Number>::value(native_base_data, face, q, quad_over);
          vector const dg = FunctionEvaluator<1, dim, Number>::value(data, face, q, quad_over);
          phi_c.submit_value(native_local_convective(phi_u, phi_w, q, g0, dg), q);
        }
        phi_c.integrate_scatter(dealii::EvaluationFlags::values, dst.block(0));
      }
    }
    dst.compress(dealii::VectorOperation::add);

    if(convective)
    {
      native_extrapolated_velocity.zero_out_ghost_values();
      fluid->pde_operator->get_convective_kernel()->get_grid_velocity().zero_out_ghost_values();
    }
  }

  // container = D^T z at all quadrature indices of the container (others and padding zero)
  void
  native_boundary_terms_transpose(Container & container, BlockVectorType const & z) const
  {
    auto const & matrix_free = *fluid_space.matrix_free;
    auto const & param       = fluid->pde_operator->get_parameters();

    container.set_zero();
    unsigned int const quad_std  = fluid->pde_operator->get_quad_index_velocity_standard();
    unsigned int const quad_over = fluid->pde_operator->get_quad_index_velocity_overintegration();

    Integrator                     phi(matrix_free, true, fluid_space.dof_index, quad_std);
    FaceIntegrator<dim, 1, Number> phi_p(matrix_free, true, fluid_pressure_dof_index, quad_std);
    Integrator                     phi_c(matrix_free, true, fluid_space.dof_index, quad_over);
    Integrator                     phi_u(matrix_free, true, fluid_space.dof_index, quad_over);
    Integrator                     phi_w(matrix_free, true, fluid_space.dof_index, quad_over);

    bool const convective = param.convective_problem();
    z.block(0).update_ghost_values();
    z.block(1).update_ghost_values();
    if(convective)
    {
      native_extrapolated_velocity.update_ghost_values();
      fluid->pde_operator->get_convective_kernel()->get_grid_velocity().update_ghost_values();
    }

    auto const store = [&](unsigned int const quad,
                           unsigned int const face,
                           unsigned int const q,
                           unsigned int const n_lanes,
                           vector const &     value) {
      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        dealii::Tensor<1, dim, double> t;
        for(unsigned int d = 0; d < dim; ++d)
          t[d] = value[d][v];
        container.set_data(quad, face, q, v, t);
      }
    };

    for(unsigned int face = fluid_space.faces_begin(); face < fluid_space.faces_end(); ++face)
    {
      if(not fluid_space.contains(face))
        continue;
      unsigned int const n_lanes = matrix_free.n_active_entries_per_face_batch(face);

      phi.reinit(face);
      phi_p.reinit(face);
      phi.gather_evaluate(z.block(0),
                          dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      phi_p.gather_evaluate(z.block(1), dealii::EvaluationFlags::values);
      fluid->pde_operator->get_viscous_kernel()->reinit_boundary_face(phi, fluid_space.dof_index);
      if(fluid->pde_operator->get_continuity_penalty_kernel())
        fluid->pde_operator->get_continuity_penalty_kernel()->reinit_boundary_face(phi);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        vector result;
        for(unsigned int d = 0; d < dim; ++d)
        {
          vector e;
          e[d]                         = scalar(1.0);
          NativeLocalTerms const terms = native_local_terms_standard(phi, q, e);
          result[d]                    = phi.JxW(q) * (terms.value * phi.get_value(q) +
                                    scalar_product(terms.gradient, phi.get_gradient(q)) +
                                    terms.pressure * phi_p.get_value(q));
        }
        store(quad_std, face, q, n_lanes, result);
      }

      if(convective)
      {
        phi_c.reinit(face);
        phi_u.reinit(face);
        phi_w.reinit(face);
        phi_c.gather_evaluate(z.block(0), dealii::EvaluationFlags::values);
        phi_u.gather_evaluate(native_extrapolated_velocity, dealii::EvaluationFlags::values);
        phi_w.gather_evaluate(fluid->pde_operator->get_convective_kernel()->get_grid_velocity(),
                              dealii::EvaluationFlags::values);
        for(unsigned int q = 0; q < phi_c.n_q_points; ++q)
        {
          vector const g0 =
            FunctionEvaluator<1, dim, Number>::value(native_base_data, face, q, quad_over);
          vector result;
          for(unsigned int d = 0; d < dim; ++d)
          {
            vector e;
            e[d] = scalar(1.0);
            result[d] =
              phi_c.JxW(q) * (native_local_convective(phi_u, phi_w, q, g0, e) * phi_c.get_value(q));
          }
          store(quad_over, face, q, n_lanes, result);
        }
      }
    }

    z.block(0).zero_out_ghost_values();
    z.block(1).zero_out_ghost_values();
    if(convective)
    {
      native_extrapolated_velocity.zero_out_ghost_values();
      fluid->pde_operator->get_convective_kernel()->get_grid_velocity().zero_out_ghost_values();
    }
  }

  /*
   * Transpose of the nodal stress interpolation (interpolate_stress_bc): the forward map writes
   * rho (nu F(u) n - p n) at the face nodes (later faces overwrite shared nodes). The transpose
   * visits the faces in reverse order and consumes each dual nodal value once (the last writer).
   */
  void
  stress_interpolation_transpose(BlockVectorType & dst, VectorType const & w) const
  {
    auto const & matrix_free = *fluid_space.matrix_free;
    auto const & param       = fluid->pde_operator->get_parameters();

    unsigned int const quad_nodal = fluid->pde_operator->get_quad_index_velocity_nodal_points();

    VectorType w_work(w);
    dst = 0.0;

    Integrator                     phi_w(matrix_free, true, fluid_space.dof_index, quad_nodal);
    Integrator                     phi_u(matrix_free, true, fluid_space.dof_index, quad_nodal);
    FaceIntegrator<dim, 1, Number> phi_p(matrix_free, true, fluid_pressure_dof_index, quad_nodal);

    std::vector<unsigned int> faces;
    for(unsigned int face = fluid_space.faces_begin(); face < fluid_space.faces_end(); ++face)
      if(fluid_space.contains(face))
        faces.push_back(face);

    std::vector<vector> s;
    for(auto it = faces.rbegin(); it != faces.rend(); ++it)
    {
      unsigned int const face              = *it;
      unsigned int const local_face_number = matrix_free.get_face_info(face).interior_face_no;
      auto const &       nodal_index = matrix_free.get_shape_info(fluid_space.dof_index, quad_nodal)
                                   .face_to_cell_index_nodal[local_face_number];

      phi_w.reinit(face);
      phi_w.read_dof_values(w_work);
      s.resize(phi_w.n_q_points);
      for(unsigned int q = 0; q < phi_w.n_q_points; ++q)
        s[q] = phi_w.get_dof_value(nodal_index[q]);
      for(unsigned int q = 0; q < phi_w.n_q_points; ++q)
        phi_w.submit_dof_value(vector(), nodal_index[q]);
      phi_w.set_dof_values(w_work);

      phi_u.reinit(face);
      phi_p.reinit(face);
      for(unsigned int q = 0; q < phi_u.n_q_points; ++q)
      {
        vector const n       = phi_u.normal_vector(q);
        scalar const inv_JxW = scalar(1.0) / phi_u.JxW(q);
        scalar const viscosity =
          fluid->pde_operator->get_viscous_kernel()->get_viscosity_boundary_face(
            phi_u.get_current_cell_index(), q);
        phi_u.submit_gradient(param.density * viscosity * inv_JxW *
                                (outer_product(s[q], n) + outer_product(n, s[q])),
                              q);
        phi_p.submit_value(-param.density * inv_JxW * (s[q] * n), q);
      }
      phi_u.integrate_scatter(dealii::EvaluationFlags::gradients, dst.block(0));
      phi_p.integrate_scatter(dealii::EvaluationFlags::values, dst.block(1));
    }
    dst.compress(dealii::VectorOperation::add);
  }

  /*
   * Forward data of the single-field solvers
   */

  // structure Neumann container = j * g at the structure quadrature points
  void
  fill_structure_load(VectorType const & g) const
  {
    auto const & matrix_free = *structure_space.matrix_free;

    if(control_space == ControlSpace::Fluid)
      fluid_to_structure->update_data(g);

    structure_load->set_zero();
    g.update_ghost_values();
    geometry_displacement.update_ghost_values();

    for(auto const quad_index : structure_load->get_quad_indices())
    {
      Integrator phi(matrix_free, true, structure_space.dof_index, quad_index);
      Integrator phi_geo(matrix_free, true, structure_space.dof_index, quad_index);

      for(unsigned int face = structure_space.faces_begin(); face < structure_space.faces_end();
          ++face)
      {
        if(not structure_space.contains(face))
          continue;

        if(control_space == ControlSpace::Structure)
        {
          phi.reinit(face);
          phi.gather_evaluate(g, dealii::EvaluationFlags::values);
        }
        if(use_area_scaling)
        {
          phi_geo.reinit(face);
          phi_geo.gather_evaluate(geometry_displacement, dealii::EvaluationFlags::gradients);
        }

        unsigned int const n_lanes = matrix_free.n_active_entries_per_face_batch(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          scalar const j = area_ratio(phi_geo, q);
          vector       value;
          if(control_space == ControlSpace::Structure)
            value = phi.get_value(q);
          else
            value = FunctionEvaluator<1, dim, Number>::value(*fluid_at_structure,
                                                             face,
                                                             q,
                                                             structure_space.quad_index);
          value = j * value;
          for(unsigned int v = 0; v < n_lanes; ++v)
          {
            dealii::Tensor<1, dim, double> t;
            for(unsigned int d = 0; d < dim; ++d)
              t[d] = value[d][v];
            structure_load->set_data(quad_index, face, q, v, t);
          }
        }
      }
    }

    g.zero_out_ghost_values();
    geometry_displacement.zero_out_ghost_values();
  }

  // fluid Neumann container = factor * g at the fluid quadrature points
  void
  fill_fluid_neumann_data(VectorType const & g, double const factor) const
  {
    if(control_space == ControlSpace::Structure)
    {
      structure_to_fluid->update_data(g);
      fluid_neumann->scale(factor);
      return;
    }

    auto const & matrix_free = *fluid_space.matrix_free;

    fluid_neumann->set_zero();
    g.update_ghost_values();
    for(auto const quad_index : fluid_neumann->get_quad_indices())
    {
      Integrator phi(matrix_free, true, fluid_space.dof_index, quad_index);
      for(unsigned int face = fluid_space.faces_begin(); face < fluid_space.faces_end(); ++face)
      {
        if(not fluid_space.contains(face))
          continue;
        phi.reinit(face);
        phi.gather_evaluate(g, dealii::EvaluationFlags::values);
        unsigned int const n_lanes = matrix_free.n_active_entries_per_face_batch(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          vector const value = phi.get_value(q);
          for(unsigned int v = 0; v < n_lanes; ++v)
          {
            dealii::Tensor<1, dim, double> t;
            for(unsigned int d = 0; d < dim; ++d)
              t[d] = factor * value[d][v];
            fluid_neumann->set_data(quad_index, face, q, v, t);
          }
        }
      }
    }
    g.zero_out_ghost_values();
  }

  /*
   * ---------------------------------------------------------------------------------------------
   * Face integrals on the trace spaces
   * ---------------------------------------------------------------------------------------------
   */

  void
  setup_trace_space(TraceSpace & space) const
  {
    auto const & matrix_free = *space.matrix_free;

    VectorType diagonal;
    matrix_free.initialize_dof_vector(diagonal, space.dof_index);
    matrix_free.initialize_dof_vector(space.mask, space.dof_index);
    matrix_free.initialize_dof_vector(space.inverse_diagonal, space.dof_index);

    Integrator                    phi(matrix_free, true, space.dof_index, space.quad_index);
    dealii::AlignedVector<scalar> local_diagonal(phi.dofs_per_cell);

    for(unsigned int face = space.faces_begin(); face < space.faces_end(); ++face)
    {
      if(not space.contains(face))
        continue;

      phi.reinit(face);
      for(unsigned int i = 0; i < phi.dofs_per_cell; ++i)
      {
        for(unsigned int j = 0; j < phi.dofs_per_cell; ++j)
          phi.begin_dof_values()[j] = scalar(0.0);
        phi.begin_dof_values()[i] = scalar(1.0);
        phi.evaluate(dealii::EvaluationFlags::values);
        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(phi.get_value(q), q);
        phi.integrate(dealii::EvaluationFlags::values);
        local_diagonal[i] = phi.begin_dof_values()[i];
      }
      for(unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = local_diagonal[i];
      phi.distribute_local_to_global(diagonal);
    }
    diagonal.compress(dealii::VectorOperation::add);

    double const max_diagonal = diagonal.linfty_norm();
    AssertThrow(max_diagonal > 0.0, dealii::ExcMessage("The FSI interface is empty."));

    for(unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
    {
      bool const on_interface     = diagonal.local_element(i) > 1.e-12 * max_diagonal;
      space.mask.local_element(i) = on_interface ? 1.0 : 0.0;
      space.inverse_diagonal.local_element(i) =
        on_interface ? 1.0 / diagonal.local_element(i) : 1.0;
    }
  }

  // dst = M src on the interface of the given side
  void
  apply_mass(TraceSpace const & space, VectorType & dst, VectorType const & src) const
  {
    auto const & matrix_free = *space.matrix_free;

    dst = 0.0;
    src.update_ghost_values();

    Integrator phi(matrix_free, true, space.dof_index, space.quad_index);
    for(unsigned int face = space.faces_begin(); face < space.faces_end(); ++face)
    {
      if(not space.contains(face))
        continue;

      phi.reinit(face);
      phi.gather_evaluate(src, dealii::EvaluationFlags::values);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);
      phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }

    dst.compress(dealii::VectorOperation::add);
    src.zero_out_ghost_values();
  }

  // operator with M on the interface dofs and the identity elsewhere (SPD on the whole space)
  struct FullMass
  {
    InterfaceProblem const * problem;
    TraceSpace const *       space;

    void
    vmult(VectorType & dst, VectorType const & src) const
    {
      problem->apply_mass(*space, dst, src);
      for(unsigned int i = 0; i < dst.locally_owned_size(); ++i)
        dst.local_element(i) += (1.0 - space->mask.local_element(i)) * src.local_element(i);
    }
  };

  // dst = M^{-1} rhs, rhs is masked to the interface dofs
  void
  solve_mass(TraceSpace const & space, VectorType & dst, VectorType const & rhs) const
  {
    VectorType b(rhs);
    b.scale(space.mask);

    dst.reinit(b);
    double const b_norm = b.l2_norm();
    if(b_norm == 0.0)
      return;

    dealii::ReductionControl           control(10000, 1.e-30 * b_norm, 1.e-13);
    dealii::SolverCG<VectorType>       solver(control);
    dealii::DiagonalMatrix<VectorType> preconditioner(space.inverse_diagonal);
    FullMass                           op{this, &space};
    solver.solve(op, dst, b, preconditioner);
    dst.scale(space.mask);
  }

  /*
   * dst = int_Gamma data . phi (optionally weighted with the area ratio j of the frozen geometry),
   * data given at the quadrature points of the given side
   */
  void
  integrate_container(TraceSpace const & space,
                      VectorType &       dst,
                      Container const &  data,
                      bool const         with_area_ratio) const
  {
    auto const & matrix_free = *space.matrix_free;

    dst = 0.0;
    if(with_area_ratio)
      geometry_displacement.update_ghost_values();

    Integrator phi(matrix_free, true, space.dof_index, space.quad_index);
    Integrator phi_geo(matrix_free, true, space.dof_index, space.quad_index);
    for(unsigned int face = space.faces_begin(); face < space.faces_end(); ++face)
    {
      if(not space.contains(face))
        continue;

      phi.reinit(face);
      if(with_area_ratio and use_area_scaling)
      {
        phi_geo.reinit(face);
        phi_geo.gather_evaluate(geometry_displacement, dealii::EvaluationFlags::gradients);
      }
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        vector value = FunctionEvaluator<1, dim, Number>::value(data, face, q, space.quad_index);
        if(with_area_ratio)
          value = area_ratio(phi_geo, q) * value;
        phi.submit_value(value, q);
      }
      phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
    dst.compress(dealii::VectorOperation::add);

    if(with_area_ratio)
      geometry_displacement.zero_out_ghost_values();
  }

  // dst = int_Gamma j src . psi on the structure side (src a structure vector)
  void
  integrate_structure_trace(VectorType &       dst,
                            VectorType const & src,
                            bool const         with_area_ratio) const
  {
    auto const & matrix_free = *structure_space.matrix_free;

    dst = 0.0;
    src.update_ghost_values();
    geometry_displacement.update_ghost_values();

    Integrator phi(matrix_free, true, structure_space.dof_index, structure_space.quad_index);
    Integrator phi_geo(matrix_free, true, structure_space.dof_index, structure_space.quad_index);
    for(unsigned int face = structure_space.faces_begin(); face < structure_space.faces_end();
        ++face)
    {
      if(not structure_space.contains(face))
        continue;

      phi.reinit(face);
      phi.gather_evaluate(src, dealii::EvaluationFlags::values);
      if(with_area_ratio and use_area_scaling)
      {
        phi_geo.reinit(face);
        phi_geo.gather_evaluate(geometry_displacement, dealii::EvaluationFlags::gradients);
      }
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value((with_area_ratio ? area_ratio(phi_geo, q) : scalar(1.0)) *
                           phi.get_value(q),
                         q);
      phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }

    dst.compress(dealii::VectorOperation::add);
    src.zero_out_ghost_values();
    geometry_displacement.zero_out_ghost_values();
  }

  /*
   * container = (j) JxW src at the quadrature points of the given side (transpose of
   * integrate_container). All other quadrature indices of the container and padding lanes of
   * incomplete face batches are zero.
   */
  void
  fill_weighted_trace(TraceSpace const & space,
                      Container &        container,
                      VectorType const & src,
                      bool const         with_area_ratio) const
  {
    auto const & matrix_free = *space.matrix_free;

    container.set_zero();
    src.update_ghost_values();
    if(with_area_ratio)
      geometry_displacement.update_ghost_values();

    Integrator phi(matrix_free, true, space.dof_index, space.quad_index);
    Integrator phi_geo(matrix_free, true, space.dof_index, space.quad_index);
    for(unsigned int face = space.faces_begin(); face < space.faces_end(); ++face)
    {
      if(not space.contains(face))
        continue;

      phi.reinit(face);
      phi.gather_evaluate(src, dealii::EvaluationFlags::values);
      if(with_area_ratio and use_area_scaling)
      {
        phi_geo.reinit(face);
        phi_geo.gather_evaluate(geometry_displacement, dealii::EvaluationFlags::gradients);
      }
      unsigned int const n_lanes = matrix_free.n_active_entries_per_face_batch(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        scalar weight = phi.JxW(q);
        if(with_area_ratio)
          weight = weight * area_ratio(phi_geo, q);
        vector const value = weight * phi.get_value(q);
        for(unsigned int v = 0; v < n_lanes; ++v)
        {
          dealii::Tensor<1, dim, double> t;
          for(unsigned int d = 0; d < dim; ++d)
            t[d] = value[d][v];
          container.set_data(space.quad_index, face, q, v, t);
        }
      }
    }

    src.zero_out_ghost_values();
    if(with_area_ratio)
      geometry_displacement.zero_out_ghost_values();
  }

  // area ratio da/dA of the frozen geometry (structure side)
  scalar
  area_ratio(Integrator const & phi_geo, unsigned int const q) const
  {
    if(not use_area_scaling)
      return scalar(1.0);

    tensor F = phi_geo.get_gradient(q);
    for(unsigned int d = 0; d < dim; ++d)
      F[d][d] += 1.0;
    vector const N      = phi_geo.normal_vector(q);
    vector const n_star = determinant(F) * (transpose(invert(F)) * N);
    return n_star.norm();
  }

  /*
   * ---------------------------------------------------------------------------------------------
   * Single-field linear solves (normalized right-hand sides: the solvers of the single-field
   * solvers use absolute tolerances tuned for the state, sensitivity problems are linear)
   * ---------------------------------------------------------------------------------------------
   */

  // linear fluid solve with the system matrix of the last time step: A dst = rhs
  void
  solve_fluid(BlockVectorType & dst, BlockVectorType const & rhs) const
  {
    dst                 = 0.0;
    double const factor = rhs.l2_norm();
    if(factor == 0.0)
      return;

    BlockVectorType rhs_scaled(rhs);
    rhs_scaled *= 1.0 / factor;

    VectorType dummy_transport_velocity;
    operator_coupled->solve_linear_problem(
      dst,
      rhs_scaled,
      dummy_transport_velocity,
      false /* update_preconditioner */,
      fluid->time_integrator->get_scaling_factor_time_derivative_term());
    dst *= factor;
  }

  // linearized structure solve at the current linearization point: K dst = rhs
  void
  solve_structure(VectorType & dst, VectorType const & rhs) const
  {
    dst                 = 0.0;
    double const factor = rhs.l2_norm();
    if(factor == 0.0)
      return;

    VectorType rhs_scaled(rhs);
    rhs_scaled *= 1.0 / factor;

    auto const & ti = *structure->time_integrator;
    structure->pde_operator->solve_linearized(dst,
                                              rhs_scaled,
                                              linearization,
                                              ti.get_scaling_factor_acceleration(),
                                              ti.get_scaling_factor_velocity(),
                                              ti.get_mid_time());
    dst *= factor;
  }

  void
  apply_structure_tangent(VectorType & dst, VectorType const & src) const
  {
    auto const & ti = *structure->time_integrator;
    structure->pde_operator->apply_linearized(dst,
                                              src,
                                              linearization,
                                              ti.get_scaling_factor_acceleration(),
                                              ti.get_scaling_factor_velocity(),
                                              ti.get_mid_time());
  }

  // K_II restricted to the interior (non-interface, unconstrained) dofs of the structure
  struct InteriorTangent
  {
    InterfaceProblem const * problem;

    void
    vmult(VectorType & dst, VectorType const & src) const
    {
      VectorType x(src);
      x.scale(problem->interior_mask);
      problem->apply_structure_tangent(dst, x);
      dst.scale(problem->interior_mask);
    }
  };

  /*
   * Steklov-Poincare operator of the linearized structure on the interface:
   * Sigma_s lambda = T K (T^T lambda + y),  K_II y = -(K T^T lambda)_I.
   */
  void
  apply_structure_steklov_poincare(VectorType & dst, VectorType const & lambda) const
  {
    VectorType x(lambda), r, rhs, y;
    x.scale(structure_space.mask);
    structure->pde_operator->initialize_dof_vector(r);
    structure->pde_operator->initialize_dof_vector(y);
    apply_structure_tangent(r, x);
    rhs = r;
    rhs.scale(interior_mask);
    rhs *= -1.0;

    double const factor = rhs.l2_norm();
    if(factor > 0.0)
    {
      rhs *= 1.0 / factor;
      InteriorTangent              op{this};
      dealii::ReductionControl     control(10000, 1.e-300, preconditioner_tolerance);
      dealii::SolverCG<VectorType> solver(control);
      solver.solve(op, y, rhs, dealii::PreconditionIdentity());
      preconditioner_inner_iterations += control.last_step();
      y *= factor;
    }

    x.add(1.0, y);
    apply_structure_tangent(dst, x);
    dst.scale(structure_space.mask);
  }

  /*
   * Dense Jacobian S in the coordinates of the interface dofs, one tangent solve per column,
   * linearized at the current state; LU factorization for the FrozenJacobian preconditioner.
   */
  void
  assemble_frozen_jacobian() const
  {
    unsigned int const n = n_dense;
    jacobian_lu.reinit(n, n);

    VectorType e, Se;
    initialize_vector(e);
    initialize_vector(Se);
    for(unsigned int k = 0; k < n; ++k)
    {
      e = 0.0;
      for(auto const & [local, position] : owned_positions)
        if(position == k)
          e.local_element(local) = 1.0;
      apply_S(Se, e);
      if(use_projection)
      {
        // T = P S P + (I - P): invertible, equal to S on the control space
        VectorType Pe(e);
        project_control(Pe);
        Se.add(1.0, e);
        Se.add(-1.0, Pe);
      }
      dealii::Vector<double> const column = gather(Se);
      for(unsigned int i = 0; i < n; ++i)
        jacobian_lu(i, k) = column[i];
    }
    jacobian_lu.compute_lu_factorization();

    preconditioner_setup_solves += n;
    ++preconditioner_assemblies;
    jacobian_needs_update = false;
  }

  /*
   * ---------------------------------------------------------------------------------------------
   * Dense Sobolev metrics on the interface dofs (structure control space)
   * ---------------------------------------------------------------------------------------------
   */

  void
  setup_interface_numbering()
  {
    VectorType const & mask = control().mask;

    std::vector<dealii::types::global_dof_index> local;
    owned_positions.clear();
    for(unsigned int i = 0; i < mask.locally_owned_size(); ++i)
      if(mask.local_element(i) > 0.5)
        local.push_back(mask.get_partitioner()->local_to_global(i));

    auto const all = dealii::Utilities::MPI::all_gather(mpi_comm, local);

    unsigned int       position = 0;
    unsigned int const my_rank  = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    for(unsigned int r = 0; r < all.size(); ++r)
      for(auto const index : all[r])
      {
        if(r == my_rank)
          owned_positions.emplace_back(mask.get_partitioner()->global_to_local(index), position);
        ++position;
      }
    n_dense = position;
  }

  dealii::Vector<double>
  gather(VectorType const & v) const
  {
    dealii::Vector<double> x(n_dense);
    for(auto const & [local, position] : owned_positions)
      x[position] = v.local_element(local);
    dealii::Utilities::MPI::sum(x, mpi_comm, x);
    return x;
  }

  void
  scatter(VectorType & v, dealii::Vector<double> const & x) const
  {
    v = 0.0;
    for(auto const & [local, position] : owned_positions)
      v.local_element(local) = x[position];
  }

  double
  dense_inner_product(dealii::FullMatrix<double> const & matrix,
                      VectorType const &                 a,
                      VectorType const &                 b) const
  {
    return matrix.matrix_scalar_product(gather(a), gather(b));
  }

  void
  dense_apply(VectorType &                       dst,
              dealii::FullMatrix<double> const & matrix,
              VectorType const &                 src) const
  {
    dealii::Vector<double> y(n_dense);
    matrix.vmult(y, gather(src));
    scatter(dst, y);
  }

  // dst = K_Gamma src, Laplace-Beltrami operator on the structure side of the interface
  void
  apply_laplace_beltrami(VectorType & dst, VectorType const & src) const
  {
    auto const & matrix_free = *structure_space.matrix_free;

    dst = 0.0;
    src.update_ghost_values();

    Integrator phi(matrix_free, true, structure_space.dof_index, structure_space.quad_index);
    for(unsigned int face = structure_space.faces_begin(); face < structure_space.faces_end();
        ++face)
    {
      if(not structure_space.contains(face))
        continue;

      phi.reinit(face);
      phi.gather_evaluate(src, dealii::EvaluationFlags::gradients);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        vector const n = phi.normal_vector(q);
        tensor       P;
        for(unsigned int i = 0; i < dim; ++i)
          for(unsigned int j = 0; j < dim; ++j)
            P[i][j] = (i == j ? 1.0 : 0.0) - n[i] * n[j];
        phi.submit_gradient(phi.get_gradient(q) * P, q);
      }
      phi.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
    }

    dst.compress(dealii::VectorOperation::add);
    src.zero_out_ghost_values();
  }

  /*
   * Spectral interpolation (34) of the report: (l^2 K + M) V = M V diag(mu), V^T M V = I,
   * A_s = M V diag(mu^s) V^T M, A_s^{-1} = V diag(mu^{-s}) V^T.
   */
  void
  setup_dense_metrics(double const s_m, double const s_c, double const length_scale)
  {
    unsigned int const n = n_dense;

    dealii::FullMatrix<double> M(n, n), K(n, n);
    {
      VectorType e, Me, Ke;
      initialize_vector(e);
      initialize_vector(Me);
      initialize_vector(Ke);
      for(unsigned int k = 0; k < n; ++k)
      {
        e = 0.0;
        for(auto const & [local, position] : owned_positions)
          if(position == k)
            e.local_element(local) = 1.0;
        apply_mass(structure_space, Me, e);
        apply_laplace_beltrami(Ke, e);
        dealii::Vector<double> const m_col = gather(Me), k_col = gather(Ke);
        for(unsigned int i = 0; i < n; ++i)
        {
          M(i, k) = m_col[i];
          K(i, k) = k_col[i];
        }
      }
    }

    dealii::LAPACKFullMatrix<double> lhs(n, n), rhs(n, n);
    for(unsigned int i = 0; i < n; ++i)
      for(unsigned int j = 0; j < n; ++j)
      {
        lhs(i, j) = length_scale * length_scale * K(i, j) + M(i, j);
        rhs(i, j) = M(i, j);
      }
    std::vector<dealii::Vector<double>> eigenvectors(n, dealii::Vector<double>(n));
    lhs.compute_generalized_eigenvalues_symmetric(rhs, eigenvectors);

    std::vector<double> mu(n);
    for(unsigned int k = 0; k < n; ++k)
      mu[k] = lhs.eigenvalue(k).real();

    // F_s = V diag(mu^s) V^T
    auto const spectral_function = [&](double const s) {
      dealii::FullMatrix<double> F(n, n);
      for(unsigned int k = 0; k < n; ++k)
      {
        double const factor = std::pow(mu[k], s);
        for(unsigned int i = 0; i < n; ++i)
          for(unsigned int j = 0; j < n; ++j)
            F(i, j) += factor * eigenvectors[k][i] * eigenvectors[k][j];
      }
      return F;
    };

    dealii::FullMatrix<double> const F_m = spectral_function(s_m), F_c = spectral_function(s_c),
                                     F_c_inv = spectral_function(-s_c);

    dealii::FullMatrix<double> tmp(n, n);
    A_dense.reinit(n, n);
    G_dense.reinit(n, n);
    pre_dense.reinit(n, n);
    post_dense.reinit(n, n);
    M.mmult(tmp, F_m);
    tmp.mmult(A_dense, M); // A = M F_m M
    M.mmult(tmp, F_c);
    tmp.mmult(G_dense, M);        // G = M F_c M
    F_m.mmult(pre_dense, M);      // M^{-1} A = F_m M
    F_c_inv.mmult(post_dense, M); // G^{-1} M = F_c^{-1} M

    metric_eigenvalue_range = {*std::min_element(mu.begin(), mu.end()),
                               *std::max_element(mu.begin(), mu.end())};
  }

  /*
   * Data
   */
  std::shared_ptr<SolverFluid<dim, Number>>     fluid;
  std::shared_ptr<SolverStructure<dim, Number>> structure;

  std::shared_ptr<IncNS::OperatorCoupled<dim, Number>> operator_coupled;

  double const       viscosity;
  unsigned int       fluid_pressure_dof_index = 0;
  unsigned int       fluid_degree             = 1;
  unsigned int       control_degree           = 1;
  bool               use_projection           = false;
  double const       fluid_density;
  bool const         use_area_scaling;
  ControlSpace const control_space;
  Formulation const  formulation;

  VectorType         native_velocity; // structure velocity of the current state (NativeTraction)
  VectorType         native_traction; // native fluid traction of the current state (NativeTraction)
  Container          native_base_data; // Dirichlet data of the current state (NativeTraction)
  mutable VectorType native_extrapolated_velocity; // velocity of the explicit convective term

  bool                                    geometry_in_state  = false;
  TangentDifference                       tangent_difference = TangentDifference::Analytic;
  std::function<void(VectorType const &)> mesh_update;

  MPI_Comm const mpi_comm;

  TraceSpace structure_space, fluid_space;
  VectorType interior_mask;

  std::shared_ptr<InterfaceCoupling<1, dim, Number>> structure_to_fluid;
  std::shared_ptr<InterfaceCoupling<1, dim, Number>> fluid_to_structure;
  std::shared_ptr<Container>                         fluid_at_structure;
  std::shared_ptr<Container>                         fluid_neumann;
  std::shared_ptr<Container>                         structure_load;

  VectorType geometry_displacement;
  VectorType linearization;

  // preconditioner
  Preconditioner preconditioner           = Preconditioner::None;
  double         preconditioner_tolerance = 1.e-8;

  // dense Sobolev metrics on the interface dofs
  bool                                               use_dense_metrics = false;
  unsigned int                                       n_dense           = 0;
  std::vector<std::pair<unsigned int, unsigned int>> owned_positions;
  dealii::FullMatrix<double>                         A_dense, G_dense, pre_dense, post_dense;

  int component = 0; // 0: full S, 1: fluid part only, 2: structure part only (diagnostics)

  // control of the state solves
  bool         warm_start             = false;
  unsigned int evaluations_in_step    = 0;
  double       state_tolerance_factor = 1.0;

  // IQN-IMVJ inverse Jacobian (dense, interface dofs)
  dealii::FullMatrix<double> H_previous;
  double                     qn_initial_relaxation = 0.3;

  // FrozenJacobian preconditioner
  mutable bool                             jacobian_needs_update = true;
  mutable dealii::LAPACKFullMatrix<double> jacobian_lu;
};

} // namespace OptimalControl
} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_INTERFACE_PROBLEM_H_ */
