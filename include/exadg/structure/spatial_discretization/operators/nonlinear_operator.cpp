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

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/structure/spatial_discretization/operators/boundary_conditions.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <exadg/structure/spatial_discretization/operators/nonlinear_operator.h>

// C/C++
#include <chrono>
#include <iomanip>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::initialize_derived()
{
  integrator_lin = std::make_shared<IntegratorCell>(*this->matrix_free,
                                                    this->operator_data.dof_index_inhomogeneous,
                                                    this->operator_data.quad_index);
  // it should not make a difference here whether we use dof_index or dof_index_inhomogeneous
  this->matrix_free->initialize_dof_vector(displacement_lin, this->operator_data.dof_index);
  displacement_lin.update_ghost_values();

  contact_active = not this->operator_data.bc->contact_bc.empty();
  if(contact_active)
  {
    AssertThrow(this->operator_data.contact_penalty > 0.0,
                dealii::ExcMessage("Contact boundaries require a positive contact penalty "
                                   "parameter (Parameters::contact_penalty_parameter)."));

    integrator_lin_face =
      std::make_shared<IntegratorFace>(*this->matrix_free,
                                       true,
                                       this->operator_data.dof_index_inhomogeneous,
                                       this->operator_data.quad_index);

    // the linearized contact term requires the values of the displacement increment on faces
    this->integrator_flags.face_evaluate =
      this->integrator_flags.face_evaluate | dealii::EvaluationFlags::values;

    if(this->operator_data.contact_method == ContactMethod::Nitsche)
    {
      // the normal traction requires the material on boundary faces, which is only implemented
      // for spatially constant material coefficients
      for(auto const & material : *this->operator_data.material_descriptor)
      {
        bool variable_coefficients = false;
        if(material.second->type == MaterialType::StVenantKirchhoff)
          variable_coefficients =
            std::static_pointer_cast<StVenantKirchhoffData<dim>>(material.second)
              ->youngs_modulus_function != nullptr;
        else if(material.second->type == MaterialType::IncompressibleNeoHookean)
          variable_coefficients =
            std::static_pointer_cast<IncompressibleNeoHookeanData<dim>>(material.second)
              ->shear_modulus_function != nullptr;

        AssertThrow(not variable_coefficients,
                    dealii::ExcMessage("Nitsche contact requires spatially constant material "
                                       "coefficients."));
      }

      // gamma = gamma_0 E (k + 1)^2 |dK| / |K|
      IP::calculate_penalty_parameter(nitsche_gamma,
                                      *this->matrix_free,
                                      this->operator_data.dof_index);

      dealii::DoFHandler<dim> const & dof_handler =
        this->matrix_free->get_dof_handler(this->operator_data.dof_index);
      ElementType const element_type =
        dof_handler.get_triangulation().all_reference_cells_are_hyper_cube() ?
          ElementType::Hypercube :
          ElementType::Simplex;
      Number const factor =
        IP::get_penalty_factor<dim, Number>(dof_handler.get_fe().degree,
                                            element_type,
                                            this->operator_data.contact_penalty);
      for(auto & gamma : nitsche_gamma)
        gamma *= factor;

      // the normal traction of the displacement increment is needed in the linearized operator
      this->integrator_flags.face_evaluate =
        this->integrator_flags.face_evaluate | dealii::EvaluationFlags::gradients;

      // for theta != 0, the contact term is also tested with the normal traction of the test
      // function, i.e. with its gradient
      if(this->operator_data.contact_nitsche_theta != 0.0)
        this->integrator_flags.face_integrate =
          this->integrator_flags.face_integrate | dealii::EvaluationFlags::gradients;

      AssertThrow(this->operator_data.contact_friction_coefficient == 0.0 or
                    this->operator_data.contact_nitsche_theta == 0.0,
                  dealii::ExcMessage("Frictional contact requires theta = 0."));
    }

    // frictional contact: history of the previous load step
    friction_active = (this->operator_data.contact_friction_coefficient > 0.0);
    if(friction_active)
    {
      this->matrix_free->initialize_dof_vector(displacement_previous,
                                               this->operator_data.dof_index);
      displacement_previous.update_ghost_values();

      integrator_previous_face =
        std::make_shared<IntegratorFace>(*this->matrix_free,
                                         true,
                                         this->operator_data.dof_index_inhomogeneous,
                                         this->operator_data.quad_index);

      if(this->operator_data.contact_method == ContactMethod::Penalty)
      {
        n_q_points_face = integrator_previous_face->n_q_points;
        traction_history.resize(this->matrix_free->n_boundary_face_batches() * n_q_points_face,
                                vector());
        traction_history_before = traction_history;
      }
    }
  }
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::ContactState
NonLinearOperator<dim, Number>::evaluate_contact_state(
  IntegratorFace const &           integrator,
  unsigned int const               q,
  dealii::Function<dim> const &    obstacle,
  dealii::types::boundary_id const boundary_id) const
{
  ContactState state;

  // current position
  vector const x = integrator.quadrature_point(q) + integrator.get_value(q);

  state.obstacle = evaluate_rigid_obstacle<dim, Number>(obstacle, x);

  scalar argument;
  if(this->operator_data.contact_method == ContactMethod::Penalty)
  {
    argument = -this->operator_data.contact_penalty * state.obstacle.gap;

    if(friction_active)
    {
      state.delta_g = calculate_relative_slip(x, q, boundary_id);

      // trial traction (I - n (x) n)(t_T^old - epsilon Delta g)
      unsigned int const face = integrator.get_current_cell_index();
      vector const &     t_old =
        traction_history[(face - this->matrix_free->n_inner_face_batches()) * n_q_points_face + q];
      state.w   = t_old - this->operator_data.contact_penalty * state.delta_g;
      state.A_T = state.w - (state.obstacle.normal * state.w) * state.obstacle.normal;
    }
  }
  else if(this->operator_data.contact_method == ContactMethod::Nitsche)
  {
    tensor const Grad_d = integrator.get_gradient(q);

    state.F = get_F<dim, Number>(Grad_d);
    state.S = this->material_handler.get_material()->second_piola_kirchhoff_stress(
      Grad_d, integrator.get_current_cell_index(), q);
    state.N     = integrator.normal_vector(q);
    state.gamma = integrator.read_cell_data(nitsche_gamma);

    // normal traction lambda(d) = n . (P N)
    tensor const P        = state.F * state.S;
    state.normal_traction = state.obstacle.normal * (P * state.N);

    argument = state.normal_traction - state.gamma * state.obstacle.gap;

    if(friction_active)
    {
      state.traction = P * state.N;
      state.delta_g  = calculate_relative_slip(x, q, boundary_id);

      // A_T = (I - n (x) n)(sigma - gamma Delta g)
      state.w   = state.traction - state.gamma * state.delta_g;
      state.A_T = state.w - (state.obstacle.normal * state.w) * state.obstacle.normal;
    }

    if(this->operator_data.contact_nitsche_theta != 0.0)
    {
      // Q = delta P[G] = F delta S[G] + G S with G = n (x) N
      tensor const G = dealii::outer_product(state.obstacle.normal, state.N);
      state.Q        = state.F * this->material_handler.get_material()
                            ->second_piola_kirchhoff_stress_displacement_derivative(
                              G, state.F, integrator.get_current_cell_index(), q) +
                G * state.S;
      state.q = state.obstacle.hessian * (P * state.N);
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("ContactMethod not implemented."));
  }

  scalar const zero = dealii::make_vectorized_array<Number>(0.0);
  scalar const one  = dealii::make_vectorized_array<Number>(1.0);

  state.pressure = std::max(argument, zero);
  state.active   = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(
    argument, zero, one, zero);

  if(friction_active)
  {
    // friction coefficient: constant or given as a function mu(x, t) for this boundary
    state.mu =
      dealii::make_vectorized_array<Number>(this->operator_data.contact_friction_coefficient);
    auto const mu_function = this->operator_data.bc->contact_friction_coefficient.find(boundary_id);
    if(mu_function != this->operator_data.bc->contact_friction_coefficient.end())
    {
      mu_function->second->set_time(this->time);
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      {
        dealii::Point<dim> point;
        for(unsigned int d = 0; d < dim; ++d)
          point[d] = x[d][v];
        state.mu[v] = mu_function->second->value(point);
      }
    }

    // projection onto the friction disk |t_T| <= mu lambda
    scalar const radius = state.mu * state.pressure;
    state.norm_A_T      = std::sqrt(state.A_T * state.A_T);
    scalar const safe_norm =
      std::max(state.norm_A_T,
               dealii::make_vectorized_array<Number>(std::numeric_limits<Number>::min()));
    state.stick = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
      state.norm_A_T, radius, one, zero);
    scalar const scaling =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
        state.norm_A_T, radius, one, radius / safe_norm);
    state.traction_tangential = scaling * state.A_T;
  }

  return state;
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::vector
NonLinearOperator<dim, Number>::calculate_relative_slip(
  vector const &                     x,
  unsigned int const                 q,
  dealii::types::boundary_id const & boundary_id) const
{
  // displacement since the previous load step
  vector delta_g =
    x - integrator_previous_face->quadrature_point(q) - integrator_previous_face->get_value(q);

  // displacement of the obstacle since the previous load step
  auto const it = this->operator_data.bc->contact_obstacle_displacement.find(boundary_id);
  if(it != this->operator_data.bc->contact_obstacle_displacement.end())
  {
    dealii::Function<dim> & obstacle_displacement = *it->second;
    for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
    {
      dealii::Point<dim> point;
      for(unsigned int d = 0; d < dim; ++d)
        point[d] = x[d][v];

      for(unsigned int d = 0; d < dim; ++d)
      {
        obstacle_displacement.set_time(this->time);
        double const current = obstacle_displacement.value(point, d);
        obstacle_displacement.set_time(time_previous);
        double const previous = obstacle_displacement.value(point, d);
        delta_g[d][v] -= current - previous;
      }
    }
    obstacle_displacement.set_time(this->time);
  }

  return delta_g;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::set_previous_solution(VectorType const & displacement,
                                                      double const       time) const
{
  if(not friction_active)
    time_previous = time;

  if(friction_active)
  {
    // penalty method: the tangential traction of the new state becomes the history (evaluated
    // with the old history and the old previous displacement)
    if(this->operator_data.contact_method == ContactMethod::Penalty)
    {
      traction_history_before = traction_history;
      update_traction_history(displacement, time);
    }

    if(displacement_before.size() != displacement_previous.size())
      this->matrix_free->initialize_dof_vector(displacement_before, this->operator_data.dof_index);
    displacement_before.copy_locally_owned_data_from(displacement_previous);
    displacement_before.update_ghost_values();
    time_before     = time_previous;
    time_previous   = time;
    history_updated = true;

    displacement_previous.copy_locally_owned_data_from(displacement);
    // the vector may contain inhomogeneous Dirichlet values, which are read with the
    // dof_index_inhomogeneous as for the linearization vector
    displacement_previous.update_ghost_values();
  }
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::VectorType const &
NonLinearOperator<dim, Number>::get_previous_solution() const
{
  return displacement_previous;
}

template<int dim, typename Number>
double
NonLinearOperator<dim, Number>::get_time_previous() const
{
  return time_previous;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::update_traction_history(VectorType const & displacement,
                                                        double const       time) const
{
  // evaluate obstacle and relative slip at the time of the displacement
  double const current_time = this->get_time();
  this->set_time(time);

  VectorType src;
  this->matrix_free->initialize_dof_vector(src, this->operator_data.dof_index);
  src.copy_locally_owned_data_from(displacement);
  src.update_ghost_values();

  IntegratorFace integrator(*this->matrix_free,
                            true,
                            this->operator_data.dof_index_inhomogeneous,
                            this->operator_data.quad_index);

  dealii::AlignedVector<vector> new_history(traction_history.size(), vector());

  unsigned int const begin = this->matrix_free->n_inner_face_batches();
  unsigned int const end   = begin + this->matrix_free->n_boundary_face_batches();
  for(unsigned int face = begin; face < end; ++face)
  {
    dealii::types::boundary_id const boundary_id = this->matrix_free->get_boundary_id(face);
    if(this->operator_data.bc->get_boundary_type(boundary_id) != BoundaryType::Contact)
      continue;

    auto const obstacle = this->operator_data.bc->contact_bc.find(boundary_id)->second;
    obstacle->set_time(this->time);

    integrator.reinit(face);
    integrator.gather_evaluate(src, dealii::EvaluationFlags::values);

    integrator_previous_face->reinit(face);
    integrator_previous_face->read_dof_values(displacement_previous);
    integrator_previous_face->evaluate(dealii::EvaluationFlags::values);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      ContactState const state = evaluate_contact_state(integrator, q, *obstacle, boundary_id);
      new_history[(face - begin) * n_q_points_face + q] = state.traction_tangential;
    }
  }

  traction_history = new_history;

  this->set_time(current_time);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::evaluate_nonlinear(VectorType & dst, VectorType const & src) const
{
  auto const start = std::chrono::steady_clock::now();

  // the current history is used from now on
  history_updated = false;

  this->matrix_free->loop(&This::cell_loop_nonlinear,
                          &This::face_loop_nonlinear,
                          &This::boundary_face_loop_nonlinear,
                          this,
                          dst,
                          src,
                          true /* zero_dst_vector */);

  if(deformable_contact)
    deformable_contact->evaluate_residual_add(dst, src);

  ++cost_statistics.n_residuals;
  cost_statistics.t_residual +=
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  auto const start = std::chrono::steady_clock::now();

  Base::vmult(dst, src);

  if(deformable_contact)
    deformable_contact->apply_jacobian_add(dst, src);

  ++cost_statistics.n_vmults;
  cost_statistics.t_vmult +=
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::calculate_system_matrix(
  dealii::TrilinosWrappers::SparseMatrix & system_matrix) const
{
  bool const contact = deformable_contact and deformable_contact->get_data().add_to_preconditioner;

  // slave-master coupling: extend the sparsity pattern if the current active pairs need couplings
  // that are not contained (the pattern of the elasticity operator is rebuilt as in
  // OperatorBase::init_system_matrix(), serial)
  if(contact and deformable_contact->get_data().preconditioner_coupling and
     not deformable_contact->pattern_contains_couplings())
  {
    auto const                      start = std::chrono::steady_clock::now();
    dealii::DoFHandler<dim> const & dof_handler =
      this->matrix_free->get_dof_handler(this->operator_data.dof_index);
    dealii::IndexSet const & owned_dofs = dof_handler.locally_owned_dofs();
    // the pattern of the elasticity operator is computed once
    if(not elasticity_pattern)
    {
      elasticity_pattern = std::make_shared<dealii::DynamicSparsityPattern>(owned_dofs.size(),
                                                                            owned_dofs.size(),
                                                                            owned_dofs);
      dealii::DoFTools::make_sparsity_pattern(dof_handler, *elasticity_pattern, *this->constraint);
    }
    // the set of contact couplings only grows, so it is added to the cached pattern in place
    // (DynamicSparsityPattern cannot be copied when it is not empty)
    deformable_contact->add_couplings_to_pattern(*elasticity_pattern);
    system_matrix.reinit(owned_dofs,
                         owned_dofs,
                         *elasticity_pattern,
                         system_matrix.get_mpi_communicator());
    cost_statistics.t_pattern +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }

  Base::calculate_system_matrix(system_matrix);

  if(contact)
    deformable_contact->add_to_matrix(system_matrix);
}
#endif

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::set_deformable_contact(
  std::shared_ptr<DeformableContact<dim, Number> const> contact)
{
  deformable_contact = contact;
}

template<int dim, typename Number>
std::shared_ptr<DeformableContact<dim, Number> const>
NonLinearOperator<dim, Number>::get_deformable_contact() const
{
  return deformable_contact;
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::CostStatistics const &
NonLinearOperator<dim, Number>::get_cost_statistics() const
{
  return cost_statistics;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::print_cost_statistics(
  dealii::ConditionalOStream const & pcout) const
{
  CostStatistics const & s = cost_statistics;
  auto const line          = [&](std::string const & name, unsigned long const n, double const t) {
    pcout << "  " << std::left << std::setw(34) << name << std::right << std::setw(10) << n
          << std::setw(14) << std::scientific << std::setprecision(4) << t << " s"
          << std::defaultfloat << std::endl;
  };
  pcout << std::endl << "Nonlinear operator (incl. contact terms):" << std::endl;
  pcout << "  " << std::left << std::setw(34) << "operation" << std::right << std::setw(10)
        << "calls" << std::setw(16) << "wall time" << std::endl;
  line("residual evaluations", s.n_residuals, s.t_residual);
  line("linearizations", s.n_linearizations, s.t_linearization);
  line("Jacobian-vector products", s.n_vmults, s.t_vmult);
  if(s.t_pattern > 0.0)
    pcout << "  sparsity pattern extensions (AMG matrix)  " << std::scientific
          << std::setprecision(4) << s.t_pattern << " s" << std::defaultfloat << std::endl;
}

template<int dim, typename Number>
bool
NonLinearOperator<dim, Number>::valid_deformation(VectorType const & displacement) const
{
  Number dst = 0.0;

  // dst has to remain zero for a valid deformation state
  this->matrix_free->cell_loop(
    &This::cell_loop_valid_deformation, this, dst, displacement, false /* zero_dst_vector */);

  // sum over all MPI processes
  Number valid = 0.0;
  valid        = dealii::Utilities::MPI::sum(
    dst, this->matrix_free->get_dof_handler(this->operator_data.dof_index).get_mpi_communicator());

  return (valid == 0.0);
}


template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::set_solution_linearization(VectorType const & vector) const
{
  // Only update linearized operator if deformation state is valid. It is better to continue
  // with an old deformation state in the linearized operator than with an invalid one.
  auto const start = std::chrono::steady_clock::now();

  if(valid_deformation(vector))
  {
    displacement_lin = vector;
    displacement_lin.update_ghost_values();

    if(deformable_contact)
      deformable_contact->set_linearization(vector);

    this->assemble_matrix_if_matrix_based();
  }

  ++cost_statistics.n_linearizations;
  cost_statistics.t_linearization +=
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::VectorType const &
NonLinearOperator<dim, Number>::get_solution_linearization() const
{
  return displacement_lin;
}

template<int dim, typename Number>
bool
NonLinearOperator<dim, Number>::contact_is_active() const
{
  return contact_active;
}

template<int dim, typename Number>
ContactQuantities<dim>
NonLinearOperator<dim, Number>::compute_contact_quantities(VectorType const & displacement,
                                                           bool const collect_point_data) const
{
  ContactQuantities<dim> result;
  result.time = this->time;

  if(not contact_active)
    return result;

  VectorType src;
  this->matrix_free->initialize_dof_vector(src, this->operator_data.dof_index);
  src.copy_locally_owned_data_from(displacement);
  src.update_ghost_values();

  // use the history of the last solution (see history_updated)
  bool const use_history_before = friction_active and history_updated;
  if(use_history_before)
  {
    displacement_previous.swap(displacement_before);
    std::swap(time_previous, time_before);
    traction_history.swap(traction_history_before);
  }

  IntegratorFace integrator(*this->matrix_free,
                            true,
                            this->operator_data.dof_index_inhomogeneous,
                            this->operator_data.quad_index);

  unsigned int const begin = this->matrix_free->n_inner_face_batches();
  unsigned int const end   = begin + this->matrix_free->n_boundary_face_batches();
  for(unsigned int face = begin; face < end; ++face)
  {
    dealii::types::boundary_id const boundary_id = this->matrix_free->get_boundary_id(face);
    if(this->operator_data.bc->get_boundary_type(boundary_id) != BoundaryType::Contact)
      continue;

    auto const obstacle = this->operator_data.bc->contact_bc.find(boundary_id)->second;
    obstacle->set_time(this->time);

    integrator.reinit(face);
    this->material_handler.reinit_face(*this->matrix_free, face);
    if(friction_active)
    {
      integrator_previous_face->reinit(face);
      integrator_previous_face->read_dof_values(displacement_previous);
      integrator_previous_face->evaluate(dealii::EvaluationFlags::values);
    }
    integrator.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

    unsigned int const n_lanes = this->matrix_free->n_active_entries_per_face_batch(face);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector const X = integrator.quadrature_point(q);

      ContactState const state = evaluate_contact_state(integrator, q, *obstacle, boundary_id);

      RigidObstacleValues<dim, Number> const & obstacle_values = state.obstacle;
      scalar const                             pressure        = state.pressure;
      scalar const                             JxW             = integrator.JxW(q);

      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        for(unsigned int d = 0; d < dim; ++d)
          result.force[d] += pressure[v] * obstacle_values.normal[d][v] * JxW[v];
        if(pressure[v] > 0.0)
          result.active_area += JxW[v];

        if(friction_active)
        {
          double norm_t = 0.0;
          for(unsigned int d = 0; d < dim; ++d)
          {
            result.force[d] += state.traction_tangential[d][v] * JxW[v];
            norm_t += state.traction_tangential[d][v] * state.traction_tangential[d][v];
          }
          result.max_tangential_traction =
            std::max(result.max_tangential_traction, std::sqrt(norm_t));
          if(pressure[v] > 0.0 and state.stick[v] > 0.5)
            result.stick_area += JxW[v];
        }
        result.max_pressure = std::max(result.max_pressure, (double)pressure[v]);
        result.min_gap      = std::min(result.min_gap, (double)obstacle_values.gap[v]);

        if(collect_point_data)
        {
          dealii::Point<dim> point;
          for(unsigned int d = 0; d < dim; ++d)
            point[d] = X[d][v];
          result.points.push_back(point);
          result.gap.push_back(obstacle_values.gap[v]);
          result.pressure.push_back(pressure[v]);
          dealii::Tensor<1, dim> t;
          if(friction_active)
            for(unsigned int d = 0; d < dim; ++d)
              t[d] = state.traction_tangential[d][v];
          result.tangential_traction.push_back(t);
        }
      }
    }
  }

  MPI_Comm const mpi_comm =
    this->matrix_free->get_dof_handler(this->operator_data.dof_index).get_mpi_communicator();
  for(unsigned int d = 0; d < dim; ++d)
    result.force[d] = dealii::Utilities::MPI::sum(result.force[d], mpi_comm);
  result.active_area  = dealii::Utilities::MPI::sum(result.active_area, mpi_comm);
  result.max_pressure = dealii::Utilities::MPI::max(result.max_pressure, mpi_comm);
  result.min_gap      = dealii::Utilities::MPI::min(result.min_gap, mpi_comm);
  result.friction     = friction_active;
  result.max_tangential_traction =
    dealii::Utilities::MPI::max(result.max_tangential_traction, mpi_comm);
  result.stick_area = dealii::Utilities::MPI::sum(result.stick_area, mpi_comm);

  if(use_history_before)
  {
    displacement_previous.swap(displacement_before);
    std::swap(time_previous, time_before);
    traction_history.swap(traction_history_before);
  }

  return result;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_cell_nonlinear(IntegratorCell &   integrator,
                                                      unsigned int const cell) const
{
  integrator.reinit(cell);

  this->material_handler.reinit(*this->matrix_free, cell);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorCell integrator_inhom(matrix_free,
                                  this->operator_data.dof_index_inhomogeneous,
                                  this->operator_data.quad_index);

  IntegratorCell integrator(matrix_free,
                            this->operator_data.dof_index,
                            this->operator_data.quad_index);

  auto const unsteady_flag = this->operator_data.unsteady ? dealii::EvaluationFlags::values :
                                                            dealii::EvaluationFlags::nothing;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    reinit_cell_nonlinear(integrator_inhom, cell);
    integrator.reinit(cell);

    integrator_inhom.gather_evaluate(src, unsteady_flag | dealii::EvaluationFlags::gradients);

    do_cell_integral_nonlinear(integrator_inhom);

    // make sure that we do not write into Dirichlet degrees of freedom
    integrator_inhom.integrate(unsteady_flag | dealii::EvaluationFlags::gradients,
                               integrator.begin_dof_values());
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop_valid_deformation(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  Number &                                dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorCell integrator(matrix_free,
                            this->operator_data.dof_index_inhomogeneous,
                            this->operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    reinit_cell_nonlinear(integrator, cell);

    integrator.gather_evaluate(src, dealii::EvaluationFlags::gradients);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // material displacement gradient
      tensor const Grad_d = integrator.get_gradient(q);

      // material deformation gradient
      tensor const F     = get_F<dim, Number>(Grad_d);
      scalar const det_F = determinant(F);
      for(unsigned int v = 0; v < det_F.size(); ++v)
      {
        // if deformation is invalid, add a positive value to dst
        if(det_F[v] <= 0.0)
          dst += 1.0;
      }
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::face_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::boundary_face_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorFace integrator_inhom(matrix_free,
                                  true,
                                  this->operator_data.dof_index_inhomogeneous,
                                  this->operator_data.quad_index);

  IntegratorFace integrator = IntegratorFace(matrix_free,
                                             true,
                                             this->operator_data.dof_index,
                                             this->operator_data.quad_index);

  // apply Neumann BCs
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(integrator_inhom, face);
    integrator.reinit(face);

    dealii::types::boundary_id const boundary_id = matrix_free.get_boundary_id(face);
    bool const                       is_contact_face =
      contact_active and
      (this->operator_data.bc->get_boundary_type(boundary_id) == BoundaryType::Contact);

    if(is_contact_face)
    {
      // the contact term depends on the current position x = X + d (and on the normal traction
      // for Nitsche's method)
      integrator_inhom.gather_evaluate(src,
                                       this->operator_data.contact_method ==
                                           ContactMethod::Nitsche ?
                                         dealii::EvaluationFlags::values |
                                           dealii::EvaluationFlags::gradients :
                                         dealii::EvaluationFlags::values);

      do_boundary_integral_contact_nonlinear(integrator_inhom, boundary_id);
    }
    else
    {
      // In case of a pull-back of the traction vector, we need to evaluate
      // the displacement gradient to obtain the surface area ratio da/dA.
      // We write the integrator flags explicitly in this case since they
      // depend on the parameter pull_back_traction.
      if(this->operator_data.pull_back_traction)
      {
        integrator_inhom.gather_evaluate(src, dealii::EvaluationFlags::gradients);
      }

      do_boundary_integral_continuous_nonlinear(integrator_inhom, boundary_id);
    }

    // make sure that we do not write into Dirichlet degrees of freedom
    integrator_inhom.integrate(this->integrator_flags.face_integrate,
                               integrator.begin_dof_values());
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral_nonlinear(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  // loop over all quadrature points
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    // material displacement gradient
    tensor const Grad_d = integrator.get_gradient(q);

    // material deformation gradient
    tensor const F = get_F<dim, Number>(Grad_d);

    // 2nd Piola-Kirchhoff stresses
    symmetric_tensor const S =
      material->second_piola_kirchhoff_stress(Grad_d, integrator.get_current_cell_index(), q);

    // 1st Piola-Kirchhoff stresses P = F * S
    tensor const P = F * S;

    // Grad_v : P
    integrator.submit_gradient(P, q);

    if(this->operator_data.unsteady)
    {
      integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                integrator.get_value(q),
                              q);
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_boundary_integral_continuous_nonlinear(
  IntegratorFace &                   integrator,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector traction = calculate_neumann_value<dim, Number>(
      q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->time);

    if(this->operator_data.pull_back_traction)
    {
      tensor F = get_F<dim, Number>(integrator.get_gradient(q));
      vector N = integrator.normal_vector(q);
      // da/dA * n = det F F^{-T} * N := n_star
      // -> da/dA = n_star.norm()
      vector n_star = determinant(F) * transpose(invert(F)) * N;
      // t_0 = da/dA * t
      traction *= n_star.norm();
    }

    integrator.submit_value(-traction, q);

    // the contact term of Nitsche's method with theta != 0 also tests with the gradient, which
    // has to be zero on the other boundaries
    if((this->integrator_flags.face_integrate & dealii::EvaluationFlags::gradients) !=
       dealii::EvaluationFlags::nothing)
      integrator.submit_gradient(tensor(), q);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_boundary_integral_contact_nonlinear(
  IntegratorFace &                   integrator,
  dealii::types::boundary_id const & boundary_id) const
{
  auto const obstacle = this->operator_data.bc->contact_bc.find(boundary_id)->second;
  obstacle->set_time(this->time);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    ContactState const state = evaluate_contact_state(integrator, q, *obstacle, boundary_id);

    // the obstacle pushes the body in direction of the normal n, t_0 = lambda * n (+ t_T)
    vector value = -state.pressure * state.obstacle.normal;
    if(friction_active)
      value -= state.traction_tangential;

    if(this->operator_data.contact_method == ContactMethod::Nitsche and
       this->operator_data.contact_nitsche_theta != 0.0)
    {
      // (theta / gamma) ([A]_+ - lambda(d)) lambda'(d; v)
      scalar const factor = this->operator_data.contact_nitsche_theta / state.gamma *
                            (state.pressure - state.normal_traction);
      value += factor * state.q;
      integrator.submit_gradient(factor * state.Q, q);
    }

    integrator.submit_value(value, q);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_cell_derived(IntegratorCell &   integrator,
                                                    unsigned int const cell) const
{
  Base::reinit_cell_derived(integrator, cell);

  integrator_lin->reinit(cell);

  integrator_lin->read_dof_values(displacement_lin);
  integrator_lin->evaluate(dealii::EvaluationFlags::gradients);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_boundary_face_derived(IntegratorFace &   integrator,
                                                             unsigned int const face) const
{
  (void)integrator;

  if(contact_active and this->operator_data.bc->get_boundary_type(
                          this->matrix_free->get_boundary_id(face)) == BoundaryType::Contact)
  {
    this->material_handler.reinit_face(*this->matrix_free, face);

    integrator_lin_face->reinit(face);
    integrator_lin_face->read_dof_values(displacement_lin);
    integrator_lin_face->evaluate(this->operator_data.contact_method == ContactMethod::Nitsche ?
                                    dealii::EvaluationFlags::values |
                                      dealii::EvaluationFlags::gradients :
                                    dealii::EvaluationFlags::values);

    if(friction_active)
    {
      integrator_previous_face->reinit(face);
      integrator_previous_face->read_dof_values(displacement_previous);
      integrator_previous_face->evaluate(dealii::EvaluationFlags::values);
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  // loop over all quadrature points
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    // kinematics
    tensor const Grad_delta = integrator.get_gradient(q);

    tensor const Grad_d_lin = integrator_lin->get_gradient(q);

    tensor const F_lin = get_F<dim, Number>(Grad_d_lin);

    // 2nd Piola-Kirchhoff stresses
    symmetric_tensor const S_lin =
      material->second_piola_kirchhoff_stress(Grad_d_lin, integrator.get_current_cell_index(), q);

    // directional derivative of 1st Piola-Kirchhoff stresses P

    // 1. elastic and initial displacement stiffness contributions
    tensor delta_P = F_lin * material->second_piola_kirchhoff_stress_displacement_derivative(
                               Grad_delta, F_lin, integrator.get_current_cell_index(), q);

    // 2. geometric (or initial stress) stiffness contribution
    delta_P += Grad_delta * S_lin;

    // Grad_v : delta_P
    integrator.submit_gradient(delta_P, q);

    if(this->operator_data.unsteady)
    {
      integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                integrator.get_value(q),
                              q);
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  if(operator_type == OperatorType::homogeneous and contact_active and
     this->operator_data.bc->get_boundary_type(boundary_id) == BoundaryType::Contact)
  {
    do_boundary_integral_contact(integrator, boundary_id);
  }
  else if(operator_type == OperatorType::homogeneous)
  {
    // The derivative with respect to the pull-back of the traction vector is neglected. We still
    // need to submit zeroes for consistency.
    vector zero;
    zero = 0.0;
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      integrator.submit_value(zero, q);

      // see do_boundary_integral_continuous_nonlinear()
      if((this->integrator_flags.face_integrate & dealii::EvaluationFlags::gradients) !=
         dealii::EvaluationFlags::nothing)
        integrator.submit_gradient(tensor(), q);
    }
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("OperatorType not supported in "
                                   "NonLinearOperator::do_boundary_integral_continuous()"));
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_boundary_integral_contact(
  IntegratorFace &                   integrator,
  dealii::types::boundary_id const & boundary_id) const
{
  auto const obstacle = this->operator_data.bc->contact_bc.find(boundary_id)->second;
  obstacle->set_time(this->time);

  bool const nitsche = (this->operator_data.contact_method == ContactMethod::Nitsche);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    // contact state at the point of linearization
    ContactState const state =
      evaluate_contact_state(*integrator_lin_face, q, *obstacle, boundary_id);

    vector const & n       = state.obstacle.normal;
    vector const   delta_d = integrator.get_value(q);

    // derivative of the normal
    vector const delta_n = state.obstacle.hessian * delta_d;

    scalar delta_argument;
    vector delta_sigma; // delta (P N), only computed for Nitsche
    tensor delta_P;     // delta P, only computed for Nitsche
    if(nitsche)
    {
      tensor const Grad_delta = integrator.get_gradient(q);

      // delta P = F delta S + Grad(delta d) S
      delta_P = state.F * this->material_handler.get_material()
                            ->second_piola_kirchhoff_stress_displacement_derivative(
                              Grad_delta, state.F, integrator.get_current_cell_index(), q) +
                Grad_delta * state.S;
      delta_sigma = delta_P * state.N;

      // delta (n . (P N)) - gamma (n . delta d)
      delta_argument =
        n * delta_sigma + delta_n * ((state.F * state.S) * state.N) - state.gamma * (n * delta_d);
    }
    else
    {
      delta_argument = -this->operator_data.contact_penalty * (n * delta_d);
    }

    // delta (lambda n) = H(A) delta A n + lambda delta n
    vector const delta_t = (state.active * delta_argument) * n + state.pressure * delta_n;

    vector value = -delta_t;

    if(friction_active)
    {
      // delta A_T = delta ((I - n (x) n) w) with
      //  Nitsche: w = sigma - gamma Delta g, delta w = delta sigma - gamma delta d,
      //  penalty: w = t_T^old - epsilon Delta g, delta w = - epsilon delta d
      // (the motion of the obstacle does not depend on the displacement)
      vector const & w       = state.w;
      vector const   delta_w = nitsche ? vector(delta_sigma - state.gamma * delta_d) :
                                         vector(-this->operator_data.contact_penalty * delta_d);
      vector const delta_A_T = delta_w - (n * delta_w) * n - (delta_n * w) * n - (n * w) * delta_n;

      // delta t_T: stick: delta A_T; slip: mu delta lambda s + mu lambda / |A_T| (I - s (x) s)
      // delta A_T with s = A_T / |A_T|
      scalar const & mu = state.mu;
      scalar const tiny = dealii::make_vectorized_array<Number>(std::numeric_limits<Number>::min());
      scalar const safe_norm    = std::max(state.norm_A_T, tiny);
      vector const s            = state.A_T / safe_norm;
      scalar const delta_lambda = state.active * delta_argument;
      vector const delta_t_slip = (mu * delta_lambda) * s + (mu * state.pressure / safe_norm) *
                                                              (delta_A_T - (s * delta_A_T) * s);
      scalar const one       = dealii::make_vectorized_array<Number>(1.0);
      vector const delta_t_T = state.stick * delta_A_T + (one - state.stick) * delta_t_slip;

      value -= delta_t_T;
    }

    if(nitsche and this->operator_data.contact_nitsche_theta != 0.0)
    {
      // (theta / gamma) ((H(A) - 1) lambda'(d; delta d) - H(A) gamma (n . delta d)) lambda'(d; v)
      // with lambda'(d; delta d) = delta A + gamma (n . delta d)
      scalar const n_delta_d    = n * delta_d;
      scalar const delta_lambda = delta_argument + state.gamma * n_delta_d;
      scalar const one          = dealii::make_vectorized_array<Number>(1.0);
      scalar const factor =
        this->operator_data.contact_nitsche_theta / state.gamma *
        ((state.active - one) * delta_lambda - state.active * state.gamma * n_delta_d);
      // second derivative (theta / gamma) ([A]_+ - lambda(d)) lambda''(d; delta d, v) with
      // lambda''(d; delta d, v) = Grad(v) : delta Q + v . delta q, where
      //  delta Q = D^2 P[Grad(delta d), n (x) N] + T[delta n (x) N],
      //  delta q = delta kappa (P N) + kappa (delta P N),
      // T[H] = F delta S[H] + H S the tangent of P and kappa = grad(grad(phi))
      scalar const weight = this->operator_data.contact_nitsche_theta / state.gamma *
                            (state.pressure - state.normal_traction);

      auto const         material   = this->material_handler.get_material();
      unsigned int const cell       = integrator.get_current_cell_index();
      tensor const       Grad_delta = integrator.get_gradient(q);
      tensor const       G          = dealii::outer_product(n, state.N);

      // step size of the central differences (exact for St. Venant-Kirchhoff, for which T is
      // quadratic in F)
      Number const step = std::cbrt(std::numeric_limits<Number>::epsilon());
      scalar const tiny = dealii::make_vectorized_array<Number>(std::numeric_limits<Number>::min());

      // D^2 P[Grad(delta d), G] by a central difference of T(F)[G] in direction Grad(delta d)
      tensor identity;
      for(unsigned int d = 0; d < dim; ++d)
        identity[d][d] = dealii::make_vectorized_array<Number>(1.0);
      tensor const Grad_d_lin = state.F - identity;
      scalar const h          = dealii::make_vectorized_array<Number>(step) /
                       std::max(std::sqrt(dealii::scalar_product(Grad_delta, Grad_delta)), tiny);
      auto const T_G = [&](tensor const & Grad_d) {
        tensor const           F_p = get_F<dim, Number>(Grad_d);
        symmetric_tensor const S_p = material->second_piola_kirchhoff_stress(Grad_d, cell, q);
        return tensor(
          F_p * material->second_piola_kirchhoff_stress_displacement_derivative(G, F_p, cell, q) +
          G * S_p);
      };
      tensor const D2P = (T_G(Grad_d_lin + h * Grad_delta) - T_G(Grad_d_lin - h * Grad_delta)) /
                         (dealii::make_vectorized_array<Number>(2.0) * h);

      // T[delta n (x) N]
      tensor const delta_n_N = dealii::outer_product(delta_n, state.N);
      tensor const T_delta_n =
        state.F * material->second_piola_kirchhoff_stress_displacement_derivative(delta_n_N,
                                                                                  state.F,
                                                                                  cell,
                                                                                  q) +
        delta_n_N * state.S;

      // delta kappa by a central difference of the Hessian of the obstacle in direction delta d
      vector const x_lin =
        integrator_lin_face->quadrature_point(q) + integrator_lin_face->get_value(q);
      scalar const h_x = dealii::make_vectorized_array<Number>(step) *
                         (dealii::make_vectorized_array<Number>(1.0) + std::sqrt(x_lin * x_lin)) /
                         std::max(std::sqrt(delta_d * delta_d), tiny);
      RigidObstacleValues<dim, Number> const plus =
        evaluate_rigid_obstacle<dim, Number>(*obstacle, x_lin + h_x * delta_d);
      RigidObstacleValues<dim, Number> const minus =
        evaluate_rigid_obstacle<dim, Number>(*obstacle, x_lin - h_x * delta_d);
      tensor const delta_kappa =
        (plus.hessian - minus.hessian) / (dealii::make_vectorized_array<Number>(2.0) * h_x);

      vector const P_N     = (state.F * state.S) * state.N;
      vector const delta_q = delta_kappa * P_N + state.obstacle.hessian * (delta_P * state.N);

      value += factor * state.q + weight * delta_q;
      integrator.submit_gradient(factor * state.Q + weight * (D2P + T_delta_n), q);
    }

    integrator.submit_value(value, q);
  }
}

template class NonLinearOperator<2, float>;
template class NonLinearOperator<2, double>;

template class NonLinearOperator<3, float>;
template class NonLinearOperator<3, double>;

} // namespace Structure
} // namespace ExaDG
