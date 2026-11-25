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

#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/structure/spatial_discretization/operators/boundary_conditions.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <exadg/structure/spatial_discretization/operators/nonlinear_operator.h>

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

    integrator_lin_face = std::make_shared<IntegratorFace>(*this->matrix_free,
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
      Number const factor = IP::get_penalty_factor<dim, Number>(
        dof_handler.get_fe().degree, element_type, this->operator_data.contact_penalty);
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
    }
  }
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::ContactState
NonLinearOperator<dim, Number>::evaluate_contact_state(IntegratorFace const &        integrator,
                                                       unsigned int const            q,
                                                       dealii::Function<dim> const & obstacle) const
{
  ContactState state;

  // current position
  vector const x = integrator.quadrature_point(q) + integrator.get_value(q);

  state.obstacle = evaluate_rigid_obstacle<dim, Number>(obstacle, x);

  scalar argument;
  if(this->operator_data.contact_method == ContactMethod::Penalty)
  {
    argument = -this->operator_data.contact_penalty * state.obstacle.gap;
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
  state.active =
    dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(argument,
                                                                                   zero,
                                                                                   one,
                                                                                   zero);

  return state;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::evaluate_nonlinear(VectorType & dst, VectorType const & src) const
{
  this->matrix_free->loop(&This::cell_loop_nonlinear,
                          &This::face_loop_nonlinear,
                          &This::boundary_face_loop_nonlinear,
                          this,
                          dst,
                          src,
                          true /* zero_dst_vector */);
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
  if(valid_deformation(vector))
  {
    displacement_lin = vector;
    displacement_lin.update_ghost_values();

    this->assemble_matrix_if_matrix_based();
  }
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
    integrator.gather_evaluate(src,
                               dealii::EvaluationFlags::values |
                                 dealii::EvaluationFlags::gradients);

    unsigned int const n_lanes = this->matrix_free->n_active_entries_per_face_batch(face);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector const X = integrator.quadrature_point(q);

      ContactState const state = evaluate_contact_state(integrator, q, *obstacle);

      RigidObstacleValues<dim, Number> const & obstacle_values = state.obstacle;
      scalar const                             pressure        = state.pressure;
      scalar const                             JxW             = integrator.JxW(q);

      for(unsigned int v = 0; v < n_lanes; ++v)
      {
        for(unsigned int d = 0; d < dim; ++d)
          result.force[d] += pressure[v] * obstacle_values.normal[d][v] * JxW[v];
        if(pressure[v] > 0.0)
          result.active_area += JxW[v];
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
    ContactState const state = evaluate_contact_state(integrator, q, *obstacle);

    // the obstacle pushes the body in direction of the normal n, t_0 = lambda * n
    vector value = -state.pressure * state.obstacle.normal;

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

  if(contact_active and
     this->operator_data.bc->get_boundary_type(this->matrix_free->get_boundary_id(face)) ==
       BoundaryType::Contact)
  {
    this->material_handler.reinit_face(*this->matrix_free, face);

    integrator_lin_face->reinit(face);
    integrator_lin_face->read_dof_values(displacement_lin);
    integrator_lin_face->evaluate(this->operator_data.contact_method == ContactMethod::Nitsche ?
                                    dealii::EvaluationFlags::values |
                                      dealii::EvaluationFlags::gradients :
                                    dealii::EvaluationFlags::values);
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
    ContactState const state = evaluate_contact_state(*integrator_lin_face, q, *obstacle);

    vector const & n       = state.obstacle.normal;
    vector const   delta_d = integrator.get_value(q);

    // derivative of the normal
    vector const delta_n = state.obstacle.hessian * delta_d;

    scalar delta_argument;
    if(nitsche)
    {
      tensor const Grad_delta = integrator.get_gradient(q);

      // delta P = F delta S + Grad(delta d) S
      tensor const delta_P =
        state.F * this->material_handler.get_material()
                    ->second_piola_kirchhoff_stress_displacement_derivative(
                      Grad_delta, state.F, integrator.get_current_cell_index(), q) +
        Grad_delta * state.S;

      // delta (n . (P N)) - gamma (n . delta d)
      delta_argument = n * (delta_P * state.N) + delta_n * ((state.F * state.S) * state.N) -
                       state.gamma * (n * delta_d);
    }
    else
    {
      delta_argument = -this->operator_data.contact_penalty * (n * delta_d);
    }

    // delta (lambda n) = H(A) delta A n + lambda delta n
    vector const delta_t = (state.active * delta_argument) * n + state.pressure * delta_n;

    vector value = -delta_t;

    if(nitsche and this->operator_data.contact_nitsche_theta != 0.0)
    {
      // (theta / gamma) ((H(A) - 1) lambda'(d; delta d) - H(A) gamma (n . delta d)) lambda'(d; v)
      // with lambda'(d; delta d) = delta A + gamma (n . delta d)
      scalar const n_delta_d     = n * delta_d;
      scalar const delta_lambda  = delta_argument + state.gamma * n_delta_d;
      scalar const one           = dealii::make_vectorized_array<Number>(1.0);
      scalar const factor        = this->operator_data.contact_nitsche_theta / state.gamma *
                            ((state.active - one) * delta_lambda -
                             state.active * state.gamma * n_delta_d);
      value += factor * state.q;
      integrator.submit_gradient(factor * state.Q, q);
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
