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

#ifndef EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_NONLINEAR_OPERATOR_H_
#define EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_NONLINEAR_OPERATOR_H_

// C/C++
#include <limits>
#include <vector>

// ExaDG
#include <exadg/structure/spatial_discretization/operators/boundary_conditions.h>
#include <exadg/structure/spatial_discretization/operators/elasticity_operator_base.h>

namespace ExaDG
{
namespace Structure
{
/*
 * Integral quantities of the frictionless contact with rigid obstacles, all measured in the
 * reference configuration.
 */
template<int dim>
struct ContactQuantities
{
  // time (or load factor) at which the obstacles are evaluated
  double time = 0.0;

  // force exerted by the obstacles on the body, int_{Gamma_C} lambda n dA
  dealii::Tensor<1, dim> force;

  // reference area of the active contact zone (g < 0)
  double active_area = 0.0;

  // maximum contact pressure lambda and minimum gap g on the contact boundaries
  double max_pressure = 0.0;
  double min_gap      = std::numeric_limits<double>::max();

  // frictional contact: maximum magnitude of the tangential traction and reference area of the
  // stick zone (active points with |t_T| < mu lambda)
  bool   friction                = false;
  double max_tangential_traction = 0.0;
  double stick_area              = 0.0;

  // optional point data on the locally owned contact faces: reference position X of the
  // quadrature points, gap g and contact pressure lambda
  std::vector<dealii::Point<dim>>     points;
  std::vector<double>                 gap;
  std::vector<double>                 pressure;
  std::vector<dealii::Tensor<1, dim>> tangential_traction;
};

template<int dim, typename Number>
class NonLinearOperator : public ElasticityOperatorBase<dim, Number>
{
private:
  typedef ElasticityOperatorBase<dim, Number> Base;

  typedef typename Base::VectorType     VectorType;
  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef NonLinearOperator<dim, Number> This;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef dealii::VectorizedArray<Number>                                  scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>          vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>          tensor;
  typedef dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>> symmetric_tensor;

private:
  /**
   * Initialize function.
   */
  void
  initialize_derived() override;

public:
  /**
   * Evaluates the non-linear operator.
   *
   * This function ensures that constrained degrees of freedom are set to zero
   * in the dst vector. Note, however, that this function does not ensure
   * that Dirichlet degrees of freedom are set to the prescribed inhomogeneous
   * boundary data in the src vector. This needs to be done separately prior to
   * calling this function.
   */
  void
  evaluate_nonlinear(VectorType & dst, VectorType const & src) const;

  /**
   * Returns true if deformation state is valid.
   */
  bool
  valid_deformation(VectorType const & displacement) const;

  /**
   * Linearized operator: Sets the linearization vector.
   */
  void
  set_solution_linearization(VectorType const & vector) const;

  /**
   * Linearized operator: Returns the linearization vector.
   */
  VectorType const &
  get_solution_linearization() const;

  /**
   * Returns true if contact boundaries with rigid obstacles are present.
   */
  bool
  contact_is_active() const;

  /**
   * Evaluates the contact quantities for a given displacement vector (inhomogeneous Dirichlet
   * degrees of freedom need to be set). The obstacles are evaluated at the time of the operator,
   * i.e. the time (or load factor) of the last solve. Integral quantities are summed over all
   * MPI processes, point data (if requested) only contains locally owned faces.
   */
  ContactQuantities<dim>
  compute_contact_quantities(VectorType const & displacement, bool const collect_point_data) const;

  /**
   * Frictional contact: sets the solution of the previous converged load step and its time. The
   * relative tangential slip is measured with respect to this state.
   */
  void
  set_previous_solution(VectorType const & displacement, double const time) const;

  VectorType const &
  get_previous_solution() const;

  double
  get_time_previous() const;

private:
  /*
   * Non-linear operator.
   */
  void
  reinit_cell_nonlinear(IntegratorCell & integrator, unsigned int const cell) const;

  void
  cell_loop_nonlinear(dealii::MatrixFree<dim, Number> const & matrix_free,
                      VectorType &                            dst,
                      VectorType const &                      src,
                      Range const &                           range) const;

  void
  face_loop_nonlinear(dealii::MatrixFree<dim, Number> const & matrix_free,
                      VectorType &                            dst,
                      VectorType const &                      src,
                      Range const &                           range) const;

  void
  boundary_face_loop_nonlinear(dealii::MatrixFree<dim, Number> const & matrix_free,
                               VectorType &                            dst,
                               VectorType const &                      src,
                               Range const &                           range) const;

  /*
   * A cell loop that checks whether the Jacobian determinant is positive for all q-points.
   * Prior to calling this function, inhomogeneous Dirichlet degrees of freedom need to be
   * set correctly.
   */
  void
  cell_loop_valid_deformation(dealii::MatrixFree<dim, Number> const & matrix_free,
                              Number &                                dst,
                              VectorType const &                      src,
                              Range const &                           range) const;

  /*
   * Calculates the integral
   *
   *  (v_h, factor * d_h)_Omega + (Grad(v_h), P_h)_Omega
   *
   * with 1st Piola-Kirchhoff stress tensor P_h
   *
   *  P_h = F_h * S_h ,
   *
   * 2nd Piola-Kirchhoff stress tensor S_h
   *
   *  S_h = function(E_h) ,
   *
   * Green-Lagrange strain tensor E_h
   *
   *  E_h = 1/2 (F_h^T * F_h - 1) ,
   *
   * material deformation gradient F_h
   *
   *  F_h = 1 + Grad(d_h) ,
   *
   * where
   *
   *  d_h denotes the displacement vector.
   */
  void
  do_cell_integral_nonlinear(IntegratorCell & integrator) const;

  /*
   * Computes the nonlinear Neumann boundary term
   *
   *  - (v_h, t_0)_{Gamma_N}
   *
   * with traction
   *
   *  t_0 = da/dA t .
   *
   * If the traction is specified as force per surface area of the underformed
   * body, the specified traction t is interpreted as t_0 = t, and no pull-back
   * is necessary.
   */
  void
  do_boundary_integral_continuous_nonlinear(IntegratorFace &                   integrator,
                                            dealii::types::boundary_id const & boundary_id) const;

  /*
   * State of the frictionless contact with a rigid obstacle at a quadrature point: gap
   * g = phi(x), normal n = grad(phi)(x) and curvature grad(grad(phi))(x) of the obstacle at the
   * current position x = X + d_h, and the contact pressure lambda = [A]_+ with
   *
   *  Penalty: A = - epsilon g ,
   *  Nitsche: A = lambda(d_h) - gamma g ,  lambda(d_h) = n . (P(d_h) N) ,
   *
   * with the first Piola-Kirchhoff stress P = F S and the outward unit normal N of the reference
   * configuration. All quantities refer to the reference configuration, i.e. lambda is a force
   * per undeformed area.
   */
  struct ContactState
  {
    RigidObstacleValues<dim, Number> obstacle;
    scalar                           pressure; // lambda = [A]_+
    scalar                           active;   // H(A), points with A = 0 are active

    // Nitsche only
    scalar           gamma;
    vector           N;
    tensor           F;
    symmetric_tensor S;
    scalar           normal_traction; // lambda(d) = n . (P N)

    // Nitsche with theta != 0: derivative of the normal traction in direction of a test
    // function v, lambda'(d; v) = Grad(v) : Q + v . q (using the major symmetry of the tangent
    // of hyperelastic materials), with Q = delta P[n (x) N] and q = grad(grad(phi)) (P N)
    tensor Q;
    vector q;

    // friction (Nitsche, theta = 0): traction sigma = P N, relative slip Delta g, trial
    // tangential traction A_T = (I - n (x) n)(sigma - gamma Delta g), tangential traction
    // t_T = Proj_{|t| <= mu lambda}(A_T) and stick indicator (|A_T| <= mu lambda)
    vector traction;
    vector delta_g;
    vector w; // Nitsche: sigma - gamma Delta g, penalty: t_T^old - epsilon Delta g
    vector A_T;
    scalar norm_A_T;
    vector traction_tangential;
    scalar stick;
  };

  /*
   * Evaluates the contact state. The integrator has to provide values of the displacement (and
   * gradients for ContactMethod::Nitsche), and the material has to be set for the current face.
   */
  ContactState
  evaluate_contact_state(IntegratorFace const &           integrator,
                         unsigned int const               q,
                         dealii::Function<dim> const &    obstacle,
                         dealii::types::boundary_id const boundary_id) const;

  /*
   * Frictional contact: relative tangential slip of the body with respect to the obstacle since
   * the previous load step, Delta g = (x - x_prev) - (u_obs(x, t) - u_obs(x, t_prev)), at
   * quadrature point q of the current boundary face (the previous displacement is evaluated in
   * reinit_boundary_face_derived()).
   */
  vector
  calculate_relative_slip(vector const &                     x,
                          unsigned int const                 q,
                          dealii::types::boundary_id const & boundary_id) const;

  /*
   * Computes the frictionless contact term with a rigid obstacle
   *
   *  - (v_h, lambda n)_{Gamma_C} ,
   *
   * see ContactState. For ContactMethod::Nitsche, this is the variant theta = 0 of
   * Chouly and Hild (2013), extended to finite strains.
   */
  void
  do_boundary_integral_contact_nonlinear(IntegratorFace &                   integrator,
                                         dealii::types::boundary_id const & boundary_id) const;

  /*
   * Linearized operator.
   */
  void
  reinit_cell_derived(IntegratorCell & integrator, unsigned int const cell) const final;

  /*
   * Evaluates the linearization vector on boundary faces (needed for the contact term).
   */
  void
  reinit_boundary_face_derived(IntegratorFace & integrator, unsigned int const face) const final;

  /*
   * Calculates the integral
   *
   *  (v_h, factor * delta d_h)_Omega + (Grad(v_h), delta P_h)_Omega
   *
   * with the directional derivative of the 1st Piola-Kirchhoff stress tensor P_h
   *
   *  delta P_h = d(P)/d(d)|_{d_lin} * delta d_h ,
   *
   * with the point of linearization
   *
   *  d_lin ,
   *
   * and displacement increment
   *
   *  delta d_h .
   *
   * Computing the linearization yields
   *
   *  delta P_h = + Grad(delta_d) * S(d_lin)
   *              + F(d_lin) * (C_lin : (F^T(d_lin) * Grad(delta d))) .
   *
   *  Note that a dependency of the Neumann BC on the displacements d through
   *  the area ratio da/dA = function(d) is neglected in the linearization.
   */
  void
  do_cell_integral(IntegratorCell & integrator) const final;

  /*
   * Compute the boundary term of the linearized operator
   *
   *  - delta (v_h, t_0)_{Gamma_N}
   *
   * with traction
   *
   *  t_0 = da/dA t .
   *
   * If the traction is specified as force per surface area of the underformed
   * body, the specified traction t is interpreted as t_0 = t, and no pull-back
   * is necessary.
   */
  void
  do_boundary_integral_continuous(IntegratorFace &                   integrator,
                                  OperatorType const &               operator_type,
                                  dealii::types::boundary_id const & boundary_id) const final;

  /*
   * Linearization of the contact term at the point of linearization d_lin
   *
   *  delta (- (v_h, lambda n)_{Gamma_C})
   *    = - (v_h, H(A) delta A n + lambda grad(grad(phi)) delta d)_{Gamma_C} ,
   *
   * with the Heaviside function H and
   *
   *  Penalty: delta A = - epsilon (n . delta d) ,
   *  Nitsche: delta A = n . (delta P N) + (grad(grad(phi)) delta d) . (P N) - gamma (n . delta d) ,
   *           delta P = F delta S + Grad(delta d) S .
   *
   * For Nitsche with theta != 0, the term
   *
   *  (theta / gamma) ([A]_+ - lambda(d), lambda'(d; v))_{Gamma_C}
   *
   * is added to the residual, and its linearization
   *
   *  (theta / gamma) ((H(A) - 1) lambda'(d; delta d) - H(A) gamma (n . delta d), lambda'(d; v))
   *
   * to the linearized operator. The second derivative lambda''(d; delta d, v), which is weighted
   * by [A]_+ - lambda(d) and vanishes for the exact solution, is neglected. With this, the
   * linearized operator is symmetric for theta = 1 and non-symmetric otherwise.
   */
  void
  do_boundary_integral_contact(IntegratorFace &                   integrator,
                               dealii::types::boundary_id const & boundary_id) const;

  mutable std::shared_ptr<IntegratorCell> integrator_lin;
  mutable VectorType                      displacement_lin;

  // contact with rigid obstacles
  bool                                    contact_active = false;
  mutable std::shared_ptr<IntegratorFace> integrator_lin_face;

  // Nitsche parameter gamma per cell
  dealii::AlignedVector<scalar> nitsche_gamma;

  // frictional contact: solution of the previous load step and its time
  bool               friction_active = false;
  mutable VectorType displacement_previous;
  mutable double     time_previous = 0.0;

  // penalty method with friction: tangential traction of the previous load step at the
  // quadrature points of the boundary face batches (index (face - n_inner_face_batches) * n_q + q)
  mutable dealii::AlignedVector<vector> traction_history, traction_history_before;
  unsigned int                          n_q_points_face = 0;

  // evaluates the tangential traction of the given displacement at the given time (the time of
  // the displacement, which may differ from the current time of the operator, e.g. on multigrid
  // levels) with the current history and stores it as the new history
  void
  update_traction_history(VectorType const & displacement, double const time) const;

  // history before the last call of set_previous_solution(): if the history has been updated
  // after the last evaluation of the residual (e.g. by the quasi-static driver after a converged
  // load step), the contact quantities of the last solution are evaluated with this history
  mutable VectorType                      displacement_before;
  mutable double                          time_before     = 0.0;
  mutable bool                            history_updated = false;
  mutable std::shared_ptr<IntegratorFace> integrator_previous_face;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_NONLINEAR_OPERATOR_H_ */
