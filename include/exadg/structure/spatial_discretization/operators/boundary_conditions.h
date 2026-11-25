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
#ifndef EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_BOUNDARY_CONDITIONS_H_
#define EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_BOUNDARY_CONDITIONS_H_

// deal.II
#include <deal.II/base/function.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/structure/user_interface/boundary_descriptor.h>

namespace ExaDG
{
namespace Structure
{
/*
 * This function calculates the Neumann boundary value.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
  calculate_neumann_value(unsigned int const                             q,
                          FaceIntegrator<dim, dim, Number> const &       integrator,
                          BoundaryType const &                           boundary_type,
                          dealii::types::boundary_id const               boundary_id,
                          std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
                          double const &                                 time)
{
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> traction;

  if(boundary_type == BoundaryType::Neumann)
  {
    auto bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
    auto q_points = integrator.quadrature_point(q);

    traction = FunctionEvaluator<1, dim, Number>::value(*bc, q_points, time);
  }
  else if(boundary_type == BoundaryType::NeumannCached)
  {
    auto bc = boundary_descriptor->get_neumann_cached_data();

    traction = FunctionEvaluator<1, dim, Number>::value(*bc,
                                                        integrator.get_current_cell_index(),
                                                        q,
                                                        integrator.get_quadrature_index());
  }
  else
  {
    // do nothing

    // Contact boundaries do not carry a prescribed traction, the contact traction is computed
    // separately (see evaluate_rigid_obstacle()).
    AssertThrow(boundary_type == BoundaryType::Dirichlet or
                  boundary_type == BoundaryType::DirichletCached or
                  boundary_type == BoundaryType::Contact,
                dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return traction;
}

/*
 * Rigid obstacle described by a signed distance function phi, evaluated at the current
 * position x = X + d of the quadrature points of a face batch:
 *
 *  gap     = phi(x)            (> 0: separated, < 0: penetration),
 *  normal  = grad(phi)(x)      (unit normal pointing from the obstacle towards the body),
 *  hessian = grad(grad(phi))(x).
 */
template<int dim, typename Number>
struct RigidObstacleValues
{
  dealii::VectorizedArray<Number>                         gap;
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> normal;
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> hessian;
};

template<int dim, typename Number>
inline RigidObstacleValues<dim, Number>
evaluate_rigid_obstacle(dealii::Function<dim> const &                                   obstacle,
                        dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & x)
{
  RigidObstacleValues<dim, Number> result;

  for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
  {
    dealii::Point<dim> point;
    for(unsigned int d = 0; d < dim; ++d)
      point[d] = x[d][v];

    result.gap[v] = obstacle.value(point);

    dealii::Tensor<1, dim> const grad = obstacle.gradient(point);
    dealii::Tensor<2, dim> const hess = obstacle.hessian(point);
    for(unsigned int d = 0; d < dim; ++d)
    {
      result.normal[d][v] = grad[d];
      for(unsigned int e = 0; e < dim; ++e)
        result.hessian[d][e][v] = hess[d][e];
    }
  }

  return result;
}

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_BOUNDARY_CONDITIONS_H_ */
