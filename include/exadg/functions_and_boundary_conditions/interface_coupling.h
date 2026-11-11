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

#ifndef EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_INTERFACE_COUPLING_H_
#define EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_INTERFACE_COUPLING_H_

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/container_interface_data.h>
#include <exadg/utilities/tensor_utilities.h>

namespace ExaDG
{
template<int rank, int dim, typename Number>
class InterfaceCoupling
{
private:
  static unsigned int const n_components = rank_to_n_components<rank, dim>();

  using quad_index = unsigned int;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  InterfaceCoupling();

  /**
   * setup() function.
   *
   * The aim of @param marked_vertices_src_ is to make the search of points on the src side
   * computationally more efficient. If no useful information can be provided for this parameter, an
   * empty vector has to be passed to this function.
   *
   * @param tolerance_ is a geometric tolerance passed to dealii::RemotePointEvaluation and used for
   * the search of points on the src side.
   */
  void
  setup(std::shared_ptr<ContainerInterfaceData<rank, dim, double>> interface_data_dst_,
        dealii::DoFHandler<dim> const &                            dof_handler_src_,
        dealii::Mapping<dim> const &                               mapping_src_,
        std::vector<bool> const &                                  marked_vertices_src_,
        double const                                               tolerance_);

  void
  update_data(VectorType const & dof_vector_src);

  /**
   * Exact transpose of update_data(): the point values currently stored in the dst-side container
   * are interpreted as coefficients of point functionals and are scattered back to the src-side
   * degrees of freedom, i.e. dof_vector_src += E^T * data, where E is the (averaged) point
   * evaluation operator applied by update_data(). Points that are found in several cells are
   * weighted by the inverse of their multiplicity, which is the transpose of the averaging in
   * update_data(). Constraints are not applied: the caller is responsible for projecting the
   * result onto the admissible space.
   */
  void
  add_transpose(VectorType & dof_vector_src) const;

  std::shared_ptr<ContainerInterfaceData<rank, dim, double>>
  get_interface_data() const
  {
    return interface_data_dst;
  }

private:
  /*
   * dst-side
   */
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>> interface_data_dst;

  /*
   *  Evaluates solution on src-side in those points specified by dst-side
   */
  std::map<quad_index, std::unique_ptr<dealii::Utilities::MPI::RemotePointEvaluation<dim>>>
    map_evaluator;

  /*
   * src-side
   */
  dealii::DoFHandler<dim> const * dof_handler_src;

  dealii::Mapping<dim> const * mapping_src;
};

} // namespace ExaDG

#endif /* EXADG_FUNCTIONS_AND_BOUNDARY_CONDITIONS_INTERFACE_COUPLING_H_ */
