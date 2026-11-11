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
// deal.II
#include <deal.II/matrix_free/fe_point_evaluation.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/postprocessor/write_output.h>

namespace ExaDG
{
template<int rank, int dim, typename Number>
InterfaceCoupling<rank, dim, Number>::InterfaceCoupling()
  : dof_handler_src(nullptr), mapping_src(nullptr)
{
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::setup(
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>> interface_data_dst_,
  dealii::DoFHandler<dim> const &                            dof_handler_src_,
  dealii::Mapping<dim> const &                               mapping_src_,
  std::vector<bool> const &                                  marked_vertices_src_,
  double const                                               tolerance_)
{
  AssertThrow(interface_data_dst_.get(),
              dealii::ExcMessage("Received uninitialized variable. Aborting."));

  if(marked_vertices_src_.size() > 0)
  {
    AssertThrow(marked_vertices_src_.size() ==
                  (unsigned int)dof_handler_src_.get_triangulation().n_vertices(),
                dealii::ExcMessage("Vector marked_vertices_src_ has invalid size."));
  }

  interface_data_dst = interface_data_dst_;
  dof_handler_src    = &dof_handler_src_;
  mapping_src        = &mapping_src_;

  for(auto quad_index : interface_data_dst->get_quad_indices())
  {
    // exchange quadrature points with their owners
    map_evaluator.emplace(
      quad_index,
      std::make_unique<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(
        typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData(
          tolerance_, false, 0, [marked_vertices_src_]() { return marked_vertices_src_; })));

    auto const * points = &interface_data_dst->get_array_q_points(quad_index);

    map_evaluator[quad_index]->reinit(*points, dof_handler_src_.get_triangulation(), mapping_src_);

    if(not map_evaluator[quad_index]->all_points_found())
    {
      // get vector of points not found
      std::vector<dealii::Point<dim>> points_not_found;
      points_not_found.reserve(points->size());
      unsigned int n_points_not_found = 0;
      for(unsigned int i = 0; i < points->size(); ++i)
      {
        if(not map_evaluator[quad_index]->point_found(i))
        {
          n_points_not_found += 1;
          points_not_found.push_back((*points)[i]);
        }
      }
      n_points_not_found =
        dealii::Utilities::MPI::sum(n_points_not_found, dof_handler_src->get_mpi_communicator());

      std::string const file_name =
        "interface_coupling_quad_index_" + dealii::Utilities::to_string(quad_index);

      write_grid(dof_handler_src->get_triangulation(),
                 mapping_src_,
                 4,
                 "./",
                 file_name,
                 0,
                 dof_handler_src->get_mpi_communicator());

      write_points_in_dummy_triangulation(
        points_not_found, "./", file_name, 0, dof_handler_src->get_mpi_communicator());

      AssertThrow(map_evaluator[quad_index]->all_points_found(),
                  dealii::ExcMessage(std::string("Setup of InterfaceCoupling was not successful: " +
                                                 std::to_string(n_points_not_found) +
                                                 " points have not been found.")));
    }
  }
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::update_data(VectorType const & dof_vector_src)
{
  dof_vector_src.update_ghost_values();

  for(auto quadrature : interface_data_dst->get_quad_indices())
  {
    auto const result =
      dealii::VectorTools::point_values<n_components>(*map_evaluator[quadrature],
                                                      *dof_handler_src,
                                                      dof_vector_src,
                                                      dealii::VectorTools::EvaluationFlags::avg);

    auto & array_solution = interface_data_dst->get_array_solution(quadrature);

    Assert(result.size() == array_solution.size(),
           dealii::ExcMessage("Vectors must have the same length."));

    for(unsigned int i = 0; i < result.size(); ++i)
      array_solution[i] = result[i];
  }
}

template<int rank, int dim, typename Number>
void
InterfaceCoupling<rank, dim, Number>::add_transpose(VectorType & dof_vector_src) const
{
  AssertThrow(dof_handler_src != nullptr and mapping_src != nullptr,
              dealii::ExcMessage("InterfaceCoupling has not been set up."));

  dof_vector_src.zero_out_ghost_values();

  std::vector<dealii::types::global_dof_index> dof_indices;
  std::vector<Number>                          dof_values;
  std::vector<Number>                          point_values;

  for(auto const quadrature : interface_data_dst->get_quad_indices())
  {
    auto const & evaluator = *map_evaluator.at(quadrature);
    auto const & data      = interface_data_dst->get_array_solution(quadrature);
    auto const & ptrs      = evaluator.get_point_ptrs();

    AssertThrow(ptrs.size() == data.size() + 1,
                dealii::ExcMessage("Inconsistent number of interface points."));

    // flatten point data and apply the transpose of the averaging over multiple cells
    point_values.assign(data.size() * n_components, Number(0.0));
    for(unsigned int i = 0; i < data.size(); ++i)
    {
      unsigned int const multiplicity = ptrs[i + 1] - ptrs[i];
      if(multiplicity == 0)
        continue;
      Number const weight = Number(1.0) / Number(multiplicity);
      if constexpr(rank == 0)
        point_values[i] = weight * Number(data[i]);
      else
        for(unsigned int c = 0; c < n_components; ++c)
          point_values[i * n_components + c] = weight * Number(data[i][c]);
    }

    auto const cell_operation =
      [&](dealii::ArrayView<Number const> const &                                       values,
          typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::CellData const & cell_data) {
        dealii::FEPointEvaluation<n_components, dim, dim, Number> fe_point(
          *mapping_src, dof_handler_src->get_fe(), dealii::update_values);

        for(unsigned int const i : cell_data.cell_indices())
        {
          auto const cell        = cell_data.get_active_cell_iterator(i);
          auto const unit_points = cell_data.get_unit_points(i);
          auto const local_data  = cell_data.get_data_view(i, values, n_components);

          fe_point.reinit(cell, unit_points);

          for(unsigned int q = 0; q < unit_points.size(); ++q)
          {
            if constexpr(n_components == 1)
            {
              fe_point.submit_value(local_data[q], q);
            }
            else
            {
              dealii::Tensor<1, n_components, Number> value;
              for(unsigned int c = 0; c < n_components; ++c)
                value[c] = local_data[q * n_components + c];
              fe_point.submit_value(value, q);
            }
          }

          dof_values.assign(dof_handler_src->get_fe().n_dofs_per_cell(), Number(0.0));
          fe_point.test_and_sum(dof_values, dealii::EvaluationFlags::values);

          auto const dof_cell = cell->as_dof_handler_iterator(*dof_handler_src);
          dof_indices.resize(dof_values.size());
          dof_cell->get_dof_indices(dof_indices);

          for(unsigned int j = 0; j < dof_indices.size(); ++j)
            dof_vector_src(dof_indices[j]) += dof_values[j];
        }
      };

    evaluator.template process_and_evaluate<Number, n_components>(point_values, cell_operation);
  }

  dof_vector_src.compress(dealii::VectorOperation::add);
}

template class InterfaceCoupling<0, 2, float>;
template class InterfaceCoupling<1, 2, float>;
template class InterfaceCoupling<0, 3, float>;
template class InterfaceCoupling<1, 3, float>;

template class InterfaceCoupling<0, 2, double>;
template class InterfaceCoupling<1, 2, double>;
template class InterfaceCoupling<0, 3, double>;
template class InterfaceCoupling<1, 3, double>;

} // namespace ExaDG
