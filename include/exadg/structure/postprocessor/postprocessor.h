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

#ifndef EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_
#define EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/structure/postprocessor/output_generator.h>
#include <exadg/structure/postprocessor/postprocessor_base.h>
#include <exadg/structure/spatial_discretization/operator.h>

namespace ExaDG
{
namespace Structure
{
/*
 * Output of contact quantities for contact with rigid obstacles: at every evaluation, a line
 * with time, contact force, active contact area, maximum pressure and minimum gap is appended to
 * directory + filename + ".txt". Optionally, the gap and the contact pressure at all quadrature
 * points of the contact boundaries are written to directory + filename + "_profile_<n>.txt".
 */
struct ContactCalculationData
{
  TimeControlData time_control_data;
  std::string     directory              = "output/";
  std::string     filename               = "contact";
  bool            write_pressure_profile = false;
};

template<int dim>
struct PostProcessorData
{
  OutputData                output_data;
  ErrorCalculationData<dim> error_data;
  ContactCalculationData    contact_data;
};

template<int dim, typename Number>
class PostProcessor : public PostProcessorBase<dim, Number>
{
public:
  typedef PostProcessorBase<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  PostProcessor(PostProcessorData<dim> const & pp_data, MPI_Comm const & mpi_comm);

  // custom destructor computing convergence tables if desired
  virtual ~PostProcessor();

  void
  setup(Operator<dim, Number> const & pde_operator_in) override;

  bool
  requires_scalar_field() const;

  void
  do_postprocessing(VectorType const &     solution,
                    double const           time             = 0.0,
                    types::time_step const time_step_number = numbers::steady_timestep) override;

private:
  void
  initialize_derived_fields();

  void
  invalidate_derived_fields();

  void
  evaluate_contact(VectorType const & solution) const;

  PostProcessorData<dim> pp_data;

  MPI_Comm const mpi_comm;

  dealii::ObserverPointer<Operator<dim, Number> const> pde_operator;

  // Fields for derived quantities
  SolutionField<dim, Number> displacement_magnitude;
  SolutionField<dim, Number> displacement_jacobian;
  SolutionField<dim, Number> max_principal_stress;

  // write output for visualization of results (e.g., using paraview)
  OutputGenerator<dim, Number> output_generator;

  // calculate errors for verification purposes for problems with known analytical solution
  ErrorCalculator<dim, Number> error_calculator;

  // contact quantities
  TimeControl contact_time_control;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_POSTPROCESSOR_POSTPROCESSOR_H_ */
