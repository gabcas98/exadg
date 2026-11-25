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

// C/C++
#include <fstream>
#include <iomanip>

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/utilities/create_directories.h>
#include <exadg/utilities/evaluate_convergence_study.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & pp_data_in,
                                          MPI_Comm const &               mpi_comm_in)
  : pp_data(pp_data_in),
    mpi_comm(mpi_comm_in),
    output_generator(OutputGenerator<dim, Number>(mpi_comm_in)),
    error_calculator(ErrorCalculator<dim, Number>(mpi_comm_in))
{
}

template<int dim, typename Number>
PostProcessor<dim, Number>::~PostProcessor()
{
  // Evaluate the convergence study.
  std::vector<std::string> error_directories;
  if(pp_data.error_data.compute_convergence_table)
    error_directories.push_back(pp_data.error_data.directory);

  evaluate_convergence_study(mpi_comm, error_directories);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(Operator<dim, Number> const & pde_operator_in)
{
  pde_operator = &pde_operator_in;

  initialize_derived_fields();

  output_generator.setup(pde_operator->get_dof_handler(),
                         pde_operator->get_mapping(),
                         pp_data.output_data);

  error_calculator.setup(pde_operator->get_dof_handler(),
                         pde_operator->get_mapping(),
                         pp_data.error_data);

  contact_time_control.setup(pp_data.contact_data.time_control_data);

  if(pp_data.contact_data.time_control_data.is_active)
  {
    AssertThrow(pde_operator->contact_is_active(),
                dealii::ExcMessage("Contact output requested, but no contact boundaries exist."));

    create_directories(pp_data.contact_data.directory, mpi_comm);

    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::ofstream f(pp_data.contact_data.directory + pp_data.contact_data.filename + ".txt",
                      std::ios::trunc);
      f << "# time";
      for(unsigned int d = 0; d < dim; ++d)
        f << " force_" << d;
      f << " active_area max_pressure min_gap";
      if(pde_operator->contact_friction_is_active())
        f << " max_tangential_traction stick_area";
      f << std::endl;
    }
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::evaluate_contact(VectorType const & solution) const
{
  ContactCalculationData const & data = pp_data.contact_data;

  ContactQuantities<dim> const contact =
    pde_operator->compute_contact_quantities(solution, data.write_pressure_profile);

  unsigned int const rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

  // values below 1e-12 are printed as zero to keep the screen output free of round-off (the
  // file output below contains the full values)
  auto const filter = [](double const value) { return std::abs(value) < 1e-12 ? 0.0 : value; };

  dealii::ConditionalOStream pcout(std::cout, rank == 0);
  pcout << std::endl << "Contact with rigid obstacles:" << std::endl << std::scientific;
  pcout << "  Contact force:            ";
  for(unsigned int d = 0; d < dim; ++d)
    pcout << " " << std::setprecision(5) << filter(contact.force[d]);
  pcout << std::endl
        << "  Active contact area:       " << std::setprecision(5) << filter(contact.active_area)
        << std::endl
        << "  Max. contact pressure:     " << std::setprecision(5) << filter(contact.max_pressure)
        << std::endl
        << "  Min. gap:                  " << std::setprecision(5) << filter(contact.min_gap)
        << std::endl;
  if(contact.friction)
    pcout << "  Max. tangential traction:  " << std::setprecision(5)
          << filter(contact.max_tangential_traction) << std::endl
          << "  Stick area:                " << std::setprecision(5)
          << filter(contact.stick_area) << std::endl;

  if(rank == 0)
  {
    std::ofstream f(data.directory + data.filename + ".txt", std::ios::app);
    f << std::scientific << std::setprecision(10) << contact.time;
    for(unsigned int d = 0; d < dim; ++d)
      f << " " << contact.force[d];
    f << " " << contact.active_area << " " << contact.max_pressure << " " << contact.min_gap;
    if(contact.friction)
      f << " " << contact.max_tangential_traction << " " << contact.stick_area;
    f << std::endl;
  }

  if(data.write_pressure_profile)
  {
    // flatten point data: X_0, ..., X_{dim-1}, gap, pressure (, t_T_0, ..., t_T_{dim-1})
    unsigned int const n_values = contact.friction ? 2 * dim + 2 : dim + 2;
    std::vector<double> local;
    local.reserve(contact.points.size() * n_values);
    for(unsigned int i = 0; i < contact.points.size(); ++i)
    {
      for(unsigned int d = 0; d < dim; ++d)
        local.push_back(contact.points[i][d]);
      local.push_back(contact.gap[i]);
      local.push_back(contact.pressure[i]);
      if(contact.friction)
        for(unsigned int d = 0; d < dim; ++d)
          local.push_back(contact.tangential_traction[i][d]);
    }

    std::vector<std::vector<double>> const all = dealii::Utilities::MPI::gather(mpi_comm, local);

    if(rank == 0)
    {
      std::ofstream f(data.directory + data.filename + "_profile_" +
                        dealii::Utilities::int_to_string(contact_time_control.get_counter(), 4) +
                        ".txt",
                      std::ios::trunc);
      f << "# time = " << contact.time << std::endl << "#";
      for(unsigned int d = 0; d < dim; ++d)
        f << " X_" << d;
      f << " gap pressure";
      if(contact.friction)
        for(unsigned int d = 0; d < dim; ++d)
          f << " t_T_" << d;
      f << std::endl;
      f << std::scientific << std::setprecision(10);
      for(auto const & values : all)
        for(unsigned int i = 0; i < values.size(); i += n_values)
        {
          for(unsigned int j = 0; j < n_values; ++j)
            f << values[i + j] << (j + 1 < n_values ? " " : "");
          f << std::endl;
        }
    }
  }
}

template<int dim, typename Number>
bool
PostProcessor<dim, Number>::requires_scalar_field() const
{
  return (pp_data.output_data.write_displacement_magnitude or
          pp_data.output_data.write_displacement_jacobian);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(VectorType const &     solution,
                                              double const           time,
                                              types::time_step const time_step_number)
{
  invalidate_derived_fields();

  /*
   *  write output
   */
  if(output_generator.time_control.needs_evaluation(time, time_step_number))
  {
    std::vector<dealii::ObserverPointer<SolutionField<dim, Number>>> additional_fields_vtu;

    if(pp_data.output_data.write_displacement_magnitude)
    {
      displacement_magnitude.evaluate(solution);
      additional_fields_vtu.push_back(&displacement_magnitude);
    }

    if(pp_data.output_data.write_displacement_jacobian)
    {
      displacement_jacobian.evaluate(solution);
      additional_fields_vtu.push_back(&displacement_jacobian);
    }

    if(pp_data.output_data.write_max_principal_stress)
    {
      max_principal_stress.evaluate(solution);
      additional_fields_vtu.push_back(&max_principal_stress);
    }

    output_generator.evaluate(solution,
                              additional_fields_vtu,
                              time,
                              Utilities::is_unsteady_timestep(time_step_number));
  }

  /*
   *  calculate error
   */
  if(error_calculator.time_control.needs_evaluation(time, time_step_number))
    error_calculator.evaluate(solution, time, Utilities::is_unsteady_timestep(time_step_number));

  /*
   *  contact quantities
   */
  if(contact_time_control.needs_evaluation(time, time_step_number))
    evaluate_contact(solution);
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::initialize_derived_fields()
{
  // displacement magnitude
  if(pp_data.output_data.write_displacement_magnitude)
  {
    displacement_magnitude.type              = SolutionFieldType::scalar;
    displacement_magnitude.name              = "displacement_magnitude";
    displacement_magnitude.dof_handler       = &pde_operator->get_dof_handler_scalar();
    displacement_magnitude.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar(dst);
    };
    displacement_magnitude.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                          const VectorType & src_vector_valued) {
      pde_operator->compute_displacement_magnitude(dst_scalar_valued, src_vector_valued);
    };

    displacement_magnitude.reinit();
  }

  // Jacobian of the displacement field
  if(pp_data.output_data.write_displacement_jacobian)
  {
    displacement_jacobian.type              = SolutionFieldType::scalar;
    displacement_jacobian.name              = "displacement_jacobian";
    displacement_jacobian.dof_handler       = &pde_operator->get_dof_handler_scalar();
    displacement_jacobian.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar(dst);
    };
    displacement_jacobian.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                         const VectorType & src_vector_valued) {
      pde_operator->compute_displacement_jacobian(dst_scalar_valued, src_vector_valued);
    };

    displacement_jacobian.reinit();
  }

  // Maximum principal stress
  if(pp_data.output_data.write_max_principal_stress)
  {
    max_principal_stress.type              = SolutionFieldType::scalar;
    max_principal_stress.name              = "max_principal_stress";
    max_principal_stress.dof_handler       = &pde_operator->get_dof_handler_scalar();
    max_principal_stress.initialize_vector = [&](VectorType & dst) {
      pde_operator->initialize_dof_vector_scalar(dst);
    };
    max_principal_stress.recompute_solution_field = [&](VectorType &       dst_scalar_valued,
                                                        const VectorType & src_vector_valued) {
      pde_operator->compute_max_principal_stress(dst_scalar_valued, src_vector_valued);
    };

    max_principal_stress.reinit();
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::invalidate_derived_fields()
{
  displacement_magnitude.invalidate();
  displacement_jacobian.invalidate();
  max_principal_stress.invalidate();
}

template class PostProcessor<2, float>;
template class PostProcessor<3, float>;

template class PostProcessor<2, double>;
template class PostProcessor<3, double>;

} // namespace Structure
} // namespace ExaDG
