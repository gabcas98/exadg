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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONERS_PRECONDITIONER_DIRECT_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONERS_PRECONDITIONER_DIRECT_H_

// deal.II
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/utilities/linear_algebra_utilities.h>

namespace ExaDG
{
/*
 * Sparse direct solver (Trilinos Amesos, KLU by default) of the assembled system matrix of the
 * operator, applied as preconditioner: exact for the matrix of the last update(), so that a Krylov
 * method converges in one iteration as long as the matrix is current and in very few iterations
 * if it is a few (time or Newton) steps old. Meant for small problems, where the assembly and the
 * factorization are cheaper than the setup and the cycles of a multigrid method.
 */
template<typename Operator, typename Number>
class PreconditionerDirect : public PreconditionerBase<Number>
{
private:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;

public:
  PreconditionerDirect(Operator const &    pde_operator,
                       bool const          initialize,
                       std::string const & solver_type = "Amesos_Klu")
    : pde_operator(pde_operator),
      solver(dealii::TrilinosWrappers::SolverDirect::AdditionalData(false, solver_type))
  {
    pde_operator.init_system_matrix(
      system_matrix, pde_operator.get_matrix_free().get_dof_handler().get_mpi_communicator());

    if(initialize)
      this->update();
  }

  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    // the factorization is in double precision
    apply_function_in_double_precision(
      dst,
      src,
      [&](dealii::LinearAlgebra::distributed::Vector<double> &       dst_double,
          dealii::LinearAlgebra::distributed::Vector<double> const & src_double) {
        solver.vmult(dst_double, src_double);
      });
  }

  void
  update() final
  {
    // calculate_system_matrix() adds to the matrix
    system_matrix *= 0.0;
    pde_operator.calculate_system_matrix(system_matrix);
    solver.initialize(system_matrix);

    this->update_needed = false;
  }

private:
  Operator const & pde_operator;

  dealii::TrilinosWrappers::SparseMatrix         system_matrix;
  mutable dealii::TrilinosWrappers::SolverDirect solver;
};

} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONERS_PRECONDITIONER_DIRECT_H_ */
