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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_H_

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/newton/newton_solver_data.h>

namespace ExaDG
{
namespace Newton
{
template<typename VectorType,
         typename NonlinearOperator,
         typename LinearOperator,
         typename LinearSolver>
class Solver
{
public:
  Solver(SolverData const &  solver_data_in,
         NonlinearOperator & nonlinear_operator_in,
         LinearOperator &    linear_operator_in,
         LinearSolver &      linear_solver_in)
    : solver_data(solver_data_in),
      nonlinear_operator(nonlinear_operator_in),
      linear_operator(linear_operator_in),
      linear_solver(linear_solver_in)
  {
  }

  std::tuple<unsigned int /* Newton iter */, unsigned int /* accumulated linear iter */>
  solve(VectorType & solution, UpdateData const & update)
  {
    unsigned int newton_iterations = 0, linear_iterations = 0;

    n_nonmonotone_accepted = 0;

    VectorType residual, increment, temporary;
    residual.reinit(solution, true /* omit_zeroing_entries */);
    increment.reinit(solution, true /* omit_zeroing_entries */);
    temporary.reinit(solution, true /* omit_zeroing_entries */);

    // evaluate residual using initial guess of solution
    nonlinear_operator.evaluate_residual(residual, solution);

    double norm_r   = residual.l2_norm();
    double norm_r_0 = norm_r;

    while(norm_r > this->solver_data.abs_tol and norm_r / norm_r_0 > solver_data.rel_tol and
          newton_iterations < solver_data.max_iter)
    {
      // reset increment
      increment = 0.0;

      // multiply by -1.0 since the linearized problem is "linear_operator * increment = - residual"
      residual *= -1.0;

      // update linear operator (set linearization point)
      linear_operator.set_solution_linearization(solution);

      // determine whether to update the operator/preconditioner of the linearized problem
      bool const update_now =
        update.do_update and (newton_iterations % update.update_every_newton_iter == 0);

      // update the preconditioner
      linear_solver.update_preconditioner(update_now);

      // solve linear problem
      unsigned int const n_iter_linear = linear_solver.solve(increment, residual);

      // damped Newton scheme
      double             omega         = 1.0; // damping factor (begin with 1)
      double             norm_r_damp   = 1.0; // norm of residual using temporary solution
      unsigned int       n_iter_damp   = 0;   // counts iteration of damping scheme
      unsigned int const max_iter_damp = solver_data.max_iter_damping; // max damping iterations
      double const       tau           = 0.5; // a parameter (has to be smaller than 1)

      // non-monotone acceptance of the full step, see SolverData::n_nonmonotone_steps
      bool full_step_accepted = false;
      if(solver_data.n_nonmonotone_steps > 0)
      {
        temporary.copy_locally_owned_data_from(solution);
        temporary.add(1.0, increment);
        nonlinear_operator.evaluate_residual(residual, temporary);
        norm_r_damp = residual.l2_norm();

        if(norm_r_damp < (1.0 - 0.5 * tau) * norm_r)
        {
          full_step_accepted = true;
        }
        else if(norm_r_damp < solver_data.nonmonotone_factor * norm_r and
                n_nonmonotone_accepted < solver_data.n_nonmonotone_steps)
        {
          full_step_accepted = true;
          ++n_nonmonotone_accepted;
        }
        else
        {
          // continue with the backtracking line search starting from omega = 1/2
          omega       = 0.5;
          n_iter_damp = 1;
        }
      }

      if(full_step_accepted)
      {
        // in terms of the variables of the backtracking line search below
        omega = 0.5;
      }
      else
      {
        do
        {
          // add increment to solution vector but scale by a factor omega <= 1
          temporary.copy_locally_owned_data_from(solution);
          temporary.add(omega, increment);

          // evaluate residual using the temporary solution
          nonlinear_operator.evaluate_residual(residual, temporary);

          // calculate norm of residual (for temporary solution)
          norm_r_damp = residual.l2_norm();

          // reduce step length
          omega = omega / 2.0;

          // increment counter
          n_iter_damp++;
        } while(norm_r_damp >= (1.0 - tau * omega) * norm_r and n_iter_damp < max_iter_damp);
      }

      AssertThrow(full_step_accepted or norm_r_damp < (1.0 - tau * omega) * norm_r,
                  dealii::ExcMessage("Damped Newton iteration did not converge. "
                                     "Maximum number of iterations exceeded!"));

      // refine the accepted damping factor 2 omega (the loop above has halved omega once more)
      // by bisection in [2 omega, 4 omega) to find the largest step that further reduces the
      // residual, see SolverData::n_line_search_refinements
      double const omega_accepted = 2.0 * omega;
      if(solver_data.n_line_search_refinements > 0 and not full_step_accepted and
         omega_accepted < 1.0)
      {
        if(best_solution.size() != solution.size())
        {
          best_solution.reinit(solution, true /* omit_zeroing_entries */);
          best_residual.reinit(solution, true /* omit_zeroing_entries */);
        }
        best_solution.copy_locally_owned_data_from(temporary);
        best_residual = residual;

        double omega_low = omega_accepted, omega_high = 2.0 * omega_accepted;
        double norm_low = norm_r_damp;
        for(unsigned int k = 0; k < solver_data.n_line_search_refinements; ++k)
        {
          double const omega_mid = 0.5 * (omega_low + omega_high);

          temporary.copy_locally_owned_data_from(solution);
          temporary.add(omega_mid, increment);
          nonlinear_operator.evaluate_residual(residual, temporary);
          double const norm_mid = residual.l2_norm();

          if(norm_mid < norm_low)
          {
            omega_low = omega_mid;
            norm_low  = norm_mid;
            best_solution.copy_locally_owned_data_from(temporary);
            best_residual = residual;
          }
          else
          {
            omega_high = omega_mid;
          }
        }

        temporary.copy_locally_owned_data_from(best_solution);
        residual    = best_residual;
        norm_r_damp = norm_low;
      }

      // update solution and residual
      solution.copy_locally_owned_data_from(temporary);
      norm_r = norm_r_damp;

      // increment iteration counter
      ++newton_iterations;
      linear_iterations += n_iter_linear;
    }

    AssertThrow(norm_r <= this->solver_data.abs_tol or norm_r / norm_r_0 <= solver_data.rel_tol,
                dealii::ExcMessage(
                  "Newton solver failed to solve nonlinear problem to given tolerance. "
                  "Maximum number of iterations exceeded!"));

    if(update.do_update and update.update_once_converged)
    {
      // update linear operator (set linearization point)
      linear_operator.set_solution_linearization(solution);
      // update preconditioner
      linear_solver.update_preconditioner(true);
    }

    return std::tuple<unsigned int, unsigned int>(newton_iterations, linear_iterations);
  }

private:
  // auxiliary vectors of the line search refinement
  VectorType best_solution, best_residual;

  // number of full steps accepted without decrease of the residual in the current solve
  unsigned int n_nonmonotone_accepted = 0;

  SolverData          solver_data;
  NonlinearOperator & nonlinear_operator;
  LinearOperator &    linear_operator;
  LinearSolver &      linear_solver;
};

} // namespace Newton
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_H_ */
