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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_DATA_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_DATA_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace Newton
{
struct SolverData
{
  SolverData()
    : max_iter(100),
      abs_tol(1.e-12),
      rel_tol(1.e-12),
      n_line_search_refinements(0),
      n_nonmonotone_steps(0),
      nonmonotone_factor(1.e2),
      max_iter_damping(10)
  {
  }

  SolverData(unsigned int const max_iter_,
             double const       abs_tol_,
             double const       rel_tol_,
             unsigned int const n_line_search_refinements_ = 0)
    : max_iter(max_iter_),
      abs_tol(abs_tol_),
      rel_tol(rel_tol_),
      n_line_search_refinements(n_line_search_refinements_),
      n_nonmonotone_steps(0),
      nonmonotone_factor(1.e2),
      max_iter_damping(10)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Maximum number of iterations", max_iter);
    print_parameter(pcout, "Absolute solver tolerance", abs_tol);
    print_parameter(pcout, "Relative solver tolerance", rel_tol);
    if(n_line_search_refinements > 0)
      print_parameter(pcout, "Line search refinements", n_line_search_refinements);
    if(n_nonmonotone_steps > 0)
    {
      print_parameter(pcout, "Non-monotone full steps", n_nonmonotone_steps);
      print_parameter(pcout, "Non-monotone factor", nonmonotone_factor);
    }
    if(max_iter_damping != 10)
      print_parameter(pcout, "Maximum number of damping steps", max_iter_damping);
  }

  unsigned int max_iter;
  double       abs_tol;
  double       rel_tol;

  // Number of bisection steps that refine the damping factor of the backtracking line search:
  // if the full Newton step is rejected and a damped step omega is accepted, the largest step in
  // [omega, 2 omega) that further reduces the residual is searched by bisection. This is useful
  // for semismooth problems (e.g. contact), where the residual decreases along the Newton direction
  // up to a kink, at which the active set changes, and increases steeply beyond. Each refinement
  // costs one residual evaluation. The default 0 gives the plain backtracking line search.
  unsigned int n_line_search_refinements;

  // Non-monotone acceptance of full Newton steps (watchdog): a full step is accepted even if it
  // does not reduce the residual, as long as the residual grows by less than nonmonotone_factor
  // and at most n_nonmonotone_steps consecutive full steps have been accepted in this way;
  // otherwise, the backtracking line search is used. For semismooth problems (e.g. contact), the
  // full step yields the solution for the current active set, and the next Newton step corrects
  // the active set (primal-dual active set strategy). The default 0 disables this option.
  unsigned int n_nonmonotone_steps;
  double       nonmonotone_factor;

  // maximum number of halvings of the damping factor in the backtracking line search
  unsigned int max_iter_damping;
};

struct UpdateData
{
  UpdateData() : do_update(true), update_every_newton_iter(1), update_once_converged(false)
  {
  }

  bool         do_update;
  unsigned int update_every_newton_iter;
  bool         update_once_converged;
};

} // namespace Newton
} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_NEWTON_NEWTON_SOLVER_DATA_H_ */
