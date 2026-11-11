/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
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

#ifndef EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_OPTIMIZERS_H_
#define EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_OPTIMIZERS_H_

// C/C++
#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/fluid_structure_interaction/optimal_control/parameters.h>

namespace ExaDG
{
namespace FSI
{
namespace OptimalControl
{
/*
 * Generic reduced-space optimisation methods for interface control problems.
 *
 * The class Problem has to provide (control and mismatch live in the same space V; the mismatch is
 * measured in the inner product (.,.)_A, the control in (.,.)_G, see section 6.2 of the report):
 *
 *   typedef ... VectorType;
 *   void   initialize_vector(VectorType & v) const;                 // zero vector in V
 *   double inner_product(VectorType const & a, VectorType const & b) const;          // (a,b)_G
 *   double inner_product_mismatch(VectorType const & a, VectorType const & b) const; // (a,b)_A
 *   void   fill_random(VectorType & v) const;                       // random admissible vector
 *   void   evaluate(VectorType & m, VectorType const & g);          // state solve -> mismatch
 *   void   apply_S(VectorType & dst, VectorType const & src) const;          // dm/dg * src
 *   void   apply_S_adjoint(VectorType & dst, VectorType const & src) const;  // (A,G)-adjoint of S
 *   bool   has_preconditioner() const;
 *   void   apply_preconditioner(VectorType & dst, VectorType const & src) const; // P^{-1} ~ S^{-1}
 *   VectorType const & get_linearization() const;   // data defining the current linearization
 *   void   set_linearization(VectorType const & data);
 *
 * evaluate() changes the state of the problem and sets the point at which apply_S() and
 * apply_S_adjoint() are linearized. After a rejected trial step, the optimizer restores the
 * linearization of the current iterate via set_linearization(). When solve() returns, the state
 * of the problem corresponds to the returned control.
 */
struct Statistics
{
  unsigned int iterations                  = 0;
  unsigned int state_evaluations           = 0;
  unsigned int tangent_solves              = 0; // applications of S
  unsigned int adjoint_solves              = 0; // applications of S^*
  unsigned int krylov_iterations           = 0;
  unsigned int preconditioner_applications = 0;
  unsigned int restarts                    = 0;
  bool         converged                   = false;
  std::string  reason                      = "max iterations";
  double       cost                        = 0.0;
  double       mismatch_norm               = 0.0;
  double       gradient_norm               = 0.0;

  void
  add(Statistics const & other)
  {
    iterations += other.iterations;
    state_evaluations += other.state_evaluations;
    tangent_solves += other.tangent_solves;
    adjoint_solves += other.adjoint_solves;
    krylov_iterations += other.krylov_iterations;
    preconditioner_applications += other.preconditioner_applications;
  }
};

inline std::string
method_to_string(Method const method)
{
  switch(method)
  {
    case Method::SteepestDescent:
      return "SteepestDescent";
    case Method::LBFGS:
      return "LBFGS";
    case Method::GaussNewton:
      return "GaussNewton";
    case Method::LevenbergMarquardt:
      return "LevenbergMarquardt";
    case Method::TrustRegion:
      return "TrustRegion";
    case Method::InterfaceNewton:
      return "InterfaceNewton";
    case Method::InterfaceQuasiNewton:
      return "InterfaceQuasiNewton";
  }
  return "Unknown";
}

template<typename Problem>
class Optimizer
{
public:
  typedef typename Problem::VectorType VectorType;

  Optimizer(Problem & problem, Parameters const & param, dealii::ConditionalOStream const & pcout)
    : problem(problem),
      param(param),
      pcout(pcout),
      lm_lambda(param.lm_lambda_initial),
      tr_radius(param.tr_radius_initial),
      qn_relaxation(param.qn_initial_relaxation)
  {
  }

  /*
   * Minimise J starting from the initial guess g. On return, g is the computed control and the
   * state of the problem corresponds to g.
   */
  Statistics
  solve(VectorType & g)
  {
    if(param.method == Method::InterfaceQuasiNewton)
      return solve_quasi_newton(g);

    stats = Statistics();
    lbfgs_s.clear();
    lbfgs_y.clear();

    VectorType m, r, d, trial_g, trial_m;
    problem.initialize_vector(m);
    problem.initialize_vector(r);
    problem.initialize_vector(d);
    problem.initialize_vector(trial_g);
    problem.initialize_vector(trial_m);

    double J = evaluate(m, g);
    save_linearization();

    double r0_norm = 0.0;

    // L-BFGS bookkeeping
    VectorType r_old, s_old;
    bool       has_pending_pair = false;

    for(unsigned int k = 0; k < param.max_iterations; ++k)
    {
      double const m_norm = norm_mismatch(m);

      double r_norm = 0.0;
      if(param.method != Method::InterfaceNewton)
      {
        gradient(r, m, g);
        r_norm = norm(r);
        if(k == 0)
          r0_norm = r_norm;
      }

      print_iteration(k, J, m_norm, r_norm);

      stats.cost          = J;
      stats.mismatch_norm = m_norm;
      stats.gradient_norm = r_norm;

      if(m_norm <= param.mismatch_tolerance)
      {
        stats.converged = true;
        stats.reason    = "mismatch tolerance";
        break;
      }
      if(param.method != Method::InterfaceNewton and
         r_norm <= param.gradient_tolerance_relative * r0_norm and k > 0)
      {
        stats.converged = true;
        stats.reason    = "gradient tolerance";
        break;
      }
      if(param.method != Method::InterfaceNewton and r_norm == 0.0)
      {
        stats.converged = true;
        stats.reason    = "zero gradient";
        break;
      }

      double const eta =
        std::min(param.forcing_term_max, std::sqrt(r0_norm > 0.0 ? r_norm / r0_norm : 1.0));

      bool step_accepted = false;

      if(param.method == Method::SteepestDescent)
      {
        d.equ(-1.0, r);
        double const alpha0 = cauchy_step_length(r);
        step_accepted       = armijo(g, m, J, d, -r_norm * r_norm, alpha0, trial_g, trial_m, false);
      }
      else if(param.method == Method::LBFGS)
      {
        if(has_pending_pair)
        {
          VectorType y(r);
          y.add(-1.0, r_old);
          double const sy = problem.inner_product(s_old, y);
          if(sy > 1.e-10 * norm(s_old) * norm(y))
          {
            lbfgs_s.push_back(s_old);
            lbfgs_y.push_back(y);
            if(lbfgs_s.size() > param.lbfgs_memory)
            {
              lbfgs_s.pop_front();
              lbfgs_y.pop_front();
            }
          }
          has_pending_pair = false;
        }

        lbfgs_direction(d, r);
        double slope = problem.inner_product(r, d);
        if(slope >= 0.0)
        {
          lbfgs_s.clear();
          lbfgs_y.clear();
          lbfgs_direction(d, r);
          slope = problem.inner_product(r, d);
        }

        VectorType g_old(g);
        r_old         = r;
        step_accepted = armijo(g, m, J, d, slope, 1.0, trial_g, trial_m, false);
        if(step_accepted)
        {
          s_old = g;
          s_old.add(-1.0, g_old);
          has_pending_pair = true;
        }
      }
      else if(param.method == Method::GaussNewton)
      {
        VectorType b(r);
        b *= -1.0;
        cg(d, b, eta, 0.0);
        step_accepted =
          armijo(g, m, J, d, problem.inner_product(r, d), 1.0, trial_g, trial_m, false);
      }
      else if(param.method == Method::LevenbergMarquardt)
      {
        step_accepted = levenberg_marquardt_step(g, m, J, r, d, eta, trial_g, trial_m);
      }
      else if(param.method == Method::TrustRegion)
      {
        step_accepted = trust_region_step(g, m, J, r, d, eta, trial_g, trial_m);
      }
      else if(param.method == Method::InterfaceNewton)
      {
        VectorType b(m), Sd;
        b *= -1.0;
        gmres(d, Sd, b, eta);
        double const slope = problem.inner_product_mismatch(m, Sd);
        if(slope >= 0.0)
        {
          stats.reason = "Newton direction is not a descent direction";
          break;
        }
        step_accepted = armijo(g, m, J, d, slope, 1.0, trial_g, trial_m, true);
      }
      else
      {
        AssertThrow(false, dealii::ExcNotImplemented());
      }

      ++stats.iterations;

      if(not step_accepted)
      {
        stats.reason = "line search / step acceptance failed";
        break;
      }
    }

    // make sure that the state of the problem corresponds to g
    if(not state_is_current)
    {
      J = evaluate(m, g);
      save_linearization();
    }

    if(param.verbose)
      pcout << "  OC " << method_to_string(param.method) << ": " << stats.reason << ", "
            << stats.iterations << " it, " << stats.state_evaluations << " states, "
            << stats.tangent_solves << " S, " << stats.adjoint_solves << " S*, "
            << stats.krylov_iterations << " Krylov, " << stats.preconditioner_applications
            << " P, ||m|| = " << std::scientific << std::setprecision(3) << stats.mismatch_norm
            << std::defaultfloat << std::endl;

    return stats;
  }

  // adaptive accuracy of the state solves: tolerances scaled by safety * ||m|| / tolerance
  void
  set_inner_tolerance(double const m_norm)
  {
    double factor = 1.0;
    if(param.adaptive_inner_tolerance and m_norm > 0.0 and param.mismatch_tolerance > 0.0)
      factor =
        std::min(1.e8,
                 std::max(1.0, param.inner_tolerance_safety * m_norm / param.mismatch_tolerance));
    problem.set_state_tolerance_factor(factor);
  }

  /*
   * Least-squares secant method (Anderson type II / IQN-ILS) for m(g) = 0:
   *   g_{k+1} = g_k - omega m_k - (dG - omega dM) c,  c = argmin ||m_k - dM c||_M,
   * with the differences dG, dM of the current solve (newest first) and of the last
   * QNReuseTimeSteps solves. The least-squares problem is solved by a QR decomposition (modified
   * Gram-Schmidt in the M inner product); nearly dependent columns are dropped.
   */
  Statistics
  solve_quasi_newton(VectorType & g)
  {
    stats = Statistics();

    VectorType m, g_old, m_old;
    problem.initialize_vector(m);
    set_inner_tolerance(qn_last_initial_norm);
    evaluate(m, g);
    save_linearization();
    qn_last_initial_norm = norm_mismatch(m);

    std::vector<VectorType> dG_now, dM_now;

    for(unsigned int k = 0; k < param.max_iterations; ++k)
    {
      double const m_norm = norm_mismatch(m);
      print_iteration(k, cost(m, g), m_norm, 0.0);
      stats.mismatch_norm = m_norm;
      if(m_norm <= param.mismatch_tolerance)
      {
        stats.converged = true;
        stats.reason    = "mismatch tolerance";
        break;
      }

      // columns: current solve (newest first), then previous solves
      std::vector<VectorType const *> cols_G, cols_M;
      for(int i = int(dM_now.size()) - 1; i >= 0; --i)
      {
        cols_G.push_back(&dG_now[i]);
        cols_M.push_back(&dM_now[i]);
      }
      for(auto it = qn_history_M.rbegin(); it != qn_history_M.rend(); ++it)
      {
        auto const   h  = std::distance(qn_history_M.rbegin(), it);
        auto const & HG = *(qn_history_G.rbegin() + h);
        for(int i = int(it->size()) - 1; i >= 0; --i)
        {
          cols_G.push_back(&HG[i]);
          cols_M.push_back(&(*it)[i]);
        }
      }

      VectorType step, H0x;
      problem.initialize_vector(step);
      problem.initialize_vector(H0x);
      // H_0 = omega I (relaxation) or the inverse Jacobian of the problem
      double const omega    = cols_M.empty() ? qn_relaxation : param.qn_mixing;
      auto const   apply_H0 = [&](VectorType & dst, VectorType const & src) {
        if(param.qn_initial_jacobian == QNInitialJacobian::Relaxation or
           not problem.has_initial_inverse_jacobian(param.qn_initial_jacobian))
          dst.equ(omega, src);
        else
          problem.apply_initial_inverse_jacobian(dst, src, param.qn_initial_jacobian);
      };
      if(cols_M.empty())
      {
        apply_H0(H0x, m);
        step.equ(-1.0, H0x);
      }
      else
      {
        // QR of dM (MGS in the M inner product), dropping dependent columns
        std::vector<VectorType>          Q;
        std::vector<unsigned int>        kept;
        std::vector<std::vector<double>> R; // R[j] = column j of R (size j+1)
        for(unsigned int j = 0; j < cols_M.size(); ++j)
        {
          VectorType          q(*cols_M[j]);
          double const        original = norm_mismatch(q);
          std::vector<double> r(Q.size() + 1, 0.0);
          for(unsigned int i = 0; i < Q.size(); ++i)
          {
            r[i] = problem.inner_product_mismatch(Q[i], q);
            q.add(-r[i], Q[i]);
          }
          double const q_norm = norm_mismatch(q);
          if(original == 0.0 or q_norm <= param.qn_filter * original)
            continue;
          r.back() = q_norm;
          q *= 1.0 / q_norm;
          Q.push_back(q);
          R.push_back(r);
          kept.push_back(j);
        }

        unsigned int const  n = Q.size();
        std::vector<double> rhs(n), c(n, 0.0);
        for(unsigned int i = 0; i < n; ++i)
          rhs[i] = problem.inner_product_mismatch(Q[i], m);
        for(int i = int(n) - 1; i >= 0; --i)
        {
          double sum = rhs[i];
          for(unsigned int j = i + 1; j < n; ++j)
            sum -= R[j][i] * c[j];
          c[i] = sum / R[i][i];
        }

        apply_H0(H0x, m);
        step.equ(-1.0, H0x);
        for(unsigned int i = 0; i < n; ++i)
        {
          step.add(-c[i], *cols_G[kept[i]]);
          apply_H0(H0x, *cols_M[kept[i]]);
          step.add(c[i], H0x);
        }
        stats.krylov_iterations += n; // number of secant pairs used
      }

      g_old = g;
      m_old = m;
      g.add(1.0, step);
      set_inner_tolerance(m_norm);
      evaluate(m, g);
      save_linearization();
      ++stats.iterations;

      // safeguards if the residual grows strongly: pairs of previous solves may be outdated and are
      // discarded (the pairs of the current solve, including the new one, are kept since they
      // carry the information on the current Jacobian); a pure relaxation step that increased the
      // residual reduces the relaxation parameter for the following solves.
      if(norm_mismatch(m) > param.qn_restart_factor * m_norm)
      {
        if(cols_M.empty())
          qn_relaxation *= 0.5;
        problem.reset_initial_inverse_jacobian();
        if(not qn_history_M.empty())
        {
          qn_history_G.clear();
          qn_history_M.clear();
          ++stats.restarts;
        }
      }

      VectorType dg(g), dm(m);
      dg.add(-1.0, g_old);
      dm.add(-1.0, m_old);
      dG_now.push_back(dg);
      dM_now.push_back(dm);
    }

    // IQN-IMVJ: multi-secant update of the inverse Jacobian with the (filtered) pairs of this solve
    if(param.qn_initial_jacobian == QNInitialJacobian::PreviousTimeStep and not dM_now.empty())
    {
      std::vector<VectorType> kept_G, kept_M, Q;
      for(int j = int(dM_now.size()) - 1; j >= 0; --j)
      {
        VectorType   q(dM_now[j]);
        double const original = norm_mismatch(q);
        for(auto const & qi : Q)
          q.add(-problem.inner_product_mismatch(qi, q), qi);
        double const q_norm = norm_mismatch(q);
        if(original == 0.0 or q_norm <= param.qn_filter * original)
          continue;
        q *= 1.0 / q_norm;
        Q.push_back(q);
        kept_G.push_back(dG_now[j]);
        kept_M.push_back(dM_now[j]);
      }
      problem.update_initial_inverse_jacobian(kept_G, kept_M, qn_relaxation);
    }

    problem.set_state_tolerance_factor(1.0);

    // archive the pairs of this solve
    if(param.qn_reuse > 0 and not dM_now.empty())
    {
      qn_history_G.push_back(dG_now);
      qn_history_M.push_back(dM_now);
      while(qn_history_M.size() > param.qn_reuse)
      {
        qn_history_G.pop_front();
        qn_history_M.pop_front();
      }
    }

    if(param.verbose)
      pcout << "  OC " << method_to_string(param.method) << ": " << stats.reason << ", "
            << stats.iterations << " it, " << stats.state_evaluations << " states, "
            << stats.krylov_iterations << " secant columns, " << stats.restarts
            << " restarts, ||m|| = " << std::scientific << std::setprecision(3)
            << stats.mismatch_norm << std::defaultfloat << std::endl;

    return stats;
  }

  /*
   * Verification of derivatives at g (tests T1/T2 of the report):
   *  - adjoint test: (S v, w)_M = (v, S^* w)_M,
   *  - tangent test: S v against finite differences of the state map,
   *  - Taylor test: |J(g + eps v) - J(g) - eps (r, v)_M| = O(eps^2).
   * On return, the state of the problem corresponds to g.
   */
  void
  verify_derivatives(VectorType const & g)
  {
    VectorType m0, m_eps, v, w, Sv, Sw, r, g_eps, diff;
    problem.initialize_vector(m0);
    problem.initialize_vector(m_eps);
    problem.initialize_vector(Sv);
    problem.initialize_vector(Sw);
    problem.initialize_vector(r);
    problem.initialize_vector(diff);
    problem.initialize_vector(v);
    problem.initialize_vector(w);

    double const J0 = evaluate(m0, g);
    save_linearization();

    problem.fill_random(v);
    problem.fill_random(w);
    v *= 1.0 / norm(v);
    w *= 1.0 / norm_mismatch(w);

    apply_S(Sv, v);
    apply_S_adjoint(Sw, w);
    double const a = problem.inner_product_mismatch(Sv, w);
    double const b = problem.inner_product(v, Sw);

    pcout << std::endl
          << "Verification of derivatives (optimal-control FSI coupling):" << std::endl
          << std::scientific << std::setprecision(6) << "  adjoint test: (Sv,w) = " << a
          << ", (v,S*w) = " << b
          << ", relative difference = " << std::abs(a - b) / std::max(std::abs(a), 1.e-300)
          << std::endl;

    gradient(r, m0, g);
    double const r_dot_v = problem.inner_product(r, v);
    double const scale   = std::max(norm(g), 1.0);

    pcout << "  tangent test (finite differences) and Taylor test (ratio -> 4):" << std::endl
          << "  " << std::setw(14) << "eps" << std::setw(16) << "|FD - Sv|/|Sv|" << std::setw(16)
          << "Taylor rem." << std::setw(10) << "ratio" << std::endl;

    double remainder_old = 0.0;
    for(unsigned int j = 0; j < 6; ++j)
    {
      double const eps = 1.e-1 * scale * std::pow(0.5, j);
      g_eps            = g;
      g_eps.add(eps, v);
      double const J_eps = evaluate(m_eps, g_eps);
      restore_linearization();

      diff = m_eps;
      diff.add(-1.0, m0);
      diff *= 1.0 / eps;
      diff.add(-1.0, Sv);
      double const tangent_error = norm_mismatch(diff) / std::max(norm_mismatch(Sv), 1.e-300);

      double const remainder = std::abs(J_eps - J0 - eps * r_dot_v);
      pcout << "  " << std::setw(14) << eps << std::setw(16) << tangent_error << std::setw(16)
            << remainder;
      if(j > 0)
        pcout << std::setw(10) << std::fixed << std::setprecision(3)
              << remainder_old / std::max(remainder, 1.e-300) << std::scientific
              << std::setprecision(6);
      pcout << std::endl;
      remainder_old = remainder;
    }
    pcout << std::defaultfloat << std::endl;

    evaluate(m0, g);
    save_linearization();
  }

private:
  double
  norm(VectorType const & v) const
  {
    return std::sqrt(std::max(0.0, problem.inner_product(v, v)));
  }

  double
  norm_mismatch(VectorType const & v) const
  {
    return std::sqrt(std::max(0.0, problem.inner_product_mismatch(v, v)));
  }

  double
  cost(VectorType const & m, VectorType const & g) const
  {
    double J = 0.5 * problem.inner_product_mismatch(m, m);
    if(param.regularization > 0.0)
      J += 0.5 * param.regularization * problem.inner_product(g, g);
    return J;
  }

  double
  evaluate(VectorType & m, VectorType const & g)
  {
    problem.evaluate(m, g);
    ++stats.state_evaluations;
    state_is_current = true;
    return cost(m, g);
  }

  void
  save_linearization()
  {
    linearization = problem.get_linearization();
  }

  void
  restore_linearization()
  {
    problem.set_linearization(linearization);
    state_is_current = false;
  }

  void
  apply_S(VectorType & dst, VectorType const & src)
  {
    problem.apply_S(dst, src);
    ++stats.tangent_solves;
  }

  void
  apply_S_adjoint(VectorType & dst, VectorType const & src)
  {
    problem.apply_S_adjoint(dst, src);
    ++stats.adjoint_solves;
  }

  // Riesz gradient r = S^* m + gamma g
  void
  gradient(VectorType & r, VectorType const & m, VectorType const & g)
  {
    apply_S_adjoint(r, m);
    if(param.regularization > 0.0)
      r.add(param.regularization, g);
  }

  // Gauss-Newton Hessian H v = S^* S v + (gamma + shift) v
  void
  hessian_vector(VectorType & dst, VectorType const & src, double const shift)
  {
    VectorType tmp;
    problem.initialize_vector(tmp);
    apply_S(tmp, src);
    apply_S_adjoint(dst, tmp);
    if(param.regularization + shift != 0.0)
      dst.add(param.regularization + shift, src);
  }

  // (d, H d) = ||S d||^2 + gamma ||d||^2 with one tangent solve
  double
  hessian_quadratic_form(VectorType const & d)
  {
    VectorType Sd;
    problem.initialize_vector(Sd);
    apply_S(Sd, d);
    return problem.inner_product_mismatch(Sd, Sd) +
           param.regularization * problem.inner_product(d, d);
  }

  // minimiser of the quadratic model along -r (exact line search for the GN model)
  double
  cauchy_step_length(VectorType const & r)
  {
    double const rr  = problem.inner_product(r, r);
    double const rHr = hessian_quadratic_form(r);
    return rHr > 0.0 ? rr / rHr : 1.0;
  }

  /*
   * Armijo backtracking. If merit_is_mismatch, the merit function is 1/2||m||^2 (interface Newton),
   * otherwise J. On success, g, m, J are updated and the state is current.
   */
  bool
  armijo(VectorType &       g,
         VectorType &       m,
         double &           J,
         VectorType const & d,
         double const       slope,
         double const       alpha0,
         VectorType &       trial_g,
         VectorType &       trial_m,
         bool const         merit_is_mismatch)
  {
    if(not(slope < 0.0))
      return false;

    double const merit = merit_is_mismatch ? 0.5 * problem.inner_product_mismatch(m, m) : J;

    double alpha = alpha0;
    for(unsigned int l = 0; l < param.line_search_max_iterations; ++l)
    {
      trial_g = g;
      trial_g.add(alpha, d);
      double const J_trial = evaluate(trial_m, trial_g);
      double const merit_trial =
        merit_is_mismatch ? 0.5 * problem.inner_product_mismatch(trial_m, trial_m) : J_trial;

      if(std::isfinite(merit_trial) and merit_trial <= merit + param.armijo_c1 * alpha * slope)
      {
        g = trial_g;
        m = trial_m;
        J = J_trial;
        save_linearization();
        last_alpha = alpha;
        return true;
      }

      restore_linearization();
      alpha *= 0.5;
    }

    return false;
  }

  /*
   * CG in the M inner product for (H + shift I) x = b, started from x = 0. Stops at relative
   * residual eta or after krylov_max_iterations.
   */
  void
  cg(VectorType & x, VectorType const & b, double const eta, double const shift)
  {
    VectorType res(b), p(b), Hp;
    problem.initialize_vector(Hp);
    x = 0.0;

    double const b_norm = norm(b);
    double       rr     = problem.inner_product(res, res);
    if(b_norm == 0.0)
      return;

    for(unsigned int j = 0; j < param.krylov_max_iterations; ++j)
    {
      if(std::sqrt(rr) <= eta * b_norm)
        break;

      hessian_vector(Hp, p, shift);
      ++stats.krylov_iterations;

      double const pHp = problem.inner_product(p, Hp);
      if(pHp <= 0.0)
        break;

      double const alpha = rr / pHp;
      x.add(alpha, p);
      res.add(-alpha, Hp);

      double const rr_new = problem.inner_product(res, res);
      p.sadd(rr_new / rr, 1.0, res);
      rr = rr_new;
    }
  }

  /*
   * Flexible GMRES, right preconditioned with P^{-1} if the problem provides one, for S x = b in
   * the mismatch inner product (S is not self-adjoint for FSI), started from x = 0. Also returns
   * S x, assembled from the Arnoldi relation S Z_k = V_{k+1} Hbar without an extra application of
   * S.
   */
  void
  gmres(VectorType & x, VectorType & Sx, VectorType const & b, double const eta)
  {
    unsigned int const max_it = param.krylov_max_iterations;

    problem.initialize_vector(x);
    problem.initialize_vector(Sx);

    double const beta = norm_mismatch(b);
    if(beta == 0.0)
      return;

    std::vector<VectorType>          V(1, b), Z;
    std::vector<std::vector<double>> H, R; // unrotated / rotated Hessenberg columns
    std::vector<double>              cs, sn, e(1, beta);
    V[0] *= 1.0 / beta;

    unsigned int k = 0;
    for(; k < max_it; ++k)
    {
      VectorType z, w;
      problem.initialize_vector(z);
      problem.initialize_vector(w);
      if(problem.has_preconditioner())
      {
        problem.apply_preconditioner(z, V[k]);
        ++stats.preconditioner_applications;
      }
      else
      {
        z = V[k];
      }
      Z.push_back(z);

      apply_S(w, z);
      ++stats.krylov_iterations;

      std::vector<double> h(k + 2, 0.0);
      for(unsigned int i = 0; i <= k; ++i)
      {
        h[i] = problem.inner_product_mismatch(w, V[i]);
        w.add(-h[i], V[i]);
      }
      h[k + 1] = norm_mismatch(w);
      H.push_back(h);

      for(unsigned int i = 0; i < k; ++i)
      {
        double const tmp = cs[i] * h[i] + sn[i] * h[i + 1];
        h[i + 1]         = -sn[i] * h[i] + cs[i] * h[i + 1];
        h[i]             = tmp;
      }
      double const denom = std::sqrt(h[k] * h[k] + h[k + 1] * h[k + 1]);
      cs.push_back(denom > 0.0 ? h[k] / denom : 1.0);
      sn.push_back(denom > 0.0 ? h[k + 1] / denom : 0.0);
      h[k] = denom;
      e.push_back(-sn[k] * e[k]);
      e[k] = cs[k] * e[k];
      R.push_back(h);

      bool const breakdown = (H.back()[k + 1] <= 1.e-14 * beta);
      if(not breakdown)
      {
        V.push_back(w);
        V.back() *= 1.0 / H.back()[k + 1];
      }

      if(std::abs(e[k + 1]) <= eta * beta or breakdown)
      {
        ++k;
        break;
      }
    }

    std::vector<double> y(k, 0.0);
    for(int i = int(k) - 1; i >= 0; --i)
    {
      double sum = e[i];
      for(unsigned int j = i + 1; j < k; ++j)
        sum -= R[j][i] * y[j];
      y[i] = sum / R[i][i];
    }

    for(unsigned int j = 0; j < k; ++j)
      x.add(y[j], Z[j]);

    for(unsigned int j = 0; j < k; ++j)
      for(unsigned int i = 0; i < H[j].size() and i < V.size(); ++i)
        Sx.add(H[j][i] * y[j], V[i]);
  }

  void
  lbfgs_direction(VectorType & d, VectorType const & r)
  {
    unsigned int const n = lbfgs_s.size();
    if(n == 0)
    {
      d.equ(-cauchy_step_length(r), r);
      return;
    }

    VectorType          q(r);
    std::vector<double> a(n), rho(n);
    for(int i = int(n) - 1; i >= 0; --i)
    {
      rho[i] = 1.0 / problem.inner_product(lbfgs_s[i], lbfgs_y[i]);
      a[i]   = rho[i] * problem.inner_product(lbfgs_s[i], q);
      q.add(-a[i], lbfgs_y[i]);
    }
    double const gamma_scaling = problem.inner_product(lbfgs_s[n - 1], lbfgs_y[n - 1]) /
                                 problem.inner_product(lbfgs_y[n - 1], lbfgs_y[n - 1]);
    q *= gamma_scaling;
    for(unsigned int i = 0; i < n; ++i)
    {
      double const beta = rho[i] * problem.inner_product(lbfgs_y[i], q);
      q.add(a[i] - beta, lbfgs_s[i]);
    }
    d.equ(-1.0, q);
  }

  bool
  levenberg_marquardt_step(VectorType &       g,
                           VectorType &       m,
                           double &           J,
                           VectorType const & r,
                           VectorType &       d,
                           double const       eta,
                           VectorType &       trial_g,
                           VectorType &       trial_m)
  {
    VectorType b(r);
    b *= -1.0;

    for(unsigned int t = 0; t < param.lm_max_tries; ++t)
    {
      cg(d, b, eta, lm_lambda);

      double const pred = -problem.inner_product(r, d) - 0.5 * hessian_quadratic_form(d);

      trial_g = g;
      trial_g.add(1.0, d);
      double const J_trial = evaluate(trial_m, trial_g);
      double const rho     = pred > 0.0 ? (J - J_trial) / pred : -1.0;

      if(std::isfinite(J_trial) and rho > param.tr_eta)
      {
        g = trial_g;
        m = trial_m;
        J = J_trial;
        save_linearization();
        if(rho > 0.75)
          lm_lambda = std::max(lm_lambda / 3.0, param.lm_lambda_min);
        else if(rho < 0.25)
          lm_lambda = std::min(2.0 * lm_lambda, param.lm_lambda_max);
        return true;
      }

      restore_linearization();
      lm_lambda = std::min(4.0 * lm_lambda, param.lm_lambda_max);
    }
    return false;
  }

  /*
   * Gauss-Newton trust-region step with the Steihaug-Toint CG. The complete CG path is stored,
   * so that a rejected step is recomputed for the reduced radius without new solves.
   */
  bool
  trust_region_step(VectorType &       g,
                    VectorType &       m,
                    double &           J,
                    VectorType const & r,
                    VectorType &       d,
                    double const       eta,
                    VectorType &       trial_g,
                    VectorType &       trial_m)
  {
    // CG path for H x = b, b = -r: iterates x_j, directions p_j, H p_j, residuals res_j = b - H x_j
    VectorType b(r);
    b *= -1.0;

    std::vector<VectorType> X, P, HP, RES;
    {
      VectorType x, res(b), p(b), Hp;
      problem.initialize_vector(x);
      problem.initialize_vector(Hp);
      double const b_norm = norm(b);
      double       rr     = problem.inner_product(res, res);

      X.push_back(x);
      RES.push_back(res);

      for(unsigned int j = 0; j < param.krylov_max_iterations; ++j)
      {
        if(std::sqrt(rr) <= eta * b_norm)
          break;
        hessian_vector(Hp, p, 0.0);
        ++stats.krylov_iterations;
        double const pHp = problem.inner_product(p, Hp);
        if(pHp <= 0.0)
          break;
        double const alpha = rr / pHp;
        P.push_back(p);
        HP.push_back(Hp);
        x.add(alpha, p);
        res.add(-alpha, Hp);
        X.push_back(x);
        RES.push_back(res);
        double const rr_new = problem.inner_product(res, res);
        p.sadd(rr_new / rr, 1.0, res);
        rr = rr_new;
        // stop the path once it leaves the largest admissible region
        if(norm(x) >= param.tr_radius_max)
          break;
      }
    }

    VectorType Hd;
    for(unsigned int t = 0; t < 30; ++t)
    {
      // Steihaug step for the current radius from the stored path
      bool on_boundary = false;
      d                = X.back();
      Hd               = b;
      Hd.add(-1.0, RES.back()); // H x = b - res
      for(unsigned int j = 0; j + 1 < X.size(); ++j)
      {
        if(norm(X[j + 1]) >= tr_radius)
        {
          // tau >= 0 with ||x_j + tau p_j|| = radius
          double const pp  = problem.inner_product(P[j], P[j]);
          double const xp  = problem.inner_product(X[j], P[j]);
          double const xx  = problem.inner_product(X[j], X[j]);
          double const tau = (-xp + std::sqrt(xp * xp + pp * (tr_radius * tr_radius - xx))) / pp;
          d                = X[j];
          d.add(tau, P[j]);
          Hd = b;
          Hd.add(-1.0, RES[j]);
          Hd.add(tau, HP[j]);
          on_boundary = true;
          break;
        }
      }

      double const pred = -problem.inner_product(r, d) - 0.5 * problem.inner_product(d, Hd);
      if(not(pred > 0.0))
        return false;

      trial_g = g;
      trial_g.add(1.0, d);
      double const J_trial = evaluate(trial_m, trial_g);
      double const rho     = (J - J_trial) / pred;
      double const d_norm  = norm(d);

      if(std::isfinite(J_trial) and rho > param.tr_eta)
      {
        g = trial_g;
        m = trial_m;
        J = J_trial;
        save_linearization();
        if(rho < 0.25)
          tr_radius = 0.25 * d_norm;
        else if(rho > 0.75 and on_boundary)
          tr_radius = std::min(2.0 * tr_radius, param.tr_radius_max);
        return true;
      }

      restore_linearization();
      tr_radius = 0.25 * (on_boundary ? tr_radius : d_norm);
    }
    return false;
  }

  void
  print_iteration(unsigned int const k, double const J, double const m_norm, double const r_norm)
  {
    if(not param.verbose)
      return;
    pcout << "    it " << std::setw(3) << k << std::scientific << std::setprecision(4)
          << "  J = " << J << "  ||m|| = " << m_norm;
    if(param.method != Method::InterfaceNewton and param.method != Method::InterfaceQuasiNewton)
      pcout << "  ||r|| = " << r_norm;
    if(param.method == Method::LevenbergMarquardt)
      pcout << "  lambda = " << lm_lambda;
    if(param.method == Method::TrustRegion)
      pcout << "  radius = " << tr_radius;
    if(k > 0 and (param.method == Method::SteepestDescent or param.method == Method::LBFGS or
                  param.method == Method::GaussNewton or param.method == Method::InterfaceNewton))
      pcout << "  alpha = " << last_alpha;
    pcout << std::defaultfloat << std::endl;
  }

  Problem &                          problem;
  Parameters const &                 param;
  dealii::ConditionalOStream const & pcout;

  Statistics stats;
  VectorType linearization;
  bool       state_is_current = false;
  double     last_alpha       = 0.0;

  // persistent across time steps: adaptive LM parameter and trust-region radius
  double lm_lambda;
  double tr_radius;

  std::deque<VectorType> lbfgs_s, lbfgs_y;

  // InterfaceQuasiNewton: secant pairs of previous solves
  std::deque<std::vector<VectorType>> qn_history_G, qn_history_M;
  double                              qn_relaxation;
  double                              qn_last_initial_norm = 0.0;
};

} // namespace OptimalControl
} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_OPTIMIZERS_H_ */
