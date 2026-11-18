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
 *   Linearization get_linearization() const;   // data defining the current linearization
 *   void   set_linearization(Linearization const & data);
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
  unsigned int model_steps_tried           = 0; // ModelNewtonSteps: model steps evaluated
  unsigned int model_steps_accepted        = 0; // ... that reached the tolerance
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
    model_steps_tried += other.model_steps_tried;
    model_steps_accepted += other.model_steps_accepted;
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
      qn_relaxation(param.qn_initial_relaxation),
      active_method(param.method)
  {
  }

  // change the method between two solves (e.g. a method schedule in time); the state that is
  // specific to a method (damping, radius, secant history) is reset
  void
  set_method(Method const method)
  {
    if(method == active_method)
      return;
    active_method = method;
    lm_lambda     = param.lm_lambda_initial;
    tr_radius     = param.tr_radius_initial;
    qn_relaxation = param.qn_initial_relaxation;
    qn_history_G.clear();
    qn_history_M.clear();
    lbfgs_s.clear();
    lbfgs_y.clear();
  }

  Method
  get_method() const
  {
    return active_method;
  }

  /*
   * Stopping tolerance for ||m||_M: absolute, relative to ||g||_M, and relative to the mismatch of
   * the initial guess (MismatchReduction). The initial guess is the extrapolation of the controls
   * of the previous time steps, a predictor whose error estimates the local time error; the
   * coupling error is then a fraction of the time error.
   */
  double
  stopping_tolerance(VectorType const & g, double const m0_norm, VectorType const & m) const
  {
    if(param.stopping_criterion == StoppingCriterion::DisplacementResidual)
    {
      // tolerance for ||m||_M equivalent to the displacement criterion at the current iterate:
      // ||m|| <= tolerance  <=>  residual <= displacement tolerance
      double const m_norm   = norm_mismatch(m);
      double const residual = problem.displacement_residual(m);
      if(residual <= 0.0)
        return std::max(m_norm, param.mismatch_tolerance);
      return m_norm * problem.displacement_tolerance() / residual;
    }

    double tolerance = param.mismatch_tolerance;
    if(param.mismatch_tolerance_relative > 0.0)
      tolerance = std::max(tolerance, param.mismatch_tolerance_relative * norm_mismatch(g));
    if(param.mismatch_reduction > 0.0)
      tolerance = std::max(tolerance, param.mismatch_reduction * m0_norm);
    return tolerance;
  }

  /*
   * Minimise J starting from the initial guess g. On return, g is the computed control and the
   * state of the problem corresponds to g.
   */
  Statistics
  solve(VectorType & g)
  {
    if(active_method == Method::InterfaceQuasiNewton)
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
    double const m0_norm = norm_mismatch(m);
    if(param.surrogate_diagnostics)
      problem.surrogate_diagnostics(m, pcout);
    if(param.cheap_tangent_diagnostics)
      problem.cheap_tangent_diagnostics(m, pcout);

    double r0_norm = 0.0;

    // after a failed step: derivatives at full accuracy, no recycled pairs of previous steps
    bool safeguarded = false;

    // ModelNewtonSteps: one step of the recycled inverse per solve, if such steps have recently
    // reached the tolerance often enough (with a probe every ModelStepProbeInterval solves)
    bool model_steps_enabled = false;
    // (not in the first time steps, where the recycled inverse has hardly any pairs)
    if(param.model_newton_steps and recycle_history.size() >= 2)
    {
      ++model_solves_since_probe;
      if(model_success_rate >= param.model_step_min_success or
         model_solves_since_probe >= param.model_step_probe_interval)
      {
        model_steps_enabled      = true;
        model_solves_since_probe = 0;
      }
    }

    // L-BFGS bookkeeping
    VectorType r_old, s_old;
    bool       has_pending_pair = false;

    for(unsigned int k = 0; k < param.max_iterations; ++k)
    {
      double const m_norm = norm_mismatch(m);

      double const tolerance = stopping_tolerance(g, m0_norm, m);

      /*
       * Model step without tangent solves (ModelNewtonSteps): d = -H m with the recycled inverse
       * H, exact on the tangent pairs of previous steps and on the secant pairs of this solve.
       * For the affine fluid step (explicit geometry) the difference m(g + d) - m(g) is S d up to
       * the nonlinearity of the structure, i.e. an exact pair for H. Tangents are only computed
       * if the model step does not reach the mismatch tolerance. The gradient methods take it
       * before their gradient (adjoint solve) and continue with their own iteration otherwise.
       */
      if(model_steps_enabled and not safeguarded and m_norm > tolerance)
      {
        rebuild_recycled_preconditioner();
        if(has_recycled_preconditioner())
        {
          apply_recycled(d, m);
          d *= -1.0;
          trial_g = g;
          trial_g.add(1.0, d);
          double const J_trial    = evaluate(trial_m, trial_g);
          double const trial_norm = norm_mismatch(trial_m);
          ++model_steps_tried;
          ++stats.model_steps_tried;
          if(param.recycled_preconditioner and std::isfinite(trial_norm))
          {
            VectorType y(trial_m);
            y.add(-1.0, m);
            recycle_now.emplace_back(d, y);
          }
          if(std::isfinite(trial_norm) and trial_norm < m_norm)
          {
            g = trial_g;
            m = trial_m;
            J = J_trial;
            save_linearization();
            ++stats.iterations;
          }
          else
            restore_linearization();
          // success: the model step alone reached the tolerance; otherwise the solve continues
          // with tangents (from the better point, if the model step reduced ||m||)
          bool const success = std::isfinite(trial_norm) and trial_norm <= tolerance;
          if(success)
          {
            ++model_steps_accepted;
            ++stats.model_steps_accepted;
          }
          model_success_rate = (1.0 - param.model_step_rate_weight) * model_success_rate +
                               param.model_step_rate_weight * (success ? 1.0 : 0.0);
          model_steps_enabled = false;
          if(std::isfinite(trial_norm) and trial_norm < m_norm)
            continue;
        }
      }

      if(safeguarded)
        problem.set_linearized_relative_tolerance(0.0);
      else if(param.linearized_tolerance_target > 0.0 and m_norm > 0.0)
        problem.set_linearized_relative_tolerance(param.linearized_tolerance_target * tolerance /
                                                  m_norm);

      double     r_norm = 0.0;
      bool const needs_gradient =
        active_method != Method::InterfaceNewton and not uses_interface_equation();
      // no gradient (adjoint solve) at the converged point
      if(needs_gradient and m_norm > tolerance)
      {
        gradient(r, m, g);
        r_norm = norm(r);
        if(r0_norm == 0.0) // first gradient of the solve (possibly after a model step)
          r0_norm = r_norm;
      }

      print_iteration(k, J, m_norm, r_norm);

      stats.cost          = J;
      stats.mismatch_norm = m_norm;
      stats.gradient_norm = r_norm;

      if(m_norm <= tolerance)
      {
        stats.converged = true;
        stats.reason    = "mismatch tolerance";
        break;
      }
      if(needs_gradient and r_norm <= param.gradient_tolerance_relative * r0_norm and k > 0)
      {
        stats.converged = true;
        stats.reason    = "gradient tolerance";
        break;
      }
      if(needs_gradient and r_norm == 0.0)
      {
        stats.converged = true;
        stats.reason    = "zero gradient";
        break;
      }

      double eta =
        std::min(param.forcing_term_max, std::sqrt(r0_norm > 0.0 ? r_norm / r0_norm : 1.0));
      // interface equation: the linear model has to reach the mismatch tolerance, not more
      if(param.forcing_term_target > 0.0 and not needs_gradient and m_norm > 0.0)
      {
        double const eta_target = std::max(1.e-12, param.forcing_term_target * tolerance / m_norm);
        // close to the tolerance, the linear model only has to reach the target
        eta = param.forcing_term_target_max > 0.0 ?
                std::min(param.forcing_term_target_max, eta_target) :
                std::min(eta, eta_target);
      }
      if(param.verbose)
        pcout << "      tolerance = " << std::scientific << std::setprecision(3) << tolerance
              << ", ||m|| / tolerance = " << m_norm / tolerance << std::defaultfloat << std::endl;

      bool step_accepted = false;

      if(active_method == Method::SteepestDescent)
      {
        double slope = 0.0;
        if(preconditioned_gradient_direction(d, r, slope))
        {
          // preconditioned steepest descent: d = -H H_A r ~ -(S^* S)^{-1} r, step length from the
          // exact quadratic model (one application of S, which also enriches H)
          problem.begin_tangent_record();
          double const dHd = hessian_quadratic_form(d);
          problem.set_state_prediction(g, d, std::vector<double>(1, 1.0));
          double const alpha0 = dHd > 0.0 ? -slope / dHd : 1.0;
          step_accepted       = armijo(g, m, J, d, slope, alpha0, trial_g, trial_m, false);
        }
        else
        {
          d.equ(-1.0, r);
          double const alpha0 = cauchy_step_length(r);
          step_accepted = armijo(g, m, J, d, -r_norm * r_norm, alpha0, trial_g, trial_m, false);
        }
      }
      else if(active_method == Method::LBFGS)
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

        // safeguards against directions that are not of descent (the recycled initial inverse
        // Hessian is not symmetric): drop the pairs of the current solve, then the recycled
        // pairs of previous time steps, finally use -r
        lbfgs_direction(d, r);
        double slope = problem.inner_product(r, d);
        if(not is_descent(slope, r, d))
        {
          lbfgs_s.clear();
          lbfgs_y.clear();
          lbfgs_direction(d, r);
          slope = problem.inner_product(r, d);
        }
        if(not is_descent(slope, r, d) and has_recycled_preconditioner())
        {
          discard_recycled_history();
          lbfgs_direction(d, r);
          slope = problem.inner_product(r, d);
        }
        if(not is_descent(slope, r, d))
        {
          d.equ(-1.0, r);
          slope = -r_norm * r_norm;
        }

        // with the recycled initial inverse Hessian: step length from the exact quadratic model
        // (one application of S, which also enriches H and predicts the fluid state)
        double alpha0 = 1.0;
        if(has_recycled_preconditioner())
        {
          problem.begin_tangent_record();
          double const dHd = hessian_quadratic_form(d);
          problem.set_state_prediction(g, d, std::vector<double>(1, 1.0));
          if(dHd > 0.0)
            alpha0 = -slope / dHd;
        }

        VectorType g_old(g);
        r_old         = r;
        step_accepted = armijo(g, m, J, d, slope, alpha0, trial_g, trial_m, false);
        if(step_accepted)
        {
          s_old = g;
          s_old.add(-1.0, g_old);
          has_pending_pair = true;
        }
      }
      else if(uses_interface_equation() and active_method != Method::GaussNewton)
      {
        // Levenberg-Marquardt, trust region in the Krylov subspace of S d = -m
        VectorType b(m), Sd;
        b *= -1.0;
        problem.begin_tangent_record();
        gmres(d, Sd, b, eta);
        step_accepted = subspace_step(g, m, J, d, trial_g, trial_m);
      }
      else if(uses_interface_equation())
      {
        // square, invertible S: the Gauss-Newton step solves S d = -m
        VectorType b(m), Sd;
        b *= -1.0;
        problem.begin_tangent_record();
        gmres(d, Sd, b, eta);
        problem.set_state_prediction(g, d, last_gmres_coefficients);
        double const slope = problem.inner_product_mismatch(m, Sd);
        if(slope >= 0.0)
        {
          stats.reason = "Gauss-Newton direction is not a descent direction";
          break;
        }
        step_accepted = armijo(g, m, J, d, slope, 1.0, trial_g, trial_m, false);
      }
      else if(active_method == Method::GaussNewton)
      {
        VectorType b(r);
        b *= -1.0;
        cg(d, b, eta, 0.0);
        step_accepted =
          armijo(g, m, J, d, problem.inner_product(r, d), 1.0, trial_g, trial_m, false);
      }
      else if(active_method == Method::LevenbergMarquardt)
      {
        step_accepted = levenberg_marquardt_step(g, m, J, r, d, eta, trial_g, trial_m);
      }
      else if(active_method == Method::TrustRegion)
      {
        step_accepted = trust_region_step(g, m, J, r, d, eta, trial_g, trial_m);
      }
      else if(active_method == Method::InterfaceNewton)
      {
        VectorType b(m), Sd;
        b *= -1.0;
        problem.begin_tangent_record();
        gmres(d, Sd, b, eta);
        problem.set_state_prediction(g, d, last_gmres_coefficients);
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
        if(not safeguarded and
           (param.linearized_tolerance_target > 0.0 or param.recycled_preconditioner))
        {
          // retry once with accurate derivatives and without the recycled pairs of previous
          // steps; the state of the current iterate is re-evaluated, so that the linearization
          // (also of the fluid) is the one of g
          if(param.verbose)
            pcout << "    step failed: retry with accurate derivatives" << std::endl;
          safeguarded = true;
          if(param.recycled_preconditioner)
            discard_recycled_history();
          J = evaluate(m, g);
          save_linearization();
          continue;
        }
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
      pcout << "  OC " << method_to_string(active_method) << ": " << stats.reason << ", "
            << stats.iterations << " it, " << stats.state_evaluations << " states, "
            << stats.tangent_solves << " S, " << stats.adjoint_solves << " S*, "
            << stats.krylov_iterations << " Krylov, " << stats.preconditioner_applications
            << " P, ||m|| = " << std::scientific << std::setprecision(3) << stats.mismatch_norm
            << std::defaultfloat << std::endl;

    archive_recycled_pairs();

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
    double const m0_norm = qn_last_initial_norm;

    std::vector<VectorType> dG_now, dM_now;

    // stagnation safeguard: best iterate so far
    VectorType   g_best(g);
    double       best_norm = norm_mismatch(m), reference_norm = best_norm;
    unsigned int last_improvement = 0;

    for(unsigned int k = 0; k < param.max_iterations; ++k)
    {
      double const m_norm = norm_mismatch(m);
      print_iteration(k, cost(m, g), m_norm, 0.0);
      stats.mismatch_norm    = m_norm;
      double const tolerance = stopping_tolerance(g, m0_norm, m);
      if(m_norm <= tolerance)
      {
        stats.converged = true;
        stats.reason    = "mismatch tolerance";
        break;
      }

      if(m_norm < best_norm)
      {
        best_norm = m_norm;
        g_best    = g;
      }
      if(best_norm < 0.9 * reference_norm)
      {
        reference_norm   = best_norm;
        last_improvement = k;
      }
      if(param.stagnation_iterations > 0 and k >= last_improvement + param.stagnation_iterations)
      {
        // noise floor of the state map reached: return the best iterate (and its state)
        if(best_norm < m_norm)
        {
          g = g_best;
          evaluate(m, g);
          save_linearization();
          stats.mismatch_norm = norm_mismatch(m);
        }
        stats.reason = "stagnation";
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
        // QR of dM (MGS in the M inner product), dropping dependent columns. With the Riesz map M
        // of the metric, (Q_i, x) = (M Q_i) * x and M q follows from linearity: one application of
        // M per column instead of one per inner product.
        bool const                       fast = problem.has_metric_map();
        std::vector<VectorType>          Q, MQ;
        std::vector<unsigned int>        kept;
        std::vector<std::vector<double>> R; // R[j] = column j of R (size j+1)
        for(unsigned int j = 0; j < cols_M.size(); ++j)
        {
          VectorType q(*cols_M[j]), Mq;
          if(fast)
            problem.apply_metric(Mq, q);
          double const original =
            fast ? std::sqrt(std::max(0.0, double(Mq * q))) : norm_mismatch(q);
          std::vector<double> r(Q.size() + 1, 0.0);
          for(unsigned int i = 0; i < Q.size(); ++i)
          {
            r[i] = fast ? MQ[i] * q : problem.inner_product_mismatch(Q[i], q);
            q.add(-r[i], Q[i]);
            if(fast)
              Mq.add(-r[i], MQ[i]);
          }
          double const q_norm = fast ? std::sqrt(std::max(0.0, double(Mq * q))) : norm_mismatch(q);
          if(original == 0.0 or q_norm <= param.qn_filter * original)
            continue;
          r.back() = q_norm;
          q *= 1.0 / q_norm;
          Q.push_back(q);
          if(fast)
          {
            Mq *= 1.0 / q_norm;
            MQ.push_back(Mq);
          }
          R.push_back(r);
          kept.push_back(j);
        }

        unsigned int const  n = Q.size();
        std::vector<double> rhs(n), c(n, 0.0);
        for(unsigned int i = 0; i < n; ++i)
          rhs[i] = fast ? MQ[i] * m : problem.inner_product_mismatch(Q[i], m);
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

    // an unconverged solve (stagnation or iteration limit) may carry pairs of a map that the secant
    // model did not capture; they and the older pairs are discarded, so that the next solve starts
    // from the initial inverse Jacobian instead of reusing them for QNReuseTimeSteps steps
    if(not stats.converged)
    {
      dM_now.clear();
      dG_now.clear();
      problem.reset_initial_inverse_jacobian();
      if(not qn_history_M.empty())
      {
        qn_history_G.clear();
        qn_history_M.clear();
        ++stats.restarts;
      }
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
      pcout << "  OC " << method_to_string(active_method) << ": " << stats.reason << ", "
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
    // every tangent application gives an exact pair for the recycled preconditioner
    if(param.recycled_preconditioner)
      recycle_now.emplace_back(src, dst);
  }

  /*
   * Recycled matrix-free preconditioner from exact tangent pairs (v_i, y_i = S v_i): with the
   * M-orthonormal factorization Y = Q R (newest pairs first, nearly dependent columns dropped),
   * H x = x + sum_j W_j (Q_j, x)_M with W = (V - Y) R^{-1}, so that H y_i = v_i for the kept
   * pairs; the M-adjoint is H^* x = x + sum_j Q_j (W_j, x)_M.
   */
  void
  rebuild_recycled_preconditioner()
  {
    dealii::Timer timer;
    rebuild_recycled_inverse(recycle_now, recycle_history, rec_Q, rec_MQ, rec_W, false);
    rebuild_recycled_inverse(adjoint_now, adjoint_history, adj_Q, adj_MQ, adj_W, true);
    time_recycled_rebuild += timer.wall_time();
  }

  /*
   * Inverse from exact pairs (v_i, y_i = A v_i): H y_i = v_i on the kept pairs, identity on the
   * complement; the images y_i are orthonormalized in the mismatch inner product (A = S) or in
   * the control inner product (A = S^*).
   */
  void
  rebuild_recycled_inverse(
    std::vector<std::pair<VectorType, VectorType>> const &             now,
    std::deque<std::vector<std::pair<VectorType, VectorType>>> const & history,
    std::vector<VectorType> &                                          rec_Q,
    std::vector<VectorType> &                                          rec_MQ,
    std::vector<VectorType> &                                          rec_W,
    bool const                                                         control_metric) const
  {
    rec_Q.clear();
    rec_MQ.clear();
    rec_W.clear();
    if(not param.recycled_preconditioner)
      return;

    // with the Riesz map M of the metric, (Q_i, x) = (M Q_i) * x: one application of M per kept
    // pair instead of one per inner product
    bool const fast  = problem.has_metric_map();
    auto const inner = [&](VectorType const & a, VectorType const & b) {
      return control_metric ? problem.inner_product(a, b) : problem.inner_product_mismatch(a, b);
    };
    auto const vnorm = [&](VectorType const & a) { return std::sqrt(std::max(0.0, inner(a, a))); };

    std::vector<std::pair<VectorType, VectorType> const *> pairs;
    for(auto it = now.rbegin(); it != now.rend(); ++it)
      pairs.push_back(&(*it));
    for(auto h = history.rbegin(); h != history.rend(); ++h)
      for(auto it = h->rbegin(); it != h->rend(); ++it)
        pairs.push_back(&(*it));

    std::vector<std::vector<double>> R; // R[j] = column j (size j+1)
    std::vector<VectorType>          D; // V - Y of the kept pairs
    for(auto const * p : pairs)
    {
      if(rec_Q.size() >= param.recycle_max_pairs)
        break;
      VectorType q(p->second), Mq;
      if(fast)
        problem.apply_metric(Mq, q);
      double const        original = fast ? std::sqrt(std::max(0.0, double(Mq * q))) : vnorm(q);
      std::vector<double> r(rec_Q.size() + 1, 0.0);
      for(unsigned int i = 0; i < rec_Q.size(); ++i)
      {
        r[i] = fast ? rec_MQ[i] * q : inner(rec_Q[i], q);
        q.add(-r[i], rec_Q[i]);
        if(fast)
          Mq.add(-r[i], rec_MQ[i]);
      }
      double const q_norm = fast ? std::sqrt(std::max(0.0, double(Mq * q))) : vnorm(q);
      if(original == 0.0 or q_norm <= param.qn_filter * original)
        continue;
      r.back() = q_norm;
      q *= 1.0 / q_norm;
      rec_Q.push_back(q);
      if(fast)
      {
        Mq *= 1.0 / q_norm;
        rec_MQ.push_back(Mq);
      }
      R.push_back(r);
      VectorType d(p->first);
      d.add(-1.0, p->second);
      D.push_back(d);
    }

    // W R = D, column by column
    for(unsigned int j = 0; j < D.size(); ++j)
    {
      VectorType w(D[j]);
      for(unsigned int i = 0; i < j; ++i)
        w.add(-R[j][i], rec_W[i]);
      w *= 1.0 / R[j][j];
      rec_W.push_back(w);
    }
  }

  bool
  has_recycled_preconditioner() const
  {
    return param.recycled_preconditioner and not rec_Q.empty();
  }

  // dst = H src
  void
  apply_recycled(VectorType & dst, VectorType const & src) const
  {
    dealii::Timer timer;
    dst = src;
    for(unsigned int j = 0; j < rec_Q.size(); ++j)
      dst.add(rec_MQ.empty() ? problem.inner_product_mismatch(rec_Q[j], src) : rec_MQ[j] * src,
              rec_W[j]);
    time_recycled_apply += timer.wall_time();
  }

  // dst = H_A src ~ S^{-*} src (from the exact adjoint pairs, control -> mismatch space)
  void
  apply_recycled_adjoint_inverse(VectorType & dst, VectorType const & src) const
  {
    dst = src;
    for(unsigned int j = 0; j < adj_Q.size(); ++j)
      dst.add(adj_MQ.empty() ? problem.inner_product(adj_Q[j], src) : adj_MQ[j] * src, adj_W[j]);
  }

  // dst = H H_A src ~ S^{-1} S^{-*} src = (S^* S)^{-1} src
  void
  apply_recycled_normal(VectorType & dst, VectorType const & src) const
  {
    VectorType tmp;
    problem.initialize_vector(tmp);
    apply_recycled_adjoint_inverse(tmp, src);
    apply_recycled(dst, tmp);
  }

  // archive the pairs of the finished solve
  void
  archive_recycled_pairs()
  {
    if(not param.recycled_preconditioner)
      return;
    if(param.recycle_time_steps > 0 and not recycle_now.empty())
    {
      recycle_history.push_back(std::move(recycle_now));
      while(recycle_history.size() > param.recycle_time_steps)
        recycle_history.pop_front();
    }
    recycle_now.clear();
    if(param.recycle_time_steps > 0 and not adjoint_now.empty())
    {
      adjoint_history.push_back(std::move(adjoint_now));
      while(adjoint_history.size() > param.recycle_time_steps)
        adjoint_history.pop_front();
    }
    adjoint_now.clear();
  }

  void
  apply_S_adjoint(VectorType & dst, VectorType const & src)
  {
    problem.apply_S_adjoint(dst, src);
    ++stats.adjoint_solves;
    // every adjoint application gives an exact pair (w, S^* w) for H_A ~ S^{-*}
    if(param.recycled_preconditioner)
      adjoint_now.emplace_back(src, dst);
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
    // preconditioned with the limited-memory quasi-Newton (Morales-Nocedal) preconditioner built
    // from exact pairs (p, A0 p) of previous CG solves (A0 = S^* S, current shift added), or with
    // (J^* J)^{-1} of the frozen Jacobian
    double const sigma        = param.regularization + shift;
    bool const   lmp          = param.recycled_preconditioner and not cg_pairs.empty();
    bool const   precondition = lmp or problem.has_normal_preconditioner();
    auto const   apply_P      = [&](VectorType & dst, VectorType const & src) {
      if(lmp)
        apply_lmp(dst, src, sigma);
      else
        problem.apply_normal_preconditioner(dst, src);
    };
    std::vector<std::pair<VectorType, VectorType>> new_pairs;

    VectorType res(b), z, p, Hp;
    problem.initialize_vector(Hp);
    problem.initialize_vector(z);
    x = 0.0;

    double const b_norm = norm(b);
    if(b_norm == 0.0)
      return;

    if(precondition)
    {
      apply_P(z, res);
      ++stats.preconditioner_applications;
    }
    else
      z = res;
    p         = z;
    double rz = problem.inner_product(res, z);

    for(unsigned int j = 0; j < param.krylov_max_iterations; ++j)
    {
      if(norm(res) <= eta * b_norm)
        break;

      hessian_vector(Hp, p, shift);
      ++stats.krylov_iterations;

      double const pHp = problem.inner_product(p, Hp);
      if(pHp <= 0.0)
        break;

      if(param.recycled_preconditioner)
      {
        VectorType A0p(Hp);
        A0p.add(-sigma, p);
        new_pairs.emplace_back(p, A0p);
      }

      double const alpha = rz / pHp;
      x.add(alpha, p);
      res.add(-alpha, Hp);

      if(precondition)
      {
        apply_P(z, res);
        ++stats.preconditioner_applications;
      }
      else
        z = res;
      double const rz_new = problem.inner_product(res, z);
      p.sadd(rz_new / rz, 1.0, z);
      rz = rz_new;
    }

    // keep the newest exact pairs for the next solves (the preconditioner is fixed within a solve)
    for(auto & pair : new_pairs)
    {
      cg_pairs.push_back(std::move(pair));
      if(cg_pairs.size() > param.recycle_max_pairs)
        cg_pairs.pop_front();
    }
  }

  /*
   * Limited-memory quasi-Newton preconditioner (Morales, Nocedal 2000) for A = A0 + sigma I from
   * exact pairs (s_i, A0 s_i): L-BFGS two-loop recursion with y_i = A0 s_i + sigma s_i and the
   * initial scaling (s, y)/(y, y) of the newest pair. Symmetric positive definite.
   */
  void
  apply_lmp(VectorType & dst, VectorType const & src, double const sigma) const
  {
    unsigned int const      n = cg_pairs.size();
    std::vector<VectorType> y(n);
    std::vector<double>     rho(n), a(n);
    for(unsigned int i = 0; i < n; ++i)
    {
      y[i] = cg_pairs[i].second;
      y[i].add(sigma, cg_pairs[i].first);
      rho[i] = 1.0 / problem.inner_product(cg_pairs[i].first, y[i]);
    }
    dst = src;
    for(int i = int(n) - 1; i >= 0; --i)
    {
      a[i] = rho[i] * problem.inner_product(cg_pairs[i].first, dst);
      dst.add(-a[i], y[i]);
    }
    dst *= problem.inner_product(cg_pairs[n - 1].first, y[n - 1]) /
           problem.inner_product(y[n - 1], y[n - 1]);
    for(unsigned int i = 0; i < n; ++i)
    {
      double const beta = rho[i] * problem.inner_product(y[i], dst);
      dst.add(a[i] - beta, cg_pairs[i].first);
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
    last_gmres_coefficients.clear();

    double const beta = norm_mismatch(b);
    if(beta == 0.0)
      return;

    rebuild_recycled_preconditioner();

    std::vector<VectorType>          V(1, b), Z;
    std::vector<std::vector<double>> H, R; // unrotated / rotated Hessenberg columns
    std::vector<double>              cs, sn, e(1, beta);
    std::vector<double>              residual_history;
    V[0] *= 1.0 / beta;

    unsigned int k = 0;
    for(; k < max_it; ++k)
    {
      VectorType z, w;
      problem.initialize_vector(z);
      problem.initialize_vector(w);
      if(has_recycled_preconditioner())
      {
        apply_recycled(z, V[k]);
        ++stats.preconditioner_applications;
      }
      else if(problem.has_preconditioner())
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
      residual_history.push_back(std::abs(e[k + 1]) / beta);

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

    if(param.verbose)
      pcout << "      GMRES: " << k << " it, eta = " << std::scientific << std::setprecision(2)
            << eta << ", ||b + S x|| / ||b|| =";
    if(param.verbose)
    {
      for(double const r : residual_history)
        pcout << " " << r;
      pcout << std::defaultfloat << std::endl;
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
    last_gmres_coefficients = y;
    krylov_beta             = beta;
    krylov_H                = H;
    krylov_Z                = Z;

    for(unsigned int j = 0; j < k; ++j)
      for(unsigned int i = 0; i < H[j].size() and i < V.size(); ++i)
        Sx.add(H[j][i] * y[j], V[i]);
  }

  void
  lbfgs_direction(VectorType & d, VectorType const & r)
  {
    unsigned int const n = lbfgs_s.size();
    rebuild_recycled_preconditioner();
    bool const recycled = has_recycled_preconditioner();
    if(n == 0)
    {
      if(recycled)
      {
        apply_recycled_normal(d, r);
        d *= -1.0;
      }
      else
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
    if(recycled)
    {
      // initial inverse Hessian H H^* from the recycled tangent pairs
      VectorType tmp(q);
      apply_recycled_normal(q, tmp);
    }
    else
    {
      double const gamma_scaling = problem.inner_product(lbfgs_s[n - 1], lbfgs_y[n - 1]) /
                                   problem.inner_product(lbfgs_y[n - 1], lbfgs_y[n - 1]);
      q *= gamma_scaling;
    }
    for(unsigned int i = 0; i < n; ++i)
    {
      double const beta = rho[i] * problem.inner_product(lbfgs_y[i], q);
      q.add(a[i] - beta, lbfgs_s[i]);
    }
    d.equ(-1.0, q);
  }

  // angle condition (r, d) <= -c ||r|| ||d||
  bool
  is_descent(double const slope, VectorType const & r, VectorType const & d) const
  {
    return slope < -1.e-4 * norm(r) * norm(d);
  }

  /*
   * Preconditioned gradient direction d = -H H_A r. The recycled inverse is not symmetric, and
   * pairs of previous time steps may be outdated (e.g. after changes of the time step): if d is
   * not a descent direction, the pairs of previous time steps are discarded and d is recomputed
   * with the pairs of the current solve. Returns false if no recycled preconditioner is available
   * or no descent direction results.
   */
  bool
  preconditioned_gradient_direction(VectorType & d, VectorType const & r, double & slope)
  {
    for(unsigned int attempt = 0; attempt < 2; ++attempt)
    {
      rebuild_recycled_preconditioner();
      if(not has_recycled_preconditioner())
        return false;
      apply_recycled_normal(d, r);
      d *= -1.0;
      slope = problem.inner_product(r, d);
      if(is_descent(slope, r, d))
        return true;
      discard_recycled_history();
    }
    return false;
  }

  void
  discard_recycled_history()
  {
    if(param.verbose)
      pcout << "    recycled preconditioner: no descent direction, pairs of previous time "
               "steps discarded"
            << std::endl;
    recycle_history.clear();
    adjoint_history.clear();
    ++recycled_restarts;
  }

  bool
  uses_interface_equation() const
  {
    return param.interface_equation and param.regularization == 0.0 and
           (active_method == Method::GaussNewton or active_method == Method::LevenbergMarquardt or
            active_method == Method::TrustRegion);
  }

  /*
   * Model of the last GMRES solve in its search directions Z (d = Z y): with the Arnoldi relation
   * S Z = V Hbar and b = -m = beta v_1, ||m + S Z y||^2 = ||Hbar y - beta e_1||^2, i.e. the
   * Gauss-Newton model 0.5 ||m + S d||^2 = 0.5 beta^2 - y.c + 0.5 y.A y with A = Hbar^T Hbar,
   * c = beta Hbar^T e_1, and the metric of the steps ||d||^2 = y.G y, G_ij = (z_i, z_j).
   */
  void
  subspace_model(std::vector<std::vector<double>> & A,
                 std::vector<double> &              c,
                 std::vector<std::vector<double>> & G) const
  {
    unsigned int const k = krylov_Z.size();
    A.assign(k, std::vector<double>(k, 0.0));
    G.assign(k, std::vector<double>(k, 0.0));
    c.assign(k, 0.0);
    auto Hbar = [&](unsigned int const i, unsigned int const j) {
      return i < krylov_H[j].size() ? krylov_H[j][i] : 0.0;
    };
    for(unsigned int i = 0; i < k; ++i)
    {
      c[i] = krylov_beta * Hbar(0, i);
      for(unsigned int j = 0; j < k; ++j)
      {
        for(unsigned int l = 0; l <= k; ++l)
          A[i][j] += Hbar(l, i) * Hbar(l, j);
        if(j >= i)
          G[i][j] = G[j][i] = problem.inner_product(krylov_Z[i], krylov_Z[j]);
      }
    }
  }

  // y = (A + lambda G)^{-1} c by Cholesky (small, symmetric positive definite)
  static std::vector<double>
  solve_small(std::vector<std::vector<double>> const & A,
              std::vector<std::vector<double>> const & G,
              std::vector<double> const &              c,
              double const                             lambda)
  {
    unsigned int const               k = c.size();
    std::vector<std::vector<double>> L(k, std::vector<double>(k, 0.0));
    for(unsigned int i = 0; i < k; ++i)
      for(unsigned int j = 0; j <= i; ++j)
      {
        double sum = A[i][j] + lambda * G[i][j];
        for(unsigned int l = 0; l < j; ++l)
          sum -= L[i][l] * L[j][l];
        if(i == j)
          L[i][i] = std::sqrt(std::max(sum, 1.e-300));
        else
          L[i][j] = sum / L[j][j];
      }
    std::vector<double> y(c);
    for(unsigned int i = 0; i < k; ++i)
    {
      for(unsigned int l = 0; l < i; ++l)
        y[i] -= L[i][l] * y[l];
      y[i] /= L[i][i];
    }
    for(int i = int(k) - 1; i >= 0; --i)
    {
      for(unsigned int l = i + 1; l < k; ++l)
        y[i] -= L[l][i] * y[l];
      y[i] /= L[i][i];
    }
    return y;
  }

  static double
  quadratic_form(std::vector<std::vector<double>> const & A, std::vector<double> const & y)
  {
    double sum = 0.0;
    for(unsigned int i = 0; i < y.size(); ++i)
      for(unsigned int j = 0; j < y.size(); ++j)
        sum += y[i] * A[i][j] * y[j];
    return sum;
  }

  /*
   * Levenberg-Marquardt and trust-region steps in the Krylov subspace of the interface equation:
   * no further applications of S or S^* (also for rejected trial steps), the Gauss-Newton model
   * of the subspace is exact up to the GMRES accuracy.
   *   LM: y = (A + lambda G)^{-1} c,
   *   TR: y = argmin of the model subject to y.G y <= radius^2 (bisection for the multiplier).
   */
  bool
  subspace_step(VectorType & g,
                VectorType & m,
                double &     J,
                VectorType & d,
                VectorType & trial_g,
                VectorType & trial_m)
  {
    if(krylov_Z.empty())
      return false;

    std::vector<std::vector<double>> A, G;
    std::vector<double>              c;
    subspace_model(A, c, G);

    bool const                lm      = (active_method == Method::LevenbergMarquardt);
    std::vector<double> const y_gn    = solve_small(A, G, c, 0.0);
    double const              gn_norm = std::sqrt(std::max(0.0, quadratic_form(G, y_gn)));

    unsigned int const max_tries = lm ? param.lm_max_tries : 30;
    for(unsigned int t = 0; t < max_tries; ++t)
    {
      std::vector<double> y;
      bool                on_boundary = false;
      if(lm)
        y = solve_small(A, G, c, lm_lambda);
      else if(gn_norm <= tr_radius)
        y = y_gn;
      else
      {
        // ||d(lambda)|| decreases monotonically in lambda: bisection in log(lambda)
        double lo = 1.e-16, hi = 1.0;
        while(std::sqrt(quadratic_form(G, solve_small(A, G, c, hi))) > tr_radius and hi < 1.e16)
          hi *= 10.0;
        for(unsigned int it = 0; it < 60; ++it)
        {
          double const mid = std::sqrt(lo * hi);
          if(std::sqrt(quadratic_form(G, solve_small(A, G, c, mid))) > tr_radius)
            lo = mid;
          else
            hi = mid;
        }
        y           = solve_small(A, G, c, hi);
        on_boundary = true;
      }

      double pred = -0.5 * quadratic_form(A, y);
      for(unsigned int i = 0; i < y.size(); ++i)
        pred += y[i] * c[i];
      if(not(pred > 0.0))
        return false;

      d = 0.0;
      for(unsigned int j = 0; j < y.size(); ++j)
        d.add(y[j], krylov_Z[j]);
      problem.set_state_prediction(g, d, y);

      trial_g = g;
      trial_g.add(1.0, d);
      double const J_trial = evaluate(trial_m, trial_g);
      double const rho     = (J - J_trial) / pred;
      double const d_norm  = std::sqrt(std::max(0.0, quadratic_form(G, y)));

      if(std::isfinite(J_trial) and rho > param.tr_eta)
      {
        g = trial_g;
        m = trial_m;
        J = J_trial;
        save_linearization();
        if(lm)
        {
          if(rho > 0.75)
            lm_lambda = std::max(lm_lambda / 3.0, param.lm_lambda_min);
          else if(rho < 0.25)
            lm_lambda = std::min(2.0 * lm_lambda, param.lm_lambda_max);
        }
        else
        {
          if(rho < 0.25)
            tr_radius = 0.25 * d_norm;
          else if(rho > 0.75 and on_boundary)
            tr_radius = std::min(2.0 * tr_radius, param.tr_radius_max);
        }
        return true;
      }

      restore_linearization();
      if(lm)
        lm_lambda = std::min(4.0 * lm_lambda, param.lm_lambda_max);
      else
        tr_radius = 0.25 * (on_boundary ? tr_radius : d_norm);
    }
    return false;
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
    if(active_method != Method::InterfaceNewton and active_method != Method::InterfaceQuasiNewton)
      pcout << "  ||r|| = " << r_norm;
    if(active_method == Method::LevenbergMarquardt)
      pcout << "  lambda = " << lm_lambda;
    if(active_method == Method::TrustRegion)
      pcout << "  radius = " << tr_radius;
    if(k > 0 and (active_method == Method::SteepestDescent or active_method == Method::LBFGS or
                  active_method == Method::GaussNewton or active_method == Method::InterfaceNewton))
      pcout << "  alpha = " << last_alpha;
    pcout << std::defaultfloat << std::endl;
  }

  Problem &                          problem;
  Parameters const &                 param;
  dealii::ConditionalOStream const & pcout;

  Statistics stats;

  // coefficients of the last GMRES solution x = sum_j c_j z_j in the search directions z_j, and
  // its Arnoldi data (unrotated Hessenberg columns, search directions, ||b||)
  std::vector<double>              last_gmres_coefficients;
  std::vector<std::vector<double>> krylov_H;
  std::vector<VectorType>          krylov_Z;
  double                           krylov_beta = 0.0;
  typename Problem::Linearization  linearization;
  bool                             state_is_current = false;
  double                           last_alpha       = 0.0;

  // persistent across time steps: adaptive LM parameter and trust-region radius
  double lm_lambda;
  double tr_radius;

  std::deque<VectorType> lbfgs_s, lbfgs_y;

  // InterfaceQuasiNewton: secant pairs of previous solves
  std::deque<std::vector<VectorType>> qn_history_G, qn_history_M;
  double                              qn_relaxation;
  Method                              active_method;

  // recycled preconditioner: exact tangent pairs of the current solve and of previous solves,
  // and the factors of H
  std::vector<std::pair<VectorType, VectorType>>             recycle_now;
  std::deque<std::vector<std::pair<VectorType, VectorType>>> recycle_history;
  std::vector<VectorType>                                    rec_Q, rec_MQ, rec_W;
  std::vector<std::pair<VectorType, VectorType>>             adjoint_now; // (w, S^* w)
  std::deque<std::vector<std::pair<VectorType, VectorType>>> adjoint_history;
  std::vector<VectorType>                                    adj_Q, adj_MQ, adj_W;

public:
  unsigned int recycled_restarts        = 0;   // history discarded (no descent direction)
  unsigned int model_steps_tried        = 0;   // ModelNewtonSteps: steps d = -H m evaluated
  unsigned int model_steps_accepted     = 0;   // ... that reached the tolerance
  double       model_success_rate       = 1.0; // moving average of the success of model steps
  unsigned int model_solves_since_probe = 0;
  // wall time of the recycled inverse: rebuild (orthogonalization) and application
  double         time_recycled_rebuild = 0.0;
  mutable double time_recycled_apply   = 0.0;

private:
  std::deque<std::pair<VectorType, VectorType>> cg_pairs; // exact pairs (p, S^* S p)
  double                                        qn_last_initial_norm = 0.0;
};

} // namespace OptimalControl
} // namespace FSI
} // namespace ExaDG

#endif /* EXADG_FLUID_STRUCTURE_INTERACTION_OPTIMAL_CONTROL_OPTIMIZERS_H_ */
