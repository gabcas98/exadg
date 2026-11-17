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

#ifndef APPLICATIONS_FSI_CYLINDER_WITH_FLAG_H_
#define APPLICATIONS_FSI_CYLINDER_WITH_FLAG_H_

#include <exadg/grid/one_sided_cylindrical_manifold.h>

namespace ExaDG
{
// Test cases FSI1, FSI2, FSI3 of Turek and Hron (2006), selected at run time by the parameter
// Application/TestCase (default: FSI2). The dependent constants (U_X_MAX, END_TIME, output
// interval) are set by set_test_case(); EndTime should be given explicitly for FSI1 and FSI3.
inline unsigned int TEST_CASE                = 2;
inline double       U_MEAN                   = 1.0;
inline double       FLUID_VISCOSITY          = 1.0e-3;
inline double       FLUID_DENSITY            = 1.0e3;
inline double       DENSITY_STRUCTURE        = 1.0e4;
inline double       POISSONS_RATIO_STRUCTURE = 0.4;
inline double       YOUNGS_MODULUS_STRUCTURE = 0.5e6 * 2.0 * (1.0 + 0.4);

// physical dimensions (diameter D and center coordinate Y_C can be varied)
double const X_0    = 0.0;  // origin (x-coordinate)
double const Y_0    = 0.0;  // origin (y-coordinate)
double const L      = 2.5;  // x-coordinate of outflow boundary
double const H      = 0.41; // height of channel
double const X_C    = 0.2;  // center of cylinder (x-coordinate)
double const Y_C    = 0.2;  // center of cylinder (y-coordinate)
double const X_2    = 2.0 * X_C;
double const D      = 0.1;                    // cylinder diameter
double const R      = D / 2.0;                // cylinder radius
double const T      = 0.02;                   // thickness of flag
double const L_FLAG = 0.35;                   // length of flag
double const X_3    = X_C + R + L_FLAG * 1.6; // only relevant for mesh
double const Y_3    = H / 3.0;                // only relevant for mesh

// boundary conditions
dealii::types::boundary_id const BOUNDARY_ID_WALLS    = 0;
dealii::types::boundary_id const BOUNDARY_ID_INFLOW   = 1;
dealii::types::boundary_id const BOUNDARY_ID_OUTFLOW  = 2;
dealii::types::boundary_id const BOUNDARY_ID_CYLINDER = 3;
dealii::types::boundary_id const BOUNDARY_ID_FLAG     = 4;

bool STRUCTURE_COVERS_FLAG_ONLY = true;

unsigned int N_CELLS_FLAG_X = 16;

inline double U_X_MAX  = 1.5 * U_MEAN;
inline double END_TIME = 4.0 * L / U_MEAN;

inline double      OUTPUT_INTERVAL_TIME                = END_TIME / 600;
unsigned int const OUTPUT_SOLVER_INFO_EVERY_TIME_STEPS = 1e2;

inline void
set_test_case(unsigned int const test_case)
{
  AssertThrow(test_case >= 1 and test_case <= 3,
              dealii::ExcMessage("Application/TestCase has to be 1, 2 or 3."));
  TEST_CASE                = test_case;
  FLUID_VISCOSITY          = 1.0e-3;
  FLUID_DENSITY            = 1.0e3;
  POISSONS_RATIO_STRUCTURE = 0.4;
  if(test_case == 1)
  {
    U_MEAN                   = 0.2;
    DENSITY_STRUCTURE        = 1.0e3;
    YOUNGS_MODULUS_STRUCTURE = 0.5e6 * 2.0 * (1.0 + POISSONS_RATIO_STRUCTURE);
  }
  else if(test_case == 2)
  {
    U_MEAN                   = 1.0;
    DENSITY_STRUCTURE        = 1.0e4;
    YOUNGS_MODULUS_STRUCTURE = 0.5e6 * 2.0 * (1.0 + POISSONS_RATIO_STRUCTURE);
  }
  else
  {
    U_MEAN                   = 2.0;
    DENSITY_STRUCTURE        = 1.0e3;
    YOUNGS_MODULUS_STRUCTURE = 2.0e6 * 2.0 * (1.0 + POISSONS_RATIO_STRUCTURE);
  }
  U_X_MAX              = 1.5 * U_MEAN;
  END_TIME             = 4.0 * L / U_MEAN;
  OUTPUT_INTERVAL_TIME = END_TIME / 600;
}

// registers Application/TestCase in the parameter handler of a sub-application
inline void
add_test_case_parameter(dealii::ParameterHandler & prm)
{
  prm.add_parameter("TestCase",
                    TEST_CASE,
                    "Turek-Hron benchmark: 1 (FSI1), 2 (FSI2) or 3 (FSI3).",
                    dealii::Patterns::Integer(1, 3));
  prm.add_action("TestCase", [](std::string const & value) { set_test_case(std::stoi(value)); });
}

double const REL_TOL = 1.e-6;
double const ABS_TOL = 1.e-8;

double const REL_TOL_LINEARIZED = 1.e-6;
double const ABS_TOL_LINEARIZED = 1.e-12;

template<int dim>
class InflowBC : public dealii::Function<dim>
{
public:
  InflowBC() : dealii::Function<dim>(dim, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)p;
    double result = 0.0;

    double const t = this->get_time();
    double const T = 2.0;

    double const y = p[1];

    if(component == 0)
      result = U_X_MAX * (y * (H - y) / std::pow(H / 2.0, 2.0)) *
               ((t < T) ? 0.5 * (1.0 - std::cos(t * dealii::numbers::PI / T)) : 1.0);

    return result;
  }
};

template<int dim>
class SpatiallyVaryingE : public dealii::Function<dim>
{
public:
  /*
   * stiffening = 0: band |y - H/2| < H/8 along the whole channel (original model)
   * stiffening = 1: E = 1 + E_max (r_0 / (r_0 + d))^2, d = distance to the flag (reference
   *                 configuration), so that the cells around the flag move almost rigidly with it
   */
  SpatiallyVaryingE(unsigned int const stiffening = 0,
                    double const       e_max      = 1000.0,
                    double const       r_0        = 0.02)
    : dealii::Function<dim>(1, 0.0), stiffening(stiffening), e_max(e_max), r_0(r_0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)component;

    if(stiffening == 1)
    {
      // distance to the flag rectangle [X_C, X_C + R + L_FLAG] x [Y_C - T/2, Y_C + T/2]
      double const dx = std::max({X_C - p[0], 0.0, p[0] - (X_C + R + L_FLAG)});
      double const dy = std::max({(Y_C - T / 2.0) - p[1], 0.0, p[1] - (Y_C + T / 2.0)});
      double const d  = std::sqrt(dx * dx + dy * dy);
      return 1.0 + e_max * std::pow(r_0 / (r_0 + d), 2.0);
    }

    double const value =
      (std::abs(p[1] - H / 2.) < H / 8.) ? std::cos(4.0 * p[1] / H * dealii::numbers::PI) : 0.0;
    double result = 1. + 100. * value * value;

    return result;
  }

private:
  unsigned int const stiffening;
  double const       e_max, r_0;
};

namespace FluidFSI
{
template<int dim, typename Number>
class Application : public FluidFSI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : FluidFSI::ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    FluidFSI::ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      add_test_case_parameter(prm);
      prm.add_parameter("EndTime",
                        end_time,
                        "End time of the simulation.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("TractionCoupling",
                        traction_coupling,
                        "Prescribe the interface traction on the fluid (NeumannCached) and on the "
                        "structure, as required by FSI/AccelerationMethod = OptimalControl.",
                        dealii::Patterns::Bool());
      prm.add_parameter(
        "PullBackTraction",
        pull_back_traction,
        "Structure: pull back the (spatial) interface traction with the current "
        "deformation. false is required by FSI/AccelerationMethod = OptimalControl.",
        dealii::Patterns::Bool());
      prm.add_parameter("CFL",
                        cfl,
                        "CFL number of the adaptive time step.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("ViscousFormulation",
                        viscous_formulation,
                        "Formulation of the viscous term of the fluid: Auto (Laplace for the "
                        "projection methods, Divergence for the coupled solver and traction "
                        "coupling), Laplace or Divergence.",
                        dealii::Patterns::Selection("Auto|Laplace|Divergence"));
      prm.add_parameter("CoupledFluidSolver",
                        coupled_fluid_solver,
                        "Use the fluid discretization of the traction coupling (coupled solver, "
                        "divergence formulation) also for Dirichlet-Neumann schemes.",
                        dealii::Patterns::Bool());
      prm.add_parameter("FluidSolverAbsTol",
                        fluid_abs_tol,
                        "Absolute tolerance of the linear solvers of the projection-type fluid "
                        "schemes (dual splitting, pressure correction).",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("FluidSolverRelTol",
                        fluid_rel_tol,
                        "Relative tolerance of the linear solvers of the projection-type fluid "
                        "schemes (dual splitting, pressure correction).",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("CoupledSolverAbsTol",
                        coupled_abs_tol,
                        "Absolute tolerance of the linear solver of the coupled fluid solver "
                        "(TractionCoupling or CoupledFluidSolver).",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("CoupledSolverRelTol",
                        coupled_rel_tol,
                        "Relative tolerance of the linear solver of the coupled fluid solver "
                        "(TractionCoupling or CoupledFluidSolver).",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("CoupledVelocityPreconditioner",
                        coupled_velocity_preconditioner,
                        "Preconditioner of the velocity block of the coupled fluid solver.",
                        Patterns::Enum<IncNS::MomentumPreconditioner>());
      prm.add_parameter("CoupledSchurPreconditioner",
                        coupled_schur_preconditioner,
                        "Preconditioner of the Schur complement of the coupled fluid solver.",
                        Patterns::Enum<IncNS::SchurComplementPreconditioner>());
      prm.add_parameter("CoupledSchurMultigridType",
                        coupled_schur_multigrid_type,
                        "Multigrid type of the Laplace operator in the Schur-complement "
                        "preconditioner of the coupled fluid solver.",
                        Patterns::Enum<MultigridType>());
      prm.add_parameter("CoupledSchurCoarseSolver",
                        coupled_schur_coarse_solver,
                        "Coarse-grid solver of that multigrid.",
                        Patterns::Enum<MultigridCoarseGridSolver>());
      prm.add_parameter("CoupledSchurCoarsePreconditioner",
                        coupled_schur_coarse_preconditioner,
                        "Preconditioner of that coarse-grid solver.",
                        Patterns::Enum<MultigridCoarseGridPreconditioner>());
      prm.add_parameter(
        "AdaptiveTimeStepping",
        adaptive_time_stepping,
        "Adaptive (CFL) time step; false: constant time step from the CFL condition "
        "with the maximum inflow velocity.",
        dealii::Patterns::Bool());
      prm.add_parameter("MeshMovementType",
                        mesh_movement_type,
                        "ALE mesh motion of the fluid.",
                        Patterns::Enum<IncNS::MeshMovementType>());
      prm.add_parameter("MappingDegree",
                        mapping_degree,
                        "Degree of the fluid mapping (0: degree of the fluid velocity).",
                        dealii::Patterns::Integer(0));
      prm.add_parameter("ALEStiffening",
                        ale_stiffening,
                        "Stiffness distribution of the ALE elasticity: 0 = band along the channel "
                        "(original), 1 = growing towards the flag.",
                        dealii::Patterns::Integer(0, 1));
      prm.add_parameter("ALEStiffeningMax",
                        ale_stiffening_max,
                        "ALEStiffening = 1: maximum relative stiffness at the flag.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("ALEStiffeningRadius",
                        ale_stiffening_radius,
                        "ALEStiffening = 1: decay length r_0 of the stiffness.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("ALEPreconditioner",
                        ale_preconditioner,
                        "Preconditioner of the ALE elasticity (Direct: sparse direct solver of the "
                        "assembled matrix, factorized once for the linear mesh motion).",
                        Patterns::Enum<Structure::Preconditioner>());
      prm.add_parameter("ALESolverAbsTol",
                        ale_abs_tol,
                        "Absolute tolerance of the linear solver of the ALE elasticity.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("ALESolverRelTol",
                        ale_rel_tol,
                        "Relative tolerance of the linear solver of the ALE elasticity.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("ALELargeDeformation",
                        ale_large_deformation,
                        "Nonlinear (large-deformation) elasticity for the ALE mesh motion.",
                        dealii::Patterns::Bool());
      prm.add_parameter("CoupledConvectiveImplicit",
                        coupled_convective_implicit,
                        "Coupled fluid solver: implicit convective term (Newton), no CFL limit of "
                        "the explicit convection (only for methods without analytic derivatives).",
                        dealii::Patterns::Bool());
      prm.add_parameter("CoupledNewtonAbsTol",
                        coupled_newton_abs_tol,
                        "CoupledConvectiveImplicit: absolute tolerance of the Newton solver.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("CoupledNewtonRelTol",
                        coupled_newton_rel_tol,
                        "CoupledConvectiveImplicit: relative tolerance of the Newton solver.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("CoupledVelocityMultigridType",
                        coupled_velocity_multigrid_type,
                        "Multigrid type of the velocity block of the coupled fluid solver.",
                        Patterns::Enum<MultigridType>());
      prm.add_parameter("CoupledSchurDirect",
                        coupled_schur_direct,
                        "Coupled fluid solver: Laplace operator of the Schur-complement "
                        "preconditioner inverted by a sparse direct solver instead of multigrid.",
                        dealii::Patterns::Bool());
      prm.add_parameter("PressurePoissonPreconditioner",
                        pressure_poisson_preconditioner,
                        "Dual splitting: preconditioner of the pressure Poisson equation (Direct: "
                        "sparse direct solver of the assembled matrix).",
                        Patterns::Enum<IncNS::PreconditionerPressurePoisson>());
      prm.add_parameter("ViscousPreconditioner",
                        viscous_preconditioner,
                        "Dual splitting: preconditioner of the viscous step.",
                        Patterns::Enum<IncNS::MomentumPreconditioner>());
      prm.add_parameter("ProjectionPreconditioner",
                        projection_preconditioner,
                        "Dual splitting: preconditioner of the penalty (projection) step.",
                        Patterns::Enum<IncNS::PreconditionerProjection>());
      prm.add_parameter("FluidPreconditionerUpdate",
                        fluid_preconditioner_update,
                        "Renewal of the matrix-based (Direct) fluid preconditioners every ... time "
                        "steps (0: never, the matrix of the first time step is kept).",
                        dealii::Patterns::Integer(0));
      prm.add_parameter("PenaltyTerms",
                        penalty_terms,
                        "Divergence and continuity penalty terms of the fluid.",
                        dealii::Patterns::Bool());
      prm.add_parameter("ContinuityPenaltyUseBoundaryData",
                        continuity_penalty_use_boundary_data,
                        "Continuity penalty: use the Dirichlet data on boundary faces.",
                        dealii::Patterns::Bool());
    }
    prm.leave_subsection();
  }

  double                               end_time               = END_TIME;
  bool                                 traction_coupling      = false;
  bool                                 coupled_fluid_solver   = false;
  std::string                          viscous_formulation    = "Auto";
  double                               cfl                    = 0.5;
  bool                                 pull_back_traction     = true;
  double                               fluid_abs_tol          = ABS_TOL;
  double                               fluid_rel_tol          = REL_TOL;
  double                               coupled_abs_tol        = 1.e-14;
  double                               coupled_rel_tol        = 1.e-10;
  bool                                 penalty_terms          = true;
  bool                                 adaptive_time_stepping = true;
  IncNS::PreconditionerPressurePoisson pressure_poisson_preconditioner =
    IncNS::PreconditionerPressurePoisson::Direct;
  IncNS::MomentumPreconditioner viscous_preconditioner =
    IncNS::MomentumPreconditioner::InverseMassMatrix;
  IncNS::PreconditionerProjection projection_preconditioner =
    IncNS::PreconditionerProjection::InverseMassMatrix;
  unsigned int                  fluid_preconditioner_update     = 0;
  bool                          coupled_schur_direct            = true;
  MultigridType                 coupled_velocity_multigrid_type = MultigridType::cphMG;
  bool                          coupled_convective_implicit     = false;
  double                        coupled_newton_abs_tol          = 1.e-12;
  double                        coupled_newton_rel_tol          = 1.e-8;
  IncNS::MomentumPreconditioner coupled_velocity_preconditioner =
    IncNS::MomentumPreconditioner::InverseMassMatrix;
  IncNS::SchurComplementPreconditioner coupled_schur_preconditioner =
    IncNS::SchurComplementPreconditioner::PressureConvectionDiffusion;
  MultigridType             coupled_schur_multigrid_type = MultigridType::hMG;
  MultigridCoarseGridSolver coupled_schur_coarse_solver  = MultigridCoarseGridSolver::Chebyshev;
  MultigridCoarseGridPreconditioner coupled_schur_coarse_preconditioner =
    MultigridCoarseGridPreconditioner::PointJacobi;
  IncNS::MeshMovementType   mesh_movement_type    = IncNS::MeshMovementType::Elasticity;
  unsigned int              mapping_degree        = 0;
  bool                      ale_large_deformation = false;
  double                    ale_abs_tol           = ABS_TOL;
  Structure::Preconditioner ale_preconditioner    = Structure::Preconditioner::Multigrid;
  double                    ale_rel_tol           = REL_TOL;
  unsigned int              ale_stiffening        = 0;
  double                    ale_stiffening_max    = 1000.0;
  double                    ale_stiffening_radius = 0.02;
  bool                      continuity_penalty_use_boundary_data = true;

  void
  set_parameters() final
  {
    using namespace IncNS;

    Parameters & param = this->param;

    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term    = FormulationConvectiveTerm::ConvectiveFormulation;
    param.use_outflow_bc_convective_term = true;
    param.right_hand_side                = false;

    // ALE
    param.ale_formulation                     = true;
    param.mesh_movement_type                  = mesh_movement_type;
    param.neumann_with_variable_normal_vector = false;

    // PHYSICAL QUANTITIES
    param.start_time = 0.0;
    param.end_time   = end_time;
    param.viscosity  = FLUID_VISCOSITY;
    param.density    = FLUID_DENSITY;

    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplitting;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;
    param.adaptive_time_stepping          = adaptive_time_stepping;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.time_step_size                  = END_TIME;
    param.max_velocity                    = U_X_MAX;
    param.cfl                             = cfl;
    param.cfl_exponent_fe_degree_velocity = 1.5;

    // output of solver information
    param.solver_info_data.interval_time_steps = OUTPUT_SOLVER_INFO_EVERY_TIME_STEPS;

    // restart
    param.restarted_simulation             = false;
    param.restart_data.write_restart       = false;
    param.restart_data.interval_time       = 0.25;
    param.restart_data.interval_wall_time  = 1.e6;
    param.restart_data.interval_time_steps = 1e8;
    param.restart_data.filename            = "output/vortex/vortex";


    // SPATIAL DISCRETIZATION
    param.grid.triangulation_type     = TriangulationType::Distributed;
    param.mapping_degree              = mapping_degree > 0 ? mapping_degree : param.degree_u;
    param.mapping_degree_coarse_grids = param.mapping_degree;
    param.degree_p                    = DegreePressure::MixedOrder;

    // convective term
    param.upwind_factor = 1.0;

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = penalty_terms;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = penalty_terms;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    param.continuity_penalty_use_boundary_data       = continuity_penalty_use_boundary_data;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;
    param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_data_pressure_poisson =
      SolverData(1000, fluid_abs_tol, fluid_rel_tol, LinearSolver::CG, 100);
    param.preconditioner_pressure_poisson = pressure_poisson_preconditioner;
    if(pressure_poisson_preconditioner == PreconditionerPressurePoisson::Direct and
       fluid_preconditioner_update > 0)
    {
      param.update_preconditioner_pressure_poisson                  = true;
      param.update_preconditioner_pressure_poisson_every_time_steps = fluid_preconditioner_update;
    }
    param.multigrid_data_pressure_poisson.type                   = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_pressure_poisson.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

    // projection step
    param.solver_data_projection = SolverData(1000, fluid_abs_tol, fluid_rel_tol, LinearSolver::CG);
    param.preconditioner_projection = projection_preconditioner;
    if(projection_preconditioner == PreconditionerProjection::Direct and
       fluid_preconditioner_update > 0)
    {
      // the penalty parameters depend on the velocity
      param.update_preconditioner_projection                  = true;
      param.update_preconditioner_projection_every_time_steps = fluid_preconditioner_update;
    }

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;
    param.formulation_convective_term_bc = FormulationConvectiveTerm::ConvectiveFormulation;

    if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplitting)
    {
      this->param.solver_data_momentum =
        SolverData(1000, fluid_abs_tol, fluid_rel_tol, LinearSolver::CG);
      this->param.preconditioner_momentum = viscous_preconditioner;
      if(viscous_preconditioner == MomentumPreconditioner::Direct and
         fluid_preconditioner_update > 0)
      {
        // the time step size changes the mass coefficient
        this->param.update_preconditioner_momentum                  = true;
        this->param.update_preconditioner_momentum_every_time_steps = fluid_preconditioner_update;
      }
    }


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation =
      std::min(2, (int)param.order_time_integrator) - 1; // J_p = J-1, but not larger than 1
    param.rotational_formulation = true;

    // momentum step
    if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      // Newton solver
      param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

      // linear solver
      if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
        param.solver_data_momentum =
          SolverData(1e4, ABS_TOL_LINEARIZED, REL_TOL_LINEARIZED, LinearSolver::FGMRES, 100);
      else
        param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, LinearSolver::FGMRES, 100);
      param.update_preconditioner_momentum = false;
      param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    }


    // COUPLED NAVIER-STOKES SOLVER
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_coupled =
        SolverData(1e4, ABS_TOL_LINEARIZED, REL_TOL_LINEARIZED, LinearSolver::FGMRES, 100);
    else
      param.solver_data_coupled = SolverData(1e4, ABS_TOL, REL_TOL, LinearSolver::FGMRES, 100);

    // preconditioner linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;
    if(fluid_preconditioner_update > 0 and
       (coupled_velocity_preconditioner == MomentumPreconditioner::Direct or coupled_schur_direct))
    {
      // the time step size changes the mass coefficient, the mesh moves (ALE)
      param.update_preconditioner_coupled                   = true;
      param.update_preconditioner_coupled_every_time_steps  = fluid_preconditioner_update;
      param.update_preconditioner_coupled_every_newton_iter = 1000;
    }

    // preconditioner momentum block
    param.preconditioner_velocity_block          = coupled_velocity_preconditioner;
    param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data_velocity_block.type     = coupled_velocity_multigrid_type;
    param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::PointJacobi;
    param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_velocity_block.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block                       = coupled_schur_preconditioner;
    param.direct_solve_of_pressure_block                      = coupled_schur_direct;
    param.multigrid_data_pressure_block.type                  = coupled_schur_multigrid_type;
    param.multigrid_data_pressure_block.coarse_problem.solver = coupled_schur_coarse_solver;
    param.multigrid_data_pressure_block.coarse_problem.preconditioner =
      coupled_schur_coarse_preconditioner;

    // Traction-controlled coupling (optimal control): the fluid step has to be affine in the
    // interface traction with a symmetric saddle point matrix, and the natural boundary condition
    // has to be the Cauchy traction.
    if(traction_coupling or coupled_fluid_solver)
    {
      param.formulation_viscous_term     = FormulationViscousTerm::DivergenceFormulation;
      param.temporal_discretization      = TemporalDiscretization::BDFCoupledSolution;
      param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
      param.apply_penalty_terms_in_postprocessing_step = false;
      // sensitivities and adjoints are only as accurate as the linear solves
      param.solver_data_coupled =
        SolverData(1e4, coupled_abs_tol, coupled_rel_tol, LinearSolver::FGMRES, 100);

      if(coupled_convective_implicit)
      {
        // nonlinear fluid step: Newton with linear solves to CoupledSolverAbs/RelTol
        param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
        param.newton_solver_data_coupled =
          Newton::SolverData(100, coupled_newton_abs_tol, coupled_newton_rel_tol);
        param.multigrid_operator_type_velocity_block =
          MultigridOperatorType::ReactionConvectionDiffusion;
        if(param.update_preconditioner_coupled)
          param.update_preconditioner_coupled_every_newton_iter = 1;
      }
    }

    // explicit choice of the viscous formulation (e.g. to compare the fluid solvers on the same
    // spatial discretization)
    if(viscous_formulation == "Laplace")
      param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    else if(viscous_formulation == "Divergence")
      param.formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation;
  }

  void
  create_triangulation(dealii::Triangulation<2> & tria)
  {
    std::vector<dealii::Triangulation<2>> tria_vec;
    tria_vec.resize(11);

    dealii::GridGenerator::general_cell(
      tria_vec[0],
      {dealii::Point<2>(X_0, 0.0),
       dealii::Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
       dealii::Point<2>(X_0, H),
       dealii::Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0))});

    dealii::GridGenerator::general_cell(
      tria_vec[1],
      {dealii::Point<2>(X_0, 0.0),
       dealii::Point<2>(X_2, 0.0),
       dealii::Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
       dealii::Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0))});

    dealii::GridGenerator::general_cell(
      tria_vec[2],
      {dealii::Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
       dealii::Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
       dealii::Point<2>(X_0, H),
       dealii::Point<2>(X_2, H)});

    dealii::GridGenerator::general_cell(
      tria_vec[3],
      {dealii::Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
       dealii::Point<2>(X_2, 0.0),
       dealii::Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
       dealii::Point<2>(X_2, Y_C - T / 2.0)});

    dealii::GridGenerator::general_cell(
      tria_vec[4],
      {dealii::Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0),
       dealii::Point<2>(X_2, Y_C + T / 2.0),
       dealii::Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
       dealii::Point<2>(X_2, H)});

    dealii::GridGenerator::subdivided_hyper_rectangle(tria_vec[5],
                                                      {1, 1} /* subdivisions x,y */,
                                                      dealii::Point<2>(X_2, 0.0),
                                                      dealii::Point<2>(X_C + R + L_FLAG,
                                                                       Y_C - T / 2.0));

    dealii::GridGenerator::subdivided_hyper_rectangle(tria_vec[6],
                                                      {1, 1} /* subdivisions x,y */,
                                                      dealii::Point<2>(X_2, Y_C + T / 2.0),
                                                      dealii::Point<2>(X_C + R + L_FLAG, H));

    dealii::GridGenerator::general_cell(tria_vec[7],
                                        {dealii::Point<2>(X_C + R + L_FLAG, 0.0),
                                         dealii::Point<2>(X_3, 0.0),
                                         dealii::Point<2>(X_C + R + L_FLAG, Y_C - T / 2.0),
                                         dealii::Point<2>(X_3, Y_3)});

    dealii::GridGenerator::general_cell(tria_vec[8],
                                        {dealii::Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0),
                                         dealii::Point<2>(X_3, 2.0 * Y_3),
                                         dealii::Point<2>(X_C + R + L_FLAG, H),
                                         dealii::Point<2>(X_3, H)});

    dealii::GridGenerator::general_cell(tria_vec[9],
                                        {dealii::Point<2>(X_C + R + L_FLAG, Y_C - T / 2.0),
                                         dealii::Point<2>(X_3, Y_3),
                                         dealii::Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0),
                                         dealii::Point<2>(X_3, 2.0 * Y_3)});

    dealii::GridGenerator::subdivided_hyper_rectangle(tria_vec[10],
                                                      {8, 3} /* subdivisions x,y */,
                                                      dealii::Point<2>(X_3, 0.0),
                                                      dealii::Point<2>(L, H));

    std::vector<dealii::Triangulation<2> const *> tria_vec_ptr(tria_vec.size());
    for(unsigned int i = 0; i < tria_vec.size(); ++i)
      tria_vec_ptr[i] = &tria_vec[i];

    dealii::GridGenerator::merge_triangulations(tria_vec_ptr, tria);
  }

  void
  create_triangulation(dealii::Triangulation<3> & tria)
  {
    (void)tria;

    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)periodic_face_pairs;
      (void)vector_local_refinements;

      create_triangulation(tria);

      // vectors of manifold_ids and face_ids
      unsigned int const        manifold_id_start = 10;
      std::vector<unsigned int> manifold_ids;
      std::vector<unsigned int> face_ids;

      dealii::Point<dim> center;
      center[0] = X_C;
      center[1] = Y_C;

      AssertThrow(
        this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
        dealii::ExcMessage(
          "Manifolds might not be applied correctly for TriangulationType::FullyDistributed. "
          "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

      for(auto cell : tria.cell_iterators())
      {
        double const TOL = 1.e-12;

        // boundary IDs
        for(auto const & f : cell->face_indices())
        {
          double const x = cell->face(f)->center()(0);
          double const y = cell->face(f)->center()(1);

          if(std::fabs(y - Y_0) < TOL or std::fabs(y - H) < TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_WALLS);
          }

          if(std::fabs(x - X_0) < TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_INFLOW);
          }

          if(std::fabs(x - L) < TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_OUTFLOW);
          }

          if(std::fabs(cell->face(f)->center().distance(center)) < R + TOL)
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_CYLINDER);
          }

          if(std::fabs(y - (Y_C - T / 2.0)) < TOL or std::fabs(y - (Y_C + T / 2.0)) < TOL or
             (std::fabs(y - Y_C) < T / 2.0 + TOL and std::fabs(x - (X_C + R + L_FLAG)) < TOL))
          {
            cell->face(f)->set_boundary_id(BOUNDARY_ID_FLAG);
          }
        }

        // manifold IDs
        for(auto const & f : cell->face_indices())
        {
          if(cell->face(f)->at_boundary())
          {
            bool face_at_sphere_boundary = true;
            for(auto const & v : cell->face(f)->vertex_indices())
            {
              if(std::abs(center.distance(cell->face(f)->vertex(v)) - R) > TOL)
              {
                face_at_sphere_boundary = false;
                break;
              }
            }

            if(face_at_sphere_boundary)
            {
              face_ids.push_back(f);
              unsigned int manifold_id = manifold_id_start + manifold_ids.size() + 1;
              cell->set_all_manifold_ids(manifold_id);
              manifold_ids.push_back(manifold_id);
              break;
            }
          }
        }
      }

      // generate vector of manifolds and apply manifold to all cells that have been marked
      std::vector<std::shared_ptr<dealii::Manifold<dim>>> manifold_vec;
      manifold_vec.resize(manifold_ids.size());

      for(unsigned int i = 0; i < manifold_ids.size(); ++i)
      {
        for(auto cell : tria.cell_iterators())
        {
          if(cell->manifold_id() == manifold_ids[i])
          {
            manifold_vec[i] =
              std::shared_ptr<dealii::Manifold<dim>>(static_cast<dealii::Manifold<dim> *>(
                new OneSidedCylindricalManifold<dim>(tria, cell, face_ids[i], center)));
            tria.set_manifold(manifold_ids[i], *(manifold_vec[i]));
          }
        }
      }

      tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    std::shared_ptr<IncNS::BoundaryDescriptor<dim>> boundary_descriptor = this->boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // fill boundary descriptor velocity
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(BOUNDARY_ID_WALLS, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(BOUNDARY_ID_INFLOW, new InflowBC<dim>()));
    boundary_descriptor->velocity->neumann_bc.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->velocity->dirichlet_bc.insert(
      pair(BOUNDARY_ID_CYLINDER, new dealii::Functions::ZeroFunction<dim>(dim)));
    // fluid-structure interface: velocity from the structure (Dirichlet-Neumann schemes) or
    // traction from the coupling scheme (optimal control, with p = 0 in the weak boundary term)
    if(traction_coupling)
      boundary_descriptor->velocity->neumann_cached_bc.insert(BOUNDARY_ID_FLAG);
    else
      boundary_descriptor->velocity->dirichlet_cached_bc.insert(BOUNDARY_ID_FLAG);

    // fill boundary descriptor pressure
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_WALLS);
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_INFLOW);
    boundary_descriptor->pressure->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(1)));
    boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_CYLINDER);
    if(traction_coupling)
      boundary_descriptor->pressure->dirichlet_bc.insert(
        pair(BOUNDARY_ID_FLAG, new dealii::Functions::ZeroFunction<dim>(1)));
    else
      boundary_descriptor->pressure->neumann_bc.insert(BOUNDARY_ID_FLAG);
  }

  void
  set_field_functions() final
  {
    std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions = this->field_functions;

    field_functions->initial_solution_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new dealii::Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = 0.0;
    pp_data.output_data.time_control_data.trigger_interval = OUTPUT_INTERVAL_TIME;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_fluid";
    pp_data.output_data.n_vtu_files        = this->output_parameters.n_vtu_files;
    pp_data.output_data.write_pvd          = this->output_parameters.write_pvd;
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree_u;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  void
  set_parameters_ale_poisson() final
  {
    using namespace Poisson;

    Parameters & param = this->ale_poisson_param;

    // MATHEMATICAL MODEL
    param.right_hand_side = false;

    // SPATIAL DISCRETIZATION
    param.spatial_discretization = SpatialDiscretization::CG;
    param.degree                 = this->param.mapping_degree;

    // SOLVER
    param.solver_data    = SolverData(1e4, ABS_TOL, REL_TOL, LinearSolver::FGMRES, 100);
    param.preconditioner = Preconditioner::Multigrid;

    param.multigrid_data.type                          = MultigridType::phMG;
    param.multigrid_data.p_sequence                    = PSequenceType::Bisect;
    param.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
  }

  void
  set_boundary_descriptor_ale_poisson() final
  {
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor =
      this->ale_poisson_boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_WALLS, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_INFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_CYLINDER, new dealii::Functions::ZeroFunction<dim>(dim)));

    // fluid-structure interface
    boundary_descriptor->dirichlet_cached_bc.insert(BOUNDARY_ID_FLAG);
  }


  void
  set_field_functions_ale_poisson() final
  {
    std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions =
      this->ale_poisson_field_functions;

    field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  void
  set_parameters_ale_elasticity() final
  {
    using namespace Structure;

    Parameters & param = this->ale_elasticity_param;

    param.problem_type         = ProblemType::Steady;
    param.body_force           = false;
    param.pull_back_body_force = false;
    param.large_deformation    = ale_large_deformation;
    param.pull_back_traction   = false;
    param.material_type        = MaterialType::StVenantKirchhoff;

    param.degree = this->param.mapping_degree;

    param.newton_solver_data = Newton::SolverData(1e4, ABS_TOL, REL_TOL);
    if(param.large_deformation)
      param.solver_data =
        SolverData(1e4, ABS_TOL_LINEARIZED, REL_TOL_LINEARIZED, LinearSolver::FGMRES, 100);
    else
      param.solver_data = SolverData(1e4, ale_abs_tol, ale_rel_tol, LinearSolver::FGMRES, 100);
    param.preconditioner                               = ale_preconditioner;
    param.multigrid_data.type                          = MultigridType::phMG;
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;

    param.update_preconditioner                         = param.large_deformation;
    param.update_preconditioner_every_newton_iterations = 10;
  }

  void
  set_boundary_descriptor_ale_elasticity() final
  {
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor =
      this->ale_elasticity_boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_WALLS, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_WALLS, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_WALLS, dealii::ComponentMask()));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_INFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_INFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_INFLOW, dealii::ComponentMask()));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_OUTFLOW, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_OUTFLOW, dealii::ComponentMask()));
    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_CYLINDER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_CYLINDER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_CYLINDER, dealii::ComponentMask()));

    // fluid-structure interface
    boundary_descriptor->dirichlet_cached_bc.insert(BOUNDARY_ID_FLAG);
  }

  void
  set_material_descriptor_ale_elasticity() final
  {
    std::shared_ptr<Structure::MaterialDescriptor> material_descriptor =
      this->ale_elasticity_material_descriptor;

    using namespace Structure;

    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    AssertThrow(this->ale_elasticity_param.material_type == MaterialType::StVenantKirchhoff,
                dealii::ExcMessage("Material parameters for ALE elasticity problem not defined."));

    MaterialType const type         = MaterialType::StVenantKirchhoff;
    Type2D const       two_dim_type = Type2D::PlaneStrain;

    double const                           youngs_modulus = 1.0;
    double const                           poissons_ratio = 0.3;
    std::shared_ptr<dealii::Function<dim>> youngs_modulus_function;
    youngs_modulus_function.reset(
      new SpatiallyVaryingE<dim>(ale_stiffening, ale_stiffening_max, ale_stiffening_radius));
    material_descriptor->insert(
      Pair(0,
           new StVenantKirchhoffData<dim>(
             type, youngs_modulus, poissons_ratio, two_dim_type, youngs_modulus_function)));
  }

  void
  set_field_functions_ale_elasticity() final
  {
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions =
      this->ale_elasticity_field_functions;

    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_displacement.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }
};
} // namespace FluidFSI

namespace StructureFSI
{
template<int dim, typename Number>
class Application : public StructureFSI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : StructureFSI::ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    StructureFSI::ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      add_test_case_parameter(prm);
      prm.add_parameter("EndTime",
                        end_time,
                        "End time of the simulation.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("TractionCoupling",
                        traction_coupling,
                        "Prescribe the interface traction on the fluid (NeumannCached) and on the "
                        "structure, as required by FSI/AccelerationMethod = OptimalControl.",
                        dealii::Patterns::Bool());
      prm.add_parameter(
        "PullBackTraction",
        pull_back_traction,
        "Structure: pull back the (spatial) interface traction with the current "
        "deformation. false is required by FSI/AccelerationMethod = OptimalControl.",
        dealii::Patterns::Bool());
      prm.add_parameter("CFL",
                        cfl,
                        "CFL number of the adaptive time step.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("CoupledFluidSolver",
                        coupled_fluid_solver,
                        "Use the fluid discretization of the traction coupling (coupled solver, "
                        "divergence formulation) also for Dirichlet-Neumann schemes.",
                        dealii::Patterns::Bool());
      prm.add_parameter("GenAlphaType",
                        gen_alpha_type,
                        "Time integrator of the structure (generalized-alpha family).",
                        Patterns::Enum<GenAlphaType>());
      prm.add_parameter("SpectralRadius",
                        spectral_radius,
                        "Spectral radius of the generalized-alpha (Bossak) structure integrator.",
                        dealii::Patterns::Double(0.0, 1.0));
      prm.add_parameter("StructurePreconditioner",
                        structure_preconditioner,
                        "Preconditioner of the linear solver of the structure (Direct: sparse "
                        "direct solver of the assembled matrix).",
                        Patterns::Enum<Structure::Preconditioner>());
      prm.add_parameter("StructurePreconditionerUpdate",
                        structure_preconditioner_update,
                        "Update of the structure preconditioner every ... time steps and Newton "
                        "iterations.",
                        dealii::Patterns::Integer(1));
      prm.add_parameter("StructureNewtonAbsTol",
                        structure_newton_abs_tol,
                        "Absolute tolerance of the Newton solver of the structure.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("StructureNewtonRelTol",
                        structure_newton_rel_tol,
                        "Relative tolerance of the Newton solver of the structure.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("StructureLinearAbsTol",
                        structure_linear_abs_tol,
                        "Absolute tolerance of the linear solver of the structure.",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter("StructureLinearRelTol",
                        structure_linear_rel_tol,
                        "Relative tolerance of the linear solver of the structure.",
                        dealii::Patterns::Double(0.0));
    }
    prm.leave_subsection();
  }

  Structure::Preconditioner structure_preconditioner        = Structure::Preconditioner::Multigrid;
  unsigned int              structure_preconditioner_update = 10;
  double                    structure_newton_abs_tol        = ABS_TOL;
  double                    structure_newton_rel_tol        = REL_TOL;
  double                    structure_linear_abs_tol        = ABS_TOL_LINEARIZED;
  double                    structure_linear_rel_tol        = REL_TOL_LINEARIZED;

  double       end_time             = END_TIME;
  bool         traction_coupling    = false;
  bool         coupled_fluid_solver = false;
  double       cfl                  = 0.5;
  bool         pull_back_traction   = true;
  double       spectral_radius      = 0.8;
  GenAlphaType gen_alpha_type       = GenAlphaType::BossakAlpha;

  // Structure
  void
  set_parameters() final
  {
    using namespace Structure;

    Parameters & param = this->param;

    param.problem_type         = ProblemType::Unsteady;
    param.body_force           = false;
    param.pull_back_body_force = false;
    param.large_deformation    = true;
    // traction-controlled coupling applies the area ratio of the frozen geometry itself
    param.pull_back_traction = pull_back_traction and not traction_coupling;
    param.material_type      = MaterialType::StVenantKirchhoff;

    param.density = DENSITY_STRUCTURE;

    param.start_time                           = 0.0;
    param.end_time                             = end_time;
    param.time_step_size                       = END_TIME / 100.0;
    param.gen_alpha_type                       = gen_alpha_type;
    param.spectral_radius                      = spectral_radius;
    param.solver_info_data.interval_time_steps = OUTPUT_SOLVER_INFO_EVERY_TIME_STEPS;

    param.grid.triangulation_type     = TriangulationType::Distributed;
    param.mapping_degree              = param.degree;
    param.mapping_degree_coarse_grids = param.mapping_degree;

    param.newton_solver_data =
      Newton::SolverData(1e4, structure_newton_abs_tol, structure_newton_rel_tol);
    if(param.large_deformation)
      param.solver_data = SolverData(
        1e4, structure_linear_abs_tol, structure_linear_rel_tol, LinearSolver::FGMRES, 100);
    else
      param.solver_data = SolverData(1e4, ABS_TOL, REL_TOL, LinearSolver::FGMRES, 100);
    param.preconditioner                               = structure_preconditioner;
    param.multigrid_data.type                          = MultigridType::phMG;
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;

    param.update_preconditioner                         = true;
    param.update_preconditioner_every_time_steps        = structure_preconditioner_update;
    param.update_preconditioner_every_newton_iterations = structure_preconditioner_update;
  }

  void
  create_triangulation_structure(dealii::Triangulation<2> & tria)
  {
    if(STRUCTURE_COVERS_FLAG_ONLY)
    {
      dealii::GridGenerator::subdivided_hyper_rectangle(
        tria,
        {N_CELLS_FLAG_X, 1} /* subdivisions x,y */,
        dealii::Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
        dealii::Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0));
    }
    else
    {
      std::vector<dealii::Triangulation<2>> tria_vec;

      tria_vec.resize(12);

      dealii::GridGenerator::general_cell(
        tria_vec[0],
        {dealii::Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
         dealii::Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
         dealii::Point<2>(X_C + T / 2.0, Y_C - T),
         dealii::Point<2>(X_C + T / 2.0 +
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C - T / 4.0)});

      dealii::GridGenerator::general_cell(
        tria_vec[1],
        {dealii::Point<2>(X_C + T / 2.0, Y_C - T),
         dealii::Point<2>(X_C + T / 2.0 +
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C - T / 4.0),
         dealii::Point<2>(X_C + T / 2.0, Y_C + T),
         dealii::Point<2>(X_C + T / 2.0 +
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C + T / 4.0)});

      dealii::GridGenerator::general_cell(
        tria_vec[2],
        {dealii::Point<2>(X_C + T / 2.0 +
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C - T / 4.0),
         dealii::Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
         dealii::Point<2>(X_C + T / 2.0 +
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C + T / 4.0),
         dealii::Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0)});

      dealii::GridGenerator::general_cell(
        tria_vec[3],
        {dealii::Point<2>(X_C + T / 2.0 +
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C + T / 4.0),
         dealii::Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0),
         dealii::Point<2>(X_C + T / 2.0, Y_C + T),
         dealii::Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0))});

      dealii::GridGenerator::general_cell(tria_vec[4],
                                          {dealii::Point<2>(X_C - T / 2.0, Y_C - T),
                                           dealii::Point<2>(X_C + T / 2.0, Y_C - T),
                                           dealii::Point<2>(X_C - T / 2.0, Y_C + T),
                                           dealii::Point<2>(X_C + T / 2.0, Y_C + T)});

      dealii::GridGenerator::general_cell(
        tria_vec[5],
        {dealii::Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
         dealii::Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
         dealii::Point<2>(X_C - T / 2.0, Y_C - T),
         dealii::Point<2>(X_C + T / 2.0, Y_C - T)});

      dealii::GridGenerator::general_cell(
        tria_vec[6],
        {dealii::Point<2>(X_C - T / 2.0, Y_C + T),
         dealii::Point<2>(X_C + T / 2.0, Y_C + T),
         dealii::Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
         dealii::Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0))});

      dealii::GridGenerator::general_cell(
        tria_vec[7],
        {dealii::Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
         dealii::Point<2>(X_C - T / 2.0, Y_C - T),
         dealii::Point<2>(X_C - R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
         dealii::Point<2>(X_C - T / 2.0 -
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C - T / 4.0)});

      dealii::GridGenerator::general_cell(
        tria_vec[8],
        {dealii::Point<2>(X_C - T / 2.0 -
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C - T / 4.0),
         dealii::Point<2>(X_C - T / 2.0, Y_C - T),
         dealii::Point<2>(X_C - T / 2.0 -
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C + T / 4.0),
         dealii::Point<2>(X_C - T / 2.0, Y_C + T)});

      dealii::GridGenerator::general_cell(
        tria_vec[9],
        {dealii::Point<2>(X_C - R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
         dealii::Point<2>(X_C - T / 2.0 -
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C - T / 4.0),
         dealii::Point<2>(X_C - R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0),
         dealii::Point<2>(X_C - T / 2.0 -
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C + T / 4.0)});

      dealii::GridGenerator::general_cell(
        tria_vec[10],
        {dealii::Point<2>(X_C - R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0),
         dealii::Point<2>(X_C - T / 2.0 -
                            (X_C + R * std::cos(std::asin(T / (2.0 * R))) - (X_C + T / 2.0)) / 2.0,
                          Y_C + T / 4.0),
         dealii::Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
         dealii::Point<2>(X_C - T / 2.0, Y_C + T)});

      dealii::GridGenerator::subdivided_hyper_rectangle(
        tria_vec[11],
        {8, 1} /* subdivisions x,y */,
        dealii::Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
        dealii::Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0));

      std::vector<dealii::Triangulation<2> const *> tria_vec_ptr(tria_vec.size());
      for(unsigned int i = 0; i < tria_vec.size(); ++i)
        tria_vec_ptr[i] = &tria_vec[i];

      dealii::GridGenerator::merge_triangulations(tria_vec_ptr, tria);
    }
  }

  void
  create_triangulation_structure(dealii::Triangulation<3> & tria)
  {
    (void)tria;

    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)periodic_face_pairs;
      (void)vector_local_refinements;

      create_triangulation_structure(tria);

      // vectors of manifold_ids and face_ids
      unsigned int const        manifold_id_start = 10;
      std::vector<unsigned int> manifold_ids;
      std::vector<unsigned int> face_ids;

      dealii::Point<dim> center;
      center[0] = X_C;
      center[1] = Y_C;

      AssertThrow(
        this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
        dealii::ExcMessage(
          "Manifolds might not be applied correctly for TriangulationType::FullyDistributed. "
          "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

      for(auto cell : tria.cell_iterators())
      {
        double const TOL = 1.e-12;

        // boundary IDs
        for(auto const & f : cell->face_indices())
        {
          double const x = cell->face(f)->center()(0);

          if(cell->face(f)->at_boundary())
          {
            if(STRUCTURE_COVERS_FLAG_ONLY)
            {
              if(x < X_C + R * std::cos(std::asin(T / (2.0 * R))) + TOL)
              {
                cell->face(f)->set_boundary_id(BOUNDARY_ID_CYLINDER);
              }
            }
            else
            {
              if(x < X_C + R * std::cos(std::asin(T / (2.0 * R))))
              {
                cell->face(f)->set_boundary_id(BOUNDARY_ID_CYLINDER);
              }
            }

            if(x > X_C + R * std::cos(std::asin(T / (2.0 * R))))
            {
              cell->face(f)->set_boundary_id(BOUNDARY_ID_FLAG);
            }
          }
        }

        // manifold IDs
        for(auto const & f : cell->face_indices())
        {
          if(cell->face(f)->at_boundary())
          {
            bool face_at_sphere_boundary = true;
            for(auto const & v : cell->face(f)->vertex_indices())
            {
              if(std::abs(center.distance(cell->face(f)->vertex(v)) - R) > TOL)
              {
                face_at_sphere_boundary = false;
                break;
              }
            }

            if(face_at_sphere_boundary)
            {
              face_ids.push_back(f);
              unsigned int manifold_id = manifold_id_start + manifold_ids.size() + 1;
              cell->set_all_manifold_ids(manifold_id);
              manifold_ids.push_back(manifold_id);
              break;
            }
          }
        }
      }

      // generate vector of manifolds and apply manifold to all cells that have been marked
      std::vector<std::shared_ptr<dealii::Manifold<dim>>> manifold_vec;
      manifold_vec.resize(manifold_ids.size());

      for(unsigned int i = 0; i < manifold_ids.size(); ++i)
      {
        for(auto cell : tria.cell_iterators())
        {
          if(cell->manifold_id() == manifold_ids[i])
          {
            manifold_vec[i] =
              std::shared_ptr<dealii::Manifold<dim>>(static_cast<dealii::Manifold<dim> *>(
                new OneSidedCylindricalManifold<dim>(tria, cell, face_ids[i], center)));
            tria.set_manifold(manifold_ids[i], *(manifold_vec[i]));
          }
        }
      }

      tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor =
      this->boundary_descriptor;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    boundary_descriptor->dirichlet_bc.insert(
      pair(BOUNDARY_ID_CYLINDER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(BOUNDARY_ID_CYLINDER, new dealii::Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(BOUNDARY_ID_CYLINDER, dealii::ComponentMask()));

    // fluid-structure interface
    boundary_descriptor->neumann_cached_bc.insert(BOUNDARY_ID_FLAG);
  }

  void
  set_field_functions() final
  {
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions = this->field_functions;

    field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_displacement.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  void
  set_material_descriptor() final
  {
    std::shared_ptr<Structure::MaterialDescriptor> material_descriptor = this->material_descriptor;

    using namespace Structure;

    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    AssertThrow(this->param.material_type == MaterialType::StVenantKirchhoff,
                dealii::ExcMessage("Material parameters for elasticity problem not defined."));

    MaterialType const type         = MaterialType::StVenantKirchhoff;
    Type2D const       two_dim_type = Type2D::PlaneStrain;

    material_descriptor->insert(Pair(0,
                                     new StVenantKirchhoffData<dim>(type,
                                                                    YOUNGS_MODULUS_STRUCTURE,
                                                                    POISSONS_RATIO_STRUCTURE,
                                                                    two_dim_type)));
  }

  std::shared_ptr<Structure::PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    using namespace Structure;

    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = 0.0;
    pp_data.output_data.time_control_data.trigger_interval = OUTPUT_INTERVAL_TIME;
    pp_data.output_data.directory   = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename    = this->output_parameters.filename + "_structure";
    pp_data.output_data.n_vtu_files = this->output_parameters.n_vtu_files;
    pp_data.output_data.write_pvd   = this->output_parameters.write_pvd;

    pp_data.output_data.write_processor_id = true;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }
};

} // namespace StructureFSI

namespace FSI
{
template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
  {
    this->structure = std::make_shared<StructureFSI::Application<dim, Number>>(input_file, comm);
    this->fluid     = std::make_shared<FluidFSI::Application<dim, Number>>(input_file, comm);
  }
};
} // namespace FSI

} // namespace ExaDG

#include <exadg/fluid_structure_interaction/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_FSI_CYLINDER_WITH_FLAG_H_ */
