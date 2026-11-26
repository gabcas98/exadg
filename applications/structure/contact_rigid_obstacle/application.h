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

#ifndef STRUCTURE_CONTACT_RIGID_OBSTACLE
#define STRUCTURE_CONTACT_RIGID_OBSTACLE

/*
 * Frictionless contact of an elastic body with a rigid obstacle (penalty or Nitsche method).
 *
 * Setup "PatchTest": a block [0,L] x [0,H] (x [0,W]) rests on the rigid plane y = 0 and is loaded
 * by a uniform traction p on the top face. Symmetry conditions on x = 0 (and z = 0) suppress the
 * rigid body motions in tangential direction. The exact solution has a uniform contact pressure
 * lambda = p, a uniform gap g = -p / epsilon (penalty) or g = 0 (Nitsche), a contact force p L (W)
 * and an active area L (W), independently of the mesh, the polynomial degree and the strain
 * magnitude.
 *
 * Setup "SlidingBlock": the block of the patch test is compressed by a prescribed displacement
 * -delta of the top face against the rigid plane y = 0 (load factor t <= 1/2), then the plane is
 * translated tangentially by s while the top face is held fixed (t > 1/2). With Coulomb friction,
 * the tangential traction satisfies |t_T| = mu lambda at all slipping points; once the whole
 * contact face slips, the tangential force equals mu times the normal force.
 *
 * Setup "Cattaneo" (2d): a rigid cylinder is pressed by delta into the body [-W, W] x [-H, 0]
 * (t <= 1/2) and then shifted tangentially by s (t > 1/2). For Q < mu P, the contact zone
 * consists of a central stick zone of half-width c = a sqrt(1 - Q / (mu P)) and two slip zones
 * (Cattaneo-Mindlin; exact for uncoupled normal and tangential problems, e.g. nu -> 1/2, and a
 * frictionless indentation, see the parameter FrictionlessIndentation).
 *
 * Setup "Hertz": a rigid cylinder (2d, plane strain) or sphere (3d) of radius R is pressed by an
 * indentation delta into the elastic body [0,W] x [-H,0] (x [0,W]). Only one half (2d) or one
 * quarter (3d) is computed using symmetry conditions on x = 0 (and z = 0), the bottom y = -H is
 * clamped. For W, H >> a, the contact half-width a, the maximum pressure p0 and the pressure
 * profile p(r) = p0 sqrt(1 - r^2/a^2) are compared to the Hertz solution for the measured load.
 */

namespace ExaDG
{
namespace Structure
{
/*
 * Rigid plane with normal n (pointing towards the body) through the point p(t) = p_0 + t * v.
 */
template<int dim>
class RigidPlane : public dealii::Function<dim>
{
public:
  RigidPlane(dealii::Point<dim> const &     point,
             dealii::Tensor<1, dim> const & normal,
             dealii::Tensor<1, dim> const & velocity)
    : dealii::Function<dim>(1), point(point), normal(normal / normal.norm()), velocity(velocity)
  {
  }

  double
  value(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)component;
    return (x - (point + this->get_time() * velocity)) * normal;
  }

  dealii::Tensor<1, dim>
  gradient(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)x;
    (void)component;
    return normal;
  }

  dealii::SymmetricTensor<2, dim>
  hessian(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)x;
    (void)component;
    return dealii::SymmetricTensor<2, dim>();
  }

private:
  dealii::Point<dim> const     point;
  dealii::Tensor<1, dim> const normal;
  dealii::Tensor<1, dim> const velocity;
};

/*
 * Rigid cylinder (2d) or sphere (3d) of radius R with center c(t) = c_0 + t * v. The body is
 * located outside of the obstacle, phi(x) = |x - c| - R.
 */
template<int dim>
class RigidBall : public dealii::Function<dim>
{
public:
  RigidBall(dealii::Point<dim> const & center, double const radius, dealii::Tensor<1, dim> velocity)
    : dealii::Function<dim>(1), center(center), radius(radius), velocity(velocity)
  {
  }

  double
  value(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)component;
    return (x - current_center()).norm() - radius;
  }

  dealii::Tensor<1, dim>
  gradient(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)component;
    dealii::Tensor<1, dim> const r = x - current_center();
    return r / r.norm();
  }

  dealii::SymmetricTensor<2, dim>
  hessian(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)component;
    dealii::Tensor<1, dim> const r      = x - current_center();
    double const                 norm_r = r.norm();
    dealii::Tensor<1, dim> const n      = r / norm_r;
    return (dealii::unit_symmetric_tensor<dim>() -
            dealii::symmetrize(dealii::outer_product(n, n))) /
           norm_r;
  }

private:
  dealii::Point<dim>
  current_center() const
  {
    return center + this->get_time() * velocity;
  }

  dealii::Point<dim> const     center;
  double const                 radius;
  dealii::Tensor<1, dim> const velocity;
};

/*
 * Displacement of a rigid ball that is first pressed into the body by delta (t <= t_press) and
 * then translated tangentially (x direction) by s (t_press < t <= 1).
 */
template<int dim>
class PressAndShiftDisplacement : public dealii::Function<dim>
{
public:
  PressAndShiftDisplacement(double const delta, double const shift, double const t_press)
    : dealii::Function<dim>(dim), delta(delta), shift(shift), t_press(t_press)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;
    double const t = this->get_time();
    if(c == 0)
      return shift * std::max(t - t_press, 0.0) / (1.0 - t_press);
    else if(c == 1)
      return -delta * std::min(t / t_press, 1.0);
    return 0.0;
  }

private:
  double const delta, shift, t_press;
};

/*
 * Rigid ball whose center moves with a given displacement function, phi(x) = |x - c(t)| - R.
 */
template<int dim>
class MovingBall : public dealii::Function<dim>
{
public:
  MovingBall(dealii::Point<dim> const &             center,
             double const                           radius,
             std::shared_ptr<dealii::Function<dim>> displacement)
    : dealii::Function<dim>(1), center(center), radius(radius), displacement(displacement)
  {
  }

  double
  value(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)component;
    return (x - current_center()).norm() - radius;
  }

  dealii::Tensor<1, dim>
  gradient(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)component;
    dealii::Tensor<1, dim> const r = x - current_center();
    return r / r.norm();
  }

  dealii::SymmetricTensor<2, dim>
  hessian(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)component;
    dealii::Tensor<1, dim> const r      = x - current_center();
    double const                 norm_r = r.norm();
    dealii::Tensor<1, dim> const n      = r / norm_r;
    return (dealii::unit_symmetric_tensor<dim>() -
            dealii::symmetrize(dealii::outer_product(n, n))) /
           norm_r;
  }

private:
  dealii::Point<dim>
  current_center() const
  {
    displacement->set_time(this->get_time());
    dealii::Point<dim> c = center;
    for(unsigned int d = 0; d < dim; ++d)
      c[d] += displacement->value(center, d);
    return c;
  }

  dealii::Point<dim> const               center;
  double const                           radius;
  std::shared_ptr<dealii::Function<dim>> displacement;
};

/*
 * Friction coefficient that vanishes during the indentation (t <= t_press) and equals mu during the
 * tangential loading (t > t_press).
 */
template<int dim>
class FrictionAfterIndentation : public dealii::Function<dim>
{
public:
  FrictionAfterIndentation(double const mu, double const t_press)
    : dealii::Function<dim>(1), mu(mu), t_press(t_press)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)p;
    (void)component;
    return (this->get_time() > t_press * (1.0 + 1e-12)) ? mu : 0.0;
  }

private:
  double const mu, t_press;
};

/*
 * Uniform traction t = (0, -p, 0) scaled by the load factor.
 */
template<int dim>
class Pressure : public dealii::Function<dim>
{
public:
  Pressure(double const pressure) : dealii::Function<dim>(dim), pressure(pressure)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;
    return (c == 1) ? -pressure * this->get_time() : 0.0;
  }

private:
  double const pressure;
};

/*
 * SlidingBlock: displacement of the top face, (0, -delta t / t_press) for t <= t_press, then
 * constant.
 */
template<int dim>
class TopDisplacement : public dealii::Function<dim>
{
public:
  TopDisplacement(double const compression, double const t_press)
    : dealii::Function<dim>(dim), compression(compression), t_press(t_press)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;
    return (c == 1) ? -compression * std::min(this->get_time() / t_press, 1.0) : 0.0;
  }

private:
  double const compression, t_press;
};

/*
 * SlidingBlock: tangential translation of the rigid plane, (s (t - t_press) / (1 - t_press), 0)
 * for t >= t_press.
 */
template<int dim>
class PlaneDisplacement : public dealii::Function<dim>
{
public:
  PlaneDisplacement(double const slip, double const t_press)
    : dealii::Function<dim>(dim), slip(slip), t_press(t_press)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;
    return (c == 0) ? slip * std::max(this->get_time() - t_press, 0.0) / (1.0 - t_press) : 0.0;
  }

private:
  double const slip, t_press;
};

enum class Setup
{
  PatchTest,
  Hertz,
  SlidingBlock,
  Cattaneo
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      prm.add_parameter("Setup", setup, "PatchTest, Hertz, SlidingBlock or Cattaneo.");
      prm.add_parameter("TangentialShift", shift, "Cattaneo: tangential shift of the ball.");
      prm.add_parameter("FrictionlessIndentation",
                        frictionless_indentation,
                        "Cattaneo: no friction during the indentation (as in Cattaneo-Mindlin).");
      prm.add_parameter("FrictionCoefficient", friction_coefficient, "Coulomb coefficient mu.");
      prm.add_parameter("Compression", compression, "SlidingBlock: compression of the block.");
      prm.add_parameter("Slip", slip, "SlidingBlock: tangential translation of the plane.");
      prm.add_parameter("YoungsModulus", youngs_modulus, "Young's modulus.");
      prm.add_parameter("PoissonsRatio", poissons_ratio, "Poisson's ratio.");
      prm.add_parameter("ContactMethod", contact_method, "Penalty or Nitsche.");
      prm.add_parameter("NitscheTheta",
                        nitsche_theta,
                        "Nitsche variant: 1 symmetric, 0 incomplete, -1 skew-symmetric.");
      prm.add_parameter("PenaltyParameter",
                        penalty_parameter,
                        "Penalty: epsilon. Nitsche: gamma_0 * E, gamma = gamma_0 E (k+1)^2 / h.");
      prm.add_parameter("Pressure", pressure, "PatchTest: traction on the top face.");
      prm.add_parameter("Radius", radius, "Hertz: radius of the rigid cylinder/sphere.");
      prm.add_parameter("Indentation", indentation, "Hertz: indentation of the obstacle.");
      prm.add_parameter("DomainSize", domain_size, "Hertz: size W = H of the elastic body.");
      prm.add_parameter("LocalRefinements",
                        n_local_refinements,
                        "Hertz: number of local refinements towards the contact zone.");
      prm.add_parameter("RefinementRadius",
                        refinement_radius,
                        "Hertz: radius of the locally refined zone around the origin.");
      prm.add_parameter("LoadIncrement", load_increment, "Load increment (quasi-static).");
      prm.add_parameter("LineSearchRefinements",
                        line_search_refinements,
                        "Bisection refinements of the damping factor of Newton's method.");
      prm.add_parameter("NonmonotoneFactor",
                        nonmonotone_factor,
                        "Maximum growth of the residual for a non-monotone full step.");
      prm.add_parameter("MaxIterDamping", max_iter_damping, "Maximum number of damping steps.");
      prm.add_parameter("NewtonAbsTol", newton_abs_tol, "Absolute tolerance of Newton's method.");
      prm.add_parameter("NonmonotoneSteps",
                        nonmonotone_steps,
                        "Consecutive full Newton steps accepted without residual decrease.");
      prm.add_parameter("Preconditioner", preconditioner, "Preconditioner for the linear system.");
      prm.add_parameter("MultigridType", multigrid_type, "Multigrid type (e.g. phMG, pMG, hMG).");
      prm.add_parameter("SmootherPreconditioner",
                        smoother_preconditioner,
                        "Preconditioner of the Chebyshev smoother (PointJacobi, AdditiveSchwarz).");
      prm.add_parameter("SmootherIterations",
                        smoother_iterations,
                        "Number of Chebyshev smoothing steps (degree).");
      prm.add_parameter("UseMatrixBasedOperator",
                        use_matrix_based_operator,
                        "Use matrix-based operators in global Krylov solver and Multigrid.");
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    this->param.problem_type         = ProblemType::QuasiStatic;
    this->param.body_force           = false;
    this->param.large_deformation    = true;
    this->param.pull_back_body_force = false;
    this->param.pull_back_traction   = false;
    this->param.material_type        = MaterialType::StVenantKirchhoff;

    this->param.contact_method               = contact_method;
    this->param.contact_nitsche_theta        = nitsche_theta;
    this->param.contact_friction_coefficient = friction_coefficient;
    this->param.contact_penalty_parameter    = penalty_parameter;

    this->param.load_increment = load_increment;

    this->param.mapping_degree              = 1;
    this->param.mapping_degree_coarse_grids = 1;

    this->param.grid.element_type                 = ElementType::Hypercube;
    this->param.grid.triangulation_type           = TriangulationType::Distributed;
    this->param.grid.create_coarse_triangulations = true;

    this->param.use_matrix_based_operator = use_matrix_based_operator;
    this->param.sparse_matrix_type        = SparseMatrixType::Trilinos;

    this->param.newton_solver_data =
      Newton::SolverData(50, newton_abs_tol, 1.e-9, line_search_refinements);
    this->param.newton_solver_data.n_nonmonotone_steps = nonmonotone_steps;
    this->param.newton_solver_data.nonmonotone_factor  = nonmonotone_factor;
    this->param.newton_solver_data.max_iter_damping    = max_iter_damping;
    // the linearized operator is not symmetric for Nitsche's method with theta != 1 and for
    // frictional contact (slip)
    bool const symmetric = (contact_method == ContactMethod::Penalty or nitsche_theta == 1.0) and
                           friction_coefficient == 0.0;
    LinearSolver const linear_solver = symmetric ? LinearSolver::CG : LinearSolver::GMRES;
    this->param.solver_data          = SolverData(1e4, 1.e-14, 1.e-6, linear_solver, 100);
    this->param.preconditioner       = preconditioner;

    this->param.multigrid_data.type                          = multigrid_type;
    this->param.multigrid_data.p_sequence                    = PSequenceType::DecreaseByOne;
    this->param.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    this->param.multigrid_data.smoother_data.preconditioner  = smoother_preconditioner;
    this->param.multigrid_data.smoother_data.iterations      = smoother_iterations;
    this->param.multigrid_data.smoother_data.smoothing_range = 20;
    this->param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::AMG;
    this->param.multigrid_data.coarse_problem.solver_data =
      SolverData(1e3, 1.e-12, 1.e-3, LinearSolver::Undefined, 100);

    this->param.update_preconditioner                         = true;
    this->param.update_preconditioner_every_time_steps        = 1;
    this->param.update_preconditioner_every_newton_iterations = 1;
    this->param.update_preconditioner_once_newton_converged   = true;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        dealii::Point<dim>        p1, p2;
        std::vector<unsigned int> repetitions(dim, 1);

        if(setup == Setup::PatchTest or setup == Setup::SlidingBlock)
        {
          // [0, 2] x [0, 1] (x [0, 1])
          p2[0]          = length;
          p2[1]          = height;
          repetitions[0] = 2;
          if(dim == 3)
            p2[2] = height;
        }
        else if(setup == Setup::Cattaneo)
        {
          AssertThrow(dim == 2, dealii::ExcMessage("Setup Cattaneo is implemented in 2d."));
          // [-W, W] x [-H, 0]
          p1[0] = -domain_size;
          p1[1] = -domain_size;
          p2[0] = domain_size;
          for(unsigned int d = 0; d < dim; ++d)
            repetitions[d] = 4;
          repetitions[0] = 8;
        }
        else if(setup == Setup::Hertz)
        {
          // [0, W] x [-H, 0] (x [0, W])
          p1[1] = -domain_size;
          p2[0] = domain_size;
          if(dim == 3)
            p2[2] = domain_size;
          for(unsigned int d = 0; d < dim; ++d)
            repetitions[d] = 4;
        }

        dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);

        /*
         * boundary ids:
         *  0: traction free
         *  1: x = 0, symmetry (u_x = 0)
         *  2: PatchTest: y = 0, contact with rigid plane  | Hertz: y = -H, clamped
         *  3: PatchTest: y = H, traction                   | Hertz: y = 0, contact with ball
         *  4: z = 0, symmetry (u_z = 0)
         */
        double const y_bottom = p1[1], y_top = p2[1];
        for(auto cell : tria.cell_iterators())
        {
          for(auto const & f : cell->face_indices())
          {
            if(not cell->at_boundary(f))
              continue;
            dealii::Point<dim> const c = cell->face(f)->center();
            if(std::abs(c[0]) < 1e-12)
              cell->face(f)->set_boundary_id(1);
            else if(std::abs(c[1] - y_bottom) < 1e-12)
              cell->face(f)->set_boundary_id(2);
            else if(std::abs(c[1] - y_top) < 1e-12)
              cell->face(f)->set_boundary_id(3);
            else if(dim == 3 and std::abs(c[dim - 1]) < 1e-12)
              cell->face(f)->set_boundary_id(4);
          }
        }

        if(global_refinements > 0)
          tria.refine_global(global_refinements);

        if(setup == Setup::Hertz or setup == Setup::Cattaneo)
        {
          // refine towards the initial contact point (origin)
          for(unsigned int i = 0; i < n_local_refinements; ++i)
          {
            for(auto const & cell : tria.active_cell_iterators())
            {
              if(cell->is_locally_owned() and
                 cell->center().norm() < std::max(refinement_radius, 2.0 * cell->diameter()))
                cell->set_refine_flag();
            }
            tria.execute_coarsening_and_refinement();
          }
        }
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

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
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    auto & bc = *this->boundary_descriptor;

    // traction free
    bc.neumann_bc.insert(pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    if(setup == Setup::Cattaneo)
    {
      // bottom clamped, sides traction free
      bc.dirichlet_bc.insert(pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_initial_acceleration.insert(
        pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_component_mask.insert(pair_mask(2, dealii::ComponentMask(dim, true)));

      // ball touching the body at the origin for t = 0, pressed and then shifted
      auto const displacement =
        std::make_shared<PressAndShiftDisplacement<dim>>(indentation, shift, t_press);
      dealii::Point<dim> center;
      center[1] = radius;
      bc.contact_bc.insert(pair(3, new MovingBall<dim>(center, radius, displacement)));
      bc.contact_obstacle_displacement.insert(
        pair(3, new PressAndShiftDisplacement<dim>(indentation, shift, t_press)));
      if(frictionless_indentation)
        bc.contact_friction_coefficient.insert(
          pair(3, new FrictionAfterIndentation<dim>(friction_coefficient, t_press)));
      return;
    }

    if(setup == Setup::SlidingBlock)
    {
      // lateral faces traction free
      bc.neumann_bc.insert(pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
      if(dim == 3)
        bc.neumann_bc.insert(pair(4, new dealii::Functions::ZeroFunction<dim>(dim)));

      // rigid plane y = 0 translating tangentially
      dealii::Tensor<1, dim> normal;
      normal[1] = 1.0;
      bc.contact_bc.insert(
        pair(2, new RigidPlane<dim>(dealii::Point<dim>(), normal, dealii::Tensor<1, dim>())));
      bc.contact_obstacle_displacement.insert(pair(2, new PlaneDisplacement<dim>(slip, t_press)));

      // top face: compression, then held fixed
      bc.dirichlet_bc.insert(pair(3, new TopDisplacement<dim>(compression, t_press)));
      bc.dirichlet_bc_initial_acceleration.insert(
        pair(3, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_component_mask.insert(pair_mask(3, dealii::ComponentMask(dim, true)));
      return;
    }

    // symmetry planes x = 0 and z = 0
    auto const add_symmetry = [&](dealii::types::boundary_id const id, unsigned int const comp) {
      std::vector<bool> mask(dim, false);
      mask[comp] = true;
      bc.dirichlet_bc.insert(pair(id, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_initial_acceleration.insert(
        pair(id, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_component_mask.insert(pair_mask(id, mask));
    };
    add_symmetry(1, 0);
    if(dim == 3)
      add_symmetry(4, 2);

    if(setup == Setup::PatchTest)
    {
      // rigid plane y = 0, the body lies above
      dealii::Tensor<1, dim> normal;
      normal[1] = 1.0;
      bc.contact_bc.insert(
        pair(2, new RigidPlane<dim>(dealii::Point<dim>(), normal, dealii::Tensor<1, dim>())));

      bc.neumann_bc.insert(pair(3, new Pressure<dim>(pressure)));
    }
    else if(setup == Setup::Hertz)
    {
      bc.dirichlet_bc.insert(pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_initial_acceleration.insert(
        pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_component_mask.insert(pair_mask(2, dealii::ComponentMask(dim, true)));

      // ball touching the body at the origin for t = 0, moving downwards by the indentation
      dealii::Point<dim> center;
      center[1] = radius;
      dealii::Tensor<1, dim> velocity;
      velocity[1] = -indentation;
      bc.contact_bc.insert(pair(3, new RigidBall<dim>(center, radius, velocity)));
    }
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    Type2D const two_dim_type = (dim == 2) ? Type2D::PlaneStrain : Type2D::Undefined;

    this->material_descriptor->insert(
      Pair(0,
           new StVenantKirchhoffData<dim>(
             MaterialType::StVenantKirchhoff, youngs_modulus, poissons_ratio, two_dim_type)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_displacement.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active  = this->output_parameters.write;
    pp_data.output_data.directory                    = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                     = this->output_parameters.filename;
    pp_data.output_data.write_displacement_magnitude = true;
    pp_data.output_data.write_higher_order           = true;
    pp_data.output_data.degree                       = this->param.degree;

    pp_data.contact_data.time_control_data.is_active = true;
    pp_data.contact_data.directory                   = this->output_parameters.directory;
    pp_data.contact_data.filename               = this->output_parameters.filename + "_contact";
    pp_data.contact_data.write_pressure_profile = true;

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }

  Setup setup = Setup::PatchTest;

  // friction
  double       friction_coefficient     = 0.0;
  double       compression              = 0.05;
  double       slip                     = 0.2;
  double const t_press                  = 0.5;
  double       shift                    = 0.0;
  bool         frictionless_indentation = false;

  double        youngs_modulus    = 1.0;
  double        poissons_ratio    = 0.3;
  ContactMethod contact_method    = ContactMethod::Penalty;
  double        nitsche_theta     = 0.0;
  double        penalty_parameter = 1.0e3;

  // PatchTest
  double const length   = 2.0;
  double const height   = 1.0;
  double       pressure = 0.01;

  // Hertz
  double       radius              = 1.0;
  double       indentation         = 0.01;
  double       domain_size         = 4.0;
  unsigned int n_local_refinements = 0;
  double       refinement_radius   = 0.2;

  double load_increment = 0.1;

  // Newton's method for the semismooth contact problem: non-monotone acceptance of at most three
  // full steps per load step (which update the active set), and a line search with up to 20
  // halvings and 6 bisection steps that steps up to the next change of the active set
  // (kink of the residual along the Newton direction). This avoids most reductions of the load
  // increment for stiff contact conditions and frictional contact, see
  // doc/structure_contact/report.pdf.
  unsigned int line_search_refinements = 6;
  unsigned int nonmonotone_steps       = 3;
  double       nonmonotone_factor      = 1.e6;
  unsigned int max_iter_damping        = 20;
  double       newton_abs_tol          = 1.e-12;

  Preconditioner preconditioner = Preconditioner::AMG;
  MultigridType  multigrid_type = MultigridType::phMG;
  // A Chebyshev degree of 8 instead of the default 5 reduces the multigrid iterations for contact
  // problems substantially (the point-Jacobi preconditioner does not capture the coupling of the
  // displacement components by the contact term n (x) n on curved obstacles).
  unsigned int smoother_iterations = 8;

  PreconditionerSmoother smoother_preconditioner   = PreconditionerSmoother::PointJacobi;
  bool                   use_matrix_based_operator = false;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
