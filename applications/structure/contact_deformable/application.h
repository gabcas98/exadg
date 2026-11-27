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

#ifndef STRUCTURE_CONTACT_DEFORMABLE
#define STRUCTURE_CONTACT_DEFORMABLE

/*
 * Frictionless contact between two deformable bodies (penalty method, Gauss-point-to-segment,
 * see DeformableContact), 2d, plane strain, linear elements.
 *
 * Setup "TwoBlocks": the lower block [0, 2] x [-0.5, 0] is clamped at y = -0.5. The upper block
 * [0.5, 1.5] x [g_0, g_0 + 0.5] (non-matching mesh) is moved by a prescribed displacement of its
 * top face: first downwards by g_0 + c (t <= 1/2: approach, contact and compression c), then
 * tangentially by s (t > 1/2: frictionless sliding over the lower block).
 *
 * ContactSurfaces = Faces: slave = bottom face of the upper block, master = top face of the lower
 * block (one pass). ContactSurfaces = Boundary: the whole boundary of each block except the
 * Dirichlet faces is a contact surface, two passes (both blocks are slave and master), which
 * increases the number of surface primitives searched.
 *
 * LowerBlock = Deformable: as described above. LowerBlock = Clamped: the top face of the lower
 * block is clamped as well, i.e. the master surface is the rigid plane y = 0, represented by mesh
 * faces and found by the search. LowerBlock = AnalyticPlane: same mesh and clamping, but the
 * contact is the existing penalty contact with the analytic rigid plane phi(x) = y on the bottom
 * face of the upper block (BoundaryDescriptor::contact_bc, no search). Clamped and AnalyticPlane
 * are the same problem and have to give the same solution (ContactSurfaces = Faces).
 */

namespace ExaDG
{
namespace Structure
{
/*
 * Displacement of the top face of the upper block.
 */
template<int dim>
class TopDisplacement : public dealii::Function<dim>
{
public:
  TopDisplacement(double const approach, double const slide, double const t_press)
    : dealii::Function<dim>(dim), approach(approach), slide(slide), t_press(t_press)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;
    double const t = this->get_time();
    if(c == 0)
      return slide * std::max(t - t_press, 0.0) / (1.0 - t_press);
    else if(c == 1)
      return -approach * std::min(t / t_press, 1.0);
    return 0.0;
  }

private:
  double const approach, slide, t_press;
};

enum class ContactSurfaces
{
  Faces,
  Boundary
};

enum class LowerBlock
{
  Deformable,
  Clamped,
  AnalyticPlane
};

/*
 * Rigid plane y = 0 with normal (0, 1), phi(x) = x_1.
 */
template<int dim>
class HorizontalPlane : public dealii::Function<dim>
{
public:
  HorizontalPlane() : dealii::Function<dim>(1)
  {
  }

  double
  value(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)component;
    return x[1];
  }

  dealii::Tensor<1, dim>
  gradient(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)x;
    (void)component;
    dealii::Tensor<1, dim> n;
    n[1] = 1.0;
    return n;
  }

  dealii::SymmetricTensor<2, dim>
  hessian(dealii::Point<dim> const & x, unsigned int const component = 0) const final
  {
    (void)x;
    (void)component;
    return dealii::SymmetricTensor<2, dim>();
  }
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
      prm.add_parameter("ContactSurfaces", contact_surfaces, "Faces or Boundary.");
      prm.add_parameter("LowerBlock",
                        lower_block,
                        "Deformable, Clamped (rigid master faces) or AnalyticPlane (existing "
                        "penalty contact with the plane y = 0).");
      prm.add_parameter("InitialGap", initial_gap, "Initial gap g_0 between the blocks.");
      prm.add_parameter("Compression", compression, "Compression c after contact.");
      prm.add_parameter("Slide", slide, "Tangential displacement s of the upper block.");
      prm.add_parameter("YoungsModulus", youngs_modulus, "Young's modulus.");
      prm.add_parameter("PoissonsRatio", poissons_ratio, "Poisson's ratio.");
      prm.add_parameter("PenaltyParameter", penalty_parameter, "Penalty parameter epsilon.");
      prm.add_parameter("ContactRadius", contact_radius, "Detection radius r_c.");
      prm.add_parameter("Skin", skin, "Skin delta of the Verlet list.");
      prm.add_parameter("ProjectionTolerance",
                        projection_tolerance,
                        "Admissible projections xi in [-tau, 1 + tau].");
      prm.add_parameter("Search", search, "Exhaustive or Verlet.");
      prm.add_parameter("ContactInPreconditioner",
                        contact_in_preconditioner,
                        "Add the contact stiffness to the AMG matrix.");
      prm.add_parameter("PreconditionerCoupling",
                        preconditioner_coupling,
                        "Slave-master coupling in the AMG matrix (extended sparsity pattern).");
      prm.add_parameter("LogContactState",
                        log_contact_state,
                        "Print the contact state changes and the Newton residuals.");
      prm.add_parameter("VerifySearch",
                        verify_search,
                        "Repeat every Verlet query with the exhaustive search.");
      prm.add_parameter("LoadIncrement", load_increment, "Load increment (quasi-static).");
      prm.add_parameter("Preconditioner", preconditioner, "Preconditioner (None, AMG, ...).");
      prm.add_parameter("NewtonAbsTol", newton_abs_tol, "Absolute tolerance of Newton's method.");
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

    // AnalyticPlane: existing penalty contact with a rigid obstacle, same penalty parameter
    if(lower_block == LowerBlock::AnalyticPlane)
    {
      this->param.contact_method            = ContactMethod::Penalty;
      this->param.contact_penalty_parameter = penalty_parameter;
    }

    this->param.deformable_contact_penalty_parameter       = penalty_parameter;
    this->param.deformable_contact_radius                  = contact_radius;
    this->param.deformable_contact_skin                    = skin;
    this->param.deformable_contact_projection_tolerance    = projection_tolerance;
    this->param.deformable_contact_search                  = search;
    this->param.deformable_contact_verify_search           = verify_search;
    this->param.deformable_contact_in_preconditioner       = contact_in_preconditioner;
    this->param.deformable_contact_preconditioner_coupling = preconditioner_coupling;
    this->param.deformable_contact_log                     = log_contact_state;

    this->param.load_increment = load_increment;

    this->param.mapping_degree              = 1;
    this->param.mapping_degree_coarse_grids = 1;

    this->param.grid.element_type                 = ElementType::Hypercube;
    this->param.grid.triangulation_type           = TriangulationType::Serial;
    this->param.grid.create_coarse_triangulations = false;

    this->param.use_matrix_based_operator = false;
    this->param.sparse_matrix_type        = SparseMatrixType::Trilinos;

    // Newton's method with the line search settings of the rigid obstacle application
    this->param.newton_solver_data = Newton::SolverData(50, newton_abs_tol, 1.e-9, 6);
    this->param.newton_solver_data.n_nonmonotone_steps = 3;
    this->param.newton_solver_data.nonmonotone_factor  = 1.e6;
    this->param.newton_solver_data.max_iter_damping    = 20;
    this->param.newton_solver_data.print_residuals     = log_contact_state;

    // the frictionless penalty contact has a symmetric tangent
    this->param.solver_data    = SolverData(1e4, 1.e-14, 1.e-6, LinearSolver::CG, 100);
    this->param.preconditioner = preconditioner;

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
    AssertThrow(dim == 2, dealii::ExcMessage("The deformable contact is implemented in 2d."));

    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        // lower block (8 x 2 cells) and upper block (5 x 2 cells): non-matching meshes
        dealii::Triangulation<dim> lower, upper;
        dealii::GridGenerator::subdivided_hyper_rectangle(lower,
                                                          {8, 2},
                                                          dealii::Point<dim>(0.0, -0.5),
                                                          dealii::Point<dim>(2.0, 0.0));
        dealii::GridGenerator::subdivided_hyper_rectangle(upper,
                                                          {5, 2},
                                                          dealii::Point<dim>(0.5, initial_gap),
                                                          dealii::Point<dim>(1.5,
                                                                             initial_gap + 0.5));
        dealii::GridGenerator::merge_triangulations(lower, upper, tria);

        /*
         * boundary ids:
         *  0: traction free
         *  1: y = -0.5, clamped
         *  2: top face of the upper block, prescribed displacement
         * 10: lower block contact surface (master)
         * 11: upper block contact surface (slave)
         */
        for(auto cell : tria.cell_iterators())
        {
          bool const upper_block = cell->center()[1] > 0.0;
          for(auto const & f : cell->face_indices())
          {
            if(not cell->at_boundary(f))
              continue;
            dealii::Point<dim> const c = cell->face(f)->center();
            if(std::abs(c[1] + 0.5) < 1e-12)
              cell->face(f)->set_boundary_id(1);
            else if(std::abs(c[1] - (initial_gap + 0.5)) < 1e-12)
              cell->face(f)->set_boundary_id(2);
            else if(contact_surfaces == ContactSurfaces::Boundary)
              cell->face(f)->set_boundary_id(upper_block ? 11 : 10);
            else if(not upper_block and std::abs(c[1]) < 1e-12)
              cell->face(f)->set_boundary_id(10);
            else if(upper_block and std::abs(c[1] - initial_gap) < 1e-12)
              cell->face(f)->set_boundary_id(11);
            else
              cell->face(f)->set_boundary_id(0);
          }
        }

        if(global_refinements > 0)
          tria.refine_global(global_refinements);
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

    bc.neumann_bc.insert(pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    bc.dirichlet_bc.insert(pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
    bc.dirichlet_bc_initial_acceleration.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));
    bc.dirichlet_bc_component_mask.insert(pair_mask(1, dealii::ComponentMask(dim, true)));

    bc.dirichlet_bc.insert(
      pair(2, new TopDisplacement<dim>(initial_gap + compression, slide, t_press)));
    bc.dirichlet_bc_initial_acceleration.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(dim)));
    bc.dirichlet_bc_component_mask.insert(pair_mask(2, dealii::ComponentMask(dim, true)));

    if(lower_block != LowerBlock::Deformable)
    {
      AssertThrow(contact_surfaces == ContactSurfaces::Faces,
                  dealii::ExcMessage("LowerBlock = Clamped/AnalyticPlane requires "
                                     "ContactSurfaces = Faces."));

      // the top face of the lower block is the rigid plane y = 0
      bc.dirichlet_bc.insert(pair(10, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_initial_acceleration.insert(
        pair(10, new dealii::Functions::ZeroFunction<dim>(dim)));
      bc.dirichlet_bc_component_mask.insert(pair_mask(10, dealii::ComponentMask(dim, true)));

      if(lower_block == LowerBlock::AnalyticPlane)
      {
        bc.contact_bc.insert(pair(11, new HorizontalPlane<dim>()));
      }
      else
      {
        bc.neumann_bc.insert(pair(11, new dealii::Functions::ZeroFunction<dim>(dim)));
        bc.deformable_contact_pairs.push_back({11, 10});
      }
      return;
    }

    // contact surfaces: traction free plus the contact term
    bc.neumann_bc.insert(pair(10, new dealii::Functions::ZeroFunction<dim>(dim)));
    bc.neumann_bc.insert(pair(11, new dealii::Functions::ZeroFunction<dim>(dim)));
    bc.deformable_contact_pairs.push_back({11, 10});
    if(contact_surfaces == ContactSurfaces::Boundary)
      bc.deformable_contact_pairs.push_back({10, 11});
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    this->material_descriptor->insert(
      Pair(0,
           new StVenantKirchhoffData<dim>(MaterialType::StVenantKirchhoff,
                                          youngs_modulus,
                                          poissons_ratio,
                                          Type2D::PlaneStrain)));
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
    pp_data.output_data.degree                       = this->param.degree;

    pp_data.contact_data.time_control_data.is_active = true;
    pp_data.contact_data.directory                   = this->output_parameters.directory;
    pp_data.contact_data.filename = this->output_parameters.filename + "_contact";

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }

  ContactSurfaces contact_surfaces = ContactSurfaces::Faces;
  LowerBlock      lower_block      = LowerBlock::Deformable;

  double       initial_gap = 0.05;
  double       compression = 0.02;
  double       slide       = 0.3;
  double const t_press     = 0.5;

  double youngs_modulus = 1.0;
  double poissons_ratio = 0.3;

  double                penalty_parameter         = 1000.0;
  double                contact_radius            = 0.05;
  double                skin                      = 0.05;
  double                projection_tolerance      = 0.01;
  ContactSearchStrategy search                    = ContactSearchStrategy::Verlet;
  bool                  verify_search             = false;
  bool                  contact_in_preconditioner = false;
  bool                  preconditioner_coupling   = true;
  bool                  log_contact_state         = false;

  double load_increment = 0.05;
  double newton_abs_tol = 1.e-8;

  Preconditioner preconditioner = Preconditioner::AMG;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
