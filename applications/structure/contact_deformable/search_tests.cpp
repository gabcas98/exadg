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

/*
 * Geometric and mechanical tests of the contact search between deformable surfaces with
 * prescribed displacement fields (no solver):
 *
 *  1. two blocks: a pair that enters the detection radius after the list build, the limit case
 *     2 D_max = delta (no rebuild, pair exactly at |g| = r_c), the first violation of the
 *     certificate, sliding, random perturbations,
 *  2. Newton trial sequence: accepted -> rejected trial (rebuild) -> return,
 *  3. U-shaped strip folding until its arms touch (self-contact),
 *  4. two adjacent elements folding about their common vertex (self-contact between faces of
 *     the first ring),
 *  5. residual and Jacobian-vector product: exhaustive vs. Verlet at the same state, and
 *     directional finite differences of the residual on a smooth branch (two blocks and the
 *     adjacent fold).
 *
 * Every query of the Verlet search is compared with the exhaustive search (master face,
 * projection xi and gap, bitwise). The program returns a non-zero exit code if a test fails.
 */

// C/C++
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

// ExaDG
#include <exadg/structure/spatial_discretization/operators/deformable_contact.h>

using namespace ExaDG::Structure;

int constexpr dim = 2;
typedef dealii::LinearAlgebra::distributed::Vector<double>                VectorType;
typedef DeformableContact<dim, double>                                    Contact;
typedef std::function<dealii::Tensor<1, dim>(dealii::Point<dim> const &)> DisplacementField;

unsigned int n_failures = 0;

void
check(bool const condition, std::string const & message)
{
  std::cout << (condition ? "  PASS  " : "  FAIL  ") << message << std::endl;
  if(not condition)
    ++n_failures;
}

/*
 * Vertex-wise interpolation of a displacement field (exact representation of a motion of the
 * vertices for Q1 elements).
 */
void
interpolate(dealii::DoFHandler<dim> const & dof_handler,
            DisplacementField const &       field,
            VectorType &                    u)
{
  u.reinit(dof_handler.n_dofs());
  for(auto const & cell : dof_handler.active_cell_iterators())
    for(unsigned int v = 0; v < cell->n_vertices(); ++v)
    {
      dealii::Tensor<1, dim> const value = field(cell->vertex(v));
      for(unsigned int c = 0; c < dim; ++c)
        u(cell->vertex_dof_index(v, c)) = value[c];
    }
}

/*
 * Compares the search results of two strategies point by point (bitwise).
 */
unsigned int
count_mismatches(std::vector<ContactProjection<dim>> const & a,
                 std::vector<ContactProjection<dim>> const & b)
{
  unsigned int n = 0;
  for(unsigned int p = 0; p < a.size(); ++p)
    if(a[p].master != b[p].master or a[p].xi != b[p].xi or a[p].gap != b[p].gap)
      ++n;
  return n;
}

unsigned int
count_found(std::vector<ContactProjection<dim>> const & a)
{
  unsigned int n = 0;
  for(auto const & p : a)
    n += p.found() ? 1 : 0;
  return n;
}

unsigned int
count_active(std::vector<ContactProjection<dim>> const & a)
{
  unsigned int n = 0;
  for(auto const & p : a)
    n += p.active() ? 1 : 0;
  return n;
}

/*
 * One query of both strategies at the same state, returns the Verlet result.
 */
std::vector<ContactProjection<dim>>
compare(Contact const & verlet, VectorType const & u, std::string const & label, bool print = true)
{
  unsigned long const                       builds_before = verlet.get_statistics().n_builds;
  std::vector<ContactProjection<dim>> const result_verlet =
    verlet.search(u, ContactSearchStrategy::Verlet);
  std::vector<ContactProjection<dim>> const result_exhaustive =
    verlet.search(u, ContactSearchStrategy::Exhaustive);
  unsigned int const n_mismatch = count_mismatches(result_verlet, result_exhaustive);
  if(print)
    std::cout << "    " << std::left << std::setw(44) << label << std::right
              << " found = " << std::setw(4) << count_found(result_verlet)
              << " active = " << std::setw(4) << count_active(result_verlet)
              << " rebuild = " << (verlet.get_statistics().n_builds - builds_before)
              << " list pairs = " << std::setw(5) << verlet.n_list_pairs()
              << " mismatches = " << n_mismatch << std::endl;
  if(n_mismatch > 0)
  {
    ++n_failures;
    std::cout << "  FAIL  Verlet and exhaustive search differ (" << label << ")" << std::endl;
  }
  return result_verlet;
}

/*
 * Two blocks: lower [0, 2] x [-0.5, 0] (8 x 2 cells), upper [0.5, 1.5] x [g0, g0 + 0.5] (5 x 2
 * cells), refined globally. Boundary ids: 10 top of the lower block (master), 11 bottom of the
 * upper block (slave), 0 otherwise.
 */
void
create_two_blocks(dealii::Triangulation<dim> & tria,
                  double const                 g0,
                  unsigned int const           refinements)
{
  dealii::Triangulation<dim> lower, upper;
  dealii::GridGenerator::subdivided_hyper_rectangle(lower,
                                                    {8, 2},
                                                    dealii::Point<dim>(0.0, -0.5),
                                                    dealii::Point<dim>(2.0, 0.0));
  dealii::GridGenerator::subdivided_hyper_rectangle(upper,
                                                    {5, 2},
                                                    dealii::Point<dim>(0.5, g0),
                                                    dealii::Point<dim>(1.5, g0 + 0.5));
  dealii::GridGenerator::merge_triangulations(lower, upper, tria);
  for(auto const & cell : tria.active_cell_iterators())
    for(unsigned int const f : cell->face_indices())
      if(cell->at_boundary(f))
      {
        double const y = cell->face(f)->center()[1];
        if(cell->center()[1] < 0.0 and std::abs(y) < 1e-12)
          cell->face(f)->set_boundary_id(10);
        else if(cell->center()[1] > 0.0 and std::abs(y - g0) < 1e-12)
          cell->face(f)->set_boundary_id(11);
        else
          cell->face(f)->set_boundary_id(0);
      }
  tria.refine_global(refinements);
}

/*
 * Test 1 and 2: two blocks with exact binary numbers (r_c = 1/16, delta = 1/8, initial gap
 * r_c + delta).
 */
void
test_two_blocks(double const projection_tolerance)
{
  std::cout << std::endl
            << "Test 1: two deformable blocks, approach and sliding (projection tolerance "
            << projection_tolerance << ")" << std::endl;

  double const r_c = 1.0 / 16.0, delta = 1.0 / 8.0, g0 = r_c + delta;

  dealii::Triangulation<dim> tria;
  create_two_blocks(tria, g0, 2);
  dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(1), dim);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DeformableContactData data;
  data.pairs                = {{11, 10}};
  data.penalty              = 100.0;
  data.radius               = r_c;
  data.skin                 = delta;
  data.search               = ContactSearchStrategy::Verlet;
  data.projection_tolerance = projection_tolerance;

  Contact verlet;
  verlet.setup(dof_handler, {}, data);

  // lower block: top surface moved up by a, linear in y; upper block: rigid translation (s, -b)
  auto const field = [&](double const a, double const b, double const s) -> DisplacementField {
    return [=](dealii::Point<dim> const & X) {
      dealii::Tensor<1, dim> u;
      if(X[1] <= 0.0)
        u[1] = a * (X[1] + 0.5) / 0.5;
      else
      {
        u[0] = s;
        u[1] = -b;
      }
      return u;
    };
  };

  VectorType u;
  interpolate(dof_handler, field(0.0, 0.0, 0.0), u);
  auto const r0 = compare(verlet, u, "build at gap r_c + delta");
  check(verlet.get_statistics().n_builds == 1 and count_found(r0) == 0 and
          verlet.n_list_pairs() > 0,
        "initial list contains the pairs at distance r_c + delta, none admissible yet");

  // limit case: both surfaces move by delta / 2 towards each other, 2 D_max = delta exactly
  interpolate(dof_handler, field(delta / 2.0, delta / 2.0, 0.0), u);
  check(verlet.ratio_displacement_to_skin(u) == 1.0, "2 D_max / delta = 1 exactly");
  auto const r1 = compare(verlet, u, "2 D_max = delta, gap = r_c");
  check(verlet.get_statistics().n_builds == 1, "no rebuild at 2 D_max = delta");
  check(count_found(r1) == verlet.n_slave_points(),
        "pairs that entered the radius after the build are found (|g| = r_c)");
  bool gap_exact = true;
  for(auto const & p : r1)
    gap_exact = gap_exact and p.gap == r_c;
  check(gap_exact, "gap = r_c exactly at all slave points");

  // first violation of the certificate
  interpolate(dof_handler, field(delta / 2.0 + 1.0 / 1024.0, delta / 2.0, 0.0), u);
  compare(verlet, u, "2 D_max > delta");
  check(verlet.get_statistics().n_builds == 2, "rebuild when 2 D_max > delta");

  // compression and sliding (also over the edge of the master mesh cells)
  unsigned int const builds_before = verlet.get_statistics().n_builds;
  for(unsigned int k = 0; k <= 20; ++k)
  {
    double const s = 0.02 * k;
    interpolate(dof_handler, field(delta / 2.0 + 0.001, delta / 2.0 + r_c + 0.002, s), u);
    compare(verlet, u, "sliding s = " + std::to_string(s), k % 5 == 0);
  }
  std::cout << "    rebuilds during sliding: " << verlet.get_statistics().n_builds - builds_before
            << " (slide 0.4, delta / 2 = " << delta / 2.0 << ")" << std::endl;

  // random perturbations of all vertices (smaller than the skin)
  std::mt19937                           generator(42);
  std::uniform_real_distribution<double> distribution(-0.01, 0.01);
  for(unsigned int k = 0; k < 20; ++k)
  {
    VectorType perturbed = u;
    for(unsigned int i = 0; i < perturbed.size(); ++i)
      perturbed(i) += distribution(generator);
    compare(verlet, perturbed, "random perturbation " + std::to_string(k), k % 10 == 0);
  }

  std::cout << std::endl
            << "Test 2: Newton trial sequence accepted -> rejected -> return" << std::endl;
  {
    Contact contact;
    contact.setup(dof_handler, {}, data);

    VectorType u_accepted, u_trial;
    interpolate(dof_handler, field(delta / 2.0, delta / 2.0 + r_c + 0.002, 0.1), u_accepted);
    interpolate(dof_handler, field(delta / 2.0, delta / 2.0 + r_c + 0.002 + 0.3, 0.1), u_trial);

    compare(contact, u_accepted, "accepted iterate u_k (build)");
    compare(contact, u_trial, "rejected trial (deep penetration)");
    check(contact.get_statistics().n_builds == 2, "the trial rebuilds the list");
    double const ratio = contact.ratio_displacement_to_skin(u_accepted);
    std::cout << "    2 D_max(u_k, u_trial) / delta = " << ratio << std::endl;
    check(ratio > 1.0, "certificate of u_k is checked against the trial configuration");
    compare(contact, u_accepted, "return to u_k");
    check(contact.get_statistics().n_builds == 3, "rebuild on return");

    // trial within the skin: no rebuild, return without rebuild
    Contact contact2;
    contact2.setup(dof_handler, {}, data);
    interpolate(dof_handler, field(delta / 2.0, delta / 2.0 + r_c + 0.002 + 0.03, 0.1), u_trial);
    compare(contact2, u_accepted, "accepted iterate u_k (build)");
    compare(contact2, u_trial, "rejected trial within the skin");
    compare(contact2, u_accepted, "return to u_k");
    check(contact2.get_statistics().n_builds == 1, "no rebuild for a trial within the skin");
  }
}

/*
 * Test 3: strip [0, 4] x [0, T] (80 x 1 cells), top face (id 1) in self-contact. The strip is
 * folded upwards about the region [a, a + l] (inner radius rho = l / theta) by the angle theta;
 * for theta > pi the right arm tilts down and its tip touches the left arm.
 */
void
test_u_strip()
{
  std::cout << std::endl << "Test 3: U-shaped strip folding (self-contact)" << std::endl;

  double const L = 4.0, T = 0.1, l = 0.4, a = 0.5 * (L - l);

  dealii::Triangulation<dim> tria;
  dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                    {80, 1},
                                                    dealii::Point<dim>(0.0, 0.0),
                                                    dealii::Point<dim>(L, T));
  for(auto const & cell : tria.active_cell_iterators())
    for(unsigned int const f : cell->face_indices())
      if(cell->at_boundary(f))
        cell->face(f)->set_boundary_id(std::abs(cell->face(f)->center()[1] - T) < 1e-12 ? 1 : 0);

  dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(1), dim);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DeformableContactData data;
  data.pairs   = {{1, 1}};
  data.penalty = 100.0;
  data.radius  = 0.05; // < T
  data.skin    = 0.05;
  data.search  = ContactSearchStrategy::Verlet;

  Contact verlet;
  verlet.setup(dof_handler, {}, data);

  auto const fold = [&](double const theta) -> DisplacementField {
    double const             rho = l / theta;
    dealii::Point<dim> const C(a, T + rho);
    return [=](dealii::Point<dim> const & X) {
      double const       d = T - X[1], r = rho + d;
      dealii::Point<dim> x;
      if(X[0] <= a)
        x = X;
      else if(X[0] <= a + l)
      {
        double const phi = (X[0] - a) / rho;
        x                = C + r * dealii::Point<dim>(std::sin(phi), -std::cos(phi));
      }
      else
        x = C + r * dealii::Point<dim>(std::sin(theta), -std::cos(theta)) +
            (X[0] - a - l) * dealii::Point<dim>(std::cos(theta), std::sin(theta));
      return dealii::Tensor<1, dim>(x - X);
    };
  };

  VectorType         u;
  unsigned int       max_active = 0, n_first_ring = 0;
  unsigned int const n_states = 69;
  for(unsigned int k = 0; k < n_states; ++k)
  {
    double const theta = M_PI * (0.98 + 0.0025 * k);
    interpolate(dof_handler, fold(theta), u);
    auto const result =
      compare(verlet, u, "theta / pi = " + std::to_string(theta / M_PI), k % 8 == 0);
    max_active = std::max(max_active, count_active(result));
    for(unsigned int p = 0; p < result.size(); ++p)
      if(result[p].found() and verlet.faces_share_vertex(verlet.slave_face(p), result[p].master))
        ++n_first_ring;
  }
  std::cout << "    states: " << n_states << ", list builds: " << verlet.get_statistics().n_builds
            << ", max active points: " << max_active << std::endl;
  check(verlet.get_statistics().n_builds < n_states, "the list is reused between states");
  check(max_active > 0, "the arms of the U come into contact");
  check(n_first_ring == 0, "no spurious pairs between neighbouring faces of the smooth fold");
}

/*
 * Test 4: two elements [0, 1] x [0, T] and [1, 2] x [0, T]. The right element is rotated about the
 * common top vertex (1, T) by beta, so that the top faces fold onto each other for beta -> pi.
 */
void
test_adjacent_fold(double & fd_error)
{
  std::cout << std::endl << "Test 4: two adjacent elements folding (first ring)" << std::endl;

  double const T = 0.25;

  dealii::Triangulation<dim> tria;
  dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                    {2, 1},
                                                    dealii::Point<dim>(0.0, 0.0),
                                                    dealii::Point<dim>(2.0, T));
  for(auto const & cell : tria.active_cell_iterators())
    for(unsigned int const f : cell->face_indices())
      if(cell->at_boundary(f))
        cell->face(f)->set_boundary_id(std::abs(cell->face(f)->center()[1] - T) < 1e-12 ? 1 : 0);

  dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(1), dim);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DeformableContactData data;
  data.pairs   = {{1, 1}};
  data.penalty = 100.0;
  data.radius  = 0.1;
  data.skin    = 0.05;
  data.search  = ContactSearchStrategy::Verlet;

  Contact verlet;
  verlet.setup(dof_handler, {}, data);

  auto const hinge = [&](double const beta) -> DisplacementField {
    dealii::Point<dim> const H(1.0, T);
    return [=](dealii::Point<dim> const & X) {
      if(X[0] < 1.5)
        return dealii::Tensor<1, dim>();
      dealii::Tensor<1, dim> const r = X - H;
      dealii::Point<dim>           x(H[0] + std::cos(beta) * r[0] - std::sin(beta) * r[1],
                           H[1] + std::sin(beta) * r[0] + std::cos(beta) * r[1]);
      return dealii::Tensor<1, dim>(x - X);
    };
  };

  VectorType   u;
  unsigned int n_first_ring_active = 0;
  for(double const beta_deg : {150.0, 170.0, 175.0, 179.0, 180.0, 181.0, 185.0, 190.0})
  {
    interpolate(dof_handler, hinge(beta_deg * M_PI / 180.0), u);
    auto const result = compare(verlet, u, "beta = " + std::to_string(beta_deg));
    for(unsigned int p = 0; p < result.size(); ++p)
      if(result[p].active() and verlet.faces_share_vertex(verlet.slave_face(p), result[p].master))
        ++n_first_ring_active;
  }
  check(n_first_ring_active > 0,
        "active contact between faces sharing a vertex is detected (excluding the whole first "
        "ring would miss it)");

  // Jacobian of the self-contact at a penetrated state (common DoFs of slave and master)
  interpolate(dof_handler, hinge(185.0 * M_PI / 180.0), u);
  std::mt19937                           generator(7);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  VectorType                             v(u);
  for(unsigned int i = 0; i < v.size(); ++i)
    v(i) = distribution(generator);

  verlet.set_linearization(u);
  VectorType Jv(u);
  Jv = 0.0;
  verlet.apply_jacobian_add(Jv, v);

  fd_error = 1e100;
  std::cout << "    directional finite differences (self-contact, beta = 185 deg):" << std::endl;
  for(double const h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7})
  {
    VectorType up(u), um(u), Rp(u), Rm(u);
    up.add(h, v);
    um.add(-h, v);
    Rp = 0.0;
    Rm = 0.0;
    verlet.evaluate_residual_add(Rp, up);
    verlet.evaluate_residual_add(Rm, um);
    VectorType fd(Rp);
    fd.add(-1.0, Rm);
    fd *= 1.0 / (2.0 * h);
    fd.add(-1.0, Jv);
    double const error = fd.linfty_norm() / Jv.linfty_norm();
    fd_error           = std::min(fd_error, error);
    std::cout << "      h = " << std::scientific << std::setprecision(1) << h
              << "  |FD - Jv| / |Jv| = " << std::setprecision(3) << error << std::defaultfloat
              << std::endl;
  }
}

/*
 * Test 5: residual and Jacobian-vector product of the two blocks.
 */
void
test_mechanics(double & fd_error)
{
  std::cout << std::endl << "Test 5: residual and tangent (two blocks)" << std::endl;

  dealii::Triangulation<dim> tria;
  create_two_blocks(tria, 0.05, 2);
  dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(1), dim);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DeformableContactData data;
  data.pairs                = {{11, 10}};
  data.penalty              = 1000.0;
  data.radius               = 0.05;
  data.skin                 = 0.05;
  data.projection_tolerance = 0.01;

  Contact exhaustive, verlet;
  data.search = ContactSearchStrategy::Exhaustive;
  exhaustive.setup(dof_handler, {}, data);
  data.search = ContactSearchStrategy::Verlet;
  verlet.setup(dof_handler, {}, data);

  // compressed and shifted upper block plus a smooth non-uniform deformation of both blocks
  VectorType u;
  interpolate(
    dof_handler,
    [](dealii::Point<dim> const & X) {
      dealii::Tensor<1, dim> d;
      if(X[1] > 0.0)
      {
        d[0] = 0.13 + 0.01 * std::sin(3.0 * X[0]);
        d[1] = -0.052 + 0.003 * std::cos(5.0 * X[0]) + 0.01 * (X[1] - 0.05);
      }
      else
      {
        d[0] = 0.005 * std::sin(2.0 * X[0]) * (X[1] + 0.5);
        d[1] = 0.002 * std::sin(4.0 * X[0]) * (X[1] + 0.5) / 0.5;
      }
      return d;
    },
    u);

  std::mt19937                           generator(3);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  VectorType                             v(u);
  for(unsigned int i = 0; i < v.size(); ++i)
    v(i) = distribution(generator);

  VectorType R_e(u), R_v(u), J_e(u), J_v(u);
  R_e = 0.0;
  R_v = 0.0;
  J_e = 0.0;
  J_v = 0.0;
  exhaustive.evaluate_residual_add(R_e, u);
  verlet.evaluate_residual_add(R_v, u);
  exhaustive.set_linearization(u);
  verlet.set_linearization(u);
  exhaustive.apply_jacobian_add(J_e, v);
  verlet.apply_jacobian_add(J_v, v);

  DeformableContactQuantities<dim> const q = exhaustive.compute_quantities(u);
  std::cout << "    active points " << q.n_active << " of " << exhaustive.n_slave_points()
            << ", contact force (" << q.force[0] << ", " << q.force[1] << ")" << std::endl;

  VectorType diff(R_e);
  diff.add(-1.0, R_v);
  check(diff.linfty_norm() == 0.0 and q.n_active > 0, "residual: exhaustive == Verlet (bitwise)");
  diff = J_e;
  diff.add(-1.0, J_v);
  check(diff.linfty_norm() == 0.0, "Jacobian-vector product: exhaustive == Verlet (bitwise)");

  // reaction: the sum of the residual over the slave DoFs equals the contact force on the slave
  // (with opposite sign), and the total residual vanishes (action = reaction)
  dealii::Tensor<1, dim> total;
  std::vector<bool>      visited(R_e.size(), false);
  for(auto const & cell : dof_handler.active_cell_iterators())
    for(unsigned int vertex = 0; vertex < cell->n_vertices(); ++vertex)
      for(unsigned int c = 0; c < dim; ++c)
      {
        auto const i = cell->vertex_dof_index(vertex, c);
        if(not visited[i])
          total[c] += R_e(i);
        visited[i] = true;
      }
  std::cout << "    sum of the contact residual (" << total[0] << ", " << total[1] << ")"
            << std::endl;
  check(total.norm() < 1e-12 * q.force.norm(), "contact forces are in equilibrium");

  fd_error = 1e100;
  std::cout << "    directional finite differences:" << std::endl;
  for(double const h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7})
  {
    VectorType up(u), um(u), Rp(u), Rm(u);
    up.add(h, v);
    um.add(-h, v);
    Rp = 0.0;
    Rm = 0.0;
    exhaustive.evaluate_residual_add(Rp, up);
    auto const active_p = count_active(exhaustive.search(up, ContactSearchStrategy::Exhaustive));
    exhaustive.evaluate_residual_add(Rm, um);
    auto const active_m = count_active(exhaustive.search(um, ContactSearchStrategy::Exhaustive));
    VectorType fd(Rp);
    fd.add(-1.0, Rm);
    fd *= 1.0 / (2.0 * h);
    fd.add(-1.0, J_e);
    double const error = fd.linfty_norm() / J_e.linfty_norm();
    if(active_p == q.n_active and active_m == q.n_active)
      fd_error = std::min(fd_error, error);
    std::cout << "      h = " << std::scientific << std::setprecision(1) << h
              << "  |FD - Jv| / |Jv| = " << std::setprecision(3) << error << std::defaultfloat
              << "  active(u +- h v) = " << active_p << ", " << active_m << std::endl;
  }
}

/*
 * Contact residual of a configuration (zero vector plus the contact term).
 */
VectorType
contact_residual(Contact const & contact, VectorType const & u)
{
  VectorType R(u);
  R = 0.0;
  contact.evaluate_residual_add(R, u);
  return R;
}

/*
 * Total force and moment (about the origin, current configuration) of the contact residual.
 */
std::pair<dealii::Tensor<1, dim>, double>
force_and_moment(dealii::DoFHandler<dim> const & dof_handler,
                 VectorType const &              u,
                 VectorType const &              R)
{
  dealii::Tensor<1, dim> force;
  double                 moment = 0.0;
  std::vector<bool>      visited(R.size(), false);
  for(auto const & cell : dof_handler.active_cell_iterators())
    for(unsigned int v = 0; v < cell->n_vertices(); ++v)
    {
      auto const i0 = cell->vertex_dof_index(v, 0), i1 = cell->vertex_dof_index(v, 1);
      if(visited[i0])
        continue;
      visited[i0]     = true;
      double const x0 = cell->vertex(v)[0] + u(i0), x1 = cell->vertex(v)[1] + u(i1);
      force[0] += R(i0);
      force[1] += R(i1);
      moment += x0 * R(i1) - x1 * R(i0);
    }
  return {force, moment};
}

/*
 * Directional finite differences of the contact residual compared to the Jacobian-vector product;
 * returns the smallest relative error over the step sizes for which the contact state (active
 * set and master faces) of u +- h v equals that of u.
 */
double
finite_difference_check(Contact const & contact, VectorType const & u, unsigned int const seed)
{
  std::mt19937                           generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  VectorType                             v(u);
  for(unsigned int i = 0; i < v.size(); ++i)
    v(i) = distribution(generator);

  auto const state = contact.search(u, ContactSearchStrategy::Exhaustive);
  contact.set_linearization(u);
  VectorType Jv(u);
  Jv = 0.0;
  contact.apply_jacobian_add(Jv, v);

  double best = 1e100;
  for(double const h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7})
  {
    VectorType up(u), um(u);
    up.add(h, v);
    um.add(-h, v);
    auto const state_p = contact.search(up, ContactSearchStrategy::Exhaustive);
    auto const state_m = contact.search(um, ContactSearchStrategy::Exhaustive);
    bool       same    = true;
    for(unsigned int p = 0; p < state.size(); ++p)
      same = same and state[p].active() == state_p[p].active() and
             state[p].active() == state_m[p].active() and
             (not state[p].active() or
              (state[p].master == state_p[p].master and state[p].master == state_m[p].master));
    VectorType fd = contact_residual(contact, up);
    fd.add(-1.0, contact_residual(contact, um));
    fd *= 1.0 / (2.0 * h);
    fd.add(-1.0, Jv);
    double const error = fd.linfty_norm() / Jv.linfty_norm();
    if(same)
      best = std::min(best, error);
    std::cout << "      h = " << std::scientific << std::setprecision(1) << h
              << "  |FD - Jv| / |Jv| = " << std::setprecision(3) << error << std::defaultfloat
              << (same ? "" : "  (contact state changes, excluded)") << std::endl;
  }
  return best;
}

/*
 * Test 6: continuous master normal (smooth_normals).
 */
void
test_smooth_normals()
{
  std::cout << std::endl << "Test 6: continuous master normal" << std::endl;

  dealii::Triangulation<dim> tria;
  create_two_blocks(tria, 0.05, 2);
  dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(1), dim);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DeformableContactData data;
  data.pairs                = {{11, 10}};
  data.penalty              = 100.0;
  data.radius               = 0.05;
  data.skin                 = 0.05;
  data.projection_tolerance = 0.01;
  data.search               = ContactSearchStrategy::Exhaustive;

  Contact segment, smooth, smooth_verlet;
  segment.setup(dof_handler, {}, data);
  data.smooth_normals = true;
  smooth.setup(dof_handler, {}, data);
  data.search = ContactSearchStrategy::Verlet;
  smooth_verlet.setup(dof_handler, {}, data);

  // (a) straight master surface: same projection, gap and residual as the segment normal
  {
    VectorType u;
    interpolate(
      dof_handler,
      [](dealii::Point<dim> const & X) {
        dealii::Tensor<1, dim> d;
        if(X[1] > 0.0)
        {
          d[0] = 0.13;
          d[1] = -0.052;
        }
        return d;
      },
      u);
    auto const a      = segment.search(u, ContactSearchStrategy::Exhaustive);
    auto const b      = smooth.search(u, ContactSearchStrategy::Exhaustive);
    double     max_xi = 0.0, max_gap = 0.0;
    bool       same_faces = true;
    for(unsigned int p = 0; p < a.size(); ++p)
    {
      same_faces = same_faces and a[p].master == b[p].master;
      if(a[p].found() and b[p].found())
      {
        max_xi  = std::max(max_xi, std::abs(a[p].xi - b[p].xi));
        max_gap = std::max(max_gap, std::abs(a[p].gap - b[p].gap));
      }
    }
    // slave side: identical forces; total force identical; master side: patch test (uniform
    // pressure p = eps b gives the nodal force p h at the interior master nodes)
    VectorType const R_seg = contact_residual(segment, u), R_smo = contact_residual(smooth, u);
    double const     p = data.penalty * 0.002, h = 0.25 / 4.0;
    double           max_slave = 0.0, max_patch_smooth = 0.0, max_patch_segment = 0.0;
    dealii::Tensor<1, dim> total_seg, total_smo;
    std::vector<bool>      visited(u.size(), false);
    for(auto const & cell : dof_handler.active_cell_iterators())
      for(unsigned int v = 0; v < cell->n_vertices(); ++v)
      {
        auto const i0 = cell->vertex_dof_index(v, 0), i1 = cell->vertex_dof_index(v, 1);
        if(visited[i0])
          continue;
        visited[i0]                = true;
        dealii::Point<dim> const X = cell->vertex(v);
        for(unsigned int c = 0; c < dim; ++c)
        {
          total_seg[c] += R_seg(cell->vertex_dof_index(v, c));
          total_smo[c] += R_smo(cell->vertex_dof_index(v, c));
        }
        if(X[1] > 0.025)
          max_slave =
            std::max(max_slave,
                     std::max(std::abs(R_seg(i0) - R_smo(i0)), std::abs(R_seg(i1) - R_smo(i1))));
        // interior master nodes: the support of their contact terms (two faces on each side)
        // lies within the contact zone [0.63, 1.63]
        else if(std::abs(X[1]) < 1e-12 and X[0] > 0.63 + 2.0 * h and X[0] < 1.63 - 2.0 * h)
        {
          max_patch_smooth =
            std::max(max_patch_smooth, std::max(std::abs(R_smo(i1) - p * h), std::abs(R_smo(i0))));
          max_patch_segment = std::max(max_patch_segment, std::abs(R_seg(i1) - p * h));
        }
      }
    std::cout << "    straight surface: active " << count_active(b) << ", max |dxi| " << max_xi
              << ", max |dg| " << max_gap << ", slave forces max |dR| " << max_slave
              << ", total forces (" << total_seg[1] << " / " << total_smo[1] << ")" << std::endl
              << "    patch test (master nodal force p h = " << p * h << "): max deviation segment "
              << max_patch_segment << ", smooth " << max_patch_smooth << std::endl;
    check(same_faces and count_active(b) > 0 and max_xi < 1e-13 and max_gap < 1e-15 and
            max_slave < 1e-14 * p * h and (total_seg - total_smo).norm() < 1e-14,
          "straight master surface: smooth == segment for projection, gap and slave forces");
    // GPTS does not pass the patch test exactly on non-matching meshes; the smoothed surface must
    // not be worse than the segment formulation
    check(max_patch_smooth <= max_patch_segment,
          "smooth normals: patch test deviation not larger than with segment normals");
  }

  // curved and non-uniformly deformed surfaces (as in test 5)
  VectorType u;
  interpolate(
    dof_handler,
    [](dealii::Point<dim> const & X) {
      dealii::Tensor<1, dim> d;
      if(X[1] > 0.0)
      {
        d[0] = 0.13 + 0.01 * std::sin(3.0 * X[0]);
        d[1] = -0.052 + 0.003 * std::cos(5.0 * X[0]) + 0.01 * (X[1] - 0.05);
      }
      else
      {
        d[0] = 0.005 * std::sin(2.0 * X[0]) * (X[1] + 0.5);
        d[1] = 0.02 * std::sin(4.0 * X[0]) * (X[1] + 0.5) / 0.5;
      }
      return d;
    },
    u);

  // (b) Verlet == exhaustive with smooth normals
  compare(smooth_verlet, u, "smooth normals, curved surfaces");
  std::mt19937                           generator(11);
  std::uniform_real_distribution<double> distribution(-0.005, 0.005);
  for(unsigned int k = 0; k < 10; ++k)
  {
    VectorType perturbed = u;
    for(unsigned int i = 0; i < perturbed.size(); ++i)
      perturbed(i) += distribution(generator);
    compare(smooth_verlet, perturbed, "smooth normals, perturbation " + std::to_string(k), k == 0);
  }

  // (c) tangent: finite differences
  std::cout << "    finite differences, two blocks, curved master:" << std::endl;
  double const fd_blocks = finite_difference_check(smooth, u, 5);
  check(fd_blocks < 1e-6,
        "smooth normals: tangent of the two blocks agrees with finite differences");

  // (d) equilibrium: total force and moment of the contact residual vanish
  for(Contact const * contact : {&segment, &smooth})
  {
    VectorType const R  = contact_residual(*contact, u);
    auto const       fm = force_and_moment(dof_handler, u, R);
    std::cout << "    " << (contact == &smooth ? "smooth " : "segment")
              << ": |sum R| / |R| = " << fm.first.norm() / R.linfty_norm()
              << ", |moment| / |R| = " << std::abs(fm.second) / R.linfty_norm() << std::endl;
    check(fm.first.norm() < 1e-12 * R.linfty_norm() and
            std::abs(fm.second) < 1e-12 * R.linfty_norm(),
          std::string(contact == &smooth ? "smooth" : "segment") +
            " normals: contact forces and moments are in equilibrium");
  }

  // (e) continuity: slave points sliding over a convex edge (kink at the master vertex x = 1)
  {
    double const alpha = 0.05, b = 0.002;
    auto const   field = [&](double const s) -> DisplacementField {
      return [=](dealii::Point<dim> const & X) {
        dealii::Tensor<1, dim> d;
        if(X[1] > 0.0)
        {
          d[0] = s;
          d[1] = -0.05 - b;
        }
        else
          d[1] = -alpha * std::abs(X[0] - 1.0) * (X[1] + 0.5) / 0.5;
        return d;
      };
    };
    for(Contact const * contact : {&segment, &smooth})
    {
      std::array<double, 2>       max_jump;
      std::array<double, 2> const steps = {{1e-4, 1e-5}};
      for(unsigned int k = 0; k < 2; ++k)
      {
        max_jump[k] = 0.0;
        VectorType u_old, u_new;
        interpolate(dof_handler, field(-0.03), u_old);
        VectorType R_old = contact_residual(*contact, u_old);
        for(double s = -0.03 + steps[k]; s <= 0.03; s += steps[k])
        {
          interpolate(dof_handler, field(s), u_new);
          VectorType R_new = contact_residual(*contact, u_new);
          VectorType diff(R_new);
          diff.add(-1.0, R_old);
          max_jump[k] = std::max(max_jump[k], diff.linfty_norm());
          R_old       = R_new;
        }
      }
      bool const smooth_case = (contact == &smooth);
      std::cout << "    " << (smooth_case ? "smooth " : "segment")
                << ": max |R(s + ds) - R(s)| = " << max_jump[0] << " (ds = 1e-4), " << max_jump[1]
                << " (ds = 1e-5), ratio " << max_jump[0] / max_jump[1] << std::endl;
      if(smooth_case)
        check(max_jump[0] / max_jump[1] > 5.0,
              "smooth normals: the residual is continuous across the master vertex");
      else
        check(max_jump[0] / max_jump[1] < 2.0,
              "segment normals: the residual jumps at the master vertex (reference behaviour)");
    }
  }
}

/*
 * Test 7: smooth normals in self-contact (U strip, averaging active on the curved inner surface).
 */
void
test_smooth_self_contact(double & fd_error)
{
  std::cout << std::endl << "Test 7: continuous master normal, self-contact (U strip)" << std::endl;

  double const L = 4.0, T = 0.1, l = 0.4, a = 0.5 * (L - l);

  dealii::Triangulation<dim> tria;
  dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                    {80, 1},
                                                    dealii::Point<dim>(0.0, 0.0),
                                                    dealii::Point<dim>(L, T));
  for(auto const & cell : tria.active_cell_iterators())
    for(unsigned int const f : cell->face_indices())
      if(cell->at_boundary(f))
        cell->face(f)->set_boundary_id(std::abs(cell->face(f)->center()[1] - T) < 1e-12 ? 1 : 0);
  dealii::FESystem<dim>   fe(dealii::FE_Q<dim>(1), dim);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DeformableContactData data;
  data.pairs          = {{1, 1}};
  data.penalty        = 100.0;
  data.radius         = 0.05;
  data.skin           = 0.05;
  data.search         = ContactSearchStrategy::Verlet;
  data.smooth_normals = true;
  Contact contact;
  contact.setup(dof_handler, {}, data);

  // fold of the strip by theta (see test 3) plus a smooth bending of the arms
  double const       theta = 1.06 * M_PI, rho = l / theta;
  dealii::Point<dim> C(a, T + rho);
  VectorType         u;
  interpolate(
    dof_handler,
    [&](dealii::Point<dim> const & X) {
      double const       d = T - X[1], r = rho + d;
      dealii::Point<dim> x;
      if(X[0] <= a)
        x = X;
      else if(X[0] <= a + l)
      {
        double const phi = (X[0] - a) / rho;
        x                = C + r * dealii::Point<dim>(std::sin(phi), -std::cos(phi));
      }
      else
        x = C + r * dealii::Point<dim>(std::sin(theta), -std::cos(theta)) +
            (X[0] - a - l) * dealii::Point<dim>(std::cos(theta), std::sin(theta));
      x[1] += 0.01 * std::sin(2.0 * X[0]);
      return dealii::Tensor<1, dim>(x - X);
    },
    u);
  auto const result = compare(contact, u, "U strip, theta / pi = 1.06");
  check(count_active(result) > 0, "smooth normals: the arms of the U are in contact");
  std::cout << "    finite differences:" << std::endl;
  fd_error = finite_difference_check(contact, u, 9);
}

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test_two_blocks(0.0);
  test_two_blocks(0.05);
  test_u_strip();
  double fd_fold = 0.0, fd_blocks = 0.0;
  test_adjacent_fold(fd_fold);
  test_mechanics(fd_blocks);
  test_smooth_normals();
  double fd_smooth_u = 0.0;
  test_smooth_self_contact(fd_smooth_u);

  std::cout << std::endl;
  check(fd_blocks < 1e-6, "tangent of the two blocks agrees with finite differences");
  check(fd_fold < 1e-6, "tangent of the adjacent fold agrees with finite differences");
  check(fd_smooth_u < 1e-6,
        "smooth normals: tangent of the self-contact (U strip) agrees with finite differences");

  std::cout << std::endl
            << (n_failures == 0 ? "All tests passed." :
                                  std::to_string(n_failures) + " test(s) failed.")
            << std::endl;
  return n_failures == 0 ? 0 : 1;
}
