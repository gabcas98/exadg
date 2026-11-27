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

#ifndef EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_DEFORMABLE_CONTACT_H_
#define EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_DEFORMABLE_CONTACT_H_

// C/C++
#include <array>
#include <limits>
#include <set>
#include <utility>
#include <vector>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#endif

// ExaDG
#include <exadg/structure/user_interface/enum_types.h>

namespace ExaDG
{
namespace Structure
{
/*
 * Frictionless contact between deformable surfaces (two bodies or self-contact) of the same
 * mesh, penalty method, Gauss-point-to-segment (GPTS) discretization, one pass:
 *
 *  - slave entities: Gauss points x_s of the slave faces (weights w_s of the reference
 *    configuration, the contact pressure is a force per undeformed area as for the rigid
 *    obstacles),
 *  - master entities: the master faces (straight segments [x_1, x_2] in the deformed
 *    configuration with outward unit normal n_m).
 *
 * A pair (x_s, F_m) is admissible if
 *  (i)   F_m is not the face of x_s (the only incidence that is unavoidable for Gauss points; the
 *        neighbours of the slave face are not excluded, so that a fold of adjacent elements is
 *        detected),
 *  (ii)  the projection xi = (x_s - x_1) . (x_2 - x_1) / |x_2 - x_1|^2 lies in [-tau, 1 + tau]
 *        (the projection tolerance tau > 0 closes the gaps between the projection domains of two
 *        segments at convex vertices of the master surface, where a sliding point would otherwise
 *        lose and regain contact and Newton's method stagnates),
 *  (iii) the normals face each other, n_s . n_m < 0,
 *  (iv)  |g| <= r_c with the signed normal gap g = (x_s - x_1) . n_m (for an admissible
 *        projection, the Euclidean distance of x_s and F_m is at most |g| + tau l with the length
 *        l of F_m, and equals |g| for xi in [0, 1]).
 * The admissible pair with the smallest |g| (ties: smallest face index) is selected, the point
 * is active if g <= 0 and the contact pressure is lambda = epsilon <-g>_+. The residual is the
 * derivative of the penalty potential (epsilon / 2) sum_s w_s <-g_s>_+^2 and the tangent its
 * second derivative, including the variation of n_m and xi (symmetric).
 *
 * The detection radius r_c is part of the formulation: it bounds the penetration that can be
 * detected (choose r_c > max penetration, of order lambda / epsilon) and must be smaller than
 * the thickness of the bodies, otherwise the opposite surface of the same body is admissible
 * (self-contact). Both search strategies below use the identical admissibility test and
 * selection, so they give bitwise identical results whenever the Verlet list is valid.
 *
 * Search strategies:
 *  - Exhaustive: all master faces are tested for each slave point (reference),
 *  - Verlet: candidate master faces per slave face, built (exhaustively, with axis-aligned
 *    bounding boxes) at the configuration u_*. The list contains all pairs of slave face F_s and
 *    master face F_m with dist(F_s(u_*), F_m(u_*)) <= R_m = r_c + delta + tau (l_m(u_*) + delta).
 *    For linear geometry, every point of a contact face moves at most D_max = max_a |x_a(u) -
 *    x_a(u_*)| (vertices a of all contact faces), hence dist(F_s(u), F_m(u)) >= dist(F_s(u_*),
 *    F_m(u_*)) - 2 D_max and l_m(u) <= l_m(u_*) + 2 D_max. As long as 2 D_max <= delta, the list
 *    therefore contains all pairs with distance <= r_c + tau l_m(u) at u, which includes all
 *    admissible pairs. This certificate is checked before every query (residual and
 *    linearization), against the configuration of the last build, and the list is rebuilt if it
 *    is violated.
 *
 * Three states with different lifetimes:
 *  - topology (setup): contact faces, vertices with their DoF indices and constraints, slave
 *    Gauss points, adjacent cell and local face number (as required for the traction of a
 *    Nitsche variant),
 *  - Verlet list: candidates, vertex positions of u_*, skin,
 *  - linearization state: projection, normal, gap and active set at u_k, used by all
 *    Jacobian-vector products J(u_k) v.
 *
 * Limitations of this first version: 2d, FE_Q(1) with a linear mapping (straight faces, so
 * that vertex motion bounds the motion of the faces), serial (the opposite surface is accessed
 * through global DoF indices), penalty method without friction. The deformable contact is not
 * included in the preconditioner (matrix-based assembly, multigrid levels).
 */
struct DeformableContactData
{
  // pairs (slave boundary id, master boundary id); slave == master: self-contact
  std::vector<std::pair<dealii::types::boundary_id, dealii::types::boundary_id>> pairs;

  // penalty parameter epsilon (force / area / length)
  double penalty = 0.0;

  // detection radius r_c, see above
  double radius = 0.0;

  // Verlet skin delta
  double skin = 0.0;

  // projection tolerance tau, see above
  double projection_tolerance = 0.0;

  ContactSearchStrategy search = ContactSearchStrategy::Verlet;

  // Verlet: repeat every query with the exhaustive search and count differences (diagnostic)
  bool verify = false;

  // number of Gauss points per slave face (0: degree + 1)
  unsigned int n_q_points = 0;

  // add the contact stiffness to the matrix of the preconditioner (AMG), see add_to_matrix()
  bool add_to_preconditioner = false;

  // include the slave-master coupling in the preconditioner matrix (requires an extension of the
  // sparsity pattern, see add_couplings_to_pattern()); otherwise only the slave-slave and
  // master-master blocks are added
  bool preconditioner_coupling = true;

  // print one line per linearization with the changes of the contact state (diagnostics)
  bool log_linearization = false;

  /*
   * Smoothed master surface (B2): on each master face, a Hermite cubic through its vertices whose
   * tangents are perpendicular to the nodal normals nu_a (normalized sum of the unit outward
   * normals of the two master faces adjacent to vertex a), and closest point projection onto this
   * curve. The surface is G1, so the gap and its gradient dg/dx_s = n (the curve normal, i.e. the
   * direction of the contact force) are continuous across master vertices. A vertex is a corner (no
   * averaging, nu_a = normal of the face) if the angle between the two face normals exceeds
   * smoothing_angle in the reference configuration (corners of the body) or in the current
   * configuration (folds). On straight surfaces, xi and g coincide with the segment formulation.
   * Residual and tangent are the first and second derivatives of the penalty potential, computed by
   * forward automatic differentiation (hyper-dual numbers) with respect to the up to six vertices
   * that g depends on. The Verlet certificate includes the deviation of the curve from the chord
   * (<= 0.16 l), which requires smoothing_angle <= pi/3 and projection_tolerance <= 0.05.
   */
  // multigrid: re-discretize the contact on the multigrid levels (see MultigridPreconditioner)
  bool on_multigrid_levels = true;

  // Jacobian without the geometric part - lambda w d^2g/du^2, i.e. the positive semi-definite
  // part epsilon w (dg/du) (x) (dg/du) only (used on the multigrid levels, where the operator
  // serves as a preconditioner and has to be positive definite also for large penetrations)
  bool penalty_part_only = false;

  bool   smooth_normals  = false;
  double smoothing_angle = 1.0471975511965976; // pi / 3
};

/*
 * Counters and accumulated wall times (seconds) of the deformable contact.
 */
struct DeformableContactStatistics
{
  // calls
  unsigned long n_residuals       = 0;
  unsigned long n_linearizations  = 0;
  unsigned long n_lin_reused      = 0; // linearization at the state of the last residual
  unsigned long n_vmults          = 0;
  unsigned long n_validity_checks = 0;
  unsigned long n_builds          = 0;

  // search
  unsigned long n_queries     = 0; // slave point queries
  unsigned long n_projections = 0; // point-segment projections
  unsigned long n_prefiltered = 0; // smoothed surface: rejected by the distance to the chord
  unsigned long n_candidates  = 0; // sum of candidates over all Verlet queries
  unsigned long n_list_pairs  = 0; // sum of face pairs in the list over all builds
  unsigned long n_active      = 0; // sum of active points over all queries
  unsigned long n_ties        = 0; // queries with two admissible faces at the same |g|
  unsigned long n_cutoff      = 0; // queries without admissible face within r_c, but with an
                                   // active admissible face beyond r_c (only counted by
                                   // exhaustive searches)

  // changes of the contact state between consecutive linearizations: points that became active
  // or inactive, active points whose master face changed, maximum angle between the old and the
  // new master normal of these points, active points with a projection in the tolerance zone
  // (xi < 0 or xi > 1) and minimum |g| of the points that changed their status
  unsigned long n_activated         = 0;
  unsigned long n_deactivated       = 0;
  unsigned long n_switched          = 0;
  double        max_switch_angle    = 0.0;
  unsigned long n_in_tolerance_zone = 0;
  double        min_abs_gap_changed = std::numeric_limits<double>::max();

  // preconditioner matrix and extensions of its sparsity pattern
  unsigned long n_matrix_updates    = 0;
  double        t_matrix            = 0.0;
  unsigned long n_pattern_updates   = 0;
  unsigned long n_pattern_couplings = 0;

  // verification (ContactSearchStrategy::Verlet with verify = true)
  unsigned long n_verified          = 0;
  unsigned long n_verify_mismatches = 0;
  double        max_ratio_D_to_skin = 0.0; // max of 2 D_max / delta over all checks

  // wall times
  double t_build    = 0.0;
  double t_check    = 0.0;
  double t_search   = 0.0; // queries incl. projections (without verification)
  double t_verify   = 0.0;
  double t_residual = 0.0; // assembly of the residual contributions
  double t_lin      = 0.0; // storage of the linearization state (without search)
  double t_vmult    = 0.0; // contact part of the Jacobian-vector products
};

/*
 * Result of the search for one slave point.
 */
template<int dim>
struct ContactProjection
{
  unsigned int           master = dealii::numbers::invalid_unsigned_int; // face index
  double                 xi     = 0.0;
  double                 gap    = std::numeric_limits<double>::max();
  double                 length = 0.0; // length of the master segment
  dealii::Tensor<1, dim> normal;       // outward normal of the master face
  dealii::Tensor<1, dim> tangent;      // (x_2 - x_1) / length

  bool
  found() const
  {
    return master != dealii::numbers::invalid_unsigned_int;
  }

  bool
  active() const
  {
    return found() and gap <= 0.0;
  }
};

/*
 * Integral quantities of the deformable contact.
 */
template<int dim>
struct DeformableContactQuantities
{
  // force exerted by the master surfaces on the slave surfaces, sum_s w_s lambda_s n_m
  dealii::Tensor<1, dim> force;
  unsigned int           n_active     = 0;
  double                 active_area  = 0.0;
  double                 min_gap      = std::numeric_limits<double>::max();
  double                 max_pressure = 0.0;
};

template<int dim, typename Number>
class DeformableContact
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  /*
   * Topology: collects the contact faces, their vertices (DoF indices of the displacement
   * components at the vertices) and the slave Gauss points. Constrained DoFs (global indices) are
   * skipped when writing into residuals and Jacobian-vector products, and constrained entries of
   * the direction of a Jacobian-vector product are treated as zero (as in OperatorBase::apply()).
   */
  void
  setup(dealii::DoFHandler<dim> const &                      dof_handler,
        std::vector<dealii::types::global_dof_index> const & constrained_dofs,
        DeformableContactData const &                        data);

  /*
   * dst += R_c(u). The displacement u has to contain the inhomogeneous Dirichlet values.
   */
  void
  evaluate_residual_add(VectorType & dst, VectorType const & u) const;

  /*
   * Search at u_k and storage of the linearization state.
   */
  void
  set_linearization(VectorType const & u) const;

  /*
   * dst += J_c(u_k) v with the linearization state of the last call of set_linearization().
   */
  void
  apply_jacobian_add(VectorType & dst, VectorType const & v) const;

  /*
   * diagonal += diagonal of J_c(u_k) (unconstrained DoFs), consistent with apply_jacobian_add()
   * (point Jacobi smoother, e.g. Chebyshev in multigrid).
   */
  void
  add_diagonal(VectorType & diagonal) const;

  /*
   * Sparsity pattern of the slave-master coupling for the preconditioner matrix: the pattern
   * contains the couplings of all vertex pairs (slave face vertex, master face vertex) of the
   * active points of all linearization states seen so far (monotonically growing set).
   * pattern_contains_couplings() returns false if the current linearization state needs couplings
   * that are not in this set; add_couplings_to_pattern() adds them to the set and writes all
   * couplings of the set into the given pattern.
   */
  bool
  pattern_contains_couplings() const;

  void
  add_couplings_to_pattern(dealii::DynamicSparsityPattern & dsp) const;

#ifdef DEAL_II_WITH_TRILINOS
  /*
   * Adds the part epsilon w (dg/du) (x) (dg/du) of the contact tangent at the linearization state
   * (rank one and positive semi-definite per point) to the matrix of the preconditioner;
   * constrained rows and columns are skipped and the geometric part - lambda w d^2 g/du^2 is
   * omitted. With DeformableContactData::preconditioner_coupling, the full block including the
   * slave-master coupling is added (the pattern must contain the couplings, see
   * add_couplings_to_pattern()); otherwise only the slave-slave and master-master blocks. Requires
   * a previous call of set_linearization() (no-op otherwise).
   */
  void
  add_to_matrix(dealii::TrilinosWrappers::SparseMatrix & matrix) const;
#endif

  DeformableContactQuantities<dim>
  compute_quantities(VectorType const & u) const;

  /*
   * Search for all slave points at u with the given strategy (for tests and diagnostics; the
   * Verlet strategy checks and possibly rebuilds the list).
   */
  std::vector<ContactProjection<dim>>
  search(VectorType const & u, ContactSearchStrategy const strategy) const;

  /*
   * 2 D_max(u, u_*) / delta for the current list (infinity if no list has been built).
   */
  double
  ratio_displacement_to_skin(VectorType const & u) const;

  // number of slave points, master faces and current face pairs in the Verlet list
  unsigned int
  n_slave_points() const;

  unsigned int
  n_list_pairs() const;

  // slave point data (face index, current position) for tests
  unsigned int
  slave_face(unsigned int const point) const;

  bool
  faces_share_vertex(unsigned int const face_a, unsigned int const face_b) const;

  dealii::Point<dim>
  slave_point_position(VectorType const & u, unsigned int const point) const;

  /*
   * Smooth normals: maximum relative difference of gap, gradient and Hessian between the
   * one-pass automatic differentiation and the reference implementation over the active points
   * at u (test).
   */
  double
  verify_smooth_derivatives(VectorType const & u) const;

  DeformableContactData const &
  get_data() const;

  DeformableContactStatistics const &
  get_statistics() const;

  void
  reset_statistics() const;

  void
  print_statistics(dealii::ConditionalOStream const & pcout) const;

private:
  /*
   * State 1: topology.
   */
  struct Vertex
  {
    dealii::Point<dim>                               X;
    std::array<dealii::types::global_dof_index, dim> dof;
    std::array<bool, dim>                            constrained;
  };

  struct Face
  {
    dealii::types::boundary_id  boundary_id;
    std::array<unsigned int, 2> vertex;
    // outward normal n = orientation * (t_1, -t_0) with t = (x_2 - x_1) / |x_2 - x_1|
    double orientation;
    // adjacent cell and local face number (data of the cell needed for tractions)
    typename dealii::DoFHandler<dim>::active_cell_iterator cell;
    unsigned int                                           face_no;
    // neighbouring face (same boundary id) at vertex[v] used for the nodal normal of the smooth
    // normal formulation (invalid: corner or end of the surface)
    std::array<unsigned int, 2> neighbor = {
      {dealii::numbers::invalid_unsigned_int, dealii::numbers::invalid_unsigned_int}};
  };

  // vertex displacements d_a and positions x_a = X_a + d_a of a configuration
  struct Configuration
  {
    std::vector<dealii::Point<dim>>     x;
    std::vector<dealii::Tensor<1, dim>> d;
  };

  /*
   * Smooth normal formulation: local vertices that the gap of a slave point and a master face
   * depends on (slave face, master face, other vertices of the neighbouring master faces; each
   * vertex once), and the local indices of the roles.
   */
  struct LocalTopology
  {
    unsigned int                n = 0;
    std::array<unsigned int, 6> vertex;
    unsigned int                slave_a, slave_b, master_1, master_2;
    double                      eta;
    double                      orientation;
    // neighbouring faces at master_1 (index 0) and master_2 (index 1): local indices of their
    // vertices in the order of the face, and orientation
    std::array<bool, 2>                        has_neighbor;
    std::array<std::array<unsigned int, 2>, 2> neighbor_vertex;
    std::array<double, 2>                      neighbor_orientation;
  };

  // linearization of the gap of an active point (smooth normal formulation)
  struct SmoothLinearization
  {
    LocalTopology           topology;
    std::array<double, 12>  gradient;
    std::array<double, 144> hessian;
  };

  LocalTopology
  local_topology(unsigned int const point, unsigned int const master) const;

  // gap of a slave point and a master face as a function of the local vertex positions X (number
  // type T, e.g. for automatic differentiation)
  template<typename T>
  T
  smooth_gap_of_positions(LocalTopology const &                   topology,
                          std::array<std::array<T, 2>, 6> const & X) const;

  // derivatives in one forward pass with respect to the local DoFs directions[0, n_directions)
  // (N >= n_directions)
  template<int N>
  void
  smooth_gap_derivatives_fixed(LocalTopology const &                topology,
                               Configuration const &                x,
                               std::array<unsigned int, 12> const & directions,
                               unsigned int const                   n_directions,
                               bool const                           hessian,
                               double &                             gap,
                               std::array<double, 12> &             gradient,
                               std::array<double, 144> *            hessian_values) const;

  // reference implementation: one hyper-dual evaluation per pair of directions (used to verify
  // smooth_gap_derivatives(), see verify_smooth_derivatives())
  void
  smooth_gap_derivatives_reference(LocalTopology const &     topology,
                                   Configuration const &     x,
                                   bool const                hessian,
                                   double &                  gap,
                                   std::array<double, 12> &  gradient,
                                   std::array<double, 144> * hessian_values) const;

  // gap of a slave point and a master face and its gradient (and Hessian) with respect to the
  // local vertex coordinates (smooth normal formulation); unless all_directions, only with respect
  // to the unconstrained local DoFs (the other entries are zero)
  void
  smooth_gap_derivatives(LocalTopology const &     topology,
                         Configuration const &     x,
                         bool const                hessian,
                         double &                  gap,
                         std::array<double, 12> &  gradient,
                         std::array<double, 144> * hessian_values,
                         bool const                all_directions = false) const;

  // true if all components of the vertex are constrained (no matrix entries)
  bool
  fully_constrained(unsigned int const vertex) const;

  // J_c v of one active point at the linearization state: read(vertex) returns the direction at a
  // vertex, write(vertex, value) receives the contribution
  template<typename Read, typename Write>
  void
  apply_point_jacobian(unsigned int const point, Read const & read, Write const & write) const;

  // local vertices of the contact term of an active point (for the sparsity pattern)
  std::vector<unsigned int>
  point_vertices(unsigned int const point, ContactProjection<dim> const & state) const;

  struct SlavePoint
  {
    unsigned int slave_face; // index into slave_faces
    double       eta;        // x_s = (1 - eta) x_a + eta x_b
    double       weight;     // reference weight w_s
  };

  struct SlaveFace
  {
    unsigned int face;
    unsigned int pair;
    unsigned int first_point;
    unsigned int n_points;
  };

  struct Pair
  {
    dealii::types::boundary_id slave, master;
    std::vector<unsigned int>  master_faces;
  };


  void
  compute_positions(VectorType const & u, Configuration & x) const;

  dealii::Point<dim>
  slave_position(Configuration const & x, unsigned int const point) const;

  dealii::Tensor<1, dim>
  face_normal(Configuration const & x, unsigned int const face) const;

  // dst += value at the unconstrained components of the vertex
  void
  add(VectorType & dst, unsigned int const vertex, dealii::Tensor<1, dim> const & value) const;

  double
  max_displacement_since_build(Configuration const & x) const;

  /*
   * Admissibility test (ii) - (iii) and projection; returns false if not admissible. The
   * radius test (iv) is applied by the caller.
   */
  bool
  project(dealii::Point<dim> const &     x_s,
          dealii::Tensor<1, dim> const & n_s,
          Configuration const &          x,
          unsigned int const             master,
          ContactProjection<dim> &       result) const;

  // returns true if a is preferable to b
  static bool
  closer(ContactProjection<dim> const & a, ContactProjection<dim> const & b);

  // count: update the counters of ties and cutoff hits
  ContactProjection<dim>
  query_exhaustive(Configuration const & x, unsigned int const point, bool const count) const;

  ContactProjection<dim>
  query_verlet(Configuration const & x, unsigned int const point) const;

  /*
   * Checks the certificate 2 D_max <= delta and rebuilds the list if it is violated.
   */
  void
  ensure_valid_list(Configuration const & x) const;

  void
  build_list(Configuration const & x) const;

  void
  search_all(Configuration const &                 x,
             ContactSearchStrategy const           strategy,
             std::vector<ContactProjection<dim>> & result) const;

  DeformableContactData data;

  std::vector<Vertex>     vertices;
  std::vector<Face>       faces;
  std::vector<Pair>       pairs;
  std::vector<SlaveFace>  slave_faces;
  std::vector<SlavePoint> slave_points;

  /*
   * State 2: Verlet list (CSR: candidates of slave face i in
   * [candidate_offsets[i], candidate_offsets[i+1])).
   */
  mutable std::vector<unsigned int>           candidate_offsets;
  mutable std::vector<unsigned int>           candidates;
  mutable std::vector<dealii::Tensor<1, dim>> build_displacement;
  mutable bool                                list_built = false;

  /*
   * State 3: linearization state at u_k.
   */
  mutable Configuration                       lin_positions;
  mutable std::vector<ContactProjection<dim>> lin_state;
  mutable std::vector<SmoothLinearization>    lin_smooth;

  // positions and search result of the last residual evaluation (reused by set_linearization()
  // if the positions are identical)
  mutable Configuration                       last_positions;
  mutable std::vector<ContactProjection<dim>> last_state;

  // vertex pairs (slave vertex, master vertex) contained in the pattern of the preconditioner
  mutable std::set<std::pair<unsigned int, unsigned int>> pattern_couplings;

  mutable DeformableContactStatistics stats;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_DEFORMABLE_CONTACT_H_ */
