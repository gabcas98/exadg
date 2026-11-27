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

// C/C++
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe.h>

// ExaDG
#include <exadg/structure/spatial_discretization/operators/deformable_contact.h>

namespace ExaDG
{
namespace Structure
{
namespace
{
class ScopedTime
{
public:
  ScopedTime(double & accumulator)
    : accumulator(accumulator), start(std::chrono::steady_clock::now())
  {
  }

  ~ScopedTime()
  {
    accumulator += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }

private:
  double &                                           accumulator;
  std::chrono::time_point<std::chrono::steady_clock> start;
};

template<int dim>
dealii::Tensor<1, dim>
rotate(dealii::Tensor<1, dim> const & t)
{
  // (t_1, -t_0), only used in 2d
  dealii::Tensor<1, dim> r;
  if constexpr(dim == 2)
  {
    r[0] = t[1];
    r[1] = -t[0];
  }
  else
  {
    (void)t;
    AssertThrow(false, dealii::ExcNotImplemented());
  }
  return r;
}
} // namespace

template<int dim, typename Number>
void
DeformableContact<dim, Number>::setup(
  dealii::DoFHandler<dim> const &                      dof_handler,
  std::vector<dealii::types::global_dof_index> const & constrained_dofs,
  DeformableContactData const &                        data_in)
{
  data = data_in;

  AssertThrow(dim == 2,
              dealii::ExcMessage("Contact between deformable surfaces is only implemented in 2d."));
  AssertThrow(dealii::Utilities::MPI::n_mpi_processes(dof_handler.get_mpi_communicator()) == 1,
              dealii::ExcMessage("Contact between deformable surfaces is only implemented for "
                                 "serial runs (the opposite surface is accessed through global "
                                 "DoF indices)."));
  AssertThrow(not dof_handler.get_triangulation().has_hanging_nodes(),
              dealii::ExcMessage("Hanging nodes are not supported by the deformable contact."));

  dealii::FiniteElement<dim> const & fe = dof_handler.get_fe();
  AssertThrow(fe.degree == 1 and fe.n_components() == dim and fe.n_dofs_per_vertex() == dim and
                fe.n_dofs_per_cell() == dim * dealii::GeometryInfo<dim>::vertices_per_cell,
              dealii::ExcMessage("The deformable contact requires FE_Q(1)^dim (straight faces in "
                                 "the deformed configuration)."));
  for(unsigned int c = 0; c < dim; ++c)
    AssertThrow(fe.system_to_component_index(c).first == c, dealii::ExcInternalError());

  AssertThrow(data.penalty > 0.0, dealii::ExcMessage("Deformable contact: penalty <= 0."));
  AssertThrow(data.radius > 0.0, dealii::ExcMessage("Deformable contact: radius <= 0."));
  AssertThrow(data.projection_tolerance >= 0.0 and data.projection_tolerance < 0.5,
              dealii::ExcMessage("Deformable contact: projection tolerance not in [0, 0.5)."));
  AssertThrow(data.search == ContactSearchStrategy::Exhaustive or data.skin > 0.0,
              dealii::ExcMessage("Deformable contact: Verlet search requires a skin > 0."));
  AssertThrow(not data.pairs.empty(), dealii::ExcMessage("Deformable contact: no pairs."));

  unsigned int const n_q = (data.n_q_points > 0) ? data.n_q_points : fe.degree + 1;

  std::set<dealii::types::global_dof_index> const constrained(constrained_dofs.begin(),
                                                              constrained_dofs.end());

  std::set<dealii::types::boundary_id> ids;
  for(auto const & pair : data.pairs)
  {
    ids.insert(pair.first);
    ids.insert(pair.second);
  }

  vertices.clear();
  faces.clear();
  pairs.clear();
  slave_faces.clear();
  slave_points.clear();

  std::map<dealii::types::global_dof_index, unsigned int>         vertex_index;
  std::map<dealii::types::boundary_id, std::vector<unsigned int>> faces_by_id;

  for(auto const & cell : dof_handler.active_cell_iterators())
  {
    for(unsigned int const f : cell->face_indices())
    {
      if(not cell->at_boundary(f))
        continue;
      dealii::types::boundary_id const id = cell->face(f)->boundary_id();
      if(ids.find(id) == ids.end())
        continue;

      Face face;
      face.boundary_id = id;
      face.cell        = cell;
      face.face_no     = f;

      for(unsigned int v = 0; v < 2; ++v)
      {
        dealii::types::global_dof_index const dof0 = cell->face(f)->vertex_dof_index(v, 0);
        auto const                            it   = vertex_index.find(dof0);
        if(it != vertex_index.end())
        {
          face.vertex[v] = it->second;
        }
        else
        {
          Vertex vertex;
          vertex.X = cell->face(f)->vertex(v);
          for(unsigned int c = 0; c < dim; ++c)
          {
            vertex.dof[c]         = cell->face(f)->vertex_dof_index(v, c);
            vertex.constrained[c] = (constrained.find(vertex.dof[c]) != constrained.end());
          }
          face.vertex[v]     = vertices.size();
          vertex_index[dof0] = vertices.size();
          vertices.push_back(vertex);
        }
      }

      // orientation of the outward normal with respect to the tangent (reference configuration)
      dealii::Tensor<1, dim> const t = vertices[face.vertex[1]].X - vertices[face.vertex[0]].X;
      dealii::Tensor<1, dim> const outward = cell->face(f)->center() - cell->center();
      face.orientation                     = (rotate<dim>(t) * outward > 0.0) ? 1.0 : -1.0;

      faces_by_id[id].push_back(faces.size());
      faces.push_back(face);
    }
  }

  dealii::QGauss<1> const quadrature(n_q);

  for(unsigned int p = 0; p < data.pairs.size(); ++p)
  {
    Pair pair;
    pair.slave        = data.pairs[p].first;
    pair.master       = data.pairs[p].second;
    pair.master_faces = faces_by_id[pair.master];
    AssertThrow(not pair.master_faces.empty() and not faces_by_id[pair.slave].empty(),
                dealii::ExcMessage("Deformable contact: no faces with the boundary id of a pair."));
    pairs.push_back(pair);

    for(unsigned int const face : faces_by_id[pair.slave])
    {
      SlaveFace slave_face;
      slave_face.face        = face;
      slave_face.pair        = p;
      slave_face.first_point = slave_points.size();
      slave_face.n_points    = n_q;

      double const length =
        (vertices[faces[face].vertex[1]].X - vertices[faces[face].vertex[0]].X).norm();
      for(unsigned int q = 0; q < n_q; ++q)
      {
        SlavePoint point;
        point.slave_face = slave_faces.size();
        point.eta        = quadrature.point(q)[0];
        point.weight     = quadrature.weight(q) * length;
        slave_points.push_back(point);
      }
      slave_faces.push_back(slave_face);
    }
  }

  list_built = false;
  lin_state.clear();
  last_state.clear();
  reset_statistics();
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::compute_positions(VectorType const & u, Configuration & x) const
{
  x.x.resize(vertices.size());
  x.d.resize(vertices.size());
  for(unsigned int a = 0; a < vertices.size(); ++a)
  {
    for(unsigned int c = 0; c < dim; ++c)
      x.d[a][c] = u(vertices[a].dof[c]);
    x.x[a] = vertices[a].X + x.d[a];
  }
}

template<int dim, typename Number>
dealii::Point<dim>
DeformableContact<dim, Number>::slave_position(Configuration const & x,
                                               unsigned int const    point) const
{
  SlavePoint const & sp   = slave_points[point];
  Face const &       face = faces[slave_faces[sp.slave_face].face];
  return dealii::Point<dim>((1.0 - sp.eta) * x.x[face.vertex[0]] + sp.eta * x.x[face.vertex[1]]);
}

template<int dim, typename Number>
dealii::Tensor<1, dim>
DeformableContact<dim, Number>::face_normal(Configuration const & x, unsigned int const face) const
{
  dealii::Tensor<1, dim> const t = x.x[faces[face].vertex[1]] - x.x[faces[face].vertex[0]];
  return faces[face].orientation * rotate<dim>(t) / t.norm();
}

template<int dim, typename Number>
bool
DeformableContact<dim, Number>::project(dealii::Point<dim> const &     x_s,
                                        dealii::Tensor<1, dim> const & n_s,
                                        Configuration const &          x,
                                        unsigned int const             master,
                                        ContactProjection<dim> &       result) const
{
  ++stats.n_projections;

  dealii::Point<dim> const &   x_1 = x.x[faces[master].vertex[0]];
  dealii::Tensor<1, dim> const a   = x.x[faces[master].vertex[1]] - x_1;
  double const                 l2  = a * a;
  if(not(l2 > 0.0))
    return false;

  // (ii) projection onto the segment
  double const xi = ((x_s - x_1) * a) / l2;
  if(xi < -data.projection_tolerance or xi > 1.0 + data.projection_tolerance)
    return false;

  double const                 length  = std::sqrt(l2);
  dealii::Tensor<1, dim> const tangent = a / length;
  dealii::Tensor<1, dim> const normal  = faces[master].orientation * rotate<dim>(tangent);

  // (iii) normals facing each other
  if(not(n_s * normal < 0.0))
    return false;

  result.master  = master;
  result.xi      = xi;
  result.gap     = (x_s - x_1) * normal;
  result.length  = length;
  result.normal  = normal;
  result.tangent = tangent;

  return true;
}

template<int dim, typename Number>
bool
DeformableContact<dim, Number>::closer(ContactProjection<dim> const & a,
                                       ContactProjection<dim> const & b)
{
  if(not b.found())
    return a.found();
  if(not a.found())
    return false;
  double const abs_a = std::abs(a.gap), abs_b = std::abs(b.gap);
  return (abs_a < abs_b) or (abs_a == abs_b and a.master < b.master);
}

template<int dim, typename Number>
ContactProjection<dim>
DeformableContact<dim, Number>::query_exhaustive(Configuration const & x,
                                                 unsigned int const    point,
                                                 bool const            count) const
{
  SlaveFace const &            slave_face = slave_faces[slave_points[point].slave_face];
  dealii::Point<dim> const     x_s        = slave_position(x, point);
  dealii::Tensor<1, dim> const n_s        = face_normal(x, slave_face.face);

  ContactProjection<dim> best, best_beyond, candidate;
  unsigned int           n_equal = 0;
  for(unsigned int const master : pairs[slave_face.pair].master_faces)
  {
    // (i) the face of the slave point
    if(master == slave_face.face)
      continue;

    if(not project(x_s, n_s, x, master, candidate))
      continue;

    if(std::abs(candidate.gap) <= data.radius)
    {
      if(best.found() and std::abs(candidate.gap) == std::abs(best.gap))
        ++n_equal;
      else if(not best.found() or std::abs(candidate.gap) < std::abs(best.gap))
        n_equal = 1;
      if(closer(candidate, best))
        best = candidate;
    }
    else if(closer(candidate, best_beyond))
    {
      best_beyond = candidate;
    }
  }

  if(count)
  {
    if(n_equal > 1)
      ++stats.n_ties;
    // the detection radius changes the result compared to a search without cutoff
    if(not best.found() and best_beyond.found() and best_beyond.gap <= 0.0)
      ++stats.n_cutoff;
  }

  return best;
}

template<int dim, typename Number>
ContactProjection<dim>
DeformableContact<dim, Number>::query_verlet(Configuration const & x,
                                             unsigned int const    point) const
{
  unsigned int const           index      = slave_points[point].slave_face;
  SlaveFace const &            slave_face = slave_faces[index];
  dealii::Point<dim> const     x_s        = slave_position(x, point);
  dealii::Tensor<1, dim> const n_s        = face_normal(x, slave_face.face);

  ContactProjection<dim> best, candidate;
  unsigned int           n_equal = 0;
  for(unsigned int i = candidate_offsets[index]; i < candidate_offsets[index + 1]; ++i)
  {
    unsigned int const master = candidates[i];
    if(not project(x_s, n_s, x, master, candidate))
      continue;

    if(std::abs(candidate.gap) <= data.radius)
    {
      if(best.found() and std::abs(candidate.gap) == std::abs(best.gap))
        ++n_equal;
      else if(not best.found() or std::abs(candidate.gap) < std::abs(best.gap))
        n_equal = 1;
      if(closer(candidate, best))
        best = candidate;
    }
  }
  stats.n_candidates += candidate_offsets[index + 1] - candidate_offsets[index];
  if(n_equal > 1)
    ++stats.n_ties;

  return best;
}

template<int dim, typename Number>
double
DeformableContact<dim, Number>::max_displacement_since_build(Configuration const & x) const
{
  double D2 = 0.0;
  for(unsigned int a = 0; a < vertices.size(); ++a)
    D2 = std::max(D2, (x.d[a] - build_displacement[a]).norm_square());
  return std::sqrt(D2);
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::ensure_valid_list(Configuration const & x) const
{
  if(not list_built)
  {
    build_list(x);
    return;
  }

  bool valid;
  {
    ScopedTime timer(stats.t_check);
    ++stats.n_validity_checks;

    // certificate 2 D_max <= delta (the displacements are compared, since the positions
    // X + u contain round-off)
    double D2 = 0.0;
    for(unsigned int a = 0; a < vertices.size(); ++a)
      D2 = std::max(D2, (x.d[a] - build_displacement[a]).norm_square());
    valid = (4.0 * D2 <= data.skin * data.skin);
    if(valid)
      stats.max_ratio_D_to_skin =
        std::max(stats.max_ratio_D_to_skin, 2.0 * std::sqrt(D2) / data.skin);
  }

  if(not valid)
    build_list(x);
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::build_list(Configuration const & x) const
{
  ScopedTime timer(stats.t_build);
  ++stats.n_builds;

  // bounding boxes of all faces
  std::vector<std::array<dealii::Point<dim>, 2>> boxes(faces.size());
  double                                         scale = 0.0;
  for(unsigned int f = 0; f < faces.size(); ++f)
  {
    dealii::Point<dim> const & x_1 = x.x[faces[f].vertex[0]];
    dealii::Point<dim> const & x_2 = x.x[faces[f].vertex[1]];
    for(unsigned int d = 0; d < dim; ++d)
    {
      boxes[f][0][d] = std::min(x_1[d], x_2[d]);
      boxes[f][1][d] = std::max(x_1[d], x_2[d]);
      scale          = std::max(scale, std::max(std::abs(x_1[d]), std::abs(x_2[d])));
    }
  }

  // conservative inflation R_m = r_c + delta + tau (l_m + delta) of the master boxes (plus a
  // margin for round-off in the projection)
  auto const inflation = [&](unsigned int const master) {
    double const length = (x.x[faces[master].vertex[1]] - x.x[faces[master].vertex[0]]).norm();
    return (data.radius + data.skin + data.projection_tolerance * (length + data.skin)) *
             (1.0 + 1e-10) +
           1e-13 * scale;
  };

  candidate_offsets.assign(slave_faces.size() + 1, 0);
  candidates.clear();
  for(unsigned int i = 0; i < slave_faces.size(); ++i)
  {
    candidate_offsets[i] = candidates.size();

    std::array<dealii::Point<dim>, 2> const & box_s = boxes[slave_faces[i].face];
    for(unsigned int const master : pairs[slave_faces[i].pair].master_faces)
    {
      if(master == slave_faces[i].face)
        continue;

      double const R       = inflation(master);
      bool         overlap = true;
      for(unsigned int d = 0; d < dim; ++d)
        overlap = overlap and (box_s[0][d] - R <= boxes[master][1][d]) and
                  (boxes[master][0][d] <= box_s[1][d] + R);
      if(overlap)
        candidates.push_back(master);
    }
  }
  candidate_offsets[slave_faces.size()] = candidates.size();

  build_displacement = x.d;
  list_built         = true;
  stats.n_list_pairs += candidates.size();
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::search_all(Configuration const &                 x,
                                           ContactSearchStrategy const           strategy,
                                           std::vector<ContactProjection<dim>> & result) const
{
  result.resize(slave_points.size());

  if(strategy == ContactSearchStrategy::Verlet)
    ensure_valid_list(x);

  {
    ScopedTime timer(stats.t_search);
    for(unsigned int p = 0; p < slave_points.size(); ++p)
    {
      result[p] = (strategy == ContactSearchStrategy::Exhaustive) ? query_exhaustive(x, p, true) :
                                                                    query_verlet(x, p);
      if(result[p].active())
        ++stats.n_active;
    }
    stats.n_queries += slave_points.size();
  }

  if(strategy == ContactSearchStrategy::Verlet and data.verify)
  {
    ScopedTime          timer(stats.t_verify);
    unsigned long const n_projections = stats.n_projections;
    std::ostringstream  diagnosis;
    unsigned int        n_mismatches = 0;
    for(unsigned int p = 0; p < slave_points.size(); ++p)
    {
      ContactProjection<dim> const reference = query_exhaustive(x, p, false);
      ++stats.n_verified;
      if(reference.master != result[p].master or reference.xi != result[p].xi or
         reference.gap != result[p].gap)
      {
        ++stats.n_verify_mismatches;
        if(++n_mismatches <= 10)
        {
          // is the partner of the exhaustive search a candidate of the Verlet list?
          unsigned int const index         = slave_points[p].slave_face;
          bool               in_candidates = false;
          for(unsigned int i = candidate_offsets[index]; i < candidate_offsets[index + 1]; ++i)
            in_candidates = in_candidates or (candidates[i] == reference.master);
          diagnosis << "  slave point " << p << " (face " << slave_faces[index].face << ") at "
                    << slave_position(x, p) << ": Verlet (face "
                    << (result[p].found() ? std::to_string(result[p].master) : "none") << ", xi "
                    << result[p].xi << ", g " << result[p].gap << "), exhaustive (face "
                    << (reference.found() ? std::to_string(reference.master) : "none") << ", xi "
                    << reference.xi << ", g " << reference.gap << "), partner "
                    << (in_candidates ? "in" : "missing from") << " the candidates, 2 D_max / "
                    << "delta = " << 2.0 * max_displacement_since_build(x) / data.skin << std::endl;
        }
      }
    }
    stats.n_projections = n_projections;

    // the quasi-static driver catches exceptions during the Newton solve, so the diagnosis is
    // also written to std::cerr
    if(n_mismatches > 0)
      std::cerr << "Deformable contact: Verlet and exhaustive search differ at " << n_mismatches
                << " slave points:" << std::endl
                << diagnosis.str();

    AssertThrow(n_mismatches == 0,
                dealii::ExcMessage("Deformable contact: the Verlet search differs from the "
                                   "exhaustive search at " +
                                   std::to_string(n_mismatches) + " slave points:\n" +
                                   diagnosis.str()));
  }
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::add(VectorType &                   dst,
                                    unsigned int const             vertex,
                                    dealii::Tensor<1, dim> const & value) const
{
  for(unsigned int c = 0; c < dim; ++c)
    if(not vertices[vertex].constrained[c])
      dst(vertices[vertex].dof[c]) += value[c];
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::evaluate_residual_add(VectorType & dst, VectorType const & u) const
{
  ++stats.n_residuals;

  compute_positions(u, last_positions);
  search_all(last_positions, data.search, last_state);

  ScopedTime timer(stats.t_residual);
  for(unsigned int p = 0; p < slave_points.size(); ++p)
  {
    ContactProjection<dim> const & state = last_state[p];
    if(not state.active())
      continue;

    SlavePoint const & sp     = slave_points[p];
    Face const &       slave  = faces[slave_faces[sp.slave_face].face];
    Face const &       master = faces[state.master];

    // R = - w lambda dg/du with lambda = epsilon (-g), dg = (dx_s - dx_m) . n
    dealii::Tensor<1, dim> const f = (sp.weight * data.penalty * (-state.gap)) * state.normal;
    add(dst, slave.vertex[0], -(1.0 - sp.eta) * f);
    add(dst, slave.vertex[1], -sp.eta * f);
    add(dst, master.vertex[0], (1.0 - state.xi) * f);
    add(dst, master.vertex[1], state.xi * f);
  }
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::set_linearization(VectorType const & u) const
{
  ++stats.n_linearizations;

  Configuration x;
  compute_positions(u, x);

  // the Newton solver linearizes at the last accepted trial, whose residual has just been
  // evaluated: reuse that search result if the positions are bitwise identical
  if(last_state.size() == slave_points.size() and x.x == last_positions.x)
  {
    ++stats.n_lin_reused;
    ScopedTime timer(stats.t_lin);
    lin_state     = last_state;
    lin_positions = x;
  }
  else
  {
    search_all(x, data.search, lin_state);
    ScopedTime timer(stats.t_lin);
    lin_positions = x;
  }
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::apply_jacobian_add(VectorType & dst, VectorType const & v) const
{
  ++stats.n_vmults;
  ScopedTime timer(stats.t_vmult);

  AssertThrow(lin_state.size() == slave_points.size(),
              dealii::ExcMessage("Deformable contact: linearization state not set."));

  auto const read = [&](unsigned int const vertex) {
    dealii::Tensor<1, dim> value;
    for(unsigned int c = 0; c < dim; ++c)
      if(not vertices[vertex].constrained[c])
        value[c] = v(vertices[vertex].dof[c]);
    return value;
  };

  double const epsilon = data.penalty;

  for(unsigned int p = 0; p < slave_points.size(); ++p)
  {
    ContactProjection<dim> const & state = lin_state[p];
    if(not state.active())
      continue;

    SlavePoint const & sp     = slave_points[p];
    Face const &       slave  = faces[slave_faces[sp.slave_face].face];
    Face const &       master = faces[state.master];

    double const                   xi = state.xi, g = state.gap, l = state.length;
    dealii::Tensor<1, dim> const & n = state.normal;
    dealii::Tensor<1, dim> const & t = state.tangent;

    // directions at the slave point and at the master vertices
    dealii::Tensor<1, dim> const delta_s =
      (1.0 - sp.eta) * read(slave.vertex[0]) + sp.eta * read(slave.vertex[1]);
    dealii::Tensor<1, dim> const delta_1 = read(master.vertex[0]);
    dealii::Tensor<1, dim> const delta_2 = read(master.vertex[1]);
    dealii::Tensor<1, dim> const delta_r = delta_s - (1.0 - xi) * delta_1 - xi * delta_2;

    // first variation Delta g = (Delta x_s - Delta x_m) . n
    double const delta_g = delta_r * n;

    // second variation of g (straight master segment, 2d):
    //  Delta delta g = - alpha / l (delta x_s - delta x_m) . t - (beta / l + g alpha / l^2)
    //                  n . (delta x_2 - delta x_1)
    // with alpha = n . (Delta x_2 - Delta x_1), beta = (Delta x_s - Delta x_m) . t
    double const alpha = n * (delta_2 - delta_1);
    double const beta  = delta_r * t;
    double const c_n   = beta / l + g * alpha / (l * l);
    double const c_t   = alpha / l;

    dealii::Tensor<1, dim> const c_s = -c_t * t;
    dealii::Tensor<1, dim> const c_1 = c_t * (1.0 - xi) * t + c_n * n;
    dealii::Tensor<1, dim> const c_2 = c_t * xi * t - c_n * n;

    // J v = w (epsilon H(-g) Delta g dg/du - lambda Delta delta g), lambda = epsilon (-g)
    double const lambda = epsilon * (-g);
    double const k      = sp.weight * epsilon * delta_g;

    dealii::Tensor<1, dim> const f_s = k * n - sp.weight * lambda * c_s;
    dealii::Tensor<1, dim> const f_1 = -k * (1.0 - xi) * n - sp.weight * lambda * c_1;
    dealii::Tensor<1, dim> const f_2 = -k * xi * n - sp.weight * lambda * c_2;

    add(dst, slave.vertex[0], (1.0 - sp.eta) * f_s);
    add(dst, slave.vertex[1], sp.eta * f_s);
    add(dst, master.vertex[0], f_1);
    add(dst, master.vertex[1], f_2);
  }
}

template<int dim, typename Number>
DeformableContactQuantities<dim>
DeformableContact<dim, Number>::compute_quantities(VectorType const & u) const
{
  // postprocessing: exhaustive search, which does not modify the Verlet list and the counters
  DeformableContactStatistics const stats_before = stats;

  Configuration x;
  compute_positions(u, x);
  std::vector<ContactProjection<dim>> state;
  search_all(x, ContactSearchStrategy::Exhaustive, state);

  stats = stats_before;

  DeformableContactQuantities<dim> result;
  for(unsigned int p = 0; p < slave_points.size(); ++p)
  {
    if(state[p].found())
      result.min_gap = std::min(result.min_gap, state[p].gap);
    if(not state[p].active())
      continue;
    double const lambda = data.penalty * (-state[p].gap);
    result.force += (slave_points[p].weight * lambda) * state[p].normal;
    result.n_active += 1;
    result.active_area += slave_points[p].weight;
    result.max_pressure = std::max(result.max_pressure, lambda);
  }
  return result;
}

template<int dim, typename Number>
std::vector<ContactProjection<dim>>
DeformableContact<dim, Number>::search(VectorType const &          u,
                                       ContactSearchStrategy const strategy) const
{
  Configuration x;
  compute_positions(u, x);
  std::vector<ContactProjection<dim>> result;
  search_all(x, strategy, result);
  return result;
}

template<int dim, typename Number>
double
DeformableContact<dim, Number>::ratio_displacement_to_skin(VectorType const & u) const
{
  if(not list_built)
    return std::numeric_limits<double>::infinity();
  Configuration x;
  compute_positions(u, x);
  return 2.0 * max_displacement_since_build(x) / data.skin;
}

template<int dim, typename Number>
unsigned int
DeformableContact<dim, Number>::n_slave_points() const
{
  return slave_points.size();
}

template<int dim, typename Number>
unsigned int
DeformableContact<dim, Number>::n_list_pairs() const
{
  return candidates.size();
}

template<int dim, typename Number>
unsigned int
DeformableContact<dim, Number>::slave_face(unsigned int const point) const
{
  return slave_faces[slave_points[point].slave_face].face;
}

template<int dim, typename Number>
bool
DeformableContact<dim, Number>::faces_share_vertex(unsigned int const face_a,
                                                   unsigned int const face_b) const
{
  for(unsigned int const a : faces[face_a].vertex)
    for(unsigned int const b : faces[face_b].vertex)
      if(a == b)
        return true;
  return false;
}

template<int dim, typename Number>
dealii::Point<dim>
DeformableContact<dim, Number>::slave_point_position(VectorType const & u,
                                                     unsigned int const point) const
{
  Configuration x;
  compute_positions(u, x);
  return slave_position(x, point);
}

template<int dim, typename Number>
DeformableContactData const &
DeformableContact<dim, Number>::get_data() const
{
  return data;
}

template<int dim, typename Number>
DeformableContactStatistics const &
DeformableContact<dim, Number>::get_statistics() const
{
  return stats;
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::reset_statistics() const
{
  stats = DeformableContactStatistics();
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::print_statistics(dealii::ConditionalOStream const & pcout) const
{
  DeformableContactStatistics const & s = stats;

  unsigned long const n_verlet_queries =
    (data.search == ContactSearchStrategy::Verlet) ? s.n_queries : 0;

  pcout << std::endl
        << "Deformable contact ("
        << (data.search == ContactSearchStrategy::Verlet ? "Verlet" : "Exhaustive")
        << (data.verify ? ", verified" : "") << "):" << std::endl
        << "  slave points / master faces / vertices = " << slave_points.size() << " / " <<
    [&]() {
      unsigned int n = 0;
      for(auto const & pair : pairs)
        n += pair.master_faces.size();
      return n;
    }() << " / "
        << vertices.size() << std::endl
        << "  r_c = " << data.radius << ", skin = " << data.skin
        << ", projection tolerance = " << data.projection_tolerance
        << ", penalty = " << data.penalty << std::endl;

  auto const line = [&](std::string const & name, unsigned long const n, double const t) {
    pcout << "  " << std::left << std::setw(34) << name << std::right << std::setw(10) << n
          << std::setw(14) << std::scientific << std::setprecision(4) << t << " s"
          << std::defaultfloat << std::endl;
  };

  pcout << "  " << std::left << std::setw(34) << "operation" << std::right << std::setw(10)
        << "calls" << std::setw(16) << "wall time" << std::endl;
  line("list build", s.n_builds, s.t_build);
  line("validity check", s.n_validity_checks, s.t_check);
  line("queries (search + projections)", s.n_queries, s.t_search);
  line("  projections", s.n_projections, 0.0);
  line("residual assembly", s.n_residuals, s.t_residual);
  line("linearization (reused search)", s.n_linearizations, s.t_lin);
  line("  reused", s.n_lin_reused, 0.0);
  line("Jacobian-vector products (contact)", s.n_vmults, s.t_vmult);
  if(data.verify)
    line("verification (exhaustive)", s.n_verified, s.t_verify);

  pcout << std::setprecision(6);
  pcout << "  mean candidates per Verlet query   = "
        << (n_verlet_queries > 0 ? double(s.n_candidates) / n_verlet_queries : 0.0) << std::endl
        << "  mean face pairs per list build     = "
        << (s.n_builds > 0 ? double(s.n_list_pairs) / s.n_builds : 0.0) << std::endl
        << "  mean projections per query         = "
        << (s.n_queries > 0 ? double(s.n_projections) / s.n_queries : 0.0) << std::endl
        << "  mean active points per query sweep = "
        << (s.n_queries > 0 ? double(s.n_active) * slave_points.size() / s.n_queries : 0.0)
        << std::endl
        << "  max 2 D_max / skin (valid lists)   = " << s.max_ratio_D_to_skin << std::endl
        << "  ties / cutoff hits                 = " << s.n_ties << " / " << s.n_cutoff
        << std::endl;
  if(data.verify)
    pcout << "  verification mismatches            = " << s.n_verify_mismatches << " of "
          << s.n_verified << std::endl;
}

template class DeformableContact<2, float>;
template class DeformableContact<2, double>;
template class DeformableContact<3, float>;
template class DeformableContact<3, double>;

} // namespace Structure
} // namespace ExaDG
