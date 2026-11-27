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
#include <type_traits>

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

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
/*
 * Hyper-dual number v + a e_1 + b e_2 + ab e_1 e_2 (e_1^2 = e_2^2 = 0) for exact first and second
 * derivatives by forward automatic differentiation: seeding a = 1 for variable i and b = 1 for
 * variable j gives a = df/dx_i, b = df/dx_j, ab = d^2f/(dx_i dx_j).
 */
struct HyperDual
{
  double v = 0.0, a = 0.0, b = 0.0, ab = 0.0;

  HyperDual() = default;
  HyperDual(double const value) : v(value)
  {
  }
  HyperDual(double const v, double const a, double const b, double const ab)
    : v(v), a(a), b(b), ab(ab)
  {
  }
};

inline HyperDual
operator+(HyperDual const & x, HyperDual const & y)
{
  return HyperDual(x.v + y.v, x.a + y.a, x.b + y.b, x.ab + y.ab);
}

inline HyperDual
operator-(HyperDual const & x, HyperDual const & y)
{
  return HyperDual(x.v - y.v, x.a - y.a, x.b - y.b, x.ab - y.ab);
}

inline HyperDual
operator-(HyperDual const & x)
{
  return HyperDual(-x.v, -x.a, -x.b, -x.ab);
}

inline HyperDual
operator*(HyperDual const & x, HyperDual const & y)
{
  return HyperDual(x.v * y.v,
                   x.a * y.v + x.v * y.a,
                   x.b * y.v + x.v * y.b,
                   x.ab * y.v + x.a * y.b + x.b * y.a + x.v * y.ab);
}

inline HyperDual
inverse(HyperDual const & y)
{
  double const inv = 1.0 / y.v;
  return HyperDual(inv,
                   -y.a * inv * inv,
                   -y.b * inv * inv,
                   -y.ab * inv * inv + 2.0 * y.a * y.b * inv * inv * inv);
}

inline HyperDual
operator/(HyperDual const & x, HyperDual const & y)
{
  return x * inverse(y);
}

inline HyperDual
sqrt(HyperDual const & x)
{
  double const r = std::sqrt(x.v);
  return HyperDual(r,
                   0.5 * x.a / r,
                   0.5 * x.b / r,
                   0.5 * x.ab / r - 0.25 * x.a * x.b / (r * r * r));
}

inline double
value(double const x)
{
  return x;
}

inline double
value(HyperDual const & x)
{
  return x.v;
}

/*
 * Forward automatic differentiation in all N directions in one pass: value and gradient.
 */
template<int N>
struct Dual
{
  double                v = 0.0;
  std::array<double, N> g{};

  Dual() = default;
  Dual(double const value) : v(value)
  {
  }
};

template<int N>
inline Dual<N>
operator+(Dual<N> const & a, Dual<N> const & b)
{
  Dual<N> r(a.v + b.v);
  for(int i = 0; i < N; ++i)
    r.g[i] = a.g[i] + b.g[i];
  return r;
}

template<int N>
inline Dual<N>
operator-(Dual<N> const & a, Dual<N> const & b)
{
  Dual<N> r(a.v - b.v);
  for(int i = 0; i < N; ++i)
    r.g[i] = a.g[i] - b.g[i];
  return r;
}

template<int N>
inline Dual<N>
operator-(Dual<N> const & a)
{
  Dual<N> r(-a.v);
  for(int i = 0; i < N; ++i)
    r.g[i] = -a.g[i];
  return r;
}

template<int N>
inline Dual<N>
operator*(Dual<N> const & a, Dual<N> const & b)
{
  Dual<N> r(a.v * b.v);
  for(int i = 0; i < N; ++i)
    r.g[i] = a.g[i] * b.v + a.v * b.g[i];
  return r;
}

// f(a) with f'(a.v) = d1
template<int N>
inline Dual<N>
chain(Dual<N> const & a, double const f, double const d1)
{
  Dual<N> r(f);
  for(int i = 0; i < N; ++i)
    r.g[i] = d1 * a.g[i];
  return r;
}

template<int N>
inline Dual<N>
operator/(Dual<N> const & a, Dual<N> const & b)
{
  double const inv = 1.0 / b.v;
  return a * chain(b, inv, -inv * inv);
}

template<int N>
inline Dual<N>
sqrt(Dual<N> const & a)
{
  double const r = std::sqrt(a.v);
  return chain(a, r, 0.5 / r);
}

template<int N>
inline double
value(Dual<N> const & a)
{
  return a.v;
}

/*
 * Forward automatic differentiation of second order in all N directions in one pass: value,
 * gradient and the upper triangle of the Hessian (packed row by row, h[k] for i <= j).
 */
template<int N>
struct Taylor2
{
  static constexpr int  M = N * (N + 1) / 2;
  double                v = 0.0;
  std::array<double, N> g{};
  std::array<double, M> h{};

  Taylor2() = default;
  Taylor2(double const value) : v(value)
  {
  }
};

template<int N>
inline Taylor2<N>
operator+(Taylor2<N> const & a, Taylor2<N> const & b)
{
  Taylor2<N> r(a.v + b.v);
  for(int i = 0; i < N; ++i)
    r.g[i] = a.g[i] + b.g[i];
  for(int k = 0; k < Taylor2<N>::M; ++k)
    r.h[k] = a.h[k] + b.h[k];
  return r;
}

template<int N>
inline Taylor2<N>
operator-(Taylor2<N> const & a, Taylor2<N> const & b)
{
  Taylor2<N> r(a.v - b.v);
  for(int i = 0; i < N; ++i)
    r.g[i] = a.g[i] - b.g[i];
  for(int k = 0; k < Taylor2<N>::M; ++k)
    r.h[k] = a.h[k] - b.h[k];
  return r;
}

template<int N>
inline Taylor2<N>
operator-(Taylor2<N> const & a)
{
  Taylor2<N> r(-a.v);
  for(int i = 0; i < N; ++i)
    r.g[i] = -a.g[i];
  for(int k = 0; k < Taylor2<N>::M; ++k)
    r.h[k] = -a.h[k];
  return r;
}

template<int N>
inline Taylor2<N>
operator*(Taylor2<N> const & a, Taylor2<N> const & b)
{
  Taylor2<N> r(a.v * b.v);
  for(int i = 0; i < N; ++i)
    r.g[i] = a.g[i] * b.v + a.v * b.g[i];
  int k = 0;
  for(int i = 0; i < N; ++i)
    for(int j = i; j < N; ++j, ++k)
      r.h[k] = a.h[k] * b.v + a.v * b.h[k] + a.g[i] * b.g[j] + a.g[j] * b.g[i];
  return r;
}

// f(a) with f'(a.v) = d1, f''(a.v) = d2
template<int N>
inline Taylor2<N>
chain(Taylor2<N> const & a, double const f, double const d1, double const d2)
{
  Taylor2<N> r(f);
  for(int i = 0; i < N; ++i)
    r.g[i] = d1 * a.g[i];
  int k = 0;
  for(int i = 0; i < N; ++i)
    for(int j = i; j < N; ++j, ++k)
      r.h[k] = d1 * a.h[k] + d2 * a.g[i] * a.g[j];
  return r;
}

template<int N>
inline Taylor2<N>
operator/(Taylor2<N> const & a, Taylor2<N> const & b)
{
  double const inv = 1.0 / b.v;
  return a * chain(b, inv, -inv * inv, 2.0 * inv * inv * inv);
}

template<int N>
inline Taylor2<N>
sqrt(Taylor2<N> const & a)
{
  double const r = std::sqrt(a.v);
  return chain(a, r, 0.5 / r, -0.25 / (r * a.v));
}

template<int N>
inline double
value(Taylor2<N> const & a)
{
  return a.v;
}

template<typename T>
using Vector2 = std::array<T, 2>;

// outward unit normal orientation * (t_1, -t_0) / |t| of the face [x_0, x_1]
template<typename T>
Vector2<T>
segment_normal(Vector2<T> const & x_0, Vector2<T> const & x_1, double const orientation)
{
  using std::sqrt;
  T const t0 = x_1[0] - x_0[0], t1 = x_1[1] - x_0[1];
  T const length = sqrt(t0 * t0 + t1 * t1);
  return {{T(orientation) * t1 / length, T(-orientation) * t0 / length}};
}

/*
 * Smoothed master surface: on the master face [m_1, m_2] (length l, orientation o), the Hermite
 * cubic p(xi) = H00 m_1 + H10 l T_1 + H01 m_2 + H11 l T_2 with the unit nodal tangents
 * T_k = o (-nu_k1, nu_k0) perpendicular to the nodal normals nu_k. The nodal normals are the
 * normalized sum of the unit outward normals of the master face and of the neighbouring face
 * [p_0, p_1] at m_1 resp. [q_0, q_1] at m_2 (if present and if the angle between the two face
 * normals does not exceed arccos(cos_max); otherwise the normal of the master face). The curves of
 * neighbouring faces have the same tangent direction at the common vertex (G1 surface); on a
 * straight surface, p(xi) is the straight segment.
 *
 * Closest point projection: Newton's method for f(xi) = (x_s - p(xi)) . p'(xi) = 0 in double
 * precision, started at the projection onto the chord; from the converged value, two Newton steps
 * in the type T, which makes the first and second derivatives with respect to the input exact for
 * T = HyperDual (the Newton map has zero derivative with respect to xi at the solution). The gap is
 * g = (x_s - p(xi)) . n(xi) with the unit normal n = o (p'_1, -p'_0) / |p'|, and dg/dx_s = n.
 * Returns false if Newton's method does not converge or the solution is not a local minimum of
 * the distance.
 */
template<typename T>
bool
smooth_projection(Vector2<T> const & x_s,
                  Vector2<T> const & m_1,
                  Vector2<T> const & m_2,
                  double const       orientation,
                  bool const         has_p,
                  Vector2<T> const & p_0,
                  Vector2<T> const & p_1,
                  double const       orientation_p,
                  bool const         has_q,
                  Vector2<T> const & q_0,
                  Vector2<T> const & q_1,
                  double const       orientation_q,
                  double const       cos_max,
                  T &                xi,
                  T &                gap,
                  Vector2<T> &       nu_hat)
{
  using std::sqrt;

  Vector2<T> const n_m = segment_normal(m_1, m_2, orientation);

  auto const nodal_normal =
    [&](bool const has, Vector2<T> const & x_0, Vector2<T> const & x_1, double const o) {
      if(not has)
        return n_m;
      Vector2<T> const n = segment_normal(x_0, x_1, o);
      if(value(n[0] * n_m[0] + n[1] * n_m[1]) < cos_max)
        return n_m;
      T const s0 = n[0] + n_m[0], s1 = n[1] + n_m[1];
      T const length = sqrt(s0 * s0 + s1 * s1);
      return Vector2<T>{{s0 / length, s1 / length}};
    };

  Vector2<T> const nu_1 = nodal_normal(has_p, p_0, p_1, orientation_p);
  Vector2<T> const nu_2 = nodal_normal(has_q, q_0, q_1, orientation_q);

  Vector2<T> const a = {{m_2[0] - m_1[0], m_2[1] - m_1[1]}};
  T const          l = sqrt(a[0] * a[0] + a[1] * a[1]);
  T const          o(orientation);

  // tangent vectors l T_k of the Hermite curve
  Vector2<T> const t_1 = {{-o * nu_1[1] * l, o * nu_1[0] * l}};
  Vector2<T> const t_2 = {{-o * nu_2[1] * l, o * nu_2[0] * l}};

  // p(xi), p'(xi), p''(xi) for the coefficient type U (double or T)
  auto const curve = [](auto const & z,
                        auto const & q_1,
                        auto const & q_2,
                        auto const & s_1,
                        auto const & s_2,
                        auto &       p,
                        auto &       dp,
                        auto &       ddp) {
    using U    = std::decay_t<decltype(z)>;
    U const z2 = z * z, z3 = z2 * z;
    U const h00 = U(2.0) * z3 - U(3.0) * z2 + U(1.0), h10 = z3 - U(2.0) * z2 + z;
    U const h01 = U(-2.0) * z3 + U(3.0) * z2, h11 = z3 - z2;
    U const d00 = U(6.0) * z2 - U(6.0) * z, d10 = U(3.0) * z2 - U(4.0) * z + U(1.0);
    U const d01 = U(-6.0) * z2 + U(6.0) * z, d11 = U(3.0) * z2 - U(2.0) * z;
    U const e00 = U(12.0) * z - U(6.0), e10 = U(6.0) * z - U(4.0);
    U const e01 = U(-12.0) * z + U(6.0), e11 = U(6.0) * z - U(2.0);
    for(unsigned int c = 0; c < 2; ++c)
    {
      p[c]   = h00 * q_1[c] + h10 * s_1[c] + h01 * q_2[c] + h11 * s_2[c];
      dp[c]  = d00 * q_1[c] + d10 * s_1[c] + d01 * q_2[c] + d11 * s_2[c];
      ddp[c] = e00 * q_1[c] + e10 * s_1[c] + e01 * q_2[c] + e11 * s_2[c];
    }
  };

  // 1. Newton's method in double precision on the values, started at the projection onto the
  // chord; tolerance well above the round-off of xi (~ eps |x| / l)
  auto const to_double = [](Vector2<T> const & v) {
    return Vector2<double>{{value(v[0]), value(v[1])}};
  };
  Vector2<double> const xd = to_double(x_s), q1 = to_double(m_1), q2 = to_double(m_2),
                        s1 = to_double(t_1), s2 = to_double(t_2);
  Vector2<double> const ad = {{q2[0] - q1[0], q2[1] - q1[1]}};
  double zd = ((xd[0] - q1[0]) * ad[0] + (xd[1] - q1[1]) * ad[1]) / (ad[0] * ad[0] + ad[1] * ad[1]);
  bool   converged = false;
  for(unsigned int k = 0; k < 50 and not converged; ++k)
  {
    Vector2<double> p, dp, ddp;
    curve(zd, q1, q2, s1, s2, p, dp, ddp);
    double const f = (xd[0] - p[0]) * dp[0] + (xd[1] - p[1]) * dp[1];
    double const df =
      (xd[0] - p[0]) * ddp[0] + (xd[1] - p[1]) * ddp[1] - (dp[0] * dp[0] + dp[1] * dp[1]);
    if(not(df < 0.0))
      return false;
    double const dz = f / df;
    zd -= dz;
    converged = (std::abs(dz) <= 1e-12);
  }
  if(not converged)
    return false;

  // 2. two Newton steps in the type T from the converged value: the Newton map has zero
  // derivative with respect to xi at the solution, so that the first and second derivatives of xi
  // are exact for T = HyperDual (and the value is unchanged up to round-off)
  Vector2<T> p, dp, ddp;
  T          z(zd);
  for(unsigned int k = 0; k < 2; ++k)
  {
    curve(z, m_1, m_2, t_1, t_2, p, dp, ddp);
    Vector2<T> const r  = {{x_s[0] - p[0], x_s[1] - p[1]}};
    T const          f  = r[0] * dp[0] + r[1] * dp[1];
    T const          df = r[0] * ddp[0] + r[1] * ddp[1] - (dp[0] * dp[0] + dp[1] * dp[1]);
    z                   = z - f / df;
  }

  xi = z;
  curve(z, m_1, m_2, t_1, t_2, p, dp, ddp);
  T const length_dp = sqrt(dp[0] * dp[0] + dp[1] * dp[1]);
  nu_hat            = {{o * dp[1] / length_dp, -o * dp[0] / length_dp}};
  gap               = (x_s[0] - p[0]) * nu_hat[0] + (x_s[1] - p[1]) * nu_hat[1];

  return true;
}

template<typename T, int dim>
Vector2<T>
to_vector2(dealii::Tensor<1, dim> const & x)
{
  return {{T(x[0]), T(x[1])}};
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

  // smooth normals: neighbouring face with the same boundary id at each vertex, unless the angle
  // between the face normals exceeds the smoothing angle in the reference configuration (corner)
  if(data.smooth_normals)
  {
    AssertThrow(data.smoothing_angle > 0.0 and
                  data.smoothing_angle <= dealii::numbers::PI / 3.0 * (1.0 + 1e-12),
                dealii::ExcMessage("Deformable contact: smoothing angle not in (0, pi/3] (bound "
                                   "of the curve deviation in the Verlet certificate)."));
    AssertThrow(data.projection_tolerance <= 0.05,
                dealii::ExcMessage("Deformable contact: smoothed surface requires a projection "
                                   "tolerance <= 0.05 (bound of the curve deviation)."));
    Configuration reference;
    reference.x.resize(vertices.size());
    reference.d.resize(vertices.size());
    for(unsigned int a = 0; a < vertices.size(); ++a)
      reference.x[a] = vertices[a].X;

    std::map<std::pair<dealii::types::boundary_id, unsigned int>, std::vector<unsigned int>>
      faces_at_vertex;
    for(unsigned int f = 0; f < faces.size(); ++f)
      for(unsigned int const v : faces[f].vertex)
        faces_at_vertex[{faces[f].boundary_id, v}].push_back(f);

    for(unsigned int f = 0; f < faces.size(); ++f)
      for(unsigned int v = 0; v < 2; ++v)
      {
        std::vector<unsigned int> const & adjacent =
          faces_at_vertex[{faces[f].boundary_id, faces[f].vertex[v]}];
        AssertThrow(adjacent.size() <= 2,
                    dealii::ExcMessage("Deformable contact: more than two faces of a contact "
                                       "surface share a vertex."));
        for(unsigned int const g : adjacent)
          if(g != f and face_normal(reference, f) * face_normal(reference, g) >=
                          std::cos(data.smoothing_angle))
            faces[f].neighbor[v] = g;
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

  // Nitsche's method: slave cell data
  nitsche_faces.clear();
  nitsche_points.clear();
  if(data.nitsche)
  {
    AssertThrow(data.youngs_modulus > 0.0 and data.poissons_ratio < 0.5,
                dealii::ExcMessage("Deformable contact (Nitsche): invalid material data."));

    dealii::MappingQ<dim>     mapping(1);
    dealii::QGauss<dim>       cell_quadrature(fe.degree + 1);
    dealii::QGauss<dim - 1>   area_quadrature(fe.degree + 1);
    dealii::QGauss<dim - 1>   point_quadrature(n_q);
    dealii::FEValues<dim>     fe_values(mapping, fe, cell_quadrature, dealii::update_JxW_values);
    dealii::FEFaceValues<dim> fe_face_area(mapping, fe, area_quadrature, dealii::update_JxW_values);
    dealii::FEFaceValues<dim> fe_face_points(mapping,
                                             fe,
                                             point_quadrature,
                                             dealii::update_gradients |
                                               dealii::update_quadrature_points |
                                               dealii::update_normal_vectors);
    double const              penalty_factor = (fe.degree + 1.0) * (fe.degree + 1.0);

    nitsche_points.resize(slave_points.size());
    for(SlaveFace const & slave_face : slave_faces)
    {
      Face const & face = faces[slave_face.face];
      NitscheFace  nitsche_face;
      nitsche_face.dofs.resize(fe.n_dofs_per_cell());
      face.cell->get_dof_indices(nitsche_face.dofs);
      for(unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
      {
        nitsche_face.component.push_back(fe.system_to_component_index(j).first);
        nitsche_face.constrained.push_back(constrained.find(nitsche_face.dofs[j]) !=
                                           constrained.end());
      }

      // gamma = gamma_0 E (k + 1)^2 |dK| / |K| (interior faces weighted by 1/2, as in
      // IP::calculate_penalty_parameter())
      fe_values.reinit(face.cell);
      double volume = 0.0, surface = 0.0;
      for(unsigned int q = 0; q < cell_quadrature.size(); ++q)
        volume += fe_values.JxW(q);
      for(unsigned int const f : face.cell->face_indices())
      {
        fe_face_area.reinit(face.cell, f);
        double const factor =
          (face.cell->at_boundary(f) and not face.cell->has_periodic_neighbor(f)) ? 1.0 : 0.5;
        for(unsigned int q = 0; q < area_quadrature.size(); ++q)
          surface += factor * fe_face_area.JxW(q);
      }
      nitsche_face.gamma = data.penalty * penalty_factor * surface / volume;
      nitsche_faces.push_back(nitsche_face);

      // slave points: match the face quadrature points by position
      fe_face_points.reinit(face.cell, face.face_no);
      for(unsigned int k = 0; k < slave_face.n_points; ++k)
      {
        unsigned int const       point = slave_face.first_point + k;
        SlavePoint const &       sp    = slave_points[point];
        dealii::Point<dim> const X((1.0 - sp.eta) * vertices[face.vertex[0]].X +
                                   sp.eta * vertices[face.vertex[1]].X);
        unsigned int             match = dealii::numbers::invalid_unsigned_int;
        for(unsigned int q = 0; q < point_quadrature.size(); ++q)
          if(fe_face_points.quadrature_point(q).distance(X) <= 1e-12 * (1.0 + X.norm()))
            match = q;
        AssertThrow(match != dealii::numbers::invalid_unsigned_int,
                    dealii::ExcMessage("Deformable contact (Nitsche): slave point not found among "
                                       "the face quadrature points."));
        nitsche_points[point].N = fe_face_points.normal_vector(match);
        for(unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          nitsche_points[point].grad_phi.push_back(fe_face_points.shape_grad(j, match));
      }
    }
  }

  AssertThrow(not(data.augmented_lagrangian and data.nitsche),
              dealii::ExcMessage("The augmented Lagrangian method is implemented for the penalty "
                                 "method."));
  multipliers.assign(slave_points.size(), 0.0);

  list_built = false;
  lin_state.clear();
  lin_smooth.clear();
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

  if(data.smooth_normals)
  {
    // exact pre-filter: an admissible pair satisfies x_s = p(xi) + g n with |g| <= r_c,
    // xi in [-tau, 1 + tau] and |p(xi) - chord(xi)| <= 0.16 l, i.e. the distance of x_s to the
    // chord [m_1, m_2] is at most r_c + (0.16 + tau) l; other pairs are rejected without the
    // projection onto the curve
    {
      dealii::Point<dim> const &   c_1   = x.x[faces[master].vertex[0]];
      dealii::Tensor<1, dim> const chord = x.x[faces[master].vertex[1]] - c_1;
      double const                 l2    = chord * chord;
      double const                 t     = std::max(0.0, std::min(1.0, ((x_s - c_1) * chord) / l2));
      double const bound = data.radius + (0.16 + data.projection_tolerance) * std::sqrt(l2);
      if((x_s - (c_1 + t * chord)).norm_square() > bound * bound * (1.0 + 1e-12))
      {
        ++stats.n_prefiltered;
        return false;
      }
    }

    Face const & m        = faces[master];
    auto const   position = [&](unsigned int const v) { return to_vector2<double, dim>(x.x[v]); };
    bool const   has_p    = (m.neighbor[0] != dealii::numbers::invalid_unsigned_int);
    bool const   has_q    = (m.neighbor[1] != dealii::numbers::invalid_unsigned_int);
    Face const & p        = has_p ? faces[m.neighbor[0]] : m;
    Face const & q        = has_q ? faces[m.neighbor[1]] : m;

    double          xi, gap;
    Vector2<double> nu;
    if(not smooth_projection<double>(to_vector2<double, dim>(x_s),
                                     position(m.vertex[0]),
                                     position(m.vertex[1]),
                                     m.orientation,
                                     has_p,
                                     position(p.vertex[0]),
                                     position(p.vertex[1]),
                                     p.orientation,
                                     has_q,
                                     position(q.vertex[0]),
                                     position(q.vertex[1]),
                                     q.orientation,
                                     std::cos(data.smoothing_angle),
                                     xi,
                                     gap,
                                     nu))
      return false;

    // (ii) projection onto the face
    if(xi < -data.projection_tolerance or xi > 1.0 + data.projection_tolerance)
      return false;

    dealii::Tensor<1, dim> normal;
    normal[0] = nu[0];
    normal[1] = nu[1];

    // (iii) normals facing each other
    if(not(n_s * normal < 0.0))
      return false;

    dealii::Tensor<1, dim> const a = x.x[m.vertex[1]] - x.x[m.vertex[0]];
    result.master                  = master;
    result.xi                      = xi;
    result.gap                     = gap;
    result.length                  = a.norm();
    result.normal                  = normal;
    result.tangent                 = a / result.length;
    return true;
  }

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
    // smoothed surface: the Hermite curve deviates from the chord by at most
    // (4/27) (|l T_1 - a| + |l T_2 - a|) <= 0.16 l for nodal normals within 30 degrees of the face
    // normal (smoothing angle <= 60 degrees) and tau <= 0.05
    double const deviation = data.smooth_normals ? 0.16 : 0.0;
    return (data.radius + data.skin +
            (data.projection_tolerance + deviation) * (length + data.skin)) *
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
typename DeformableContact<dim, Number>::LocalTopology
DeformableContact<dim, Number>::local_topology(unsigned int const point,
                                               unsigned int const master) const
{
  LocalTopology topology;
  auto const    add_vertex = [&](unsigned int const v) {
    for(unsigned int k = 0; k < topology.n; ++k)
      if(topology.vertex[k] == v)
        return k;
    topology.vertex[topology.n] = v;
    return topology.n++;
  };

  SlavePoint const & sp    = slave_points[point];
  Face const &       slave = faces[slave_faces[sp.slave_face].face];
  Face const &       m     = faces[master];

  topology.slave_a     = add_vertex(slave.vertex[0]);
  topology.slave_b     = add_vertex(slave.vertex[1]);
  topology.master_1    = add_vertex(m.vertex[0]);
  topology.master_2    = add_vertex(m.vertex[1]);
  topology.eta         = sp.eta;
  topology.orientation = m.orientation;
  for(unsigned int v = 0; v < 2; ++v)
  {
    topology.has_neighbor[v]    = (m.neighbor[v] != dealii::numbers::invalid_unsigned_int);
    Face const & neighbor       = topology.has_neighbor[v] ? faces[m.neighbor[v]] : m;
    topology.neighbor_vertex[v] = {
      {add_vertex(neighbor.vertex[0]), add_vertex(neighbor.vertex[1])}};
    topology.neighbor_orientation[v] = neighbor.orientation;
  }
  return topology;
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::smooth_gap_derivatives_reference(
  LocalTopology const &     topology,
  Configuration const &     x,
  bool const                hessian,
  double &                  gap,
  std::array<double, 12> &  gradient,
  std::array<double, 144> * hessian_values) const
{
  unsigned int const n_dofs = 2 * topology.n;

  // gap with the hyper-dual seeds a on local DoF i and b on local DoF j (invalid: no seed)
  auto const evaluate = [&](unsigned int const i, unsigned int const j) {
    std::array<Vector2<HyperDual>, 6> X;
    for(unsigned int k = 0; k < topology.n; ++k)
      for(unsigned int c = 0; c < 2; ++c)
        X[k][c] = HyperDual(x.x[topology.vertex[k]][c],
                            (2 * k + c == i) ? 1.0 : 0.0,
                            (2 * k + c == j) ? 1.0 : 0.0,
                            0.0);

    double const             eta = topology.eta;
    Vector2<HyperDual> const x_s = {
      {HyperDual(1.0 - eta) * X[topology.slave_a][0] + HyperDual(eta) * X[topology.slave_b][0],
       HyperDual(1.0 - eta) * X[topology.slave_a][1] + HyperDual(eta) * X[topology.slave_b][1]}};

    HyperDual          xi, g;
    Vector2<HyperDual> nu;
    bool const         found = smooth_projection<HyperDual>(x_s,
                                                    X[topology.master_1],
                                                    X[topology.master_2],
                                                    topology.orientation,
                                                    topology.has_neighbor[0],
                                                    X[topology.neighbor_vertex[0][0]],
                                                    X[topology.neighbor_vertex[0][1]],
                                                    topology.neighbor_orientation[0],
                                                    topology.has_neighbor[1],
                                                    X[topology.neighbor_vertex[1][0]],
                                                    X[topology.neighbor_vertex[1][1]],
                                                    topology.neighbor_orientation[1],
                                                    std::cos(data.smoothing_angle),
                                                    xi,
                                                    g,
                                                    nu);
    AssertThrow(found, dealii::ExcInternalError());
    return g;
  };

  gradient.fill(0.0);
  unsigned int const invalid = dealii::numbers::invalid_unsigned_int;
  if(not hessian)
  {
    for(unsigned int i = 0; i < n_dofs; ++i)
    {
      HyperDual const g = evaluate(i, invalid);
      gradient[i]       = g.a;
      gap               = g.v;
    }
  }
  else
  {
    hessian_values->fill(0.0);
    for(unsigned int i = 0; i < n_dofs; ++i)
      for(unsigned int j = i; j < n_dofs; ++j)
      {
        HyperDual const g             = evaluate(i, j);
        (*hessian_values)[i * 12 + j] = g.ab;
        (*hessian_values)[j * 12 + i] = g.ab;
        if(i == j)
          gradient[i] = g.a;
        gap = g.v;
      }
  }
}

template<int dim, typename Number>
template<typename T>
T
DeformableContact<dim, Number>::smooth_gap_of_positions(
  LocalTopology const &                   topology,
  std::array<std::array<T, 2>, 6> const & X) const
{
  T                gap;
  std::array<T, 2> normal;
  smooth_projection_of_positions(topology, X, gap, normal);
  return gap;
}

template<int dim, typename Number>
template<int N>
void
DeformableContact<dim, Number>::smooth_gap_derivatives_fixed(
  LocalTopology const &                topology,
  Configuration const &                x,
  std::array<unsigned int, 12> const & directions,
  unsigned int const                   n_directions,
  bool const                           hessian,
  double &                             gap,
  std::array<double, 12> &             gradient,
  std::array<double, 144> *            hessian_values) const
{
  // seed direction d on the local DoF directions[d] (d < n_directions <= N)
  gradient.fill(0.0);
  if(hessian)
  {
    std::array<std::array<Taylor2<N>, 2>, 6> X;
    for(unsigned int k = 0; k < topology.n; ++k)
      for(unsigned int c = 0; c < 2; ++c)
        X[k][c] = Taylor2<N>(x.x[topology.vertex[k]][c]);
    for(unsigned int d = 0; d < n_directions; ++d)
      X[directions[d] / 2][directions[d] % 2].g[d] = 1.0;
    Taylor2<N> const g = smooth_gap_of_positions(topology, X);
    gap                = g.v;
    hessian_values->fill(0.0);
    int k = 0;
    for(int a = 0; a < N; ++a)
      for(int b = a; b < N; ++b, ++k)
        if(a < int(n_directions) and b < int(n_directions))
        {
          (*hessian_values)[directions[a] * 12 + directions[b]] = g.h[k];
          (*hessian_values)[directions[b] * 12 + directions[a]] = g.h[k];
        }
    for(unsigned int d = 0; d < n_directions; ++d)
      gradient[directions[d]] = g.g[d];
  }
  else
  {
    std::array<std::array<Dual<N>, 2>, 6> X;
    for(unsigned int k = 0; k < topology.n; ++k)
      for(unsigned int c = 0; c < 2; ++c)
        X[k][c] = Dual<N>(x.x[topology.vertex[k]][c]);
    for(unsigned int d = 0; d < n_directions; ++d)
      X[directions[d] / 2][directions[d] % 2].g[d] = 1.0;
    Dual<N> const g = smooth_gap_of_positions(topology, X);
    gap             = g.v;
    for(unsigned int d = 0; d < n_directions; ++d)
      gradient[directions[d]] = g.g[d];
  }
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::smooth_gap_derivatives(LocalTopology const &     topology,
                                                       Configuration const &     x,
                                                       bool const                hessian,
                                                       double &                  gap,
                                                       std::array<double, 12> &  gradient,
                                                       std::array<double, 144> * hessian_values,
                                                       bool const all_directions) const
{
  // derivatives in one forward pass with respect to the local DoFs, by default only the
  // unconstrained ones (the entries of constrained DoFs are never used: residual and matrix skip
  // them, Jacobian-vector products treat them as zero); entries of skipped DoFs are zero
  std::array<unsigned int, 12> directions;
  unsigned int                 n_directions = 0;
  for(unsigned int k = 0; k < topology.n; ++k)
    for(unsigned int c = 0; c < 2; ++c)
      if(all_directions or not vertices[topology.vertex[k]].constrained[c])
        directions[n_directions++] = 2 * k + c;

  switch((n_directions + 1) / 2)
  {
    case 0:
    case 1:
      smooth_gap_derivatives_fixed<2>(
        topology, x, directions, n_directions, hessian, gap, gradient, hessian_values);
      break;
    case 2:
      smooth_gap_derivatives_fixed<4>(
        topology, x, directions, n_directions, hessian, gap, gradient, hessian_values);
      break;
    case 3:
      smooth_gap_derivatives_fixed<6>(
        topology, x, directions, n_directions, hessian, gap, gradient, hessian_values);
      break;
    case 4:
      smooth_gap_derivatives_fixed<8>(
        topology, x, directions, n_directions, hessian, gap, gradient, hessian_values);
      break;
    case 5:
      smooth_gap_derivatives_fixed<10>(
        topology, x, directions, n_directions, hessian, gap, gradient, hessian_values);
      break;
    case 6:
      smooth_gap_derivatives_fixed<12>(
        topology, x, directions, n_directions, hessian, gap, gradient, hessian_values);
      break;
    default:
      AssertThrow(false, dealii::ExcInternalError());
  }
}

template<int dim, typename Number>
double
DeformableContact<dim, Number>::verify_smooth_derivatives(VectorType const & u) const
{
  AssertThrow(data.smooth_normals, dealii::ExcMessage("Requires smooth normals."));

  Configuration x;
  compute_positions(u, x);
  DeformableContactStatistics const   stats_before = stats;
  std::vector<ContactProjection<dim>> state;
  search_all(x, ContactSearchStrategy::Exhaustive, state);
  stats = stats_before;

  double max_difference = 0.0;
  for(unsigned int p = 0; p < state.size(); ++p)
  {
    if(not state[p].active())
      continue;
    LocalTopology const     topology = local_topology(p, state[p].master);
    double                  gap_ref, gap_fast, gap_grad;
    std::array<double, 12>  grad_ref, grad_fast, grad_only;
    std::array<double, 144> hess_ref, hess_fast;
    smooth_gap_derivatives_reference(topology, x, true, gap_ref, grad_ref, &hess_ref);
    smooth_gap_derivatives(topology, x, true, gap_fast, grad_fast, &hess_fast, true);
    smooth_gap_derivatives(topology, x, false, gap_grad, grad_only, nullptr, true);

    // unconstrained directions only: equal to the reference on the unconstrained entries
    double                  gap_free;
    std::array<double, 12>  grad_free;
    std::array<double, 144> hess_free;
    smooth_gap_derivatives(topology, x, true, gap_free, grad_free, &hess_free, false);
    auto const free = [&](unsigned int const i) {
      return not vertices[topology.vertex[i / 2]].constrained[i % 2];
    };

    double scale_g = 0.0, scale_h = 0.0, diff_g = 0.0, diff_h = 0.0;
    for(unsigned int i = 0; i < 2 * topology.n; ++i)
    {
      scale_g = std::max(scale_g, std::abs(grad_ref[i]));
      if(free(i))
        diff_g = std::max(diff_g, std::abs(grad_ref[i] - grad_free[i]));
      diff_g = std::max(diff_g,
                        std::max(std::abs(grad_ref[i] - grad_fast[i]),
                                 std::abs(grad_ref[i] - grad_only[i])));
      for(unsigned int j = 0; j < 2 * topology.n; ++j)
      {
        scale_h = std::max(scale_h, std::abs(hess_ref[i * 12 + j]));
        diff_h  = std::max(diff_h, std::abs(hess_ref[i * 12 + j] - hess_fast[i * 12 + j]));
        if(free(i) and free(j))
          diff_h = std::max(diff_h, std::abs(hess_ref[i * 12 + j] - hess_free[i * 12 + j]));
      }
    }
    max_difference = std::max(max_difference,
                              std::max(std::abs(gap_ref - gap_fast) + std::abs(gap_ref - gap_grad),
                                       std::max(diff_g / scale_g, diff_h / scale_h)));
  }
  return max_difference;
}

template<int dim, typename Number>
dealii::Tensor<2, dim>
DeformableContact<dim, Number>::svk_stress(dealii::Tensor<2, dim> const & E) const
{
  // as StVenantKirchhoff (2d, plane strain or plane stress), coefficients times Young's modulus
  double const nu = data.poissons_ratio;
  double       f0, f1, f2;
  if(data.plane_stress)
  {
    f0 = 1. / (1. - nu * nu);
    f1 = nu / (1. - nu * nu);
    f2 = (1. - nu) / (2.0 * (1. - nu * nu));
  }
  else
  {
    f0 = (1. - nu) / ((1. + nu) * (1. - 2. * nu));
    f1 = nu / ((1. + nu) * (1. - 2. * nu));
    f2 = (1. - 2. * nu) / (2.0 * (1. + nu) * (1. - 2. * nu));
  }
  f0 *= data.youngs_modulus;
  f1 *= data.youngs_modulus;
  f2 *= data.youngs_modulus;

  dealii::Tensor<2, dim> S;
  S[0][0] = f0 * E[0][0] + f1 * E[1][1];
  S[1][1] = f1 * E[0][0] + f0 * E[1][1];
  S[0][1] = f2 * (E[0][1] + E[1][0]);
  S[1][0] = S[0][1];
  return S;
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::slave_traction(unsigned int const       point,
                                               VectorType const &       u,
                                               dealii::Tensor<2, dim> & F,
                                               dealii::Tensor<2, dim> & S,
                                               dealii::Tensor<1, dim> & traction) const
{
  NitscheFace const &  face = nitsche_faces[slave_points[point].slave_face];
  NitschePoint const & np   = nitsche_points[point];

  // displacement gradient H = sum_j u_j e_{c_j} (x) grad phi_j
  dealii::Tensor<2, dim> H;
  for(unsigned int j = 0; j < face.dofs.size(); ++j)
    for(unsigned int d = 0; d < dim; ++d)
      H[face.component[j]][d] += u(face.dofs[j]) * np.grad_phi[j][d];

  dealii::Tensor<2, dim> const E = 0.5 * (H + transpose(H) + transpose(H) * H);
  F                              = H;
  for(unsigned int d = 0; d < dim; ++d)
    F[d][d] += 1.0;
  S        = svk_stress(E);
  traction = (F * S) * np.N;
}

template<int dim, typename Number>
double
DeformableContact<dim, Number>::pressure_argument(unsigned int const             point,
                                                  ContactProjection<dim> const & projection,
                                                  VectorType const &             u,
                                                  NitscheState * nitsche_state) const
{
  if(not data.nitsche)
    return (data.augmented_lagrangian ? multipliers[point] : 0.0) +
           data.penalty * (activation_shift - projection.gap);

  NitscheState state;
  slave_traction(point, u, state.F, state.S, state.traction);
  if(nitsche_state)
    *nitsche_state = state;
  return projection.normal * state.traction -
         nitsche_faces[slave_points[point].slave_face].gamma * projection.gap;
}

template<int dim, typename Number>
double
DeformableContact<dim, Number>::contact_stiffness(unsigned int const point) const
{
  return data.nitsche ? nitsche_faces[slave_points[point].slave_face].gamma : data.penalty;
}

template<int dim, typename Number>
template<typename T>
void
DeformableContact<dim, Number>::smooth_projection_of_positions(
  LocalTopology const &                   topology,
  std::array<std::array<T, 2>, 6> const & X,
  T &                                     gap,
  std::array<T, 2> &                      normal) const
{
  double const     eta = topology.eta;
  Vector2<T> const x_s = {
    {T(1.0 - eta) * X[topology.slave_a][0] + T(eta) * X[topology.slave_b][0],
     T(1.0 - eta) * X[topology.slave_a][1] + T(eta) * X[topology.slave_b][1]}};

  T          xi;
  bool const found = smooth_projection<T>(x_s,
                                          X[topology.master_1],
                                          X[topology.master_2],
                                          topology.orientation,
                                          topology.has_neighbor[0],
                                          X[topology.neighbor_vertex[0][0]],
                                          X[topology.neighbor_vertex[0][1]],
                                          topology.neighbor_orientation[0],
                                          topology.has_neighbor[1],
                                          X[topology.neighbor_vertex[1][0]],
                                          X[topology.neighbor_vertex[1][1]],
                                          topology.neighbor_orientation[1],
                                          std::cos(data.smoothing_angle),
                                          xi,
                                          gap,
                                          normal);
  AssertThrow(found, dealii::ExcInternalError());
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::smooth_normal_derivatives(
  LocalTopology const &                   topology,
  Configuration const &                   x,
  std::array<std::array<double, 12>, 2> & dn) const
{
  std::array<std::array<Dual<12>, 2>, 6> X;
  for(unsigned int k = 0; k < topology.n; ++k)
    for(unsigned int c = 0; c < 2; ++c)
    {
      X[k][c]              = Dual<12>(x.x[topology.vertex[k]][c]);
      X[k][c].g[2 * k + c] = 1.0;
    }
  Dual<12>                gap;
  std::array<Dual<12>, 2> normal;
  smooth_projection_of_positions(topology, X, gap, normal);
  for(unsigned int c = 0; c < 2; ++c)
    for(unsigned int i = 0; i < 12; ++i)
      dn[c][i] = normal[c].g[i];
}

template<int dim, typename Number>
std::vector<unsigned int>
DeformableContact<dim, Number>::point_vertices(unsigned int const             point,
                                               ContactProjection<dim> const & state) const
{
  if(data.smooth_normals)
  {
    LocalTopology const topology = local_topology(point, state.master);
    return std::vector<unsigned int>(topology.vertex.begin(), topology.vertex.begin() + topology.n);
  }
  Face const & slave = faces[slave_faces[slave_points[point].slave_face].face];
  return {slave.vertex[0],
          slave.vertex[1],
          faces[state.master].vertex[0],
          faces[state.master].vertex[1]};
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::evaluate_residual_add(VectorType & dst, VectorType const & u) const
{
  ++stats.n_residuals;

  compute_positions(u, last_positions);
  search_all(last_positions, data.search, last_state);

  // diagnostics: master switches and status changes of this residual evaluation (e.g. a line
  // search trial) with respect to the linearization state
  if(data.log_linearization and lin_state.size() == last_state.size())
  {
    unsigned int n_switched = 0, n_status = 0;
    for(unsigned int p = 0; p < last_state.size(); ++p)
    {
      if(last_state[p].active() != lin_state[p].active())
        ++n_status;
      else if(last_state[p].active() and last_state[p].master != lin_state[p].master)
        ++n_switched;
    }
    std::cout << "      residual: master switches " << n_switched << ", status changes " << n_status
              << " with respect to the linearization state" << std::endl;
  }

  ScopedTime timer(stats.t_residual);
  for(unsigned int p = 0; p < slave_points.size(); ++p)
  {
    ContactProjection<dim> const & state = last_state[p];
    SlavePoint const &             sp    = slave_points[p];

    if(not state.found())
    {
      // Nitsche: the detection radius changes the result if the normal traction exceeds
      // gamma r_c (the point would be active for any gap below this distance)
      if(data.nitsche)
      {
        dealii::Tensor<2, dim> F, S;
        dealii::Tensor<1, dim> traction;
        slave_traction(p, u, F, S, traction);
        if(-(nitsche_points[p].N * traction) > nitsche_faces[sp.slave_face].gamma * data.radius)
          ++stats.n_cutoff;
      }
      continue;
    }

    // contact pressure lambda = [A]_+ (penalty: A = - epsilon g, Nitsche: n . (P N) - gamma g)
    double const lambda = std::max(pressure_argument(p, state, u, nullptr), 0.0);
    if(lambda == 0.0)
      continue;

    if(data.smooth_normals)
    {
      // R = - w lambda dg/du (gradient by automatic differentiation)
      LocalTopology const    topology = local_topology(p, state.master);
      double                 gap;
      std::array<double, 12> gradient;
      smooth_gap_derivatives(topology, last_positions, false, gap, gradient, nullptr);
      double const factor = -sp.weight * lambda;
      for(unsigned int k = 0; k < topology.n; ++k)
      {
        dealii::Tensor<1, dim> value;
        for(unsigned int c = 0; c < dim; ++c)
          value[c] = factor * gradient[2 * k + c];
        add(dst, topology.vertex[k], value);
      }
      continue;
    }

    Face const & slave  = faces[slave_faces[sp.slave_face].face];
    Face const & master = faces[state.master];

    // R = - w lambda dg/du with dg = (dx_s - dx_m) . n
    dealii::Tensor<1, dim> const f = (sp.weight * lambda) * state.normal;
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

  std::vector<ContactProjection<dim>> const previous = lin_state;

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

  // active set and contact pressure (and the slave traction for Nitsche's method)
  {
    ScopedTime timer(stats.t_lin);
    lin_active.assign(lin_state.size(), 0);
    lin_lambda.assign(lin_state.size(), 0.0);
    if(data.nitsche)
      lin_nitsche.resize(lin_state.size());
    for(unsigned int p = 0; p < lin_state.size(); ++p)
      if(lin_state[p].found())
      {
        double const A =
          pressure_argument(p, lin_state[p], u, data.nitsche ? &lin_nitsche[p] : nullptr);
        if(A >= 0.0)
        {
          lin_active[p] = 1;
          lin_lambda[p] = A;
        }
      }
  }

  // smooth normals: gradient and Hessian of the gap of the active points
  if(data.smooth_normals)
  {
    ScopedTime timer(stats.t_lin);
    lin_smooth.resize(lin_state.size());
    for(unsigned int p = 0; p < lin_state.size(); ++p)
      if(lin_active[p])
      {
        lin_smooth[p].topology = local_topology(p, lin_state[p].master);
        double gap;
        // the Hessian is only needed for the geometric part of the Jacobian
        smooth_gap_derivatives(lin_smooth[p].topology,
                               x,
                               not data.penalty_part_only,
                               gap,
                               lin_smooth[p].gradient,
                               &lin_smooth[p].hessian);
        if(data.nitsche and not data.penalty_part_only)
          smooth_normal_derivatives(lin_smooth[p].topology, x, lin_nitsche[p].dn);
      }
  }

  // changes of the contact state with respect to the previous linearization
  if(previous.size() == lin_state.size())
  {
    unsigned int n_activated = 0, n_deactivated = 0, n_switched = 0, n_zone = 0, n_active = 0;
    double       max_angle = 0.0, min_gap = std::numeric_limits<double>::max();
    for(unsigned int p = 0; p < lin_state.size(); ++p)
    {
      ContactProjection<dim> const &old_state = previous[p], &new_state = lin_state[p];
      if(new_state.active())
      {
        ++n_active;
        if(new_state.xi < 0.0 or new_state.xi > 1.0)
          ++n_zone;
      }
      if(new_state.active() != old_state.active())
      {
        if(new_state.active())
          ++n_activated;
        else
          ++n_deactivated;
        min_gap = std::min(min_gap,
                           std::min(old_state.found() ? std::abs(old_state.gap) : min_gap,
                                    new_state.found() ? std::abs(new_state.gap) : min_gap));
      }
      else if(new_state.active() and new_state.master != old_state.master)
      {
        ++n_switched;
        double const cosine = std::max(-1.0, std::min(1.0, new_state.normal * old_state.normal));
        max_angle           = std::max(max_angle, std::acos(cosine));
      }
    }
    stats.n_activated += n_activated;
    stats.n_deactivated += n_deactivated;
    stats.n_switched += n_switched;
    stats.n_in_tolerance_zone += n_zone;
    stats.max_switch_angle    = std::max(stats.max_switch_angle, max_angle);
    stats.min_abs_gap_changed = std::min(stats.min_abs_gap_changed, min_gap);

    // active points with a second admissible master face (overlap of the projection domains at a
    // master vertex): angle between the two normals and relative difference of |g|
    unsigned int n_overlap         = 0;
    double       max_overlap_angle = 0.0, min_relative_difference = 1.0;
    if(data.log_linearization)
      for(unsigned int p = 0; p < lin_state.size(); ++p)
      {
        if(not lin_state[p].active())
          continue;
        SlaveFace const &            slave_face = slave_faces[slave_points[p].slave_face];
        dealii::Point<dim> const     x_s        = slave_position(x, p);
        dealii::Tensor<1, dim> const n_s        = face_normal(x, slave_face.face);
        ContactProjection<dim>       candidate;
        bool                         overlap = false;
        for(unsigned int const master : pairs[slave_face.pair].master_faces)
        {
          if(master == slave_face.face or master == lin_state[p].master)
            continue;
          if(project(x_s, n_s, x, master, candidate) and std::abs(candidate.gap) <= data.radius)
          {
            overlap = true;
            double const cosine =
              std::max(-1.0, std::min(1.0, candidate.normal * lin_state[p].normal));
            max_overlap_angle = std::max(max_overlap_angle, std::acos(cosine));
            min_relative_difference =
              std::min(min_relative_difference,
                       (std::abs(candidate.gap) - std::abs(lin_state[p].gap)) /
                         std::max(std::abs(lin_state[p].gap), 1e-300));
          }
        }
        n_overlap += overlap ? 1 : 0;
      }

    if(data.log_linearization)
      std::cout << "    overlap: " << n_overlap
                << " active points with two admissible faces (max angle " << std::scientific
                << std::setprecision(2) << max_overlap_angle << ", min relative |g| difference "
                << min_relative_difference << ")" << std::defaultfloat << std::endl;

    if(data.log_linearization)
      std::cout << "    contact state: active " << n_active << " (+" << n_activated << " -"
                << n_deactivated << "), master switches " << n_switched << " (max angle "
                << std::scientific << std::setprecision(2) << max_angle << "), tolerance zone "
                << n_zone << ", min |g| of status changes "
                << (min_gap < std::numeric_limits<double>::max() ? min_gap : 0.0)
                << std::defaultfloat << std::endl;
  }
}

template<int dim, typename Number>
bool
DeformableContact<dim, Number>::fully_constrained(unsigned int const vertex) const
{
  for(unsigned int c = 0; c < dim; ++c)
    if(not vertices[vertex].constrained[c])
      return false;
  return true;
}

template<int dim, typename Number>
bool
DeformableContact<dim, Number>::pattern_contains_couplings() const
{
  for(unsigned int p = 0; p < lin_state.size(); ++p)
  {
    if(not lin_active[p])
      continue;
    std::vector<unsigned int> const local = point_vertices(p, lin_state[p]);
    for(unsigned int const a : local)
      for(unsigned int const b : local)
        if(a < b and not fully_constrained(a) and not fully_constrained(b) and
           pattern_couplings.find({a, b}) == pattern_couplings.end())
          return false;
  }
  return true;
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::add_couplings_to_pattern(dealii::DynamicSparsityPattern & dsp) const
{
  for(unsigned int p = 0; p < lin_state.size(); ++p)
  {
    if(not lin_active[p])
      continue;
    std::vector<unsigned int> const local = point_vertices(p, lin_state[p]);
    for(unsigned int const a : local)
      for(unsigned int const b : local)
        if(a < b and not fully_constrained(a) and not fully_constrained(b))
          pattern_couplings.insert({a, b});
  }

  for(auto const & pair : pattern_couplings)
    for(unsigned int c = 0; c < dim; ++c)
      for(unsigned int d = 0; d < dim; ++d)
      {
        dsp.add(vertices[pair.first].dof[c], vertices[pair.second].dof[d]);
        dsp.add(vertices[pair.second].dof[d], vertices[pair.first].dof[c]);
      }

  ++stats.n_pattern_updates;
  stats.n_pattern_couplings = pattern_couplings.size();
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number>
void
DeformableContact<dim, Number>::add_to_matrix(dealii::TrilinosWrappers::SparseMatrix & matrix) const
{
  if(lin_state.size() != slave_points.size())
    return;

  ScopedTime timer(stats.t_matrix);
  ++stats.n_matrix_updates;

  // epsilon w (N_i n) (x) (N_j n) for the vertices i, j of one face with weights N
  auto const add_block = [&](std::array<unsigned int, 2> const & vertex,
                             std::array<double, 2> const &       N,
                             dealii::Tensor<1, dim> const &      n,
                             double const                        factor) {
    for(unsigned int i = 0; i < 2; ++i)
      for(unsigned int c = 0; c < dim; ++c)
      {
        if(vertices[vertex[i]].constrained[c])
          continue;
        for(unsigned int j = 0; j < 2; ++j)
          for(unsigned int d = 0; d < dim; ++d)
          {
            if(vertices[vertex[j]].constrained[d])
              continue;
            matrix.add(vertices[vertex[i]].dof[c],
                       vertices[vertex[j]].dof[d],
                       factor * N[i] * N[j] * n[c] * n[d]);
          }
      }
  };

  for(unsigned int p = 0; p < slave_points.size(); ++p)
  {
    ContactProjection<dim> const & state = lin_state[p];
    if(not lin_active[p])
      continue;

    SlavePoint const & sp     = slave_points[p];
    double const       factor = contact_stiffness(p) * sp.weight;
    if(data.smooth_normals)
    {
      // epsilon w (dg/du) (x) (dg/du) on all local vertices (with coupling) or on the slave and
      // master face blocks
      SmoothLinearization const & lin = lin_smooth[p];
      for(unsigned int k = 0; k < lin.topology.n; ++k)
        for(unsigned int c = 0; c < dim; ++c)
        {
          Vertex const & vk = vertices[lin.topology.vertex[k]];
          if(vk.constrained[c])
            continue;
          for(unsigned int l = 0; l < lin.topology.n; ++l)
          {
            if(not data.preconditioner_coupling)
            {
              bool const slave_k = (k == lin.topology.slave_a or k == lin.topology.slave_b);
              bool const slave_l = (l == lin.topology.slave_a or l == lin.topology.slave_b);
              bool const face_k  = (k == lin.topology.master_1 or k == lin.topology.master_2);
              bool const face_l  = (l == lin.topology.master_1 or l == lin.topology.master_2);
              if(not((slave_k and slave_l) or (face_k and face_l)))
                continue;
            }
            for(unsigned int d = 0; d < dim; ++d)
            {
              Vertex const & vl = vertices[lin.topology.vertex[l]];
              if(vl.constrained[d])
                continue;
              matrix.add(vk.dof[c],
                         vl.dof[d],
                         factor * lin.gradient[2 * k + c] * lin.gradient[2 * l + d]);
            }
          }
        }
      continue;
    }
    if(data.preconditioner_coupling)
    {
      // dg/du = (N_a n, N_b n, -N_1 n, -N_2 n) for the slave vertices a, b and the master
      // vertices 1, 2
      std::array<unsigned int, 2> const & slave_vertex =
        faces[slave_faces[sp.slave_face].face].vertex;
      std::array<unsigned int, 4> const vertex = {{slave_vertex[0],
                                                   slave_vertex[1],
                                                   faces[state.master].vertex[0],
                                                   faces[state.master].vertex[1]}};
      std::array<double, 4> const       N = {{1.0 - sp.eta, sp.eta, -(1.0 - state.xi), -state.xi}};
      for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int c = 0; c < dim; ++c)
        {
          if(vertices[vertex[i]].constrained[c])
            continue;
          for(unsigned int j = 0; j < 4; ++j)
            for(unsigned int d = 0; d < dim; ++d)
            {
              if(vertices[vertex[j]].constrained[d])
                continue;
              matrix.add(vertices[vertex[i]].dof[c],
                         vertices[vertex[j]].dof[d],
                         factor * N[i] * N[j] * state.normal[c] * state.normal[d]);
            }
        }
    }
    else
    {
      add_block(faces[slave_faces[sp.slave_face].face].vertex,
                {{1.0 - sp.eta, sp.eta}},
                state.normal,
                factor);
      add_block(faces[state.master].vertex, {{1.0 - state.xi, state.xi}}, state.normal, factor);
    }
  }

  matrix.compress(dealii::VectorOperation::add);
}
#endif

template<int dim, typename Number>
template<typename ReadDoF, typename Write>
void
DeformableContact<dim, Number>::apply_point_jacobian(unsigned int const point,
                                                     ReadDoF const &    read_dof,
                                                     Write const &      write) const
{
  ContactProjection<dim> const & state  = lin_state[point];
  SlavePoint const &             sp     = slave_points[point];
  double const                   w      = sp.weight;
  double const                   lambda = lin_lambda[point];

  // direction at a vertex of the contact (constrained components are zero)
  auto const read = [&](unsigned int const vertex) {
    dealii::Tensor<1, dim> value;
    for(unsigned int c = 0; c < dim; ++c)
      if(not vertices[vertex].constrained[c])
        value[c] = read_dof(vertices[vertex].dof[c]);
    return value;
  };

  // Nitsche: delta (n . (P N)) = delta n . t + n . (delta P N) with delta P = F delta S + G S,
  // G = Grad(delta d) in the slave cell
  auto const delta_normal_traction = [&](dealii::Tensor<1, dim> const & delta_n) {
    NitscheState const &   ns   = lin_nitsche[point];
    NitscheFace const &    face = nitsche_faces[sp.slave_face];
    NitschePoint const &   np   = nitsche_points[point];
    dealii::Tensor<2, dim> G;
    for(unsigned int j = 0; j < face.dofs.size(); ++j)
      if(not face.constrained[j])
      {
        double const value = read_dof(face.dofs[j]);
        for(unsigned int d = 0; d < dim; ++d)
          G[face.component[j]][d] += value * np.grad_phi[j][d];
      }
    dealii::Tensor<2, dim> H = ns.F;
    for(unsigned int d = 0; d < dim; ++d)
      H[d][d] -= 1.0;
    dealii::Tensor<2, dim> const delta_E =
      0.5 * (G + transpose(G) + transpose(G) * H + transpose(H) * G);
    dealii::Tensor<2, dim> const delta_P = ns.F * svk_stress(delta_E) + G * ns.S;
    return delta_n * ns.traction + state.normal * (delta_P * np.N);
  };

  // J v = alpha dg/du - beta d^2g/du^2 v with
  //  penalty: alpha = w epsilon (dg . v), beta = w lambda,
  //  Nitsche: alpha = - w delta A, delta A = delta (n . (P N)) - gamma (dg . v), beta = w lambda,
  //  positive semi-definite part only: alpha = w kappa (dg . v), beta = 0
  double const kappa = contact_stiffness(point);
  double const beta  = data.penalty_part_only ? 0.0 : w * lambda;

  if(data.smooth_normals)
  {
    SmoothLinearization const & lin    = lin_smooth[point];
    unsigned int const          n_dofs = 2 * lin.topology.n;
    std::array<double, 12>      v_local;
    for(unsigned int k = 0; k < lin.topology.n; ++k)
    {
      dealii::Tensor<1, dim> const value = read(lin.topology.vertex[k]);
      for(unsigned int c = 0; c < dim; ++c)
        v_local[2 * k + c] = value[c];
    }
    double delta_g = 0.0;
    for(unsigned int i = 0; i < n_dofs; ++i)
      delta_g += lin.gradient[i] * v_local[i];

    double alpha = w * kappa * delta_g;
    if(data.nitsche and not data.penalty_part_only)
    {
      dealii::Tensor<1, dim> delta_n;
      for(unsigned int c = 0; c < dim; ++c)
        for(unsigned int i = 0; i < n_dofs; ++i)
          delta_n[c] += lin_nitsche[point].dn[c][i] * v_local[i];
      alpha = -w * (delta_normal_traction(delta_n) - kappa * delta_g);
    }

    for(unsigned int k = 0; k < lin.topology.n; ++k)
    {
      dealii::Tensor<1, dim> value;
      for(unsigned int c = 0; c < dim; ++c)
      {
        unsigned int const i  = 2 * k + c;
        double             Hv = 0.0;
        if(not data.penalty_part_only)
          for(unsigned int j = 0; j < n_dofs; ++j)
            Hv += lin.hessian[i * 12 + j] * v_local[j];
        value[c] = alpha * lin.gradient[i] - beta * Hv;
      }
      write(lin.topology.vertex[k], value);
    }
    return;
  }

  Face const & slave  = faces[slave_faces[sp.slave_face].face];
  Face const & master = faces[state.master];

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
  //  Delta delta g = - a / l (delta x_s - delta x_m) . t - (b / l + g a / l^2)
  //                  n . (delta x_2 - delta x_1)
  // with a = n . (Delta x_2 - Delta x_1), b = (Delta x_s - Delta x_m) . t
  double const a   = n * (delta_2 - delta_1);
  double const b   = delta_r * t;
  double const c_n = b / l + g * a / (l * l);
  double const c_t = a / l;

  dealii::Tensor<1, dim> const c_s = -c_t * t;
  dealii::Tensor<1, dim> const c_1 = c_t * (1.0 - xi) * t + c_n * n;
  dealii::Tensor<1, dim> const c_2 = c_t * xi * t - c_n * n;

  double alpha = w * kappa * delta_g;
  if(data.nitsche and not data.penalty_part_only)
  {
    // variation of the master normal: delta n = - t (n . (delta x_2 - delta x_1)) / l
    dealii::Tensor<1, dim> const delta_n = -(a / l) * t;
    alpha                                = -w * (delta_normal_traction(delta_n) - kappa * delta_g);
  }

  dealii::Tensor<1, dim> const f_s = alpha * n - beta * c_s;
  dealii::Tensor<1, dim> const f_1 = -alpha * (1.0 - xi) * n - beta * c_1;
  dealii::Tensor<1, dim> const f_2 = -alpha * xi * n - beta * c_2;

  write(slave.vertex[0], (1.0 - sp.eta) * f_s);
  write(slave.vertex[1], sp.eta * f_s);
  write(master.vertex[0], f_1);
  write(master.vertex[1], f_2);
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::apply_jacobian_add(VectorType & dst, VectorType const & v) const
{
  ++stats.n_vmults;
  ScopedTime timer(stats.t_vmult);

  AssertThrow(lin_state.size() == slave_points.size(),
              dealii::ExcMessage("Deformable contact: linearization state not set."));

  // constrained entries of v are treated as zero (in apply_point_jacobian()), constrained rows of
  // dst are not written
  auto const read_dof = [&](dealii::types::global_dof_index const i) { return double(v(i)); };
  auto const write    = [&](unsigned int const vertex, dealii::Tensor<1, dim> const & value) {
    add(dst, vertex, value);
  };

  for(unsigned int p = 0; p < slave_points.size(); ++p)
    if(lin_active[p])
      apply_point_jacobian(p, read_dof, write);
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::add_diagonal(VectorType & diagonal) const
{
  if(lin_state.size() != slave_points.size())
    return;

  for(unsigned int p = 0; p < slave_points.size(); ++p)
  {
    if(not lin_active[p])
      continue;

    // diagonal entry of the point contribution for each unconstrained DoF of the vertices of
    // the contact (the rows of the contact term): apply the point Jacobian to the unit vector of
    // this DoF (vertices that appear in several roles, e.g. common vertices of slave and master
    // faces in self-contact, are one DoF; for Nitsche's method, the unit vector also acts on the
    // traction of the slave cell)
    std::vector<unsigned int> local = point_vertices(p, lin_state[p]);
    std::sort(local.begin(), local.end());
    local.erase(std::unique(local.begin(), local.end()), local.end());
    for(unsigned int const vertex : local)
      for(unsigned int c = 0; c < dim; ++c)
      {
        if(vertices[vertex].constrained[c])
          continue;
        dealii::types::global_dof_index const dof = vertices[vertex].dof[c];

        auto const read_dof = [&](dealii::types::global_dof_index const i) {
          return i == dof ? 1.0 : 0.0;
        };
        double     entry = 0.0;
        auto const write = [&](unsigned int const w, dealii::Tensor<1, dim> const & value) {
          if(w == vertex)
            entry += value[c];
        };
        apply_point_jacobian(p, read_dof, write);
        diagonal(dof) += entry;
      }
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
    if(not state[p].found())
      continue;
    result.min_gap = std::min(result.min_gap, state[p].gap);
    double const A = pressure_argument(p, state[p], u, nullptr);
    if(A < 0.0)
      continue;
    double const lambda = A;
    if(data.smooth_normals)
    {
      // force on the slave face: w lambda dg/dx_s (derivatives with respect to the slave vertices)
      LocalTopology const    topology = local_topology(p, state[p].master);
      double                 gap;
      std::array<double, 12> gradient;
      smooth_gap_derivatives(topology, x, false, gap, gradient, nullptr, true);
      for(unsigned int c = 0; c < dim; ++c)
        result.force[c] +=
          slave_points[p].weight * lambda *
          (gradient[2 * topology.slave_a + c] + gradient[2 * topology.slave_b + c]);
    }
    else
      result.force += (slave_points[p].weight * lambda) * state[p].normal;
    result.n_active += 1;
    result.active_area += slave_points[p].weight;
    result.max_pressure = std::max(result.max_pressure, lambda);
  }
  return result;
}

template<int dim, typename Number>
bool
DeformableContact<dim, Number>::update_multipliers(VectorType const & u,
                                                   double &           max_penetration) const
{
  AssertThrow(data.augmented_lagrangian, dealii::ExcInternalError());

  Configuration x;
  compute_positions(u, x);
  std::vector<ContactProjection<dim>> state;
  search_all(x, data.search, state);

  std::vector<double> updated(slave_points.size(), 0.0);
  max_penetration = 0.0;
  for(unsigned int p = 0; p < slave_points.size(); ++p)
    if(state[p].found())
    {
      updated[p] = std::max(pressure_argument(p, state[p], u, nullptr), 0.0);
      if(updated[p] > 0.0)
        max_penetration = std::max(max_penetration, -state[p].gap);
    }

  if(max_penetration <= data.uzawa_gap_tolerance)
    return true;

  multipliers = updated;
  ++stats.n_uzawa;
  return false;
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::set_activation_shift(double const shift) const
{
  activation_shift = shift;
}

template<int dim, typename Number>
std::vector<double> const &
DeformableContact<dim, Number>::get_multipliers() const
{
  return multipliers;
}

template<int dim, typename Number>
void
DeformableContact<dim, Number>::set_multipliers(std::vector<double> const & values) const
{
  AssertThrow(values.size() == multipliers.size(), dealii::ExcInternalError());
  multipliers = values;
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
  if(data.smooth_normals)
    line("    rejected by the pre-filter", s.n_prefiltered, 0.0);
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
        << "  Uzawa updates (augmented Lagrangian) = " << s.n_uzawa << std::endl
        << "  ties / cutoff hits                 = " << s.n_ties << " / " << s.n_cutoff
        << std::endl;
  pcout << "  state changes between linearizations: activated " << s.n_activated << ", deactivated "
        << s.n_deactivated << ", master switches " << s.n_switched << " (max angle "
        << s.max_switch_angle << "), tolerance zone " << s.n_in_tolerance_zone << std::endl;
  if(data.add_to_preconditioner)
    pcout << "  preconditioner matrix: " << s.n_matrix_updates << " updates, " << s.t_matrix
          << " s, coupling " << (data.preconditioner_coupling ? "yes" : "no")
          << ", pattern extensions " << s.n_pattern_updates << " (" << s.n_pattern_couplings
          << " vertex pairs)" << std::endl;
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
