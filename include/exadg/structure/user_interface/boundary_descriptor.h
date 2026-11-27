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

#ifndef EXADG_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_
#define EXADG_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/component_mask.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/container_interface_data.h>

namespace ExaDG
{
namespace Structure
{
enum class BoundaryType
{
  Undefined,
  Dirichlet,
  DirichletCached,
  Neumann,
  NeumannCached,
  Contact
};

template<int dim>
struct BoundaryDescriptor
{
  // Dirichlet
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_bc;

  // Initial acceleration prescribed on Dirichlet boundary:
  // This data structure will only be used if the initial_acceleration is not set in FieldFunctions.
  // Moreover, this data structure will only be used for unsteady problems.
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    dirichlet_bc_initial_acceleration;

  // ComponentMask
  // If a certain boundary ID is not inserted into this map, it is assumed that all components are
  // active, in analogy to the default constructor of dealii::ComponentMask.
  std::map<dealii::types::boundary_id, dealii::ComponentMask> dirichlet_bc_component_mask;

  // Another type of Dirichlet boundary condition where the Dirichlet values come
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Dirichlet boundary condition
  // is required for the ALE mesh deformation problem in fluid-structure interaction).
  // ComponentMask is not implemented/available for this type of boundary condition.
  std::set<dealii::types::boundary_id> dirichlet_cached_bc;

  // Neumann
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> neumann_bc;

  // another type of Neumann boundary condition where the traction force comes
  // from the solution on another domain that is in contact with the actual domain
  // of interest at the given boundary (this type of Neumann boundary condition
  // is required for fluid-structure interaction problems)
  std::set<dealii::types::boundary_id> neumann_cached_bc;

  // Frictionless contact with a rigid obstacle (penalty method, see
  // Parameters::contact_penalty_parameter). The function describes the obstacle by a signed
  // distance function phi(x) evaluated in the current configuration x = X + d:
  //
  //  - value(x)    : gap g = phi(x), positive if the point x is separated from the obstacle,
  //                  negative in case of penetration,
  //  - gradient(x) : unit normal n = grad(phi) pointing from the obstacle towards the body,
  //  - hessian(x)  : curvature of the obstacle, grad(grad(phi)) (zero for a plane).
  //
  // All three functions have to be implemented. The time of the function is set to the current
  // time (or load factor in case of quasi-static problems), which allows moving obstacles.
  // Contact is only available for the nonlinear operator (large_deformation = true).
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> contact_bc;

  // Optional for contact boundaries with friction: displacement of the material points of the
  // obstacle (dim components) as a function of the current position x and the time, used to
  // compute the relative slip between body and obstacle. For a rigid translation c(t) of the
  // obstacle, the function returns c(t) independently of x. If no function is given for a contact
  // boundary, the obstacle does not move tangentially.
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    contact_obstacle_displacement;

  // Optional for contact boundaries with friction: friction coefficient mu(x, t) (scalar function
  // of the current position and the time), which replaces Parameters::contact_friction_coefficient
  // on this boundary (friction has to be activated by a positive
  // Parameters::contact_friction_coefficient). This allows, e.g., a frictionless loading phase.
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    contact_friction_coefficient;

  // Frictionless contact between deformable surfaces of the same mesh (penalty method, see
  // DeformableContact and Parameters::deformable_contact_*): pairs (slave boundary id, master
  // boundary id), slave == master for self-contact. The contact term is added to the boundary
  // condition of these boundaries, which therefore need a boundary type of their own (typically
  // Neumann with zero traction).
  std::vector<std::pair<dealii::types::boundary_id, dealii::types::boundary_id>>
    deformable_contact_pairs;

  inline DEAL_II_ALWAYS_INLINE //
    BoundaryType
    get_boundary_type(dealii::types::boundary_id const & boundary_id) const
  {
    if(this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
      return BoundaryType::Dirichlet;
    else if(this->dirichlet_cached_bc.find(boundary_id) != this->dirichlet_cached_bc.end())
      return BoundaryType::DirichletCached;
    else if(this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
      return BoundaryType::Neumann;
    else if(this->neumann_cached_bc.find(boundary_id) != this->neumann_cached_bc.end())
      return BoundaryType::NeumannCached;
    else if(this->contact_bc.find(boundary_id) != this->contact_bc.end())
      return BoundaryType::Contact;

    AssertThrow(false,
                dealii::ExcMessage(
                  "Could not find a boundary type to the specified boundary_id = " +
                  std::to_string(boundary_id) +
                  ". A possible reason is that you "
                  "forgot to define a boundary condition for this boundary_id, or "
                  "that the boundary type associated to this boundary has not been "
                  "implemented."));

    return BoundaryType::Undefined;
  }

  inline DEAL_II_ALWAYS_INLINE //
    void
    verify_boundary_conditions(
      dealii::types::boundary_id const             boundary_id,
      std::set<dealii::types::boundary_id> const & periodic_boundary_ids) const
  {
    unsigned int counter = 0;
    if(dirichlet_bc.find(boundary_id) != dirichlet_bc.end())
    {
      counter++;

      AssertThrow(
        dirichlet_bc_component_mask.find(boundary_id) != dirichlet_bc_component_mask.end(),
        dealii::ExcMessage(
          "dirichlet_bc_component_mask must contain the same boundary IDs as dirichlet_bc."));

      AssertThrow(
        dirichlet_bc_initial_acceleration.find(boundary_id) !=
          dirichlet_bc_initial_acceleration.end(),
        dealii::ExcMessage(
          "dirichlet_bc_initial_acceleration must contain the same boundary IDs as dirichlet_bc."));
    }

    if(dirichlet_cached_bc.find(boundary_id) != dirichlet_cached_bc.end())
      counter++;

    if(neumann_bc.find(boundary_id) != neumann_bc.end())
      counter++;

    if(neumann_cached_bc.find(boundary_id) != neumann_cached_bc.end())
      counter++;

    if(contact_bc.find(boundary_id) != contact_bc.end())
      counter++;

    if(periodic_boundary_ids.find(boundary_id) != periodic_boundary_ids.end())
      counter++;

    AssertThrow(counter == 1,
                dealii::ExcMessage("Boundary face with non-unique boundary type found."));
  }

  void
  set_dirichlet_cached_data(
    std::shared_ptr<ContainerInterfaceData<1, dim, double> const> interface_data) const
  {
    dirichlet_cached_data = interface_data;
  }

  void
  set_neumann_cached_data(
    std::shared_ptr<ContainerInterfaceData<1, dim, double> const> interface_data) const
  {
    neumann_cached_data = interface_data;
  }

  std::shared_ptr<ContainerInterfaceData<1, dim, double> const>
  get_dirichlet_cached_data() const
  {
    AssertThrow(dirichlet_cached_data.get(),
                dealii::ExcMessage("Pointer to ContainerInterfaceData has not been initialized."));

    return dirichlet_cached_data;
  }

  std::shared_ptr<ContainerInterfaceData<1, dim, double> const>
  get_neumann_cached_data() const
  {
    AssertThrow(neumann_cached_data.get(),
                dealii::ExcMessage("Pointer to ContainerInterfaceData has not been initialized."));

    return neumann_cached_data;
  }

private:
  mutable std::shared_ptr<ContainerInterfaceData<1, dim, double> const> dirichlet_cached_data;
  mutable std::shared_ptr<ContainerInterfaceData<1, dim, double> const> neumann_cached_data;
};

} // namespace Structure
} // namespace ExaDG

#endif /* EXADG_STRUCTURE_USER_INTERFACE_BOUNDARY_DESCRIPTOR_H_ */
