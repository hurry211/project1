#ifndef MFEM_AGENT_POISSON_DIAGNOSTICS_HPP
#define MFEM_AGENT_POISSON_DIAGNOSTICS_HPP

#include "mfem.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace poisson_diagnostics
{

struct Residual
{
   double absolute_l2;
   double rhs_l2;
   double relative_l2;
};

// Use the actual constrained linear system, including its Dirichlet rows.
// Applying Operator::Mult also works for a partially assembled operator.
inline Residual MeasureResidual(const mfem::Operator &A, const mfem::Vector &X,
                                const mfem::Vector &B)
{
   mfem::Vector residual(B.Size());
   A.Mult(X, residual);
   residual -= B;
   const double undefined = std::numeric_limits<double>::quiet_NaN();
   // MFEM's scaled Euclidean norm can skip NaN entries during reduction.
   const double absolute = residual.CheckFinite() == 0 ? residual.Norml2() : undefined;
   const double rhs = B.CheckFinite() == 0 ? B.Norml2() : undefined;
   const bool normalizable = rhs > 0.0 && std::isfinite(rhs) && std::isfinite(absolute);
   return {absolute, rhs, normalizable ? absolute / rhs : undefined};
}

struct BoundaryFlux
{
   // Physical sides in bottom, right, top, left order (independent of tag IDs).
   std::array<double, 4> outward_by_side{};
   double source_integral = 0.0;

   double OutwardTotal() const
   {
      return std::accumulate(outward_by_side.begin(), outward_by_side.end(), 0.0);
   }
};

inline int UnitSquareSide(const mfem::real_t *a, const mfem::real_t *b)
{
   constexpr double tol = 1e-12;
   if (std::abs(a[1]) <= tol && std::abs(b[1]) <= tol) { return 0; }
   if (std::abs(a[0] - 1.0) <= tol && std::abs(b[0] - 1.0) <= tol) { return 1; }
   if (std::abs(a[1] - 1.0) <= tol && std::abs(b[1] - 1.0) <= tol) { return 2; }
   if (std::abs(a[0]) <= tol && std::abs(b[0]) <= tol) { return 3; }
   return -1;
}

inline BoundaryFlux MeasureBoundaryFlux(mfem::Mesh &mesh, const mfem::GridFunction &u,
                                        bool field_is_finite, mfem::Coefficient &source,
                                        const mfem::IntegrationRule &volume_rule,
                                        const mfem::IntegrationRule &boundary_rule,
                                        mfem::Coefficient *diffusivity,
                                        const mfem::Array<int> &boundary_sides,
                                        double diffusion_x = 1.0)
{
   BoundaryFlux flux;
   // Integrate the volume coefficient actually used by the native RHS,
   // including any injected source fault. Boundary loads are separate.
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      auto *T = mesh.GetElementTransformation(e);
      for (int q = 0; q < volume_rule.GetNPoints(); q++)
      {
         const auto &ip = volume_rule.IntPoint(q);
         T->SetIntPoint(&ip);
         flux.source_integral += ip.weight * T->Weight() * source.Eval(*T, ip);
      }
   }

   if (!field_is_finite)
   {
      flux.outward_by_side.fill(std::numeric_limits<double>::quiet_NaN());
      return flux;
   }

   mfem::Vector gradient(mesh.SpaceDimension()), normal(mesh.SpaceDimension());
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      auto *face = mesh.GetBdrFaceTransformations(be);
      double &side = flux.outward_by_side.at(boundary_sides[be]);
      for (int q = 0; q < boundary_rule.GetNPoints(); q++)
      {
         const auto &ip = boundary_rule.IntPoint(q);
         face->SetAllIntPoints(&ip);
         u.GetGradient(*face->Elem1, gradient);
         gradient(0) *= diffusion_x;
         // On an exterior face CalcOrtho gives the outward normal scaled by
         // the surface Jacobian. The trace comes from the adjacent element.
         mfem::CalcOrtho(face->Jacobian(), normal);
         const double kappa = diffusivity ?
                              diffusivity->Eval(*face->Elem1, face->GetElement1IntPoint()) : 1.0;
         side -= ip.weight * kappa * (gradient * normal);
      }
   }
   return flux;
}

} // namespace poisson_diagnostics

#endif
