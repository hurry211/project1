// manufactured_cases.hpp
//
// Registry of manufactured solutions compiled into the MFEM solver.
//
// Design note: the solver does NOT parse arbitrary expressions. Each case is a
// reviewed C++ implementation of u, grad(u), kappa and f = -div(kappa grad u). The
// canonical expression strings below are emitted by `poisson_mms --list-cases`
// so that the Python layer can bind a ProblemSpec (which stores expression
// strings) to a compiled case and refuse to run on any textual mismatch.
//
// The independent verifier never uses these C++ functions: it re-evaluates the
// expression strings stored in the frozen ProblemSpec with its own evaluator.

#ifndef MFEM_AGENT_MANUFACTURED_CASES_HPP
#define MFEM_AGENT_MANUFACTURED_CASES_HPP

#include "mfem.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace mms
{

// The natural load is kappa grad(u).n. The outward diffusive flux is its negative.
struct Formulation
{
   const char *equation;
   const char *domain;
   const char *space;
   const char *bilinear_integrator;
   const char *dirichlet_trace;
   const char *neumann_trace;
   const char *kappa = "1";
};

inline constexpr Formulation UnitSquarePoisson =
{
   "-laplacian(u) = f", "unit_square_2d", "scalar_H1", "DiffusionIntegrator",
   "u_exact", "grad(u_exact).n"
};

inline constexpr Formulation UnitSquareVariableDiffusion =
{
   "-div(kappa*grad(u)) = f", "unit_square_2d", "scalar_H1", "DiffusionIntegrator",
   "u_exact", "kappa*grad(u_exact).n", "1+x"
};

struct ManufacturedCase
{
   std::string id;
   std::string exact_solution;   // canonical expression string for u(x,y)
   std::string source;           // canonical expression string for f
   mfem::real_t (*u)(const mfem::Vector &);
   void (*grad_u)(const mfem::Vector &, mfem::Vector &);
   mfem::real_t (*f)(const mfem::Vector &);
   const Formulation &form = UnitSquarePoisson;
   mfem::real_t (*diffusivity)(const mfem::Vector &) = nullptr;
   mfem::real_t (*source_without_coefficient_gradient)(const mfem::Vector &) = nullptr;
};

// ---------------------------------------------------------------------------
// Case "sin_sin": u = sin(pi x) sin(pi y), homogeneous on the unit square.
// ---------------------------------------------------------------------------
inline mfem::real_t sin_sin_u(const mfem::Vector &p)
{
   return std::sin(M_PI * p(0)) * std::sin(M_PI * p(1));
}
inline void sin_sin_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = M_PI * std::cos(M_PI * p(0)) * std::sin(M_PI * p(1));
   g(1) = M_PI * std::sin(M_PI * p(0)) * std::cos(M_PI * p(1));
}
inline mfem::real_t sin_sin_f(const mfem::Vector &p)
{
   return 2.0 * M_PI * M_PI * std::sin(M_PI * p(0)) * std::sin(M_PI * p(1));
}

// ---------------------------------------------------------------------------
// Case "cos_exp": u = cos(pi x) exp(y). Non-zero and different on every side
// of the unit square, so boundary-label swaps and wrong Neumann data change
// the discrete solution (an all-zero Dirichlet test cannot detect swaps).
// -lap(u) = (pi^2 - 1) cos(pi x) exp(y).
// ---------------------------------------------------------------------------
inline mfem::real_t cos_exp_u(const mfem::Vector &p)
{
   return std::cos(M_PI * p(0)) * std::exp(p(1));
}
inline void cos_exp_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = -M_PI * std::sin(M_PI * p(0)) * std::exp(p(1));
   g(1) = std::cos(M_PI * p(0)) * std::exp(p(1));
}
inline mfem::real_t cos_exp_f(const mfem::Vector &p)
{
   return (M_PI * M_PI - 1.0) * std::cos(M_PI * p(0)) * std::exp(p(1));
}

inline mfem::real_t one_plus_x(const mfem::Vector &p)
{
   return 1.0 + p(0);
}

inline mfem::real_t variable_u(const mfem::Vector &p)
{
   return std::sin(M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
}
inline void variable_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = M_PI * std::cos(M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
   g(1) = 2.0 * M_PI * std::sin(M_PI * p(0)) * std::cos(2.0 * M_PI * p(1));
}
inline mfem::real_t variable_laplacian_term(const mfem::Vector &p)
{
   return 5.0 * M_PI * M_PI * one_plus_x(p) * variable_u(p);
}
inline mfem::real_t variable_f(const mfem::Vector &p)
{
   // grad(kappa) = (1,0), so the product rule contributes -du/dx.
   return variable_laplacian_term(p)
          - M_PI * std::cos(M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
}

inline const std::vector<ManufacturedCase> &Registry()
{
   static const std::vector<ManufacturedCase> cases =
   {
      {
         "sin_sin",
         "sin(pi*x)*sin(pi*y)",
         "2*pi**2*sin(pi*x)*sin(pi*y)",
         sin_sin_u, sin_sin_grad, sin_sin_f
      },
      {
         "cos_exp",
         "cos(pi*x)*exp(y)",
         "(pi**2 - 1)*cos(pi*x)*exp(y)",
         cos_exp_u, cos_exp_grad, cos_exp_f
      },
      {
         "sin_sin_variable",
         "sin(pi*x)*sin(2*pi*y)",
         "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
         variable_u, variable_grad, variable_f,
         UnitSquareVariableDiffusion, one_plus_x, variable_laplacian_term
      }
   };
   return cases;
}

inline const ManufacturedCase *Find(const std::string &id)
{
   for (const auto &c : Registry())
   {
      if (c.id == id) { return &c; }
   }
   return nullptr;
}

} // namespace mms

#endif
