// manufactured_cases.hpp
//
// Registry of manufactured solutions compiled into the MFEM solver.
//
// Design note: the solver does NOT parse arbitrary expressions. Each case is a
// reviewed, hard-coded C++ implementation of u, grad(u) and f = -lap(u). The
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

// Both cases use the same weak form and trace rules. The study request assigns
// individual sides to those rules. A Neumann datum is the outward normal
// derivative in the load integral; the outward diffusive flux has opposite sign.
struct Formulation
{
   const char *equation;
   const char *domain;
   const char *space;
   const char *bilinear_integrator;
   const char *dirichlet_trace;
   const char *neumann_trace;
};

inline constexpr Formulation UnitSquarePoisson =
{
   "-laplacian(u) = f", "unit_square_2d", "scalar_H1", "DiffusionIntegrator",
   "u_exact", "grad(u_exact).n"
};

struct ManufacturedCase
{
   std::string id;
   std::string exact_solution;   // canonical expression string for u(x,y)
   std::string source;           // canonical expression string for f = -lap(u)
   mfem::real_t (*u)(const mfem::Vector &);
   void (*grad_u)(const mfem::Vector &, mfem::Vector &);
   mfem::real_t (*f)(const mfem::Vector &);
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
