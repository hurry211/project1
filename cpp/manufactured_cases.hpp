// manufactured_cases.hpp
//
// Registry of manufactured solutions compiled into the MFEM solver.
//
// Design note: the solver does NOT parse arbitrary expressions. Each case is a
// reviewed, hard-coded C++ implementation of u, grad(u) and f. The
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

struct ManufacturedCase
{
   std::string id;
   std::string exact_solution;   // canonical expression string for u(x,y)
   std::string source;           // canonical expression string for f
   mfem::real_t (*u)(const mfem::Vector &);
   void (*grad_u)(const mfem::Vector &, mfem::Vector &);
   mfem::real_t (*f)(const mfem::Vector &);
   std::string equation = "-laplacian(u) = f";
   double diffusion_x = 1.0;
   bool layered = false;
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
inline mfem::real_t anisotropic_f(const mfem::Vector &p)
{
   return 1001.0 * M_PI * M_PI * sin_sin_u(p);
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

inline mfem::real_t layered_a(const mfem::Vector &p) { return p(0) < 0.5 ? 1.0 : 10000.0; }
inline mfem::real_t layered_u(const mfem::Vector &p)
{
   const double flux = 2.0 / (1.0 + 1.0 / 10000.0);
   return p(0) < 0.5 ? flux * p(0) : 1.0 + flux * (p(0) - 1.0) / 10000.0;
}
inline void layered_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = (2.0 / (1.0 + 1.0 / 10000.0)) / layered_a(p);
   g(1) = 0.0;
}
inline mfem::real_t zero_source(const mfem::Vector &) { return 0.0; }

inline double gaussian_r2(const mfem::Vector &p)
{
   const double x = p(0) - 0.375, y = p(1) - 0.625;
   return x*x + y*y;
}
inline mfem::real_t gaussian_u(const mfem::Vector &p) { return std::exp(-25.0 * gaussian_r2(p)); }
inline void gaussian_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   const double scale = -50.0 * gaussian_u(p);
   g(0) = scale * (p(0) - 0.375);
   g(1) = scale * (p(1) - 0.625);
}
inline mfem::real_t gaussian_f(const mfem::Vector &p)
{
   return (100.0 - 2500.0 * gaussian_r2(p)) * gaussian_u(p);
}

inline const std::vector<ManufacturedCase> &Registry()
{
   static const std::vector<ManufacturedCase> cases =
   {
      {
         "gaussian_offset", "exp(-25*((x-0.375)**2+(y-0.625)**2))",
         "(100-2500*((x-0.375)**2+(y-0.625)**2))*exp(-25*((x-0.375)**2+(y-0.625)**2))",
         gaussian_u, gaussian_grad, gaussian_f
      },
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
         "anisotropic_1000_sin_sin",
         "sin(pi*x)*sin(pi*y)",
         "1001*pi**2*sin(pi*x)*sin(pi*y)",
         sin_sin_u, sin_sin_grad, anisotropic_f,
         "-div(diag(1000,1)*grad(u)) = f", 1000.0
      },
      {
         "layered_unit_drop_10000",
         "(10001*x-9999*(sqrt((x-0.5)**2)-0.5))/10001", "0",
         layered_u, layered_grad, zero_source,
         "-div(a(x)*grad(u)) = f, a(x)=1 (x<0.5), 10000 (x>=0.5)", 1.0, true
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
