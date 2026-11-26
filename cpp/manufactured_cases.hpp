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

inline constexpr Formulation LShapePoisson =
{
   "-laplacian(u) = f", "lshape_2d", "scalar_H1", "DiffusionIntegrator",
   "u_exact", "grad(u_exact).n"
};

inline constexpr Formulation UnitSquareVariableDiffusion =
{
   "-div(kappa*grad(u)) = f", "unit_square_2d", "scalar_H1", "DiffusionIntegrator",
   "u_exact", "kappa*grad(u_exact).n", "1+x"
};

inline constexpr Formulation UnitSquareAnisotropic =
{
   "-div(diag(1000,1)*grad(u)) = f", "unit_square_2d", "scalar_H1", "DiffusionIntegrator",
   "u_exact", "(A*grad(u_exact)).n", "diag(1000,1)"
};

inline constexpr Formulation UnitSquareLayered =
{
   "-div(a(x)*grad(u)) = f, a(x)=1 (x<0.5), 10000 (x>=0.5)", "unit_square_2d",
   "scalar_H1", "DiffusionIntegrator", "u_exact", "a(x)*grad(u_exact).n",
   "1 (x<0.5), 10000 (x>=0.5)"
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
   double diffusion_x = 1.0;
   bool layered = false;
   bool e3_only = false;
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

inline mfem::real_t gaussian_steep_u(const mfem::Vector &p) { return std::exp(-120.0 * gaussian_r2(p)); }
inline void gaussian_steep_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   const double scale = -240.0 * gaussian_steep_u(p);
   g(0) = scale * (p(0) - 0.375);
   g(1) = scale * (p(1) - 0.625);
}
inline mfem::real_t gaussian_steep_f(const mfem::Vector &p)
{
   return (480.0 - 57600.0 * gaussian_r2(p)) * gaussian_steep_u(p);
}

inline double gaussian_mid_r2(const mfem::Vector &p)
{
   const double x = p(0) - 0.45, y = p(1) - 0.55;
   return x * x + y * y;
}
inline mfem::real_t gaussian_mid_u(const mfem::Vector &p) { return std::exp(-40.0 * gaussian_mid_r2(p)); }
inline void gaussian_mid_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   const double scale = -80.0 * gaussian_mid_u(p);
   g(0) = scale * (p(0) - 0.45);
   g(1) = scale * (p(1) - 0.55);
}
inline mfem::real_t gaussian_mid_f(const mfem::Vector &p)
{
   return (160.0 - 6400.0 * gaussian_mid_r2(p)) * gaussian_mid_u(p);
}

inline mfem::real_t sin_2x_2y_u(const mfem::Vector &p)
{
   return std::sin(2.0 * M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
}
inline void sin_2x_2y_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = 2.0 * M_PI * std::cos(2.0 * M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
   g(1) = 2.0 * M_PI * std::sin(2.0 * M_PI * p(0)) * std::cos(2.0 * M_PI * p(1));
}
inline mfem::real_t sin_2x_2y_f(const mfem::Vector &p) { return 8.0 * M_PI * M_PI * sin_2x_2y_u(p); }

inline mfem::real_t sin_2x_3y_u(const mfem::Vector &p)
{
   return std::sin(2.0 * M_PI * p(0)) * std::sin(3.0 * M_PI * p(1));
}
inline void sin_2x_3y_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = 2.0 * M_PI * std::cos(2.0 * M_PI * p(0)) * std::sin(3.0 * M_PI * p(1));
   g(1) = 3.0 * M_PI * std::sin(2.0 * M_PI * p(0)) * std::cos(3.0 * M_PI * p(1));
}
inline mfem::real_t sin_2x_3y_f(const mfem::Vector &p) { return 13.0 * M_PI * M_PI * sin_2x_3y_u(p); }

inline mfem::real_t lshape_u(const mfem::Vector &p)
{
   const double dx = p(0) - 0.5, dy = p(1) - 0.5;
   const double r2 = dx * dx + dy * dy;
   if (r2 == 0.0) { return 0.0; }
   double phi = std::atan2(dy, dx);
   if (phi < M_PI_2 - 1e-12) { phi += 2.0 * M_PI; }
   const double theta = phi - M_PI_2;
   return std::pow(r2, 1.0 / 3.0) * std::sin((2.0 / 3.0) * theta);
}
inline void lshape_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   const double dx = p(0) - 0.5, dy = p(1) - 0.5;
   const double r2 = dx * dx + dy * dy;
   if (r2 < 1e-30) { g(0) = 0.0; g(1) = 0.0; return; }
   double phi = std::atan2(dy, dx);
   if (phi < M_PI_2 - 1e-12) { phi += 2.0 * M_PI; }
   const double theta = phi - M_PI_2;
   const double r = std::sqrt(r2);
   const double coef = (2.0 / 3.0) * std::pow(r, -1.0 / 3.0);
   const double ur = coef * std::sin((2.0 / 3.0) * theta);
   const double uth_over_r = coef * std::cos((2.0 / 3.0) * theta);
   const double ux = ur * (dx / r) - uth_over_r * (dy / r);
   const double uy = ur * (dy / r) + uth_over_r * (dx / r);
   g(0) = ux;
   g(1) = uy;
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

inline mfem::real_t sin_3x_2y_u(const mfem::Vector &p)
{
   return std::sin(3.0 * M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
}
inline void sin_3x_2y_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = 3.0 * M_PI * std::cos(3.0 * M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
   g(1) = 2.0 * M_PI * std::sin(3.0 * M_PI * p(0)) * std::cos(2.0 * M_PI * p(1));
}
inline mfem::real_t sin_3x_2y_f(const mfem::Vector &p) { return 13.0 * M_PI * M_PI * sin_3x_2y_u(p); }

inline mfem::real_t var22_u(const mfem::Vector &p)
{
   return std::sin(2.0 * M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
}
inline void var22_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = 2.0 * M_PI * std::cos(2.0 * M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
   g(1) = 2.0 * M_PI * std::sin(2.0 * M_PI * p(0)) * std::cos(2.0 * M_PI * p(1));
}
inline mfem::real_t var22_laplacian_term(const mfem::Vector &p)
{
   return 8.0 * M_PI * M_PI * one_plus_x(p) * var22_u(p);
}
inline mfem::real_t var22_f(const mfem::Vector &p)
{
   return var22_laplacian_term(p) - 2.0 * M_PI * std::cos(2.0 * M_PI * p(0)) * std::sin(2.0 * M_PI * p(1));
}

inline double gaussian_sharp_r2(const mfem::Vector &p)
{
   const double x = p(0) - 0.625, y = p(1) - 0.375;
   return x * x + y * y;
}
inline mfem::real_t gaussian_sharp_u(const mfem::Vector &p) { return std::exp(-80.0 * gaussian_sharp_r2(p)); }
inline void gaussian_sharp_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   const double scale = -160.0 * gaussian_sharp_u(p);
   g(0) = scale * (p(0) - 0.625);
   g(1) = scale * (p(1) - 0.375);
}
inline mfem::real_t gaussian_sharp_f(const mfem::Vector &p)
{
   return (320.0 - 25600.0 * gaussian_sharp_r2(p)) * gaussian_sharp_u(p);
}

inline mfem::real_t sin_cos_u(const mfem::Vector &p)
{
   return std::sin(M_PI * p(0)) * std::cos(M_PI * p(1));
}
inline void sin_cos_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = M_PI * std::cos(M_PI * p(0)) * std::cos(M_PI * p(1));
   g(1) = -M_PI * std::sin(M_PI * p(0)) * std::sin(M_PI * p(1));
}
inline mfem::real_t sin_cos_f(const mfem::Vector &p) { return 2.0 * M_PI * M_PI * sin_cos_u(p); }

inline mfem::real_t cos_2x_sin_y_var_u(const mfem::Vector &p)
{
   return std::cos(2.0 * M_PI * p(0)) * std::sin(M_PI * p(1));
}
inline void cos_2x_sin_y_var_grad(const mfem::Vector &p, mfem::Vector &g)
{
   g.SetSize(2);
   g(0) = -2.0 * M_PI * std::sin(2.0 * M_PI * p(0)) * std::sin(M_PI * p(1));
   g(1) = M_PI * std::cos(2.0 * M_PI * p(0)) * std::cos(M_PI * p(1));
}
inline mfem::real_t cos_2x_sin_y_var_lap(const mfem::Vector &p)
{
   return 5.0 * M_PI * M_PI * one_plus_x(p) * cos_2x_sin_y_var_u(p);
}
inline mfem::real_t cos_2x_sin_y_var_f(const mfem::Vector &p)
{
   return cos_2x_sin_y_var_lap(p) + 2.0 * M_PI * std::sin(2.0 * M_PI * p(0)) * std::sin(M_PI * p(1));
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
         "gaussian_steep", "exp(-120*((x-0.375)**2+(y-0.625)**2))",
         "(480-57600*((x-0.375)**2+(y-0.625)**2))*exp(-120*((x-0.375)**2+(y-0.625)**2))",
         gaussian_steep_u, gaussian_steep_grad, gaussian_steep_f,
         UnitSquarePoisson, nullptr, nullptr, 1.0, false, true
      },
      {
         "gaussian_mid", "exp(-40*((x-0.45)**2+(y-0.55)**2))",
         "(160-6400*((x-0.45)**2+(y-0.55)**2))*exp(-40*((x-0.45)**2+(y-0.55)**2))",
         gaussian_mid_u, gaussian_mid_grad, gaussian_mid_f,
         UnitSquarePoisson, nullptr, nullptr, 1.0, false, true
      },
      {
         "sin_2x_2y", "sin(2*pi*x)*sin(2*pi*y)",
         "8*pi**2*sin(2*pi*x)*sin(2*pi*y)",
         sin_2x_2y_u, sin_2x_2y_grad, sin_2x_2y_f,
         UnitSquarePoisson, nullptr, nullptr, 1.0, false, true
      },
      {
         "sin_2x_3y", "sin(2*pi*x)*sin(3*pi*y)",
         "13*pi**2*sin(2*pi*x)*sin(3*pi*y)",
         sin_2x_3y_u, sin_2x_3y_grad, sin_2x_3y_f,
         UnitSquarePoisson, nullptr, nullptr, 1.0, false, true
      },
      {
         "lshape_singularity",
         "((x-0.5)**2+(y-0.5)**2)**(1/3)*sin((2/3)*theta_lshape)",
         "0",
         lshape_u, lshape_grad, zero_source,
         LShapePoisson, nullptr, nullptr, 1.0, false, true
      },
      {
         "sin_3x_2y", "sin(3*pi*x)*sin(2*pi*y)",
         "13*pi**2*sin(3*pi*x)*sin(2*pi*y)",
         sin_3x_2y_u, sin_3x_2y_grad, sin_3x_2y_f,
         UnitSquarePoisson, nullptr, nullptr, 1.0, false, true
      },
      {
         "sin_2x_2y_var",
         "sin(2*pi*x)*sin(2*pi*y)",
         "8*pi**2*(1+x)*sin(2*pi*x)*sin(2*pi*y)-2*pi*cos(2*pi*x)*sin(2*pi*y)",
         var22_u, var22_grad, var22_f,
         UnitSquareVariableDiffusion, one_plus_x, var22_laplacian_term, 1.0, false, true
      },
      {
         "gaussian_sharp",
         "exp(-80*((x-0.625)**2+(y-0.375)**2))",
         "(320-25600*((x-0.625)**2+(y-0.375)**2))*exp(-80*((x-0.625)**2+(y-0.375)**2))",
         gaussian_sharp_u, gaussian_sharp_grad, gaussian_sharp_f,
         UnitSquarePoisson, nullptr, nullptr, 1.0, false, true
      },
      {
         "sin_cos",
         "sin(pi*x)*cos(pi*y)",
         "2*pi**2*sin(pi*x)*cos(pi*y)",
         sin_cos_u, sin_cos_grad, sin_cos_f,
         UnitSquarePoisson, nullptr, nullptr, 1.0, false, true
      },
      {
         "cos_2x_sin_y_var",
         "cos(2*pi*x)*sin(pi*y)",
         "5*pi**2*(1+x)*cos(2*pi*x)*sin(pi*y)+2*pi*sin(2*pi*x)*sin(pi*y)",
         cos_2x_sin_y_var_u, cos_2x_sin_y_var_grad, cos_2x_sin_y_var_f,
         UnitSquareVariableDiffusion, one_plus_x, cos_2x_sin_y_var_lap, 1.0, false, true
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
         UnitSquareAnisotropic, nullptr, nullptr, 1000.0
      },
      {
         "layered_unit_drop_10000",
         "(10001*x-9999*(sqrt((x-0.5)**2)-0.5))/10001", "0",
         layered_u, layered_grad, zero_source,
         UnitSquareLayered, layered_a, nullptr, 1.0, true
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
