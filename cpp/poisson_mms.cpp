// poisson_mms.cpp
//
// MFEM-based solver for the scalar Poisson problem on the unit square
//
//     -lap(u) = f            in Omega = (0,1)^2
//           u = u_exact      on Gamma_D  (boundary attributes in --dirichlet)
//   grad(u).n = grad(u_exact).n on Gamma_N (boundary attributes in --neumann)
//
// with a manufactured solution selected from manufactured_cases.hpp.
//
// The structure follows MFEM Example 1 (examples/ex1.cpp, MFEM v4.10), with
// these deliberate differences:
//   * the source term is the manufactured f instead of the constant 1;
//   * every mesh boundary attribute must be assigned explicitly to exactly one
//     of --dirichlet / --neumann (no silent natural-BC default);
//   * CGSolver is used directly and its convergence flag is checked (ex1 uses
//     the PCG() helper, whose tolerances are squared, with no status check);
//   * raw nodal data, element connectivity and boundary geometry are exported
//     so an external verifier can recompute errors without trusting this code.
//
// Exit codes (stable contract with the Python runner):
//   0 success, 2 invalid arguments, 3 boundary-coverage error,
//   4 unsupported assembly/solver combination, 5 linear solver not converged,
//   6 non-finite values in the solution.
//
// Fault injection (--inject-fault) exists ONLY to test that the downstream
// verifier rejects wrong results. Every injected fault is recorded in
// result.json under "injected_fault".

#include "mfem.hpp"
#include "manufactured_cases.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

std::vector<int> ParseIntList(const std::string &s, bool &ok)
{
   std::vector<int> out;
   ok = true;
   if (s.empty() || s == "none") { return out; }
   std::stringstream ss(s);
   std::string tok;
   while (std::getline(ss, tok, ','))
   {
      try
      {
         size_t pos = 0;
         int v = std::stoi(tok, &pos);
         if (pos != tok.size() || v <= 0) { ok = false; return out; }
         out.push_back(v);
      }
      catch (...) { ok = false; return out; }
   }
   return out;
}

std::string JsonEscape(const std::string &s)
{
   std::string o;
   for (char c : s)
   {
      switch (c)
      {
         case '"': o += "\\\""; break;
         case '\\': o += "\\\\"; break;
         case '\n': o += "\\n"; break;
         default: o += c;
      }
   }
   return o;
}

std::string JsonNum(double v)
{
   if (!std::isfinite(v)) { return "null"; }
   std::ostringstream os;
   os << std::setprecision(17) << v;
   return os.str();
}

std::string JsonIntList(const std::vector<int> &v)
{
   std::ostringstream os;
   os << "[";
   for (size_t i = 0; i < v.size(); i++) { os << (i ? ", " : "") << v[i]; }
   os << "]";
   return os.str();
}

void ListCases()
{
   std::cout << "{\n  \"mfem_version\": \"" << GetVersionStr() << "\",\n"
             << "  \"cases\": [\n";
   const auto &reg = mms::Registry();
   for (size_t i = 0; i < reg.size(); i++)
   {
      std::cout << "    {\"id\": \"" << reg[i].id << "\", \"exact_solution\": \""
                << reg[i].exact_solution << "\", \"source\": \""
                << reg[i].source << "\"}" << (i + 1 < reg.size() ? "," : "")
                << "\n";
   }
   std::cout << "  ]\n}\n";
}

} // namespace

int main(int argc, char *argv[])
{
   const char *case_id = "sin_sin";
   const char *mesh_type = "tri";
   int n = 8;
   int order = 1;
   const char *dirichlet_str = "1,2,3,4";
   const char *neumann_str = "none";
   const char *assembly = "full";
   const char *solver = "cg_gs";
   double rel_tol = 1e-12;
   double abs_tol = 0.0;
   int max_iter = 5000;
   const char *out_dir = ".";
   const char *fault = "none";
   bool list_cases = false;

   OptionsParser args(argc, argv);
   args.AddOption(&case_id, "-c", "--case", "Manufactured case id.");
   args.AddOption(&mesh_type, "-mt", "--mesh-type", "tri or quad.");
   args.AddOption(&n, "-n", "--n", "Elements per side of the unit square.");
   args.AddOption(&order, "-o", "--order", "H1 polynomial order.");
   args.AddOption(&dirichlet_str, "-dir", "--dirichlet",
                  "Comma-separated Dirichlet boundary attributes or 'none'.");
   args.AddOption(&neumann_str, "-neu", "--neumann",
                  "Comma-separated Neumann boundary attributes or 'none'.");
   args.AddOption(&assembly, "-a", "--assembly", "full or partial.");
   args.AddOption(&solver, "-s", "--solver", "cg_gs, cg_jacobi or cg_none.");
   args.AddOption(&rel_tol, "-rtol", "--rel-tol",
                  "CG relative tolerance (not squared).");
   args.AddOption(&abs_tol, "-atol", "--abs-tol", "CG absolute tolerance.");
   args.AddOption(&max_iter, "-maxit", "--max-iter", "CG iteration cap.");
   args.AddOption(&out_dir, "-out", "--out", "Existing output directory.");
   args.AddOption(&fault, "-fault", "--inject-fault",
                  "none, wrong_source, drop_dirichlet_attr, nan_value, "
                  "early_stop_lie, self_report_lie (testing only).");
   args.AddOption(&list_cases, "-list", "--list-cases", "-no-list",
                  "--no-list-cases", "Print compiled cases as JSON and exit.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(std::cerr); return 2; }
   if (list_cases) { ListCases(); return 0; }

   const std::string fault_s(fault);
   const std::set<std::string> known_faults =
   {
      "none", "wrong_source", "drop_dirichlet_attr", "nan_value",
      "early_stop_lie", "self_report_lie"
   };
   if (!known_faults.count(fault_s))
   {
      std::cerr << "ERROR: unknown fault mode '" << fault_s << "'\n";
      return 2;
   }

   const mms::ManufacturedCase *mc = mms::Find(case_id);
   if (!mc) { std::cerr << "ERROR: unknown case '" << case_id << "'\n"; return 2; }
   const std::string mt(mesh_type);
   if (mt != "tri" && mt != "quad")
   {
      std::cerr << "ERROR: mesh-type must be tri or quad\n"; return 2;
   }
   if (n < 1 || n > 4096) { std::cerr << "ERROR: n out of range\n"; return 2; }
   if (order < 1 || order > 8) { std::cerr << "ERROR: order out of range\n"; return 2; }
   if (!(rel_tol > 0.0) || abs_tol < 0.0 || max_iter < 1)
   {
      std::cerr << "ERROR: invalid solver tolerances\n"; return 2;
   }
   const std::string asm_s(assembly), sol_s(solver);
   if (asm_s != "full" && asm_s != "partial")
   {
      std::cerr << "ERROR: assembly must be full or partial\n"; return 2;
   }
   if (sol_s != "cg_gs" && sol_s != "cg_jacobi" && sol_s != "cg_none")
   {
      std::cerr << "ERROR: unknown solver '" << sol_s << "'\n"; return 2;
   }
   // GSSmoother needs an explicitly assembled SparseMatrix; partial assembly
   // only provides a matrix-free operator. Reject instead of guessing.
   if (asm_s == "partial" && sol_s == "cg_gs")
   {
      std::cerr << "ERROR: UNSUPPORTED_COMBINATION assembly=partial with "
                   "solver=cg_gs (GSSmoother requires an assembled "
                   "SparseMatrix)\n";
      return 4;
   }

   bool ok_d = true, ok_n = true;
   std::vector<int> dir_attrs = ParseIntList(dirichlet_str, ok_d);
   std::vector<int> neu_attrs = ParseIntList(neumann_str, ok_n);
   if (!ok_d || !ok_n)
   {
      std::cerr << "ERROR: malformed boundary attribute list\n"; return 2;
   }

   auto t0 = std::chrono::steady_clock::now();

   // 1. Mesh: n x n Cartesian mesh of the unit square. Boundary attributes
   //    (MFEM v4.10 Mesh::Make2D): 1 = y=0, 2 = x=1, 3 = y=1, 4 = x=0.
   const Element::Type etype = (mt == "tri") ? Element::TRIANGLE
                                             : Element::QUADRILATERAL;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, etype, true, 1.0, 1.0);
   const int dim = mesh.Dimension();

   // 2. Boundary coverage: each mesh attribute exactly once; no unknown ones.
   std::set<int> mesh_attrs;
   for (int i = 0; i < mesh.bdr_attributes.Size(); i++)
   {
      mesh_attrs.insert(mesh.bdr_attributes[i]);
   }
   std::set<int> dset(dir_attrs.begin(), dir_attrs.end());
   std::set<int> nset(neu_attrs.begin(), neu_attrs.end());
   bool bc_ok = (dset.size() == dir_attrs.size()) &&
                (nset.size() == neu_attrs.size());
   for (int a : dset) { if (nset.count(a) || !mesh_attrs.count(a)) { bc_ok = false; } }
   for (int a : nset) { if (!mesh_attrs.count(a)) { bc_ok = false; } }
   for (int a : mesh_attrs) { if (!dset.count(a) && !nset.count(a)) { bc_ok = false; } }
   if (!bc_ok)
   {
      std::cerr << "ERROR: BOUNDARY_COVERAGE mesh attributes {";
      for (int a : mesh_attrs) { std::cerr << " " << a; }
      std::cerr << " } must be partitioned exactly by --dirichlet "
                << dirichlet_str << " and --neumann " << neumann_str << "\n";
      return 3;
   }
   if (dset.empty())
   {
      std::cerr << "ERROR: BOUNDARY_COVERAGE pure Neumann problem has a "
                   "constant nullspace; not supported by this solver\n";
      return 3;
   }

   // 3. Finite element space.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   const int ndofs = fespace.GetTrueVSize();

   const int max_attr = mesh.bdr_attributes.Max();
   Array<int> ess_bdr(max_attr), neu_bdr(max_attr);
   ess_bdr = 0; neu_bdr = 0;
   for (int a : dset) { ess_bdr[a - 1] = 1; }
   for (int a : nset) { neu_bdr[a - 1] = 1; }
   if (fault_s == "drop_dirichlet_attr")
   {
      // Silently turn the largest Dirichlet attribute into a natural
      // (homogeneous Neumann) boundary: the classic "forgotten BC" bug.
      ess_bdr[*dset.rbegin() - 1] = 0;
   }
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 4. Coefficients and integration rules.
   FunctionCoefficient u_coef(mc->u);
   VectorFunctionCoefficient grad_coef(dim, mc->grad_u);
   const double src_scale = (fault_s == "wrong_source") ? 1.1 : 1.0;
   FunctionCoefficient f_coef([mc, src_scale](const Vector &p)
   {
      return src_scale * mc->f(p);
   });
   const Geometry::Type geom = mesh.GetTypicalElementGeometry();
   const int rhs_q = 2 * order + 4;
   const IntegrationRule &ir_dom = IntRules.Get(geom, rhs_q);
   const IntegrationRule &ir_bdr = IntRules.Get(Geometry::SEGMENT, rhs_q);

   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(f_coef, &ir_dom));
   if (!nset.empty())
   {
      auto *bn = new BoundaryNormalLFIntegrator(grad_coef);
      bn->SetIntRule(&ir_bdr);
      b.AddBoundaryIntegrator(bn, neu_bdr);
   }
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(u_coef, ess_bdr);

   BilinearForm a(&fespace);
   if (asm_s == "partial") { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   auto t1 = std::chrono::steady_clock::now();

   // 5. Solve with explicit convergence checking.
   CGSolver cg;
   cg.SetRelTol(rel_tol);
   cg.SetAbsTol(abs_tol);
   int it_cap = max_iter;
   if (fault_s == "early_stop_lie") { it_cap = 3; }
   cg.SetMaxIter(it_cap);
   cg.SetPrintLevel(IterativeSolver::PrintLevel().Errors().Warnings());
   std::unique_ptr<Solver> prec;
   if (sol_s == "cg_gs")
   {
      prec.reset(new GSSmoother(*A.As<SparseMatrix>()));
   }
   else if (sol_s == "cg_jacobi")
   {
      prec.reset(new OperatorJacobiSmoother(a, ess_tdof_list));
   }
   cg.SetOperator(*A);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.Mult(B, X);
   auto t2 = std::chrono::steady_clock::now();

   bool converged = cg.GetConverged();
   const int iters = cg.GetNumIterations();
   const double init_norm = cg.GetInitialNorm();
   const double final_norm = cg.GetFinalNorm();
   if (fault_s == "early_stop_lie") { converged = true; }  // deliberate lie

   a.RecoverFEMSolution(X, b, x);
   if (fault_s == "nan_value") { x(ndofs / 2) = std::numeric_limits<double>::quiet_NaN(); }

   bool finite = true;
   for (int i = 0; i < x.Size(); i++) { if (!std::isfinite(x(i))) { finite = false; break; } }

   // 6. Self-reported errors from MFEM (the verifier recomputes these).
   const IntegrationRule *irs[Geometry::NumGeom];
   const int err_q = 2 * order + 4;
   for (int g = 0; g < Geometry::NumGeom; g++) { irs[g] = &IntRules.Get(g, err_q); }
   double l2_err = finite ? x.ComputeL2Error(u_coef, irs) : NAN;
   double h1s_err = finite ? x.ComputeGradError(&grad_coef, irs) : NAN;
   if (fault_s == "self_report_lie") { l2_err = 0.0; h1s_err = 0.0; }
   auto t3 = std::chrono::steady_clock::now();

   // 7. Export raw data for the independent verifier.
   const std::string od(out_dir);
   {
      std::ofstream f(od + "/elements.txt");
      f << std::setprecision(17);
      f << "# elem geom nverts v0x v0y v1x v1y ... (MFEM vertex order)\n";
      Array<int> v;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         mesh.GetElementVertices(e, v);
         f << e << " " << (mt == "tri" ? "tri" : "quad") << " " << v.Size();
         for (int k = 0; k < v.Size(); k++)
         {
            const real_t *c = mesh.GetVertex(v[k]);
            f << " " << c[0] << " " << c[1];
         }
         f << "\n";
      }
   }
   {
      std::ofstream f(od + "/element_dofs.txt");
      f << std::setprecision(17);
      f << "# elem local_index global_dof x y\n";
      Array<int> dofs;
      Vector xp;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         fespace.GetElementDofs(e, dofs);
         const FiniteElement *fe = fespace.GetFE(e);
         const IntegrationRule &nodes = fe->GetNodes();
         ElementTransformation *T = mesh.GetElementTransformation(e);
         for (int k = 0; k < dofs.Size(); k++)
         {
            T->SetIntPoint(&nodes.IntPoint(k));
            T->Transform(nodes.IntPoint(k), xp);
            const int gd = dofs[k] >= 0 ? dofs[k] : -1 - dofs[k];
            f << e << " " << k << " " << gd << " " << xp(0) << " " << xp(1) << "\n";
         }
      }
   }
   {
      std::ofstream f(od + "/dof_values.txt");
      f << std::setprecision(17);
      f << "# global_dof value\n";
      for (int i = 0; i < x.Size(); i++) { f << i << " " << x(i) << "\n"; }
   }
   {
      std::ofstream f(od + "/boundary.txt");
      f << std::setprecision(17);
      f << "# bdr_elem attribute x0 y0 x1 y1\n";
      Array<int> v;
      for (int be = 0; be < mesh.GetNBE(); be++)
      {
         mesh.GetBdrElementVertices(be, v);
         const real_t *c0 = mesh.GetVertex(v[0]);
         const real_t *c1 = mesh.GetVertex(v[1]);
         f << be << " " << mesh.GetBdrAttribute(be) << " " << c0[0] << " "
           << c0[1] << " " << c1[0] << " " << c1[1] << "\n";
      }
   }
   {
      std::ofstream mf(od + "/mesh.mesh");
      mf.precision(17);
      mesh.Print(mf);
      std::ofstream sf(od + "/solution.gf");
      sf.precision(17);
      x.Save(sf);
   }

   auto ms = [](auto a_, auto b_)
   {
      return std::chrono::duration<double, std::milli>(b_ - a_).count();
   };
   {
      std::ofstream f(od + "/result.json");
      f << "{\n"
        << "  \"producer\": \"poisson_mms\",\n"
        << "  \"mfem_version\": \"" << JsonEscape(GetVersionStr()) << "\",\n"
        << "  \"mfem_git\": \"" << JsonEscape(GetGitStr()) << "\",\n"
        << "  \"case\": \"" << mc->id << "\",\n"
        << "  \"case_exact_solution\": \"" << JsonEscape(mc->exact_solution) << "\",\n"
        << "  \"case_source\": \"" << JsonEscape(mc->source) << "\",\n"
        << "  \"mesh_type\": \"" << mt << "\",\n"
        << "  \"n\": " << n << ",\n"
        << "  \"h\": " << JsonNum(1.0 / n) << ",\n"
        << "  \"order\": " << order << ",\n"
        << "  \"num_elements\": " << mesh.GetNE() << ",\n"
        << "  \"num_boundary_elements\": " << mesh.GetNBE() << ",\n"
        << "  \"num_dofs\": " << ndofs << ",\n"
        << "  \"num_essential_dofs\": " << ess_tdof_list.Size() << ",\n"
        << "  \"mesh_boundary_attributes\": "
        << JsonIntList(std::vector<int>(mesh_attrs.begin(), mesh_attrs.end())) << ",\n"
        << "  \"dirichlet_attributes\": " << JsonIntList(dir_attrs) << ",\n"
        << "  \"neumann_attributes\": " << JsonIntList(neu_attrs) << ",\n"
        << "  \"assembly\": \"" << asm_s << "\",\n"
        << "  \"solver\": \"" << sol_s << "\",\n"
        << "  \"rel_tol\": " << JsonNum(rel_tol) << ",\n"
        << "  \"abs_tol\": " << JsonNum(abs_tol) << ",\n"
        << "  \"max_iter\": " << max_iter << ",\n"
        << "  \"rhs_quadrature_order\": " << rhs_q << ",\n"
        << "  \"error_quadrature_order\": " << err_q << ",\n"
        << "  \"solver_converged\": " << (converged ? "true" : "false") << ",\n"
        << "  \"solver_iterations\": " << iters << ",\n"
        << "  \"solver_initial_norm\": " << JsonNum(init_norm) << ",\n"
        << "  \"solver_final_norm\": " << JsonNum(final_norm) << ",\n"
        << "  \"solution_finite\": " << (finite ? "true" : "false") << ",\n"
        << "  \"self_reported_l2_error\": " << JsonNum(l2_err) << ",\n"
        << "  \"self_reported_h1_semi_error\": " << JsonNum(h1s_err) << ",\n"
        << "  \"injected_fault\": \"" << fault_s << "\",\n"
        << "  \"time_setup_ms\": " << JsonNum(ms(t0, t1)) << ",\n"
        << "  \"time_solve_ms\": " << JsonNum(ms(t1, t2)) << ",\n"
        << "  \"time_error_ms\": " << JsonNum(ms(t2, t3)) << "\n"
        << "}\n";
   }

   std::cout << std::setprecision(6)
             << "poisson_mms case=" << mc->id << " mesh=" << mt << " n=" << n
             << " order=" << order << " dofs=" << ndofs
             << " cg_iters=" << iters << " converged=" << converged
             << " L2=" << l2_err << " H1semi=" << h1s_err
             << " fault=" << fault_s << "\n";

   if (!finite) { std::cerr << "ERROR: NON_FINITE solution values\n"; return 6; }
   if (!converged) { std::cerr << "ERROR: SOLVER_NOT_CONVERGED\n"; return 5; }
   return 0;
}
