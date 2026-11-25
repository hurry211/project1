// poisson_mms.cpp
//
// MFEM-based solver for scalar diffusion on the unit square
//
//     -div(kappa grad(u)) = f in Omega = (0,1)^2
//           u = u_exact      on Gamma_D  (boundary attributes in --dirichlet)
//   kappa grad(u).n = kappa grad(u_exact).n on Gamma_N (--neumann)
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
//   6 non-finite values in the solution, 7 artifact output failure.
//
// Fault injection (--inject-fault) exists ONLY to test that the downstream
// verifier rejects wrong results. Every injected fault is recorded in
// result.json under "injected_fault".

#include "mfem.hpp"
#include "mfem_build.hpp"
#include "manufactured_cases.hpp"
#include "poisson_diagnostics.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef __linux__
#include <sys/resource.h>
#endif

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

bool FinishExport(std::ofstream &stream, const char *name)
{
   stream.close();
   if (stream.fail()) { std::cerr << "ERROR: cannot write " << name << "\n"; }
   return !stream.fail();
}

std::string PeakRssKiB()
{
   // Linux VmHWM belongs to the current executable's address space. A fork
   // before exec can inflate getrusage(RUSAGE_SELF).ru_maxrss.
   std::ifstream status("/proc/self/status");
   for (std::string line; std::getline(status, line); )
   {
      if (line.rfind("VmHWM:", 0) == 0)
      {
         long kib;
         std::istringstream value(line.substr(6));
         if (value >> kib) { return std::to_string(kib); }
         break;
      }
   }
   return "null";
}

void PrintFormulation(const mms::Formulation &form, bool include_kappa)
{
   std::cout << "{\"equation\": \"" << JsonEscape(form.equation)
             << "\", \"domain\": \"" << JsonEscape(form.domain)
             << "\", \"space\": \"" << JsonEscape(form.space)
             << "\", \"bilinear_integrator\": \"" << JsonEscape(form.bilinear_integrator)
             << "\", \"dirichlet_trace\": \"" << JsonEscape(form.dirichlet_trace)
             << "\", \"neumann_trace\": \"" << JsonEscape(form.neumann_trace) << "\"";
   if (include_kappa) { std::cout << ", \"kappa\": \"" << JsonEscape(form.kappa) << "\""; }
   std::cout << "}";
}

void ListCases(bool include_e3 = false)
{
   std::cout << "{\n  \"mfem_version\": \"" << GetVersionStr() << "\",\n"
             << "  \"mfem_git\": \"" << JsonEscape(GetGitStr()) << "\",\n"
             << "  \"mfem_config\": " << MfemBuildFlags << ",\n"
             << "  \"mesh_sources\": [\"cartesian\", \"gmsh22\"],\n"
             << "  \"registry_version\": 2,\n  \"formulation\": ";
   PrintFormulation(mms::UnitSquarePoisson, false);
   std::cout << ",\n  \"cases\": [\n";
   const auto &reg = mms::Registry();
   std::vector<const mms::ManufacturedCase *> visible;
   for (const auto &c : reg)
   {
      if (include_e3 || !c.e3_only) { visible.push_back(&c); }
   }
   for (size_t i = 0; i < visible.size(); i++)
   {
      const auto *c = visible[i];
      std::cout << "    {\"id\": \"" << JsonEscape(c->id) << "\", \"exact_solution\": \""
                << JsonEscape(c->exact_solution) << "\", \"source\": \""
                << JsonEscape(c->source) << "\", \"equation\": \""
                << JsonEscape(c->form.equation) << "\"";
      if (&c->form != &mms::UnitSquarePoisson)
      {
         std::cout << ", \"formulation\": ";
         PrintFormulation(c->form, true);
      }
      std::cout << "}" << (i + 1 < visible.size() ? "," : "") << "\n";
   }
   std::cout << "  ]\n}\n";
}

} // namespace

int main(int argc, char *argv[])
{
   const char *case_id = "sin_sin";
   const char *mesh_type = "tri";
   const char *mesh_topology = "cartesian";
   const char *mesh_file = "";
   int mesh_refinements = 0;
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
   bool include_e3 = false;

   OptionsParser args(argc, argv);
   args.AddOption(&case_id, "-c", "--case", "Manufactured case id.");
   args.AddOption(&mesh_type, "-mt", "--mesh-type", "tri or quad.");
   args.AddOption(&mesh_topology, "-topo", "--mesh-topology", "cartesian or unstructured.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Preflighted unit-square Gmsh input.");
   args.AddOption(&mesh_refinements, "-mr", "--mesh-refinements", "Uniform refinements of imported mesh.");
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
                  "early_stop_lie, self_report_lie, wrong_diffusion, omit_coefficient_gradient, "
                  "omit_neumann_coefficient (testing only).");
   args.AddOption(&list_cases, "-list", "--list-cases", "-no-list",
                  "--no-list-cases", "Print compiled cases as JSON and exit.");
   args.AddOption(&include_e3, "-e3", "--include-e3", "-no-e3",
                  "--no-include-e3", "Include E3 extension cases in --list-cases output.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(std::cerr); return 2; }
   if (list_cases) { ListCases(include_e3); return 0; }

   const std::string fault_s(fault);
   const std::set<std::string> known_faults =
   {
      "none", "wrong_source", "drop_dirichlet_attr", "nan_value",
      "early_stop_lie", "self_report_lie", "wrong_diffusion", "omit_coefficient_gradient",
      "omit_neumann_coefficient"
   };
   if (!known_faults.count(fault_s))
   {
      std::cerr << "ERROR: unknown fault mode '" << fault_s << "'\n";
      return 2;
   }

   const mms::ManufacturedCase *mc = mms::Find(case_id);
   if (!mc) { std::cerr << "ERROR: unknown case '" << case_id << "'\n"; return 2; }
   if ((fault_s == "omit_coefficient_gradient" && !mc->source_without_coefficient_gradient)
       || (fault_s == "omit_neumann_coefficient" && !mc->source_without_coefficient_gradient))
   {
      std::cerr << "ERROR: this fault requires a variable diffusion case\n"; return 2;
   }
   const std::string mt(mesh_type);
   if (mt != "tri" && mt != "quad")
   {
      std::cerr << "ERROR: mesh-type must be tri or quad\n"; return 2;
   }
   const std::string topo_s(mesh_topology);
   if (topo_s != "cartesian" && topo_s != "unstructured")
   {
      std::cerr << "ERROR: mesh-topology must be cartesian or unstructured\n"; return 2;
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

   // 1. Construct the default Cartesian grid or load the submitted topology
   //    and physical attributes directly through MFEM's mesh reader.
   const Element::Type etype = (mt == "tri") ? Element::TRIANGLE
                                             : Element::QUADRILATERAL;
   const bool imported = mesh_file[0] != '\0';
   if (mesh_refinements < 0 || mesh_refinements > 12 || (!imported && mesh_refinements))
   {
      std::cerr << "ERROR: invalid mesh refinement count\n"; return 2;
   }
   if (topo_s == "unstructured" && !imported)
   {
      std::cerr << "ERROR: unstructured topology requires an imported --mesh\n"; return 2;
   }
   Mesh mesh = imported ? Mesh(mesh_file, 1, 1) : Mesh::MakeCartesian2D(n, n, etype, true, 1.0, 1.0);
   if (imported)
   {
      if (mesh.Dimension() != 2 || mesh.SpaceDimension() != 2)
      {
         std::cerr << "ERROR: imported mesh must be 2D\n"; return 2;
      }
      if (topo_s == "cartesian")
      {
         const int factor = 1 << mesh_refinements;
         const int base = n / factor;
         if (n % factor || mesh.GetNE() != base * base * (mt == "tri" ? 2 : 1) ||
             mesh.GetNBE() != 4 * base)
         {
            std::cerr << "ERROR: imported mesh dimensions or element counts differ from request\n"; return 2;
         }
      }
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         if (mesh.GetElementType(e) != etype)
         {
            std::cerr << "ERROR: imported mesh element type differs from request\n"; return 2;
         }
      }
      for (int k = 0; k < mesh_refinements; ++k) { mesh.UniformRefinement(); }
   }
   const int dim = mesh.Dimension();

   Array<int> boundary_sides(mesh.GetNBE()), boundary_vertices;
   for (int be = 0; be < mesh.GetNBE(); be++)
   {
      mesh.GetBdrElementVertices(be, boundary_vertices);
      if (boundary_vertices.Size() != 2)
      {
         std::cerr << "ERROR: expected boundary segments in the unit square\n"; return 3;
      }
      boundary_sides[be] = poisson_diagnostics::UnitSquareSide(
         mesh.GetVertex(boundary_vertices[0]), mesh.GetVertex(boundary_vertices[1]));
      if (boundary_sides[be] < 0)
      {
         std::cerr << "ERROR: boundary segment is off the unit square\n"; return 3;
      }
   }

   // 2. Boundary coverage: each mesh attribute exactly once; no unknown ones.
   std::set<int> mesh_attrs;
   for (int i = 0; i < mesh.bdr_attributes.Size(); i++)
   {
      mesh_attrs.insert(mesh.bdr_attributes[i]);
   }
   std::set<int> dset(dir_attrs.begin(), dir_attrs.end());
   std::set<int> nset(neu_attrs.begin(), neu_attrs.end());
   if (fault_s == "omit_neumann_coefficient" && nset.empty())
   {
      std::cerr << "ERROR: omit_neumann_coefficient requires a Neumann boundary\n";
      return 2;
   }
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
   if (mc->diffusion_x != 1.0 &&
       (imported || mt != "quad" || order != 1 || asm_s != "full" || sol_s != "cg_jacobi" ||
        !nset.empty()))
   {
      std::cerr << "ERROR: anisotropic case requires full Q1 CG+Jacobi and all Dirichlet\n";
      return 4;
   }
   if (mc->layered && (imported || mt != "quad" || order != 1 || asm_s != "full" || sol_s != "cg_gs" ||
                       n < 4 || n > 128 || (n & (n-1)) != 0 ||
                       dset != std::set<int>{2, 4} || nset != std::set<int>{1, 3}))
   {
      std::cerr << "ERROR: layered case requires aligned Q1, full CG+GS and a horizontal unit drop\n";
      return 4;
   }
   if (fault_s == "wrong_diffusion" && !mc->layered) { return 4; }
   if (mc->id == "gaussian_offset" && (imported || mt != "quad" || order != 2 || sol_s != "cg_jacobi" ||
                                       !nset.empty() || n < 32 || n > 256 || (n & (n-1)) != 0))
   {
      std::cerr << "ERROR: localized Gaussian requires all-Dirichlet Q2 CG+Jacobi on reviewed levels\n";
      return 4;
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
   // A null coefficient selects MFEM's original unit-diffusion assembly path.
   std::unique_ptr<FunctionCoefficient> diffusivity;
   if (mc->diffusivity) { diffusivity = std::make_unique<FunctionCoefficient>(mc->diffusivity); }
   VectorFunctionCoefficient conormal_coef(dim, mc->grad_u, diffusivity.get());
   const double src_scale = (fault_s == "wrong_source") ? 1.1 : 1.0;
   const auto source = fault_s == "omit_coefficient_gradient" ?
                       mc->source_without_coefficient_gradient : mc->f;
   FunctionCoefficient f_coef([source, src_scale, mc](const Vector &p)
   {
      // Scaling a homogeneous source is a no-op. Use a unit load for this control.
      if (mc->layered && src_scale != 1.0) { return 1.0; }
      return src_scale * source(p);
   });
   const Geometry::Type geom = mesh.GetTypicalElementGeometry();
   const int rhs_q = (topo_s == "unstructured") ? std::max(2 * order + 4, 12) : (2 * order + 4);
   const IntegrationRule &ir_dom = IntRules.Get(geom, rhs_q);
   const IntegrationRule &ir_bdr = IntRules.Get(Geometry::SEGMENT, rhs_q);

   const auto rhs_start = std::chrono::steady_clock::now();
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(f_coef, &ir_dom));
   // Both horizontal boundaries in the layered benchmark have exactly zero flux.
   if (!nset.empty() && !mc->layered)
   {
      auto &natural_flux = fault_s == "omit_neumann_coefficient" ? grad_coef : conormal_coef;
      auto *bn = new BoundaryNormalLFIntegrator(natural_flux);
      bn->SetIntRule(&ir_bdr);
      b.AddBoundaryIntegrator(bn, neu_bdr);
   }
   b.Assemble();
   const auto rhs_end = std::chrono::steady_clock::now();

   GridFunction x(&fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(u_coef, ess_bdr);

   const auto assembly_start = std::chrono::steady_clock::now();
   DenseMatrix diffusion_tensor(dim);
   diffusion_tensor.Diag(1.0, dim);
   diffusion_tensor(0, 0) = mc->diffusion_x;
   MatrixConstantCoefficient diffusion(diffusion_tensor);
   BilinearForm a(&fespace);
   if (asm_s == "partial") { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   const int stiffness_q = (topo_s == "unstructured")
                             ? ((mt == "quad") ? std::max(2 * order + 7, 11) : std::max(2 * order + 3, 9))
                             : (2 * order + 1);
   if (mc->source_without_coefficient_gradient || (topo_s == "unstructured" && mt == "quad"))
   {
      // Linear kappa raises the polynomial degree; non-affine unstructured quads
      // also introduce a rational 1/det(J) factor requiring higher quadrature.
      auto *stiffness = diffusivity ? new DiffusionIntegrator(*diffusivity)
                                    : new DiffusionIntegrator;
      stiffness->SetIntRule(&IntRules.Get(geom, stiffness_q));
      a.AddDomainIntegrator(stiffness);
   }
   else if (mc->layered && fault_s != "wrong_diffusion")
   {
      a.AddDomainIntegrator(new DiffusionIntegrator(*diffusivity));
   }
   else if (mc->diffusion_x != 1.0) { a.AddDomainIntegrator(new DiffusionIntegrator(diffusion)); }
   else { a.AddDomainIntegrator(new DiffusionIntegrator); }
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
   const auto solve_start = std::chrono::steady_clock::now();
   cg.Mult(B, X);
   auto t2 = std::chrono::steady_clock::now();
   const auto algebraic_residual = poisson_diagnostics::MeasureResidual(*A, X, B);
   auto error_start = std::chrono::steady_clock::now();

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
   const int err_q = rhs_q;
   for (int g = 0; g < Geometry::NumGeom; g++) { irs[g] = &IntRules.Get(g, err_q); }
   double l2_err = finite ? x.ComputeL2Error(u_coef, irs) : NAN;
   double h1s_err = finite ? x.ComputeGradError(&grad_coef, irs) : NAN;
   if (fault_s == "self_report_lie") { l2_err = 0.0; h1s_err = 0.0; }
   auto t3 = std::chrono::steady_clock::now();
   const auto flux = poisson_diagnostics::MeasureBoundaryFlux(
      mesh, x, finite, f_coef, ir_dom, ir_bdr,
      fault_s == "wrong_diffusion" ? nullptr : diffusivity.get(), boundary_sides, mc->diffusion_x);
   const double outward_total = flux.OutwardTotal();

   // 7. Export raw data for the independent verifier.
   const auto export_start = std::chrono::steady_clock::now();
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
      if (!FinishExport(f, "elements.txt")) { return 7; }
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
      if (!FinishExport(f, "element_dofs.txt")) { return 7; }
   }
   {
      std::ofstream f(od + "/dof_values.txt");
      f << std::setprecision(17);
      f << "# global_dof value\n";
      for (int i = 0; i < x.Size(); i++) { f << i << " " << x(i) << "\n"; }
      if (!FinishExport(f, "dof_values.txt")) { return 7; }
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
      if (!FinishExport(f, "boundary.txt")) { return 7; }
   }
   {
      std::ofstream mf(od + "/mesh.mesh");
      mf.precision(17);
      mesh.Print(mf);
      if (!FinishExport(mf, "mesh.mesh")) { return 7; }
      std::ofstream sf(od + "/solution.gf");
      sf.precision(17);
      x.Save(sf);
      if (!FinishExport(sf, "solution.gf")) { return 7; }
   }

   auto ms = [](auto a_, auto b_)
   {
      return std::chrono::duration<double, std::milli>(b_ - a_).count();
   };
   const auto export_end = std::chrono::steady_clock::now();
   {
      std::ofstream f(od + "/result.json");
      f << "{\n"
        << "  \"producer\": \"poisson_mms\",\n"
        << "  \"mfem_version\": \"" << JsonEscape(GetVersionStr()) << "\",\n"
        << "  \"mfem_git\": \"" << JsonEscape(GetGitStr()) << "\",\n"
        << "  \"case\": \"" << mc->id << "\",\n"
        << "  \"case_exact_solution\": \"" << JsonEscape(mc->exact_solution) << "\",\n"
        << "  \"case_source\": \"" << JsonEscape(mc->source) << "\",\n"
        << "  \"case_equation\": \"" << JsonEscape(mc->form.equation) << "\",\n";
      if (mc->source_without_coefficient_gradient)
      {
         f << "  \"case_diffusivity\": \"" << JsonEscape(mc->form.kappa) << "\",\n"
           << "  \"stiffness_quadrature_order\": " << stiffness_q << ",\n";
      }
      f << "  \"mesh_type\": \"" << mt << "\",\n"
        << "  \"mesh_topology\": \"" << topo_s << "\",\n"
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
        << "  \"diagnostics\": {\n"
        << "    \"schema_version\": 1,\n"
        << "    \"algebraic_residual\": {\"absolute_l2\": " << JsonNum(algebraic_residual.absolute_l2)
        << ", \"rhs_l2\": " << JsonNum(algebraic_residual.rhs_l2)
        << ", \"relative_l2\": " << JsonNum(algebraic_residual.relative_l2) << "},\n"
        << "    \"boundary_flux\": {\"outward_by_side\": {"
        << "\"bottom\": " << JsonNum(flux.outward_by_side[0])
        << ", \"right\": " << JsonNum(flux.outward_by_side[1])
        << ", \"top\": " << JsonNum(flux.outward_by_side[2])
        << ", \"left\": " << JsonNum(flux.outward_by_side[3]) << "},\n"
        << "      \"outward_total\": " << JsonNum(outward_total)
        << ", \"source_integral\": " << JsonNum(flux.source_integral)
        << ", \"signed_imbalance\": " << JsonNum(outward_total - flux.source_integral) << "}\n"
        << "  },\n"
        << "  \"injected_fault\": \"" << fault_s << "\",\n"
        << "  \"time_setup_ms\": " << JsonNum(ms(t0, t1)) << ",\n"
        << "  \"time_rhs_ms\": " << JsonNum(ms(rhs_start, rhs_end)) << ",\n"
        << "  \"time_assembly_ms\": " << JsonNum(ms(assembly_start, t1)) << ",\n"
        << "  \"time_preconditioner_ms\": " << JsonNum(ms(t1, solve_start)) << ",\n"
        << "  \"time_cg_ms\": " << JsonNum(ms(solve_start, t2)) << ",\n"
        << "  \"time_solve_ms\": " << JsonNum(ms(t1, t2)) << ",\n"
        << "  \"time_error_ms\": " << JsonNum(ms(error_start, t3)) << ",\n"
        << "  \"time_export_ms\": " << JsonNum(ms(t3, export_end)) << ",\n"
        << "  \"peak_rss_kib\": " << PeakRssKiB() << "\n"
        << "}\n";
      if (!FinishExport(f, "result.json")) { return 7; }
   }
   if (std::getenv("MFEM_STUDY_PROFILE"))
   {
      const auto export_end = std::chrono::steady_clock::now();
      std::string peak_rss = "null";
#ifdef __linux__
      struct rusage usage;
      if (getrusage(RUSAGE_SELF, &usage) == 0) { peak_rss = std::to_string(usage.ru_maxrss); }
#endif
      std::cerr << "MFEM_PROFILE_V1 {\"peak_rss_kib\":" << peak_rss
                << ",\"setup_seconds\":" << JsonNum(ms(t0, t1) / 1000)
                << ",\"solve_seconds\":" << JsonNum(ms(t1, t2) / 1000)
                << ",\"error_seconds\":" << JsonNum(ms(error_start, t3) / 1000)
                << ",\"export_seconds\":" << JsonNum(ms(export_start, export_end) / 1000)
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
