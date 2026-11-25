// poisson_amr.cpp
//
// MFEM Example-6-derived Adaptive Mesh Refinement (AMR) solver for scalar
// diffusion problems on 2D domains (unit square [0,1]^2 or unit L-shape):
//
//     -div(kappa grad(u)) = f   in Omega
//                       u = u_exact on Gamma_D
//          kappa grad(u).n = kappa grad(u_exact).n on Gamma_N
//
// Uses MFEM's ZienkiewiczZhuEstimator + ThresholdRefiner (examples/ex6.cpp),
// supporting both conforming simplex refinement and non-conforming quad NCMesh
// hanging-node meshes, full or partial assembly, and order p=1..8.
//
// Exports each AMR step under <out>/step_XX/ (same raw artifact contract as
// poisson_mms) plus <out>/amr_summary.json for independent Python verification.

#include "mfem.hpp"
#include "mfem_build.hpp"
#include "manufactured_cases.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef __linux__
#include <sys/resource.h>
#endif

using namespace mfem;
namespace fs = std::filesystem;

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

} // namespace

int main(int argc, char *argv[])
{
   const char *case_id = "gaussian_steep";
   const char *mesh_type = "quad";
   const char *mesh_file = "";
   int n = 4;
   int order = 2;
   const char *dirichlet_str = "1,2,3,4";
   const char *neumann_str = "none";
   const char *assembly = "full";
   const char *solver = "cg_gs";
   double rel_tol = 1e-12;
   double abs_tol = 0.0;
   int max_iter = 20000;
   int max_steps = 8;
   int max_dofs = 60000;
   double theta = 0.7;
   const char *out_dir = ".";
   const char *fault = "none";
   bool force_nc = false;
   bool anisotropic_zz = true;
   bool ls_zz = false;

   OptionsParser args(argc, argv);
   args.AddOption(&case_id, "-c", "--case", "Manufactured case id.");
   args.AddOption(&mesh_type, "-mt", "--mesh-type", "tri or quad.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Optional initial mesh file (e.g. l-shape.mesh).");
   args.AddOption(&n, "-n", "--n", "Initial elements per side when Cartesian.");
   args.AddOption(&order, "-o", "--order", "H1 polynomial order.");
   args.AddOption(&dirichlet_str, "-dir", "--dirichlet", "Comma-separated Dirichlet attributes or 'none'.");
   args.AddOption(&neumann_str, "-neu", "--neumann", "Comma-separated Neumann attributes or 'none'.");
   args.AddOption(&assembly, "-a", "--assembly", "full or partial.");
   args.AddOption(&solver, "-s", "--solver", "cg_gs, cg_jacobi, cg_none.");
   args.AddOption(&rel_tol, "-rtol", "--rel-tol", "CG relative tolerance (not squared).");
   args.AddOption(&abs_tol, "-atol", "--abs-tol", "CG absolute tolerance.");
   args.AddOption(&max_iter, "-maxit", "--max-iter", "CG iteration cap.");
   args.AddOption(&max_steps, "-ms", "--max-steps", "Maximum AMR solve/refine cycles.");
   args.AddOption(&max_dofs, "-md", "--max-dofs", "Stop once true DOFs exceed max_dofs.");
   args.AddOption(&theta, "-th", "--theta", "ThresholdRefiner total error fraction (0..1).");
   args.AddOption(&out_dir, "-out", "--out", "Output directory for step_XX artifacts.");
   args.AddOption(&fault, "-fault", "--inject-fault",
                  "none, wrong_source, drop_dirichlet_attr, self_report_lie, break_hanging_node.");
   args.AddOption(&force_nc, "-nc", "--nc-mesh", "-no-nc", "--no-nc-mesh",
                  "Force non-conforming NCMesh on initial mesh.");
   args.AddOption(&anisotropic_zz, "-az", "--anisotropic-zz", "-no-az", "--no-anisotropic-zz",
                  "Enable anisotropic refinement in ZienkiewiczZhuEstimator.");
   args.AddOption(&ls_zz, "-ls", "--ls-zz", "-no-ls", "--no-ls-zz",
                  "Use LSZienkiewiczZhuEstimator instead of ZienkiewiczZhuEstimator.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(std::cerr); return 2; }

   const std::string fault_s(fault);
   const std::set<std::string> known_faults = {
      "none", "wrong_source", "drop_dirichlet_attr", "self_report_lie", "break_hanging_node"
   };
   if (!known_faults.count(fault_s))
   {
      std::cerr << "ERROR: unknown fault mode '" << fault_s << "'\n"; return 2;
   }

   const mms::ManufacturedCase *mc = mms::Find(case_id);
   if (!mc) { std::cerr << "ERROR: unknown case '" << case_id << "'\n"; return 2; }

   const std::string mt(mesh_type), asm_s(assembly), sol_s(solver);
   if (mt != "tri" && mt != "quad") { std::cerr << "ERROR: mesh-type must be tri or quad\n"; return 2; }
   if (asm_s != "full" && asm_s != "partial") { std::cerr << "ERROR: invalid assembly\n"; return 2; }
   if (sol_s != "cg_gs" && sol_s != "cg_jacobi" && sol_s != "cg_none")
   {
      std::cerr << "ERROR: unknown solver\n"; return 2;
   }
   if (asm_s == "partial" && sol_s == "cg_gs")
   {
      std::cerr << "ERROR: UNSUPPORTED_COMBINATION partial + cg_gs\n"; return 4;
   }
   if (n < 1 || order < 1 || order > 8 || max_steps < 1 || max_dofs < 1 || theta <= 0.0 || theta >= 1.0)
   {
      std::cerr << "ERROR: invalid AMR parameters\n"; return 2;
   }

   bool ok_d = true, ok_n = true;
   std::vector<int> dir_attrs = ParseIntList(dirichlet_str, ok_d);
   std::vector<int> neu_attrs = ParseIntList(neumann_str, ok_n);
   if (!ok_d || !ok_n || dir_attrs.empty())
   {
      std::cerr << "ERROR: invalid Dirichlet/Neumann boundary attributes\n"; return 2;
   }

   const auto total_t0 = std::chrono::steady_clock::now();
   const Element::Type etype = (mt == "tri") ? Element::TRIANGLE : Element::QUADRILATERAL;
   const bool imported = mesh_file[0] != '\0';
   Mesh mesh = imported ? Mesh(mesh_file, 1, 1) : Mesh::MakeCartesian2D(n, n, etype, true, 1.0, 1.0);
   if (mesh.Dimension() != 2 || mesh.SpaceDimension() != 2)
   {
      std::cerr << "ERROR: expected 2D mesh\n"; return 2;
   }
   if (force_nc) { mesh.EnsureNCMesh(mt == "tri"); }

   std::set<int> mesh_attrs;
   for (int i = 0; i < mesh.bdr_attributes.Size(); i++) { mesh_attrs.insert(mesh.bdr_attributes[i]); }
   std::set<int> dset(dir_attrs.begin(), dir_attrs.end());
   std::set<int> nset(neu_attrs.begin(), neu_attrs.end());
   for (int a : dset) { if (nset.count(a) || !mesh_attrs.count(a)) { std::cerr << "ERROR: BOUNDARY_COVERAGE\n"; return 3; } }
   for (int a : nset) { if (!mesh_attrs.count(a)) { std::cerr << "ERROR: BOUNDARY_COVERAGE\n"; return 3; } }
   for (int a : mesh_attrs) { if (!dset.count(a) && !nset.count(a)) { std::cerr << "ERROR: BOUNDARY_COVERAGE\n"; return 3; } }

   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();
   const int max_attr = mesh.bdr_attributes.Max();
   Array<int> ess_bdr(max_attr), neu_bdr(max_attr);
   ess_bdr = 0; neu_bdr = 0;
   for (int a : dset) { ess_bdr[a - 1] = 1; }
   for (int a : nset) { neu_bdr[a - 1] = 1; }
   if (fault_s == "drop_dirichlet_attr") { ess_bdr[*dset.rbegin() - 1] = 0; }

   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   FunctionCoefficient u_coef(mc->u);
   VectorFunctionCoefficient grad_coef(dim, mc->grad_u);
   std::unique_ptr<FunctionCoefficient> diffusivity;
   if (mc->diffusivity) { diffusivity = std::make_unique<FunctionCoefficient>(mc->diffusivity); }
   VectorFunctionCoefficient conormal_coef(dim, mc->grad_u, diffusivity.get());
   const double src_scale = (fault_s == "wrong_source") ? 1.1 : 1.0;
   const auto source = mc->f;
   FunctionCoefficient f_coef([source, src_scale](const Vector &p) { return src_scale * source(p); });

   const Geometry::Type geom = mesh.GetTypicalElementGeometry();
   const int rhs_q = std::max(2 * order + 4, 20);
   const int err_q = rhs_q;
   const int stiffness_q = mc->source_without_coefficient_gradient ? (2 * order + 3) : (2 * order + 1);
   const IntegrationRule &ir_dom = IntRules.Get(geom, rhs_q);
   const IntegrationRule &ir_bdr = IntRules.Get(Geometry::SEGMENT, rhs_q);

   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(f_coef, &ir_dom));
   if (!nset.empty())
   {
      auto *bn = new BoundaryNormalLFIntegrator(conormal_coef);
      bn->SetIntRule(&ir_bdr);
      b.AddBoundaryIntegrator(bn, neu_bdr);
   }

   BilinearForm a(&fespace);
   if (asm_s == "partial")
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.SetDiagonalPolicy(Operator::DIAG_ONE);
   }
   BilinearFormIntegrator *integ = nullptr;
   if (mc->source_without_coefficient_gradient && diffusivity)
   {
      integ = new DiffusionIntegrator(*diffusivity);
      integ->SetIntRule(&IntRules.Get(geom, stiffness_q));
   }
   else if (diffusivity)
   {
      integ = new DiffusionIntegrator(*diffusivity);
   }
   else
   {
      integ = new DiffusionIntegrator;
   }
   a.AddDomainIntegrator(integ);

   GridFunction x(&fespace);
   x = 0.0;

   std::unique_ptr<ErrorEstimator> estimator;
   if (ls_zz)
   {
      estimator = std::make_unique<LSZienkiewiczZhuEstimator>(*integ, x);
   }
   else
   {
      auto *flux_fes = new FiniteElementSpace(&mesh, &fec, sdim);
      auto *zz = new ZienkiewiczZhuEstimator(*integ, x, flux_fes);
      if (anisotropic_zz) { zz->SetAnisotropic(); }
      estimator.reset(zz);
   }
   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(theta);

   fs::create_directories(out_dir);
   struct StepRecord
   {
      int step;
      int num_elements;
      int num_boundary_elements;
      int num_dofs;
      int num_true_dofs;
      bool conforming;
      int iters;
      bool converged;
      double l2;
      double h1s;
      double step_ms;
      double estimator_total;
      std::string dir;
   };
   std::vector<StepRecord> history;
   auto ms = [](auto a_, auto b_) { return std::chrono::duration<double, std::milli>(b_ - a_).count(); };

   const IntegrationRule *irs[Geometry::NumGeom];
   for (int g = 0; g < Geometry::NumGeom; g++) { irs[g] = &IntRules.Get(g, err_q); }

   for (int it = 0; it < max_steps; it++)
   {
      const auto st0 = std::chrono::steady_clock::now();
      const int v_dofs = fespace.GetNDofs();
      const int t_dofs = fespace.GetTrueVSize();

      b.Assemble();
      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(u_coef, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      a.Assemble();
      OperatorPtr A;
      Vector B, X;
      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

      CGSolver cg;
      cg.SetRelTol(rel_tol);
      cg.SetAbsTol(abs_tol);
      cg.SetMaxIter(max_iter);
      cg.SetPrintLevel(IterativeSolver::PrintLevel().Errors().Warnings());
      std::unique_ptr<Solver> prec;
      if (sol_s == "cg_gs") { prec.reset(new GSSmoother(*A.As<SparseMatrix>())); }
      else if (sol_s == "cg_jacobi")
      {
         if (asm_s == "full") { prec.reset(new DSmoother(*A.As<SparseMatrix>())); }
         else { prec.reset(new OperatorJacobiSmoother(a, ess_tdof_list)); }
      }
      cg.SetOperator(*A);
      if (prec) { cg.SetPreconditioner(*prec); }
      cg.Mult(B, X);

      const bool converged = cg.GetConverged();
      const int iters = cg.GetNumIterations();
      const double init_norm = cg.GetInitialNorm();
      const double final_norm = cg.GetFinalNorm();
      a.RecoverFEMSolution(X, b, x);

      if (fault_s == "break_hanging_node" && !mesh.Conforming())
      {
         const auto *P = dynamic_cast<const SparseMatrix *>(fespace.GetProlongationMatrix());
         if (P)
         {
            for (int i = 0; i < v_dofs; i++)
            {
               if (P->RowSize(i) > 1) { x(i) += 0.05; break; }
            }
         }
      }

      bool finite = true;
      for (int i = 0; i < x.Size(); i++) { if (!std::isfinite(x(i))) { finite = false; break; } }
      double l2_err = finite ? x.ComputeL2Error(u_coef, irs) : NAN;
      double h1s_err = finite ? x.ComputeGradError(&grad_coef, irs) : NAN;
      if (fault_s == "self_report_lie") { l2_err = 0.0; h1s_err = 0.0; }

      std::ostringstream sname;
      sname << "step_" << std::setw(2) << std::setfill('0') << it;
      const std::string sd = std::string(out_dir) + "/" + sname.str();
      fs::create_directories(sd);

      {
         std::ofstream f(sd + "/elements.txt");
         f << std::setprecision(17);
         f << "# elem geom nverts v0x v0y ...\n";
         Array<int> v;
         for (int e = 0; e < mesh.GetNE(); e++)
         {
            mesh.GetElementVertices(e, v);
            f << e << " " << (mesh.GetElementType(e) == Element::TRIANGLE ? "tri" : "quad") << " " << v.Size();
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
         std::ofstream f(sd + "/element_dofs.txt");
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
         std::ofstream f(sd + "/dof_values.txt");
         f << std::setprecision(17);
         f << "# global_dof value\n";
         for (int i = 0; i < x.Size(); i++) { f << i << " " << x(i) << "\n"; }
         if (!FinishExport(f, "dof_values.txt")) { return 7; }
      }
      {
         std::ofstream f(sd + "/boundary.txt");
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
         std::ofstream mf(sd + "/mesh.mesh");
         mf.precision(17);
         mesh.Print(mf);
         if (!FinishExport(mf, "mesh.mesh")) { return 7; }
         std::ofstream sf(sd + "/solution.gf");
         sf.precision(17);
         x.Save(sf);
         if (!FinishExport(sf, "solution.gf")) { return 7; }
      }

      const auto st1 = std::chrono::steady_clock::now();
      const double est_total = estimator->GetTotalError();
      {
         std::ofstream f(sd + "/result.json");
         f << "{\n"
           << "  \"producer\": \"poisson_amr\",\n"
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
           << "  \"mesh_topology\": \"amr\",\n"
           << "  \"conforming\": " << (mesh.Conforming() ? "true" : "false") << ",\n"
           << "  \"amr_step\": " << it << ",\n"
           << "  \"order\": " << order << ",\n"
           << "  \"num_elements\": " << mesh.GetNE() << ",\n"
           << "  \"num_boundary_elements\": " << mesh.GetNBE() << ",\n"
           << "  \"num_dofs\": " << v_dofs << ",\n"
           << "  \"num_true_dofs\": " << t_dofs << ",\n"
           << "  \"num_essential_dofs\": " << ess_tdof_list.Size() << ",\n"
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
           << "  \"zz_total_error_estimate\": " << JsonNum(est_total) << ",\n"
           << "  \"injected_fault\": \"" << fault_s << "\",\n"
           << "  \"step_wall_ms\": " << JsonNum(ms(st0, st1)) << "\n"
           << "}\n";
         if (!FinishExport(f, "result.json")) { return 7; }
      }

      history.push_back({it, mesh.GetNE(), mesh.GetNBE(), v_dofs, t_dofs, mesh.Conforming(),
                         iters, converged, l2_err, h1s_err, ms(st0, st1), est_total, sname.str()});
      std::cout << std::setprecision(6)
                << "poisson_amr step=" << it << " elems=" << mesh.GetNE()
                << " vdofs=" << v_dofs << " tdofs=" << t_dofs
                << " conforming=" << mesh.Conforming()
                << " cg_iters=" << iters << " L2=" << l2_err << " H1semi=" << h1s_err << "\n";

      if (!finite) { return 6; }
      if (!converged) { return 5; }
      if (it + 1 >= max_steps || t_dofs > max_dofs) { break; }

      refiner.Apply(mesh);
      if (refiner.Stop()) { break; }
      fespace.Update();
      x.Update();
      a.Update();
      b.Update();
   }

   const auto total_t1 = std::chrono::steady_clock::now();
   {
      std::ofstream f(std::string(out_dir) + "/amr_summary.json");
      f << "{\n"
        << "  \"producer\": \"poisson_amr\",\n"
        << "  \"case\": \"" << mc->id << "\",\n"
        << "  \"mesh_type\": \"" << mt << "\",\n"
        << "  \"order\": " << order << ",\n"
        << "  \"assembly\": \"" << asm_s << "\",\n"
        << "  \"solver\": \"" << sol_s << "\",\n"
        << "  \"theta\": " << JsonNum(theta) << ",\n"
        << "  \"total_wall_ms\": " << JsonNum(ms(total_t0, total_t1)) << ",\n"
        << "  \"peak_rss_kib\": " << PeakRssKiB() << ",\n"
        << "  \"steps\": [\n";
      for (size_t i = 0; i < history.size(); i++)
      {
         const auto &s = history[i];
         f << "    {\"step\": " << s.step << ", \"dir\": \"" << s.dir << "\""
           << ", \"num_elements\": " << s.num_elements
           << ", \"num_dofs\": " << s.num_dofs
           << ", \"num_true_dofs\": " << s.num_true_dofs
           << ", \"conforming\": " << (s.conforming ? "true" : "false")
           << ", \"solver_iterations\": " << s.iters
           << ", \"self_reported_l2_error\": " << JsonNum(s.l2)
           << ", \"self_reported_h1_semi_error\": " << JsonNum(s.h1s)
           << ", \"step_wall_ms\": " << JsonNum(s.step_ms) << "}"
           << (i + 1 < history.size() ? "," : "") << "\n";
      }
      f << "  ]\n}\n";
      if (!FinishExport(f, "amr_summary.json")) { return 7; }
   }
   return 0;
}
