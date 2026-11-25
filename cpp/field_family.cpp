// Manufactured heat, wave and isotropic elasticity studies on tensor H1 spaces.
#include "mfem.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mfem;
namespace {
const double pi = std::acos(-1.);
constexpr double end_time = 1.0 / 32.0;

double mode(const Vector &x)
{
   double value = 1.;
   for (int i = 0; i < x.Size(); ++i) { value *= std::sin(pi * x[i]); }
   return value;
}

// For lambda = mu = 1 and displacement (s,0,...), f = -Delta u - 2 grad div u.
void elastic_load(const Vector &x, Vector &out)
{
   out.SetSize(x.Size());
   out[0] = (x.Size() + 2) * pi * pi * mode(x);
   for (int i = 1; i < x.Size(); ++i)
   {
      double value = -2. * pi * pi;
      for (int j = 0; j < x.Size(); ++j)
      {
         value *= (j == 0 || j == i) ? std::cos(pi*x[j]) : std::sin(pi*x[j]);
      }
      out[i] = value;
   }
}

void configure(CGSolver &cg, const SparseMatrix &A, GSSmoother &prec)
{
   cg.SetOperator(A);
   cg.SetPreconditioner(prec);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(4000);
   cg.SetPrintLevel(0);
}
void solve(CGSolver &cg, const Vector &rhs, Vector &field)
{
   field.SetSize(rhs.Size());
   field = 0.;
   cg.Mult(rhs, field);
   if (!cg.GetConverged()) { throw std::runtime_error("CG did not converge"); }
}

std::vector<Vector> evolve(FiniteElementSpace &fes, const Array<int> &boundary, bool wave, int steps)
{
   const double dt = end_time / steps;
   BilinearForm mass(&fes), stiff(&fes);
   mass.AddDomainIntegrator(new MassIntegrator);
   stiff.AddDomainIntegrator(new DiffusionIntegrator);
   mass.Assemble(); mass.Finalize();
   stiff.Assemble(); stiff.Finalize();
   SparseMatrix M, K;
   mass.FormSystemMatrix(boundary, M);
   stiff.FormSystemMatrix(boundary, K);
   const double factor = wave ? dt*dt/4 : dt/2;
   std::unique_ptr<SparseMatrix> A(Add(1., M, factor, K));
   GSSmoother prec(*A);
   CGSolver cg;
   configure(cg, *A, prec);
   FunctionCoefficient initial(mode);
   GridFunction u(&fes);
   u.ProjectCoefficient(initial);
   u.SetSubVector(boundary, 0.);
   Vector v(u.Size()), next(u.Size()), rhs(u.Size()), work(u.Size());
   v = 0.;
   std::vector<Vector> snapshots;
   for (int k = 1; k <= steps; ++k)
   {
      if (wave)
      {
         M.Mult(v, rhs);
         K.Mult(v, work); rhs.Add(-factor, work);
         K.Mult(u, work); rhs.Add(-dt, work);
         solve(cg, rhs, next);
         u.Add(dt/2, v); u.Add(dt/2, next);
         v = next;
      }
      else
      {
         M.Mult(u, rhs);
         K.Mult(u, work); rhs.Add(-factor, work);
         solve(cg, rhs, next);
         u = next;
      }
      if (k == steps/2 || k == steps)
      {
         snapshots.emplace_back(u);
         if (wave) { snapshots.emplace_back(v); }
      }
   }
   return snapshots;
}

std::vector<Vector> elasticity(Mesh &mesh, H1_FECollection &collection, int order)
{
   const int d = mesh.Dimension();
   FiniteElementSpace vfes(&mesh, &collection, d, Ordering::byNODES);
   Array<int> markers(mesh.bdr_attributes.Max()), ess;
   markers = 1;
   vfes.GetEssentialTrueDofs(markers, ess);
   ConstantCoefficient one(1.);
   VectorFunctionCoefficient load(d, elastic_load);
   LinearForm b(&vfes);
   auto *lf = new VectorDomainLFIntegrator(load);
   lf->SetIntRule(&IntRules.Get(mesh.GetElementBaseGeometry(0), 2*order+12));
   b.AddDomainIntegrator(lf); b.Assemble();
   BilinearForm a(&vfes);
   a.AddDomainIntegrator(new ElasticityIntegrator(one, one)); a.Assemble();
   GridFunction solution(&vfes);
   solution = 0.;
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess, solution, b, A, X, B);
   GSSmoother prec(A);
   CGSolver cg; configure(cg, A, prec); solve(cg, B, X);
   a.RecoverFEMSolution(X, b, solution);
   std::vector<Vector> components(d, Vector(vfes.GetNDofs()));
   for (int c = 0; c < d; ++c)
   {
      for (int i = 0; i < vfes.GetNDofs(); ++i) { components[c][i] = solution[vfes.DofToVDof(i, c)]; }
   }
   return components;
}
}

int main(int argc, char **argv)
{
   try
   {
      if (argc != 7) { throw std::invalid_argument("family dimension cells order steps field.tsv"); }
      const std::string family(argv[1]);
      const int d=std::stoi(argv[2]), n=std::stoi(argv[3]), p=std::stoi(argv[4]), steps=std::stoi(argv[5]);
      if ((family!="heat" && family!="wave" && family!="elasticity") || (d!=2 && d!=3) ||
          n<2 || n>32 || (p!=1 && p!=2) || steps<2 || steps>128 || steps%2)
      { throw std::invalid_argument("unsupported manufactured study"); }
      Mesh mesh = d==2 ? Mesh::MakeCartesian2D(n,n,Element::QUADRILATERAL,true) :
                         Mesh::MakeCartesian3D(n,n,n,Element::HEXAHEDRON);
      H1_FECollection fec(p,d);
      FiniteElementSpace fes(&mesh,&fec);
      Array<int> markers(mesh.bdr_attributes.Max()), ess;
      markers=1; fes.GetEssentialTrueDofs(markers,ess);
      auto fields = family=="elasticity" ? elasticity(mesh,fec,p) : evolve(fes,ess,family=="wave",steps);
      std::vector<Vector> coordinates;
      for (int axis=0; axis<d; ++axis)
      {
         FunctionCoefficient x([axis](const Vector &v){return v[axis];});
         GridFunction nodes(&fes); nodes.ProjectCoefficient(x);
         coordinates.emplace_back(nodes);
      }
      std::ofstream out(argv[6]);
      out.exceptions(std::ios::badbit | std::ios::failbit);
      out << std::setprecision(17);
      for (int i=0; i<fes.GetNDofs(); ++i)
      {
         for (const auto &x : coordinates) { out << x[i] << '\t'; }
         for (size_t c=0; c<fields.size(); ++c) { if (c) { out << '\t'; } out << fields[c][i]; }
         out << '\n';
      }
      out.close();
      return 0;
   }
   catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}
