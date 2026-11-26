// Native P1 primal proposal for the CFAC original-P0-load contract.
#include "mfem.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace mfem;

int main(int argc, char **argv)
{
   if (argc != 7) { return 2; }
   const double diffusion = std::stod(argv[3]);
   const double tolerance = std::stod(argv[4]);
   const int steps = std::stoi(argv[5]);
   if (!std::isfinite(diffusion) || diffusion <= 0 || !std::isfinite(tolerance) ||
       tolerance < 0 || tolerance >= 1 || steps <= 0) { return 2; }
   Mesh mesh(argv[1], 1, 1);
   if (mesh.Dimension() != 2 || mesh.GetTypicalElementGeometry() != Geometry::TRIANGLE) { return 2; }
   H1_FECollection elements(1, 2);
   FiniteElementSpace space(&mesh, &elements);
   std::ifstream data(argv[2]); // Cell-ordered P0 load, then vertex-ordered warm start.
   std::vector<double> loads(mesh.GetNE());
   for (double &value : loads)
   {
      if (!(data >> value) || !std::isfinite(value)) { return 2; }
   }
   GridFunction field(&space);
   field = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh.GetNV(); i++)
   {
      double value;
      if (!(data >> value) || !std::isfinite(value)) { return 2; }
      space.GetVertexDofs(i, dofs);
      field(dofs[0]) = value;
   }
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      if (mesh.GetAttribute(e) != e + 1) { return 2; }
   }
   Vector by_attribute(loads.data(), loads.size());
   PWConstCoefficient load(by_attribute);
   ConstantCoefficient a(diffusion), zero(0.);
   Array<int> boundary(mesh.bdr_attributes.Max()), essential;
   boundary = 1;
   field.ProjectBdrCoefficient(zero, boundary);
   space.GetEssentialTrueDofs(boundary, essential);

   LinearForm rhs(&space);
   rhs.AddDomainIntegrator(new DomainLFIntegrator(load));
   rhs.Assemble();
   BilinearForm stiffness(&space);
   stiffness.AddDomainIntegrator(new DiffusionIntegrator(a));
   stiffness.Assemble();
   OperatorPtr A;
   Vector X, B;
   stiffness.FormLinearSystem(essential, field, rhs, A, X, B, 1);
   OperatorJacobiSmoother jacobi(stiffness, essential);
   CGSolver cg;
   cg.SetOperator(*A);
   cg.SetPreconditioner(jacobi);
   cg.iterative_mode = true;
   cg.SetRelTol(tolerance);
   cg.SetAbsTol(1e-30);
   cg.SetMaxIter(steps);
   cg.SetPrintLevel(-1);
   cg.Mult(B, X);
   Vector residual(B.Size());
   A->Mult(X, residual);
   residual -= B;
   stiffness.RecoverFEMSolution(X, rhs, field);

   std::ofstream out(argv[6]);
   out << std::setprecision(17) << cg.GetNumIterations() << " " << residual.Norml2() << "\n";
   for (int i = 0; i < mesh.GetNV(); i++)
   {
      space.GetVertexDofs(i, dofs);
      if (!std::isfinite(field(dofs[0]))) { return 3; }
      out << mesh.GetVertex(i)[0] << " " << mesh.GetVertex(i)[1] << " " << field(dofs[0]) << "\n";
   }
   out.close();
   return out ? 0 : 4;
}
