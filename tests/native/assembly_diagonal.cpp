// Compare every Jacobi inverse diagonal entry after essential elimination.
#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
   using namespace mfem;
   auto mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL);
   FunctionCoefficient kappa([](const Vector &x) -> real_t { return 1.0 + x(0); });
   for (int p : {1, 2})
   {
      H1_FECollection fec(p, 2);
      FiniteElementSpace fes(&mesh, &fec);
      Array<int> essential(mesh.bdr_attributes.Max()), tdofs;
      essential = 1;
      essential[2] = 0; // Mixed boundary conditions.
      fes.GetEssentialTrueDofs(essential, tdofs);
      for (bool variable : {false, true})
      {
         Vector ones(fes.GetTrueVSize()), reference(ones.Size()), action(ones.Size());
         ones = 1.0;
         for (bool partial : {false, true})
         {
            BilinearForm a(&fes);
            if (partial) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
            auto *integrator = variable ? new DiffusionIntegrator(kappa) : new DiffusionIntegrator();
            if (variable) { integrator->SetIntRule(&IntRules.Get(Geometry::SQUARE, 2*p+1)); }
            a.AddDomainIntegrator(integrator);
            a.Assemble();
            OperatorPtr op;
            a.FormSystemMatrix(tdofs, op);
            OperatorJacobiSmoother jacobi(a, tdofs);
            jacobi.Mult(ones, action);
            if (!partial)
            {
               // MFEM may retain an original diagonal on constrained rows.
               // The smoother explicitly uses unit action on those entries.
               op.As<SparseMatrix>()->GetDiag(reference);
               for (int i = 0; i < reference.Size(); ++i) { reference(i) = 1.0/reference(i); }
               reference.SetSubVector(tdofs, 1.0);
            }
            action -= reference;
            const double gap = action.Normlinf()/reference.Normlinf();
            std::cout << std::setprecision(17) << p << " " << variable << " "
                      << partial << " " << gap << "\n";
            if (!std::isfinite(gap) || gap > 1e-12) { return 1; }
         }
      }
   }
}
