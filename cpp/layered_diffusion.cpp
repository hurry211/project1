// Experimental high-contrast, interface-aligned diffusion benchmark.
// MFEM computes the field; the independent Python checker reconstructs it.
// -div(a grad u)=0, a=1 (x<1/2), kappa (x>1/2).
// u=0 on x=0, u=1 on x=1, zero natural flux on y=0,1.
#include "mfem.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <sys/resource.h>

using namespace mfem;

int main(int argc, char **argv)
{
   int n=8;
   double kappa=100.0;
   const char *out=".";
   OptionsParser args(argc, argv);
   args.AddOption(&n,"-n","--n","Even number of elements per side.");
   args.AddOption(&kappa,"-k","--kappa","Positive right/left diffusion ratio, 1..1e6.");
   args.AddOption(&out,"-out","--out","Existing output directory.");
   args.Parse();
   if (!args.Good() || n<2 || n>128 || n%2 ||
       !std::isfinite(kappa) || kappa<1.0 || kappa>1e6)
   {
      args.PrintUsage(std::cerr); return 2;
   }

   const auto t0=std::chrono::steady_clock::now();
   Mesh mesh=Mesh::MakeCartesian2D(n,n,Element::QUADRILATERAL,true,1.0,1.0);
   H1_FECollection fec(1,2);
   FiniteElementSpace fes(&mesh,&fec);
   if (mesh.bdr_attributes.Size()!=4 || mesh.bdr_attributes.Max()!=4)
   { std::cerr<<"Unexpected boundary attributes\n"; return 3; }
   // The independently checked field must also satisfy the actual geometric BC.
   Array<int> ess(4); ess=0; ess[1]=1; ess[3]=1;
   Array<int> tdofs;
   fes.GetEssentialTrueDofs(ess,tdofs);

   FunctionCoefficient acoef([kappa](const Vector &p)
   { return p(0)<0.5 ? 1.0 : kappa; });
   FunctionCoefficient bc([](const Vector &p){return p(0);});
   GridFunction u(&fes); u=0.0;
   u.ProjectBdrCoefficient(bc,ess);
   LinearForm b(&fes); b=0.0;
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(acoef));
   a.Assemble();
   OperatorPtr A;
   Vector B,X;
   a.FormLinearSystem(tdofs,u,b,A,X,B);
   const auto t1=std::chrono::steady_clock::now();

   CGSolver cg;
   GSSmoother prec(*A.As<SparseMatrix>());
   cg.SetOperator(*A);
   cg.SetPreconditioner(prec);
   cg.SetRelTol(1e-13);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(20000);
   cg.SetPrintLevel(IterativeSolver::PrintLevel().Errors().Warnings());
   cg.Mult(B,X);
   const auto t2=std::chrono::steady_clock::now();
   if (!cg.GetConverged()) {std::cerr<<"CG did not converge\n";return 5;}
   a.RecoverFEMSolution(X,b,u);

   // Export actual MFEM cell-local field, not an analytic reconstruction.
   std::ofstream field(std::string(out)+"/field.txt");
   field<<std::setprecision(17);
   Array<int> dofs;
   Vector xp;
   for (int e=0;e<mesh.GetNE();++e)
   {
      fes.GetElementDofs(e,dofs);
      const FiniteElement *fe=fes.GetFE(e);
      const IntegrationRule &nodes=fe->GetNodes();
      ElementTransformation *T=mesh.GetElementTransformation(e);
      if (dofs.Size()!=4 || nodes.GetNPoints()!=4) {return 3;}
      for (int k=0;k<dofs.Size();++k)
      {
         T->SetIntPoint(&nodes.IntPoint(k));
         T->Transform(nodes.IntPoint(k),xp);
         if (dofs[k]<0 || !std::isfinite(u(dofs[k]))) {return 6;}
         field<<e<<" "<<xp(0)<<" "<<xp(1)<<" "<<u(dofs[k])<<"\n";
      }
   }
   field.close();
   if (field.fail()) {return 7;}
   struct rusage usage{};
   if (getrusage(RUSAGE_SELF,&usage)!=0) {return 7;}
   auto ms=[](auto x,auto y){return std::chrono::duration<double,std::milli>(y-x).count();};
   std::ofstream record(std::string(out)+"/result.json");
   record<<std::setprecision(17)
         <<"{\"n\":"<<n<<",\"kappa\":"<<kappa
         <<",\"dofs\":"<<fes.GetTrueVSize()
         <<",\"iterations\":"<<cg.GetNumIterations()
         <<",\"setup_ms\":"<<ms(t0,t1)
         <<",\"solve_ms\":"<<ms(t1,t2)
         <<",\"peak_rss_kib\":"<<usage.ru_maxrss<<"}\n";
   record.close();
   return record.fail()?7:0;
}
