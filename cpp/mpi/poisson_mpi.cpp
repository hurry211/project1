// Fully assembled P1 Poisson on the same unit-square triangles as poisson_mms.
#include "manufactured_cases.hpp"

#if !defined(MFEM_USE_MPI) || !defined(MFEM_USE_METIS_5) || !defined(MFEM_USE_METIS) || !defined(MFEM_USE_DOUBLE)
#error "poisson_mpi requires double-precision MFEM configured with MPI and METIS 5"
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace mfem;

namespace
{

void BuildInfo()
{
   // CMake requires the MPI/METIS5/double configuration. Obtain version and
   // git from the linked library; MFEM supplies the hypre version at build time.
   std::cout << "{\"schema_version\":1,\"mfem_version\":\""
             << GetVersionMajor() << '.' << GetVersionMinor() << '.' << GetVersionPatch()
             << "\",\"mfem_git\":\"" << GetGitStr() << "\",\"mfem_use_mpi\":true,"
             << "\"mfem_use_metis\":true,\"mfem_use_metis_5\":true,\"mfem_use_double\":true,"
             << "\"hypre_version\":" << MFEM_HYPRE_VERSION
             << ",\"case\":\"sin_sin\",\"solver\":\"cg_boomeramg\"}\n";
}

bool ExportPartition(const fs::path &directory, int rank, ParMesh &mesh,
                     ParFiniteElementSpace &space, const ParGridFunction &field)
{
   const std::string suffix = "." + std::to_string(rank) + ".tsv";
   std::ofstream cells(directory / ("cells" + suffix));
   std::ofstream boundary(directory / ("boundary" + suffix));
   cells << std::setprecision(17);
   boundary << std::setprecision(17);
   Array<int> vertices, dofs;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mesh.GetElementVertices(e, vertices);
      for (int j = 0; j < vertices.Size(); ++j)
      {
         int v = vertices[j];
         const double *xy = mesh.GetVertex(v);
         space.GetVertexDofs(v, dofs);
         if (dofs.Size() != 1 || !std::isfinite(field(dofs[0]))) { return false; }
         if (j) { cells << '\t'; }
         cells << xy[0] << '\t' << xy[1] << '\t' << field(dofs[0]);
      }
      cells << '\n';
   }
   for (int e = 0; e < mesh.GetNBE(); ++e)
   {
      mesh.GetBdrElementVertices(e, vertices);
      boundary << mesh.GetBdrAttribute(e);
      for (int v : vertices)
      {
         const double *xy = mesh.GetVertex(v);
         boundary << '\t' << xy[0] << '\t' << xy[1];
      }
      boundary << '\n';
   }
   cells.close();
   boundary.close();
   return cells.good() && boundary.good();
}

int Solve(int argc, char **argv)
{
   const MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank(), ranks = Mpi::WorldSize();
   const char *output = "";
   int n = 8, max_iter = 20000;
   double tol = 1e-12;
   OptionsParser args(argc, argv);
   args.AddOption(&n, "-n", "--n", "Number of square cells per side.");
   args.AddOption(&tol, "-rtol", "--rel-tol", "Relative PCG tolerance.");
   args.AddOption(&max_iter, "-mi", "--max-iter", "Maximum PCG iterations.");
   args.AddOption(&output, "-o", "--out", "Existing, empty output directory.", true);
   args.Parse();
   if (!args.Good() || n < 2 || n > 4096 || ranks > 2*n*n ||
       !std::isfinite(tol) || tol <= 0 || tol >= 1 || max_iter < 1)
   {
      if (rank == 0) { args.PrintUsage(std::cerr); }
      return 2;
   }
   const fs::path directory(output), marker = directory / ".mpi-in-progress";
   int ready = 1;
   if (rank == 0)
   {
      std::error_code ec;
      ready = fs::is_directory(directory, ec) && fs::is_empty(directory, ec) &&
              fs::create_directory(marker, ec);
      if (!ready) { std::cerr << "Expected an existing, empty output directory\n"; }
   }
   MPI_Bcast(&ready, 1, MPI_INT, 0, comm);
   if (!ready) { return 7; }

   MPI_Barrier(comm);
   const double start = MPI_Wtime();
   // The small serial input mesh is replicated before METIS partitions it.
   Mesh serial = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE, true, 1., 1.);
   ParMesh mesh(comm, serial);
   H1_FECollection fec(1, 2);
   ParFiniteElementSpace space(&mesh, &fec);
   Array<int> essential_boundary(mesh.bdr_attributes.Max()), essential_dofs;
   essential_boundary = 1;
   space.GetEssentialTrueDofs(essential_boundary, essential_dofs);

   const auto *problem = mms::Find("sin_sin");
   FunctionCoefficient exact(problem->u), source(problem->f);
   VectorFunctionCoefficient gradient(2, problem->grad_u);
   constexpr int quadrature_order = 6;
   const IntegrationRule &rule = IntRules.Get(Geometry::TRIANGLE, quadrature_order);
   ParLinearForm load(&space);
   load.AddDomainIntegrator(new DomainLFIntegrator(source, &rule));
   load.Assemble();
   ParGridFunction field(&space);
   field = 0.;
   field.ProjectBdrCoefficient(exact, essential_boundary);

   ParBilinearForm stiffness(&space);
   stiffness.AddDomainIntegrator(new DiffusionIntegrator);
   stiffness.Assemble();
   OperatorPtr A;
   Vector X, B;
   stiffness.FormLinearSystem(essential_dofs, field, load, A, X, B);
   HypreBoomerAMG amg;
   amg.SetPrintLevel(0);
   CGSolver cg(comm);
   cg.SetPreconditioner(amg);
   cg.SetOperator(*A);
   cg.SetRelTol(tol);
   cg.SetAbsTol(0.);
   cg.SetMaxIter(max_iter);
   cg.SetPrintLevel(0);
   const double setup_seconds = MPI_Wtime() - start;
   MPI_Barrier(comm);
   const double solve_start = MPI_Wtime();
   cg.Mult(B, X);
   const double local_times[] = {setup_seconds, MPI_Wtime() - solve_start};
   double times[2];
   MPI_Reduce(local_times, times, 2, MPI_DOUBLE, MPI_MAX, 0, comm);

   Vector residual(B.Size());
   A->Mult(X, residual);
   residual -= B;
   const double residual_norm = std::sqrt(InnerProduct(comm, residual, residual));
   const double rhs_norm = std::sqrt(InnerProduct(comm, B, B));
   stiffness.RecoverFEMSolution(X, load, field);
   const IntegrationRule *rules[Geometry::NumGeom] = {};
   rules[Geometry::TRIANGLE] = &rule;
   const double l2 = field.ComputeL2Error(exact, rules);
   const double h1 = field.ComputeGradError(&gradient, rules);
   const auto dofs = space.GlobalTrueVSize();
   const int local_elements = mesh.GetNE();
   int global_elements = 0;
   MPI_Reduce(&local_elements, &global_elements, 1, MPI_INT, MPI_SUM, 0, comm);

   int success = std::isfinite(residual_norm) && std::isfinite(rhs_norm) &&
                 std::isfinite(l2) && std::isfinite(h1) &&
                 std::isfinite(cg.GetInitialNorm()) && std::isfinite(cg.GetFinalNorm()) &&
                 ExportPartition(directory, rank, mesh, space, field);
   int all_ok = 0;
   MPI_Reduce(&success, &all_ok, 1, MPI_INT, MPI_MIN, 0, comm);
   if (rank == 0 && all_ok)
   {
      std::ofstream out(directory / "result.json");
      out << std::setprecision(17) << "{\n"
          << "  \"case\": \"sin_sin\", \"order\": 1, \"solver\": \"cg_boomeramg\",\n"
          << "  \"n\": " << n << ", \"ranks\": " << ranks << ",\n"
          << "  \"global_dofs\": " << dofs << ", \"global_elements\": " << global_elements << ",\n"
          << "  \"rhs_quadrature_order\": " << quadrature_order
          << ", \"error_quadrature_order\": " << quadrature_order << ",\n"
          << "  \"rel_tol\": " << tol << ", \"abs_tol\": 0, \"max_iter\": " << max_iter << ",\n"
          << "  \"solver_converged\": " << std::boolalpha << cg.GetConverged()
          << ", \"solver_iterations\": " << cg.GetNumIterations() << ",\n"
          << "  \"solver_initial_norm\": " << cg.GetInitialNorm()
          << ", \"solver_final_norm\": " << cg.GetFinalNorm() << ",\n"
          << "  \"rhs_l2\": " << rhs_norm << ", \"residual_l2\": " << residual_norm << ",\n"
          << "  \"l2_error\": " << l2 << ", \"h1_semi_error\": " << h1 << ",\n"
          << "  \"setup_seconds_max\": " << times[0] << ", \"solve_seconds_max\": " << times[1]
          << "\n}\n";
      out.close();
      std::error_code ec;
      all_ok = out.good() && fs::remove(marker, ec);
   }
   MPI_Bcast(&all_ok, 1, MPI_INT, 0, comm);
   if (!all_ok)
   {
      if (rank == 0) { std::cerr << "Nonfinite solution or incomplete output\n"; }
      return 7;
   }
   return cg.GetConverged() ? 0 : 5;
}
} // namespace

int main(int argc, char **argv)
{
   if (argc == 2 && std::string(argv[1]) == "--build-info")
   {
      BuildInfo();
      return 0;
   }
   Mpi::Init(argc, argv);
   Hypre::Init();
   return Solve(argc, argv);
}
