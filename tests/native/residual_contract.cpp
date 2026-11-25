#include "poisson_diagnostics.hpp"

#include <cmath>
#include <iostream>
#include <limits>

int main()
{
   mfem::DenseMatrix A(2);
   A = 0.0;
   A(0, 0) = 3.0;
   A(1, 1) = 2.0;
   mfem::Vector x(2), b(2);
   x(0) = 1.0;
   x(1) = 2.0;
   b = 0.0;

   auto zero = poisson_diagnostics::MeasureResidual(A, x, b);
   if (zero.absolute_l2 != 5.0 || zero.rhs_l2 != 0.0 || !std::isnan(zero.relative_l2))
   {
      std::cerr << "zero RHS should preserve ||Ax-b||=5 and leave the ratio undefined\n";
      return 1;
   }

   b(0) = 3.0;
   auto nonzero = poisson_diagnostics::MeasureResidual(A, x, b);
   if (nonzero.absolute_l2 != 4.0 || nonzero.rhs_l2 != 3.0 || nonzero.relative_l2 != 4.0 / 3.0)
   {
      std::cerr << "nonzero RHS should give ||Ax-b||/||b||=4/3\n";
      return 1;
   }

   x(0) = std::numeric_limits<double>::quiet_NaN();
   auto invalid = poisson_diagnostics::MeasureResidual(A, x, b);
   if (!std::isnan(invalid.absolute_l2) || invalid.rhs_l2 != 3.0 || !std::isnan(invalid.relative_l2))
   {
      std::cerr << "nonfinite candidate should leave only the finite RHS norm available\n";
      return 1;
   }
}
