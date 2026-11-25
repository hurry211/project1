cmake_minimum_required(VERSION 3.16)
project(mfem_scientific_agent_solvers CXX)

# Point MFEM_DIR at an MFEM install prefix (lib/cmake/mfem) or build tree.
# Example: cmake -S cpp -B build/cpp -DMFEM_DIR=$HOME/third_party/mfem-install/lib/cmake/mfem
if(DEFINED MFEM_DIR AND NOT EXISTS "${MFEM_DIR}/MFEMConfig.cmake")
  message(FATAL_ERROR "MFEMConfig.cmake is missing from MFEM_DIR=${MFEM_DIR}")
endif()
find_package(MFEM 4.10.0 EXACT REQUIRED CONFIG NAMES MFEM mfem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(poisson_mms poisson_mms.cpp)
# Archive the linked package configuration alongside the compilation unit.
set(config_fields)
foreach(feature MPI METIS METIS_5 SUITESPARSE LAPACK OPENMP CUDA HIP RAJA OCCA PETSC ZLIB DOUBLE SINGLE)
  set(enabled false)
  if(MFEM_USE_${feature})
    set(enabled true)
  endif()
  list(APPEND config_fields "\"MFEM_USE_${feature}\":${enabled}")
endforeach()
list(JOIN config_fields "," config_json)
configure_file(mfem_build.hpp.in mfem_build.hpp @ONLY)
target_include_directories(poisson_mms PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(poisson_mms PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(poisson_mms PRIVATE ${MFEM_LIBRARIES})
target_compile_options(poisson_mms PRIVATE -Wall -Wextra)
