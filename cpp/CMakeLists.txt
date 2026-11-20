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
target_include_directories(poisson_mms PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(poisson_mms PRIVATE ${MFEM_LIBRARIES})
target_compile_options(poisson_mms PRIVATE -Wall -Wextra)
