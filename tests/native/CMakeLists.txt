cmake_minimum_required(VERSION 3.16)
project(diagnostic_contract_test LANGUAGES CXX)
find_package(MFEM 4.10.0 EXACT REQUIRED CONFIG NAMES MFEM mfem)

add_executable(residual_contract residual_contract.cpp)
target_compile_features(residual_contract PRIVATE cxx_std_17)
target_include_directories(residual_contract PRIVATE ../../cpp ${MFEM_INCLUDE_DIRS})
target_link_libraries(residual_contract PRIVATE ${MFEM_LIBRARIES})
