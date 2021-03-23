cmake_minimum_required(VERSION 3.16)

project(AsyncMPI CXX C Fortran)

find_package(MPI REQUIRED C CXX)
find_package(TBB REQUIRED)
find_package(unifex REQUIRED)
find_package(AMReX REQUIRED)

add_library(AsyncMPI INTERFACE)
target_sources(AsyncMPI 
  INTERFACE 
    include/ampi/bulk_finally.hpp
    include/ampi/bulk_on.hpp
    include/ampi/bulk_sequence.hpp
    include/ampi/for_each.hpp
    include/ampi/mpi_abort_on_error.hpp
    include/ampi/tbb_task_scheduler.hpp
    include/ampi/amrex/FillBoundary.hpp
    include/ampi/amrex/ReduceMax.hpp)

target_include_directories(AsyncMPI INTERFACE 
                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/>)
target_link_libraries(AsyncMPI INTERFACE unifex::unifex MPI::MPI_C MPI::MPI_CXX TBB::tbb)
if (AMReX_FOUND)
  target_link_libraries(AsyncMPI INTERFACE AMReX::amrex)
endif()

target_compile_features(AsyncMPI INTERFACE cxx_std_20)

option(AMPI_WITH_EXAMPLES "Compile examples for AsyncMPI" ON)
if (AMPI_WITH_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif()