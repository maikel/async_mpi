cmake_minimum_required(VERSION 3.19)

project(AsyncMPI CXX C)

find_package(MPI REQUIRED C CXX)

find_package(TBB REQUIRED)

find_package(unifex REQUIRED)

add_library(AsyncMPI INTERFACE)
target_sources(AsyncMPI 
  INTERFACE 
    include/ampi/bulk_sequence.hpp
    include/ampi/bulk_finally.hpp
    include/ampi/for_each.hpp
    include/ampi/mpi_abort_on_error.hpp
    include/ampi/tbb_task_scheduler.hpp)

target_include_directories(AsyncMPI INTERFACE 
                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/>)
target_link_libraries(AsyncMPI INTERFACE unifex::unifex MPI::MPI_C MPI::MPI_CXX TBB::tbb)
target_compile_features(AsyncMPI INTERFACE cxx_std_20)

option(AMPI_WITH_EXAMPLES "Compile examples for AsyncMPI" ON)
if (AMPI_WITH_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif()