cmake_minimum_required(VERSION 3.19)

project(AsyncMPI CXX C)

find_package(MPI REQUIRED C CXX)

find_package(TBB REQUIRED)

option(BUILD_TESTING "" OFF)
option(UNIFEX_BUILD_EXAMPLES "" OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/libunifex")

add_library(AsyncMPI INTERFACE)
target_sources(AsyncMPI 
  INTERFACE 
    include/ampi/for_each.hpp
    include/ampi/tbb_task_group_context.hpp)
target_include_directories(AsyncMPI INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_link_libraries(AsyncMPI INTERFACE unifex MPI::MPI_C MPI::MPI_CXX TBB::tbb)
target_compile_features(AsyncMPI INTERFACE cxx_std_20)

add_executable(for_each examples/for_each.cpp) 
target_link_libraries(for_each PRIVATE AsyncMPI)

add_executable(bulk_finally examples/bulk_finally.cpp) 
target_link_libraries(bulk_finally PRIVATE AsyncMPI)