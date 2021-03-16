cmake_minimum_required(VERSION 3.19)

project(AsyncMPI CXX C)

find_package(MPI REQUIRED C CXX)

find_package(TBB)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/libunifex")

add_library(AsyncMPI INTERFACE)
target_sources(AsyncMPI 
  INTERFACE 
    include/ampi/wait_all.hpp
    include/ampi/tbb_task_group_context.hpp)
target_include_directories(AsyncMPI INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_link_libraries(AsyncMPI INTERFACE unifex MPI::MPI_C MPI::MPI_CXX TBB::tbb)
target_compile_features(AsyncMPI INTERFACE cxx_std_20)

add_executable(Recv examples/recv.cpp) 
target_link_libraries(Recv PRIVATE AsyncMPI)