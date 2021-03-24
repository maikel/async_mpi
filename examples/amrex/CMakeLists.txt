if (TBB_FOUND)
  add_executable(FillBoundary FillBoundary.cpp)
  target_link_libraries(FillBoundary PRIVATE AsyncMPI)

  add_executable(SolveEulerEquations_TBB SolveEulerEquations_TBB.cpp)
  target_link_libraries(SolveEulerEquations_TBB PUBLIC AsyncMPI)
endif()

add_executable(FillBoundary_old FillBoundary_old.cpp)
target_link_libraries(FillBoundary_old PRIVATE AsyncMPI)

add_executable(SolveEulerEquations SolveEulerEquations.cpp)
target_link_libraries(SolveEulerEquations PUBLIC AsyncMPI)

add_executable(SolveEulerEquations_old SolveEulerEquations_old.cpp)
target_link_libraries(SolveEulerEquations_old PUBLIC AsyncMPI)