#include <ampi/bulk_finally.hpp>

#include <unifex/scope_guard.hpp>

#include <mpi.h>

int main()
{
  MPI_Init(nullptr, nullptr);
  unifex::scope_guard scope_guard = []() noexcept {
    MPI_Finalize();
  };
}