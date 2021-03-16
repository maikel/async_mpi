#pragma once

#include <mpi.h>

namespace ampi
{
  struct mpi_abort_on_error {
    MPI_Comm comm;
    template <typename... Ts>
    void set_value(Ts&&...) const noexcept {}
    void set_done() const noexcept { MPI_Abort(comm, MPI_ERR_OTHER); }
    template <typename... Err>
    void set_error(Err&&...) const noexcept {
      MPI_Abort(comm, MPI_ERR_OTHER);
    }
  };

}  // namespace ampi