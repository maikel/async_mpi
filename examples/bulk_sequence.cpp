#include <ampi/bulk_sequence.hpp>
#include <ampi/for_each.hpp>
#include <ampi/bulk_on.hpp>

#include <unifex/single_thread_context.hpp>
#include <unifex/bulk_join.hpp>
#include <unifex/bulk_transform.hpp>
#include <unifex/sync_wait.hpp>

#include <ranges>

#include <mpi.h>

#define AMPI_CALL_MPI(fun)                                          \
  if (int errc = (fun); errc != MPI_SUCCESS) {                      \
    std::fprintf(stderr, "MPI failed with error code %d.\n", errc); \
    std::terminate();                                               \
  }

using namespace unifex;
using namespace ampi;

void print_received(int index, int rank) {
  const std::size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
  std::printf("%d-%zu: Received request at index %d.\n", rank, thread_id, index);
}

int main() {
  MPI_Init(nullptr, nullptr);
  scope_guard mpi_finalize = []() noexcept {
    MPI_Finalize();
  };

  single_thread_context ctx;
  auto scheduler = ctx.get_scheduler();

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size = -1;
  int comm_rank = -1;
  AMPI_CALL_MPI(MPI_Comm_size(comm, &comm_size));
  AMPI_CALL_MPI(MPI_Comm_rank(comm, &comm_rank));

  const int size = 100;
  const int tag = 0;
  const int root_rank = 0;

  std::vector<int> values(size);
  std::ranges::fill(values, comm_rank);
  std::printf("%d: Sending values to %d ...\n", comm_rank, root_rank);
  MPI_Request send_request = MPI_REQUEST_NULL;
  AMPI_CALL_MPI(MPI_Isend(values.data(), size, MPI_INT, root_rank, tag, comm, &send_request));

  if (comm_rank == root_rank) {
    std::vector<int> all_data(comm_size * size);
    std::vector<std::span<int>> datas(comm_size);
    std::vector<MPI_Request> requests(comm_size);
    std::span<int> data(all_data);
    for (std::size_t i = 0; i < datas.size(); ++i) {
      datas[i] = data.subspan(0, size);
      data = data.subspan(size);
      const int from_rank = i;
      const std::size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
      std::printf("%d-%zu: Post Irecv values from %d ...\n", comm_rank, thread_id, from_rank);
      AMPI_CALL_MPI(MPI_Irecv(datas[i].data(), size, MPI_INT, from_rank, tag, comm, &requests[i]));
    }
    auto sender = bulk_join(bulk_transform(
        bulk_on(scheduler, for_each(requests, comm, tag)),
        [comm_rank](int index) { print_received(index, comm_rank); },
        par_unseq));

    sync_wait(std::move(sender));
  }

  AMPI_CALL_MPI(MPI_Wait(&send_request, MPI_STATUS_IGNORE));
  std::printf("%d: Send done.\n", comm_rank);
}