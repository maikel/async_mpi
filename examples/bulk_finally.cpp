#include <ampi/bulk_finally.hpp>

#include <ampi/mpi_abort_on_error.hpp>
#include <ampi/tbb_task_scheduler.hpp>
#include <ampi/for_each.hpp>

#include <mpi.h>

#include <unifex/bulk_join.hpp>
#include <unifex/bulk_transform.hpp>
#include <unifex/just.hpp>
#include <unifex/on.hpp>
#include <unifex/scope_guard.hpp>
#include <unifex/single_thread_context.hpp>
#include <unifex/static_thread_pool.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>

#include <ranges>
#include <span>
#include <vector>

#define AMPI_CALL_MPI(fun)                                          \
  if (int errc = (fun); errc != MPI_SUCCESS) {                      \
    std::fprintf(stderr, "MPI failed with error code %d.\n", errc); \
    std::terminate();                                               \
  }

int main() {
  MPI_Init(nullptr, nullptr);
  unifex::scope_guard scope_guard = []() noexcept {
    MPI_Finalize();
  };
  using namespace unifex;
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
    for (int i = 0; i < datas.size(); ++i) {
      datas[i] = data.subspan(0, size);
      data = data.subspan(size);
      const int from_rank = i;
      const std::size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
      std::printf("%d-%zu: Post Irecv values from %d ...\n", comm_rank, thread_id, from_rank);
      AMPI_CALL_MPI(MPI_Irecv(datas[i].data(), size, MPI_INT, from_rank, tag, comm, &requests[i]));
    }

    single_thread_context communication_thread{};

    tbb::task_arena arena{};
    ampi::tbb_task_scheduler intel_tbb{arena};

    submit(
        transform(
            schedule(intel_tbb),
            [comm_rank] {
              const std::size_t thread_id =
                  std::hash<std::thread::id>{}(std::this_thread::get_id());
              std::printf("%d-%zu: Hello TBB #1.\n", comm_rank, thread_id);
            }),
        ampi::mpi_abort_on_error{comm});

    submit(
        on(bulk_join(ampi::bulk_finally(
               ampi::for_each(comm, std::move(requests), tag),
               [comm_rank, &intel_tbb](int index) {
                 const std::size_t thread_id =
                     std::hash<std::thread::id>{}(std::this_thread::get_id());
                 std::printf(
                     "%d-%zu: Received request at index %d.\n", comm_rank, thread_id, index);
                 auto work = transform(unifex::schedule(intel_tbb), [comm_rank] {
                   const std::size_t thread_id =
                       std::hash<std::thread::id>{}(std::this_thread::get_id());
                   std::printf("%d-%zu: On TBB thread.\n", comm_rank, thread_id);
                 });
                 return std::tuple{work, work, work};
               })),
           communication_thread.get_scheduler()),
        ampi::mpi_abort_on_error{comm});

    submit(
        transform(
            schedule(intel_tbb),
            [comm_rank] {
              const std::size_t thread_id =
                  std::hash<std::thread::id>{}(std::this_thread::get_id());
              std::printf("%d-%zu: Hello TBB #2.\n", comm_rank, thread_id);
            }),
        ampi::mpi_abort_on_error{comm});
  }

  AMPI_CALL_MPI(MPI_Wait(&send_request, MPI_STATUS_IGNORE));
  std::printf("%d: Send done.\n", comm_rank);
}