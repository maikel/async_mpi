#include <ampi/amrex/FillBoundary.hpp>

#include <ampi/bulk_sequence.hpp>
#include <ampi/for_each.hpp>
#include <ampi/bulk_on.hpp>
#include <ampi/tbb_task_scheduler.hpp>

#include <unifex/single_thread_context.hpp>
#include <unifex/bulk_join.hpp>
#include <unifex/bulk_transform.hpp>
#include <unifex/sync_wait.hpp>

#include <ranges>

#include <mpi.h>

#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace unifex;
using namespace ampi;

inline constexpr auto then = bulk_transform;
inline constexpr auto when_all = bulk_join;

void my_main(MPI_Comm comm) {
  using namespace amrex;
  Box domain{IntVect(0), IntVect(63)};
  BoxArray ba{domain};
  ba.maxSize(32);

  DistributionMapping dm(ba);

  // Make one component with one ghost cell width
  IntVect ones = IntVect::TheUnitVector();
  IntVect ngrow = ones;
  MultiFab mf(ba, dm, 1, ngrow);
  mf.setVal(0.0, ngrow);
  // Fill only inner cells, not ghost cells
  mf.setVal(1.0, IntVect(0));

  // for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
  //   Array4<const Real> array = mf.const_array(mfi);
  //   LoopConcurrentOnCpu(mfi.growntilebox(), [=](int i, int j, int k) { AMREX_ASSERT(array(i, j, k) == 1.0); });
  // }

  // Fill all domain boundaries periodically
    RealBox real_box{{AMREX_D_DECL(-0.5, -0.5, -0.5)}, {AMREX_D_DECL(0.5, 0.5, 0.5)}};

    Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    Geometry geom{domain, real_box, CoordSys::cartesian, is_periodic};

  // Fill the first component
  NonLocalBC::PackComponents components{.dest_component = 0, .src_component = 0, .n_components = 1};
  // Get comm meta data from amrex
  const FabArrayBase::FB& cmd = mf.getFB(ngrow, geom.periodicity(), false, false);
  auto handler = FillBoundary_nowait(mf, cmd, components);
  
  tbb::task_arena arena{};
  tbb_task_scheduler oneTBB(arena);


  const std::size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
  Print() << thread_id << '\n';

  // Do some work asynchronously while receiving data
  auto async_test_for_unity = then(
      FillBoundary_finish(mf, std::move(handler), cmd, components),
      [&mf](int index, const Box& box) {
        const std::size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        Print() << thread_id << '\n';
        Array4<const Real> array = mf.const_array(index);
        LoopConcurrentOnCpu(box, [=](int i, int j, int k) {
          AMREX_ASSERT(array(i, j, k) == 1.0); 
        });
      });
  // Wait for everything being done.
  sync_wait(with_query_value(when_all(std::move(async_test_for_unity)), get_scheduler, oneTBB));
}

int main(int argc, char** argv) {
  // Initialize MPI
  const int required = MPI_THREAD_MULTIPLE;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required) {
    std::printf("MPI could not provide enough parallelism.\n");
    return 1;
  }
  scope_guard mpi_finalize = []() noexcept {
    MPI_Finalize();
  };
  MPI_Comm comm = MPI_COMM_WORLD;

  // Initialize AMReX
  amrex::Initialize(comm, std::cout, std::cerr, [](const char* msg) { throw std::runtime_error(msg); });
  scope_guard amrex_finalize = []() noexcept {
    amrex::Finalize();
  };

  // Call our Application
  my_main(comm);
}
