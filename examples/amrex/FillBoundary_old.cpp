#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include <unifex/scope_guard.hpp>

void my_main() {
  using namespace amrex;
  Box domain{IntVect(0), IntVect(127)};
  BoxArray ba{domain};
  ba.maxSize(64);

  DistributionMapping dm(ba);

  // Make one component with one ghost cell width
  IntVect ones = IntVect::TheUnitVector();
  IntVect ngrow = ones;
  MultiFab mf(ba, dm, 1, ngrow);
  mf.setVal(0.0, ngrow);
  // Fill only inner cells, not ghost cells
  mf.setVal(1.0, IntVect(0));

  RealBox real_box{{AMREX_D_DECL(-0.5, -0.5, -0.5)}, {AMREX_D_DECL(0.5, 0.5, 0.5)}};

  Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
  Geometry geom{domain, real_box, CoordSys::cartesian, is_periodic};

  mf.FillBoundary(0, 1, geom.periodicity());
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    Box box = mfi.validbox();
    Array4<const Real> array = mf.const_array(mfi);
    LoopConcurrentOnCpu(
        box, [=](int i, int j, int k) { AMREX_ALWAYS_ASSERT(array(i, j, k) == 1.0); });
  }
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
  unifex::scope_guard mpi_finalize = []() noexcept {
    MPI_Finalize();
  };

  // Initialize AMReX
  amrex::Initialize(
      MPI_COMM_WORLD, std::cout, std::cerr, [](const char* msg) { throw std::runtime_error(msg); });
  unifex::scope_guard amrex_finalize = []() noexcept {
    amrex::Finalize();
  };

  // Call our Application
  my_main();
}
