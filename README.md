![Build Status](https://github.com/maikel/async_mpi/actions/workflows/cmake.yml/badge.svg)

# Asynchronous Communication and Computation

This library wraps the blocking communication procudure `amrex::FillBoundary` in an internal `MPI_WaitAny` loop that allows asynchronous dispatch with concepts found in [libunifex](https://github.com/facebookexperimental/libunifex).
Consider a blocking call to `FillBoundary` as in:

```cpp
// Block the current thread until all pending requests are done
void EulerAmrCore::Advance(double dt) {
  // Blocking call to FillBoundary.
  // This waits for all MPI_Request to be done.
  // states is a MultiFab member variable
  states.FillBoundary();
  // Parallel for loop over all FABs in the MultiFab
#ifdef AMREX_USE_OPENMP
#pragma omp for
#endif
  for (MFIter mfi(states); mfi.isValid(); ++mfi) {
    const Box box = mfi.growntilebox();
    auto advance_dir = [&](Direction dir) {
      const int dir_v = int(dir);
      auto csarray = states.const_array(mfi);
      auto farray = fluxes[dir_v].array(mfi);
      Box faces = shrink(convert(box, dim_vec(dir)), dir_v, 1);
      ComputeNumericFluxes(faces, farray, csarray, dir);
      auto cfarray = fluxes[dir_v].const_array(K);
      auto sarray = states.array(K);
      UpdateConservatively(inner_box, sarray, farray, dt_over_dx[dir_v], dir);
    };
    // first order accurate operator splitting
    advance_dir(Direction::x);
    advance_dir(Direction::y);
    advance_dir(Direction::z);
  }
  // implicit join of all OpenMP threads here
}
```

This library provides thin wrappers around this FillBoundary call and enables the usage of the Sender/Receiver model that is developed in [libunifex](https://github.com/facebookexperimental/libunifex).

The above example could read instead
```cpp
void EulerAmrCore::AsyncAdvance(double dt) {
  auto advance = unifex::bulk_join(unifex::bulk_transform(
    // tbb_scheduler to schedule work items and comm_scheduler to schedule MPI_WaitAny/All threads
    // The feedback funcction (Box, int)  will be called for every box thas it ready,
    // i.e. all its ghost cells are filled.
    FillBoundary(tbb_scheduler, comm_scheduler, states), 
    [this, dt_over_dx](const Box& box, int K) {
      auto advance_dir = [&](Direction dir) {
        auto csarray = states.const_array(K);
        auto farray = fluxes[dir_v].array(K);
        Box faces = shrink(convert(box, unit(dir)), 1, unit(dir));
        ComputeNumericFluxes(faces, farray, csarray, dir);
        auto cfarray = fluxes[dir_v].const_array(K);
        auto sarray = states.array(K);
        UpdateConservatively(inner_box, sarray, farray, dt_over_dx[dir_v], dir);
      };
      // first order accurate operator splitting
      advance_dir(Direction::x);
      advance_dir(Direction::y);
      advance_dir(Direction::z);
    }, unifex::par_unseq));
  // Explicitly wait here until the above is done for all boxes
  unifex::sync_wait(std::move(advance));
}

```

The intent is to try out a structural parallel programming model in classical HPC applications such as in a finite volume flow solver on structured grids.
