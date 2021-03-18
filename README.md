![Build Status](https://github.com/maikel/async_mpi/actions/workflows/cmake.yml/badge.svg)

# Asynchronous Communication and Computation

This library attempts to wrap communication procudures of MPI with concepts found in [libunifex](https://github.com/facebookexperimental/libunifex).
Consider a blocking call to `MPI_WaitAll` as in:

```cpp
// Block the current thread until all pending requests are done
template <typename F>
void WaitAll_and_do_something(std::vector<MPI_Request>& pending_requests, MPI_Comm comm, int tag) {
  int n_reqs = static_cast<int>(pending_requests.size());
  MPI_WaitAll(comm, n_reqs, pending_requests.data(), tag, MPI_IGNORE_STATUSES);
  do_something();
}
```

This library provides thin wrappers around those MPI calls and enables the usage of the Sender/Receiver model that is developed in [libunifex](https://github.com/facebookexperimental/libunifex).

The above example could read instead
```cpp
// Returns a ManySender type that will lazily start on an user-defined executor thread.
auto async_do_something(std::vector<MPI_Request> pending_requests, MPI_Comm, comm int tag) {
  return ampi::for_each(std::move(pending_requests), comm, tag) 
         | unifex::bulk_transform([](int index) { do_something(index); });
}
```

It's intent is to try out a structural parallel programming model in classical HPC applications such as a finite volume flow solver on structured grids.

The examples will include a test that uses this programming model in conjunction with the [AMReX](https://github.com/AMReX-Codes/amrex) framework which is used to distribute a multi dimensional grid to multiple compute nodes of a cluster, which uses MPI under the hood. 

AMReX's usual strategy to CPU parallelization involves using parallel OpenMP blocks that are mostly separate from the MPI communication procedures.

The functions in this library enable other parallel programming models beside the OpenMP model, which are based on tasks.
