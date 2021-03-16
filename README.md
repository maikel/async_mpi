# Asynchronous Communication and Computation

This library attempts to wrap communication procudures of MPI with concepts found in [libunifex](https://github.com/facebookexperimental/libunifex).
Consider a blocking call to `MPI_WaitAll` as in:

```cpp
// Block the current thread until all pending requests are done
void WaitAll(std::vector<MPI_Request>& pending_requests, int tag) {
  int n_reqs = static_cast<int>(pending_requests.size());
  MPI_WaitAll(n_reqs, pending_requests.data(), tag, MPI_IGNORE_STATUSES);
}
```

This library provides thin wrappers around those MPI calls and enables the usage of the Sender/Receiver model that is developed in [libunifex](https://github.com/facebookexperimental/libunifex).

The above example could read instead
```cpp
// Returns a ManySender wait-all type that will lazily start an a user-defined specified executor thread.
// This will send a request index for each request that finished execution
auto WaitAll(std::vector<MPI_Request> pending_request, int tag) {
  return ampi::wait_all(std::move(pending_requset), tag);
}
```

It's intent is to try out a structural parallel programming model in classical HPC applications such as finite volume flow solver on structured grids.

The examples include a test that uses this programming model in conjunction with the [AMReX](https://github.com/AMReX-Codes/amrex) framework which is used to distribute an adaptively mesh refined grid. 
AMReX's usual strategy to CPU parallelization involves using OpenMP blocks that are mostly separate from the MPI communication procedures that exchange dependend data across compute nodes.

The functions in this library enable another parallel programming models beside the OpenMP model.
