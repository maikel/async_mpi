#pragma once

#include <AMReX_FabArrayBase.H>
#include <AMReX_NonLocalBC.H>

#include <unifex/bulk_schedule.hpp>

namespace amrex {

template <typename FAB>
[[nodiscard]] NonLocalBC::CommHandler FillBoundary_nowait(
    FabArray<FAB>& fa, const FabArrayBase::FB& cmd, const NonLocalBC::PackComponents& components) {
  return NonLocalBC::ParallelCopy_nowait(NonLocalBC::no_local_copy, fa, fa, cmd, components);
}

namespace fill_boundary_finish_ {
struct ReceiveLocalCopy;
struct SendLocalCopy;

template <typename Receiver>
struct CopyOperations {
  NonLocalBC::CommHandler handler_;
  const FabArrayBase* fa_;
  const FabArrayBase::FB* meta_data_;
  std::vector<std::atomic<int>> job_count_per_box_;
  std::atomic<int> total_job_count_{};
  Receiver receiver_;

  CopyOperations(
      NonLocalBC::CommHandler handler, const FabArrayBase* fa, const FabArrayBase::FB* meta_data)
    : handler_(std::move(handler))
    , fa_(fa)
    , meta_data_(meta_data)
    , job_count_per_box_(fa_->local_size()) {
    // count local jobs per box
    for (const FabArrayBase::CopyComTag& tag : *meta_data_->m_LocTags) {
      const int i = fa->localindex(tag.dstIndex);
      job_count_per_box_[i] += 1;
      total_job_count_ += 1;
    }
    // count jobs per box that depend on MPI receives
    for (auto [from, tags] : *meta_data_->m_RcvTags) {
      for (const FabArrayBase::CopyComTag& tag : tags) {
        const int i = fa->localindex(tag.dstIndex);
        job_count_per_box_[i] += 1;
        total_job_count_ += 1;
      }
    }
    auto scheduler = unifex::get_scheduler(receiver_);
    auto local_copies_sender = unifex::bulk_schedule(scheduler, meta_data_->m_LocTags->size());
    MPI_Comm comm = ParallelDescriptor::Communicator();
    ampi::for_each(std::move(handler_.recv.request), comm, handler_.mpi_tag);
  }
};

inline constexpr struct fn {
  template <typename FAB>
  sender operator()(
      FabArray<FAB>& fa,
      NonLocalBC::CommHandler handler,
      const FabArrayBase::FB& cmd,
      const NonLocalBC::PackComponents& components) const noexcept {}
} FillBoundary_finish;

}  // namespace fill_boundary_finish_

using fill_boundary_finish_::FillBoundary_finish;

}  // namespace amrex
