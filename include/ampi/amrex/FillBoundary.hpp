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
template <typename FAB, typename Receiver>
struct CopyOperations;

template <typename FAB, typename Receiver>
struct ReceiveLocalCopy {
  CopyOperations<FAB, Receiver>* op_;

  void set_next(int i) const noexcept;

  void set_value() const noexcept {}

  template <typename Error>
  [[noreturn]] void set_error(Error&&) const noexcept { std::terminate(); }

  [[noreturn]] void set_done() const noexcept { std::terminate(); }
};

template <typename FAB, typename Receiver>
struct CopyOperations {
  FabArray<FAB>* fa_;
  NonLocalBC::CommHandler handler_;
  const FabArrayBase::FB* meta_data_;
  NonLocalBC::PackComponents components_;
  Receiver receiver_;
  std::vector<std::atomic<int>> job_count_per_box_{};
  std::atomic<int> total_job_count_{};

  using SendLocalCopy = decltype(unifex::bulk_schedule(std::declval<unifex::get_scheduler_result_t<Receiver&>&>(), std::size_t{}));
  using LocalCopyOperation = unifex::connect_result_t<SendLocalCopy, ReceiveLocalCopy<FAB, Receiver>>; 

  std::optional<LocalCopyOperation> local_copy_op_{};

  CopyOperations(
      FabArray<FAB>* fa, NonLocalBC::CommHandler handler, const FabArrayBase::FB* meta_data, NonLocalBC::PackComponents components, Receiver&& r) noexcept
    : fa_(fa)
    , handler_(std::move(handler))
    , meta_data_(meta_data)
    , components_{components}
    , job_count_per_box_(fa_->local_size())
    , receiver_(std::move(r)) {
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
    // MPI_Comm comm = ParallelDescriptor::Communicator();
    // ampi::for_each(std::move(handler_.recv.request), comm, handler_.mpi_tag);

    if (meta_data_->m_LocTags->size() > 0) {
      auto scheduler = unifex::get_scheduler(receiver_);
      auto local_copies_sender = unifex::bulk_schedule(scheduler, meta_data_->m_LocTags->size());
      local_copy_op_.emplace(unifex::connect(std::move(local_copies_sender), ReceiveLocalCopy<FAB, Receiver>{this}));
    }

    if (meta_data_->m_LocTags->empty() && meta_data_->m_RcvTags->empty()) {
      unifex::set_value(std::move(receiver_));
    }
  }

  void start() & noexcept {
    local_copy_op_->start();
  }
};

template <typename FAB, typename Receiver>
void ReceiveLocalCopy<FAB, Receiver>::set_next(int i) const noexcept {
  auto& local_tags = *op_->meta_data_->m_LocTags;
  const FabArrayBase::CopyComTag& tag = local_tags[i];
  FabArray<FAB>& fa = *op_->fa_;
  const FAB& src = fa[tag.srcIndex];
  FAB& dest = fa[tag.dstIndex];
  dest.template copy<RunOn::Host>(src, tag.sbox, op_->components_.src_component, tag.dbox, op_->components_.dest_component, op_->components_.n_components);

  const int local_index = fa.localindex(tag.dstIndex);
  const int old_box_count = op_->job_count_per_box_[local_index].fetch_sub(1, std::memory_order_acquire);
  // we are the last job for this box
  if (old_box_count == 1) {
    unifex::set_next(op_->receiver_, tag.dstIndex, dest.box());
  }
  // we are the last job overall
  const int old_count = op_->total_job_count_.fetch_sub(1, std::memory_order_acquire);
  if (old_count == 1) {
    unifex::set_value(std::move(op_->receiver_));
  }
}

template <typename FAB>
struct Sender {
  template <template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant, template <typename...> class Tuple>
  using next_types = Variant<Tuple<int, Box>>;

  template <template <typename...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = false;

  FabArray<FAB>* fa_;
  NonLocalBC::CommHandler handler_;
  const FabArrayBase::FB* meta_data_;
  NonLocalBC::PackComponents components_;

  Sender(FabArray<FAB>& fa, NonLocalBC::CommHandler handler, const FabArrayBase::FB& meta_data, NonLocalBC::PackComponents components) noexcept
    : fa_{&fa}, handler_(std::move(handler)), meta_data_{&meta_data}, components_{components} {}

  template <typename Receiver>
  CopyOperations<FAB, std::remove_cvref_t<Receiver>> connect(Receiver&& r) && noexcept {
    return {fa_, std::move(handler_), meta_data_, components_, std::move(r)};
  }

  template <typename Receiver>
  CopyOperations<FAB, std::remove_cvref_t<Receiver>> connect(Receiver&& r) & {
    return {fa_, handler_, meta_data_, components_, std::move(r)};
  }
};

inline constexpr struct fn {
  template <typename FAB>
  Sender<FAB> operator()(
      FabArray<FAB>& fa,
      NonLocalBC::CommHandler handler,
      const FabArrayBase::FB& cmd,
      const NonLocalBC::PackComponents& components) const noexcept {
        return Sender<FAB>{fa, std::move(handler), cmd, components};
      }
} FillBoundary_finish;

}  // namespace fill_boundary_finish_

using fill_boundary_finish_::FillBoundary_finish;

}  // namespace amrex
