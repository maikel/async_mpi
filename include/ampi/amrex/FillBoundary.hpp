#pragma once

#include <ampi/bulk_finally.hpp>
#include <ampi/bulk_on.hpp>
#include <ampi/for_each.hpp>
#include <ampi/mpi_abort_on_error.hpp>

#include <AMReX_FabArrayBase.H>
#include <AMReX_NonLocalBC.H>

#include <unifex/bulk_schedule.hpp>
#include <unifex/bulk_transform.hpp>
#include <unifex/bulk_join.hpp>
#include <unifex/transform.hpp>
#include <unifex/when_all.hpp>

#include <unifex/finally.hpp>

namespace amrex {

template <typename FAB>
[[nodiscard]] NonLocalBC::CommHandler FillBoundary_nowait(
    FabArray<FAB>& fa, const FabArrayBase::FB& cmd, const NonLocalBC::PackComponents& components) {
  return NonLocalBC::ParallelCopy_nowait(NonLocalBC::no_local_copy, fa, fa, cmd, components);
}

namespace fill_boundary_finish_ {
template <typename FAB, typename Receiver>
struct CopyOpBase {
  FabArray<FAB>* fa_;
  NonLocalBC::CommHandler handler_;
  const FabArrayBase::FB* meta_data_;
  NonLocalBC::PackComponents components_;
  Receiver receiver_;
  std::vector<std::atomic<int>> job_count_per_box_{};
  std::atomic<int> total_job_count_{};

  CopyOpBase(
      FabArray<FAB>* fa,
      NonLocalBC::CommHandler handler,
      const FabArrayBase::FB* meta_data,
      const NonLocalBC::PackComponents& comps,
      Receiver&& receiver)
    : fa_(fa)
    , handler_(std::move(handler))
    , meta_data_{meta_data}
    , components_{comps}
    , receiver_{std::move(receiver)}
    , job_count_per_box_(fa_->local_size()) {
    // count local jobs per box
    int total_job_count = 0;
    for (const FabArrayBase::CopyComTag& tag : *meta_data_->m_LocTags) {
      const int i = fa->localindex(tag.dstIndex);
      job_count_per_box_[i].fetch_add(1, std::memory_order::relaxed);
      total_job_count += 1;
    }
    // count jobs per box that depend on MPI receives
    for (auto [from, tags] : *meta_data_->m_RcvTags) {
      for (const FabArrayBase::CopyComTag& tag : tags) {
        const int i = fa->localindex(tag.dstIndex);
        job_count_per_box_[i].fetch_add(1, std::memory_order::relaxed);
        total_job_count += 1;
      }
    }
    if (meta_data_->m_SndTags->size() > 0) {
      total_job_count += 1;
    }
    total_job_count_.store(total_job_count, std::memory_order::relaxed);
  }
};

template <typename FAB, typename Receiver>
void NotifyReadyBox(CopyOpBase<FAB, Receiver>& op, int i) {
  const int local_index = op.fa_->localindex(i);
  const int old_box_count =
      op.job_count_per_box_[local_index].fetch_sub(1, std::memory_order_relaxed);
  // we are the last job for this box
  if (old_box_count == 1) {
    unifex::set_next(op.receiver_, i, (*op.fa_)[i].box());
  }
}

template <typename FAB, typename Receiver>
auto LocalCopies(CopyOpBase<FAB, Receiver>& op) {
  auto scheduler = unifex::get_scheduler(op.receiver_);
  return unifex::bulk_join(unifex::bulk_transform(
      unifex::bulk_schedule(scheduler, op.meta_data_->m_LocTags->size()),
      [&op](std::size_t i) {
        if (op.meta_data_->m_threadsafe_loc) {
          auto& local_tags = *op.meta_data_->m_LocTags;
          const FabArrayBase::CopyComTag& tag = local_tags[i];
          FabArray<FAB>& fa = *op.fa_;
          const FAB& src = fa[tag.srcIndex];
          FAB& dest = fa[tag.dstIndex];
          dest.template copy<RunOn::Host>(
              src,
              tag.sbox,
              op.components_.src_component,
              tag.dbox,
              op.components_.dest_component,
              op.components_.n_components);
          NotifyReadyBox(op, tag.dstIndex); 
        } else {
          MPI_Abort(amrex::ParallelDescriptor::Communicator(), 1);
        }
      },
      unifex::par_unseq));
}

template <typename FAB, typename Receiver>
struct UnpackReceivesFactory {
  CopyOpBase<FAB, Receiver>* op_;

  auto operator()(int index) const noexcept {
    auto scheduler = unifex::get_scheduler(op_->receiver_);
    const std::size_t n_cctc = op_->handler_.recv.cctc[index]->size();
    auto do_all_copies_for_this_recv = unifex::bulk_transform(
        unifex::bulk_schedule(scheduler, n_cctc),
        [index, this](std::size_t i) noexcept {
          const auto& tags = *op_->handler_.recv.cctc[index];
          char* dptr = op_->handler_.recv.data[index];
          const int ncomp = op_->components_.n_components;
          // compute offset of the i-th tag
          for (std::size_t prev = 0; prev < i; ++prev) {
            using T = typename FAB::value_type;
            dptr += tags[prev].sbox.numPts() * ncomp * sizeof(T);
          }
          const FabArrayBase::CopyComTag& tag = (*op_->handler_.recv.cctc[index])[i];
          FabArray<FAB>& fa = *op_->fa_;
          using T = typename FAB::value_type;
          Array4 darray = fa.array(tag.dstIndex);
          Array4 sarray = amrex::makeArray4((T const*)(dptr), tag.dbox, ncomp);
          const int dcomp = op_->components_.dest_component;
          amrex::LoopConcurrentOnCpu(tag.dbox, ncomp, [=](int i, int j, int k, int n) noexcept {
            darray(i, j, k, dcomp + n) = sarray(i, j, k, n);
          });
          NotifyReadyBox(*op_, tag.dstIndex);
        },
        unifex::par_unseq);
    return unifex::bulk_join(do_all_copies_for_this_recv);
  }
};

template <typename FAB, typename Receiver>
auto UnpackReceives(CopyOpBase<FAB, Receiver>& op) {
  MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
  auto comm_scheduler = ampi::get_comm_scheduler(op.receiver_);
  return unifex::bulk_join(ampi::bulk_finally(
      ampi::bulk_on(
          comm_scheduler,
          ampi::for_each(std::move(op.handler_.recv.request), comm, op.handler_.mpi_tag)),
      UnpackReceivesFactory<FAB, Receiver>{&op}));
}

template <typename FAB, typename Receiver>
auto CopyAndWaitAll(CopyOpBase<FAB, Receiver>& op) {
  auto local_copies = LocalCopies(op);
  auto unpack_receives = UnpackReceives(op);
  auto comm_scheduler = ampi::get_comm_scheduler(op.receiver_);
  auto wait_all_sends = unifex::transform(unifex::schedule(comm_scheduler), [&op]() noexcept {
    MPI_Waitall(
        op.handler_.send.request.size(), op.handler_.send.request.data(), MPI_STATUSES_IGNORE);
  });
  return unifex::connect(
      unifex::transform(
          unifex::when_all(
              std::move(local_copies), std::move(unpack_receives), std::move(wait_all_sends)),
          [&op](auto&&...) { unifex::set_value(std::move(op.receiver_)); }),
      ampi::mpi_abort_on_error{ParallelDescriptor::Communicator()});
}

template <typename FAB, typename Receiver>
struct CopyOperations : CopyOpBase<FAB, Receiver> {
  using CopyAndWaitAllOp = decltype(CopyAndWaitAll(std::declval<CopyOpBase<FAB, Receiver>&>()));
  CopyAndWaitAllOp op_;

  CopyOperations(
      FabArray<FAB>* fa,
      NonLocalBC::CommHandler handler,
      const FabArrayBase::FB* meta_data,
      NonLocalBC::PackComponents components,
      Receiver&& r) noexcept
    : CopyOpBase<FAB, Receiver>(fa, std::move(handler), meta_data, components, std::move(r))
    , op_(CopyAndWaitAll(*this)) {}

  CopyOperations(CopyOperations&& other) noexcept;

  void start() & noexcept {
    if (this->total_job_count_.load(std::memory_order::relaxed) == 0) {
      unifex::set_value(std::move(this->receiver_));
    }
    op_.start();
  }
};

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

  Sender(
      FabArray<FAB>& fa,
      NonLocalBC::CommHandler handler,
      const FabArrayBase::FB& meta_data,
      NonLocalBC::PackComponents components) noexcept
    : fa_{&fa}
    , handler_(std::move(handler))
    , meta_data_{&meta_data}
    , components_{components} {}

  template <typename Receiver>
  auto connect(Receiver&& r) && noexcept {
    return CopyOperations<FAB, std::remove_cvref_t<Receiver>>(
        fa_, std::move(handler_), meta_data_, components_, std::move(r));
  }

  template <typename Receiver>
  auto connect(Receiver&& r) & {
    return CopyOperations<FAB, std::remove_cvref_t<Receiver>>(
        fa_, handler_, meta_data_, components_, std::move(r));
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

template <typename FAB>
auto FillBoundary_async(FabArray<FAB>& fa, const Periodicity& period)
{
  NonLocalBC::PackComponents components{.dest_component = 0, .src_component = 0, .n_components = fa.nComp()};
  // Get comm meta data from amrex
  const FabArrayBase::FB& cmd = fa.getFB(fa.nGrowVect(), period, false, false);
  auto handler = FillBoundary_nowait(fa, cmd, components);
  return FillBoundary_finish(fa, std::move(handler), cmd, components);
}

}  // namespace amrex
