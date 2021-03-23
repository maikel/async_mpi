#pragma once

#include <AMReX_FabArrayUtility.H>

#include <unifex/scheduler_concepts.hpp>
#include <unifex/transform.hpp>
#include <unifex/when_all.hpp>
#include <unifex/just.hpp>
#include <unifex/on.hpp>
#include <unifex/any_sender_of.hpp>
#include <unifex/sync_wait.hpp>

namespace ampi {
namespace reduce_ns {
template <typename Scheduler, typename F, typename ResultType, typename BinaryOp>
unifex::any_sender_of<ResultType>
reduce(const Scheduler& scheduler, F f, const ResultType& x0, BinaryOp binary_op, int lo, int hi) {
  assert(lo < hi);
  if (hi == lo + 1) {
    return unifex::just(f(lo));
  }
  const int mid = (lo + hi) / 2;
  auto lo_val = unifex::on(reduce(scheduler, f, x0, binary_op, lo, mid), scheduler);
  auto hi_val = unifex::on(reduce(scheduler, f, x0, binary_op, mid, hi), scheduler);
  return unifex::transform(
      unifex::when_all(std::move(lo_val), std::move(hi_val)),
      [binary_op = std::move(binary_op)](
         std::variant<std::tuple<ResultType>> x, std::variant<std::tuple<ResultType>> y) -> ResultType {
        return std::move(binary_op)(
            std::get<0>(*std::get_if<0>(&x)), std::get<0>(*std::get_if<0>(&y)));
      });
}


}  // namespace reduce_ns

template <typename Scheduler, typename FAB, typename F, typename ResultType, typename BinaryOp>
  requires(amrex::IsBaseFab<FAB>::value)
ResultType reduce(
    const Scheduler& scheduler,
    const amrex::FabArray<FAB>& fa,
    F&& f,
    const ResultType& x0,
    BinaryOp&& binary_op) {
  const int local_size = fa.local_size();
  auto local_index_to_value = [f = std::forward<F>(f), &fa](int local_K) {
    const int K = fa.IndexArray()[local_K];
    amrex::Box box = fa.box(K);
    return f(box, K);
  };
  assert(local_size > 0);
  std::optional<ResultType> maybe_value = unifex::sync_wait(
      reduce_ns::reduce(scheduler, std::move(local_index_to_value), x0, std::forward<BinaryOp>(binary_op), 0, local_size));
  AMREX_ALWAYS_ASSERT(maybe_value);
  ResultType value = *maybe_value;
  return value;
}

template <typename Scheduler, typename FAB, typename F>
typename FAB::value_type reduce_max(Scheduler&& scheduler, const amrex::FabArray<FAB>& fa, F&& f) {
  using T = typename FAB::value_type;
  return reduce(
      std::forward<Scheduler>(scheduler),
      fa,
      std::forward<F>(f),
      std::numeric_limits<T>::lowest(),
      [](T max, T val) { return max < val ? val : max; });
}

}