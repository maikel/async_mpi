#pragma once

#include <unifex/scheduler_concepts.hpp>
#include <unifex/sender_concepts.hpp>

#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>

namespace ampi {
namespace tbb_task_scheduler_ns {
template <typename R>
struct operation;
template <typename R>
using op = operation<std::remove_cvref_t<R>>;

struct sender;
struct tbb_task_scheduler {
  tbb::task_arena* arena_;

  tbb_task_scheduler(tbb::task_arena& arena) : arena_{&arena} {}

  sender schedule() const noexcept;

  friend auto operator<=>(const tbb_task_scheduler&, const tbb_task_scheduler&) = default;
};

template <typename Receiver>
struct operation {
  [[no_unique_address]] Receiver receiver_;
  tbb::task_arena* arena_;

  static_assert(unifex::receiver_of<Receiver>);

  void start() & noexcept {
    try {
      arena_->enqueue([&] { unifex::set_value(std::move(receiver_)); });
    } catch (...) {
      unifex::set_error(std::move(receiver_), std::current_exception());
    }
  }
};

struct sender {
  tbb::task_arena* arena_;

  template <template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant, template <typename...> class Tuple>
  using next_types = Variant<Tuple<>>;

  template <template <typename...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = true;

  template <typename R>
  auto connect(R&& receiver) const& noexcept {
    return operation<std::remove_cvref_t<R>>{std::move(receiver), arena_};
  }
};

inline sender tbb_task_scheduler::schedule() const noexcept {
  return sender{arena_};
}

static_assert(unifex::typed_sender<sender>);
static_assert(unifex::scheduler<tbb_task_scheduler>);

}  // namespace tbb_task_scheduler_ns

using tbb_task_scheduler_ns::tbb_task_scheduler;

}  // namespace ampi