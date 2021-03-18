#pragma once

#include <unifex/scheduler_concepts.hpp>
#include <unifex/sender_concepts.hpp>

#include <tbb/task_arena.h>
#include <cassert>

namespace ampi {
namespace tbb_task_scheduler_ns {

struct sender;
class tbb_task_scheduler {
public:
  tbb_task_scheduler(tbb::task_arena& arena) : arena_{&arena} {}

  sender schedule() const noexcept;

  friend auto operator<=>(const tbb_task_scheduler&, const tbb_task_scheduler&) = default;

  tbb::task_arena& arena() noexcept {
    return *arena_;
  }

private:
  tbb::task_arena* arena_;
};

template <typename Receiver>
struct operation {
  [[no_unique_address]] Receiver receiver_;
  tbb::task_arena* arena_;

  static_assert(unifex::receiver_of<Receiver>);

  void start() & noexcept {
    try {
      assert(arena_);
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

  template <template <typename...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = false;

  template <typename R>
  auto connect(R&& receiver) const& noexcept {
    assert(arena_);
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