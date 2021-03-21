#pragma once

#include <unifex/scheduler_concepts.hpp>
#include <unifex/sender_concepts.hpp>
#include <unifex/bulk_schedule.hpp>

#include <tbb/task_arena.h>
#include <cassert>

namespace ampi {
namespace tbb_task_scheduler_ns {

struct sender;
struct many_sender;
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

  friend many_sender tag_invoke(unifex::tag_t<unifex::bulk_schedule>, const tbb_task_scheduler& s, std::size_t n);
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

template <typename Receiver> struct many_operation;
template <typename Receiver> void parallel_enqueue_task(many_operation<Receiver>& op, std::size_t lo, std::size_t hi);

template <typename Receiver>
struct many_operation {
  [[no_unique_address]] Receiver receiver_;
  tbb::task_arena* arena_;
  std::atomic<std::size_t> n_tasks_;

  many_operation(Receiver&& r, tbb::task_arena* arena, std::atomic<std::size_t> n)
    : receiver_{std::move(r)}, arena_{arena}, n_tasks_{n.load(std::memory_order::relaxed)} {}
  
  many_operation(const many_operation&) = delete;
  many_operation& operator=(const many_operation&) = delete;

  many_operation(many_operation&& other) noexcept
    : receiver_(std::move(other.receiver_)), arena_(other.arena_), n_tasks_(other.n_tasks_.load(std::memory_order::relaxed)) {}
  many_operation& operator=(many_operation&& other) = delete;

  ~many_operation() noexcept = default;


  void start() & noexcept {
    try {
      arena_->enqueue([this] { 
        const std::size_t n = n_tasks_.load(std::memory_order::relaxed);
        parallel_enqueue_task(*this, 0, n); 
      });
    } catch (...) {
      unifex::set_error(std::move(receiver_), std::current_exception());
    }
  }
};

template <typename Receiver>
void parallel_enqueue_task(many_operation<Receiver>& op, std::size_t lo, std::size_t hi) {
  if (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (lo < mid) {
      op.arena_->enqueue([&op, lo, mid] {  parallel_enqueue_task(op, lo, mid); });
    } 
    if (mid + 1 < hi) {
      op.arena_->enqueue([&op, mid, hi] {  parallel_enqueue_task(op, mid + 1, hi); });
    }
    unifex::set_next(op.receiver_, mid);
    const std::size_t previous_value = op.n_tasks_.fetch_sub(1, std::memory_order::acquire);
    if (previous_value == 1) {
      unifex::set_value(std::move(op.receiver_));
    }
  }
}

struct many_sender {
  template <template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant, template <typename...> class Tuple>
  using next_types = Variant<Tuple<std::size_t>>;

  template <template <typename...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = false;

  tbb::task_arena* arena_;
  std::size_t n_tasks_;

  template <typename Receiver>
  many_operation<std::remove_cvref_t<Receiver>> connect(Receiver&& receiver) && noexcept {
    return {std::move(receiver), arena_, n_tasks_};
  }
};

inline sender tbb_task_scheduler::schedule() const noexcept {
  return sender{arena_};
}

inline many_sender tag_invoke(unifex::tag_t<unifex::bulk_schedule>, const tbb_task_scheduler& s, std::size_t n)
{
  return many_sender{s.arena_, n};
}

static_assert(unifex::typed_sender<sender>);
static_assert(unifex::scheduler<tbb_task_scheduler>);

}  // namespace tbb_task_scheduler_ns

using tbb_task_scheduler_ns::tbb_task_scheduler;

}  // namespace ampi