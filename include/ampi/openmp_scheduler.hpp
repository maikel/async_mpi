#pragma once

#include <unifex/scheduler_concepts.hpp>
#include <unifex/sender_concepts.hpp>
#include <unifex/bulk_schedule.hpp>

#include <cassert>

namespace ampi {
namespace openmp_scheduler_ns {

struct sender;
struct many_sender;
class openmp_scheduler {
public:
  sender schedule() const noexcept;

  friend auto operator<=>(const openmp_scheduler&, const openmp_scheduler&) = default;
  friend many_sender tag_invoke(unifex::tag_t<unifex::bulk_schedule>, const openmp_scheduler& s, std::size_t n);
};

template <typename Receiver>
struct operation {
  [[no_unique_address]] Receiver receiver_;

  static_assert(unifex::receiver_of<Receiver>);

  void start() & noexcept {
    #pragma omp task
    unifex::set_value(std::move(receiver_));
  }
};

struct sender {
  template <template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = false;

  template <typename R>
  auto connect(R&& receiver) const& noexcept {
    return operation<std::remove_cvref_t<R>>{std::move(receiver)};
  }
};

template <typename Receiver> struct many_operation;

template <typename Receiver>
struct many_operation {
  [[no_unique_address]] Receiver receiver_;
  std::size_t n_tasks_;

  void start() & noexcept {
    for (std::size_t i = 0; i < n_tasks_; ++i) {
      #pragma omp task
      unifex::set_next(receiver_, i);
    }
    #pragma omp taskwait
    unifex::set_value(std::move(receiver_));
  }
};

struct many_sender {
  template <template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant, template <typename...> class Tuple>
  using next_types = Variant<Tuple<std::size_t>>;

  template <template <typename...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = false;

  std::size_t n_tasks_;

  template <typename Receiver>
  many_operation<std::remove_cvref_t<Receiver>> connect(Receiver&& receiver) && noexcept {
    return {std::move(receiver), n_tasks_};
  }
};

inline sender openmp_scheduler::schedule() const noexcept {
  return sender{};
}

inline many_sender tag_invoke(unifex::tag_t<unifex::bulk_schedule>, const openmp_scheduler&, std::size_t n)
{
  return many_sender{n};
}

static_assert(unifex::typed_sender<sender>);
static_assert(unifex::scheduler<openmp_scheduler>);

}  // namespace openmp_scheduler_ns

using openmp_scheduler_ns::openmp_scheduler;

}  // namespace ampi