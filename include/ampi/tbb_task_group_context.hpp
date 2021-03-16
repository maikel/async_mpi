#pragma once

#include <unifex/scheduler_concepts.hpp>
#include <unifex/sender_concepts.hpp>

#include <tbb/parallel_invoke.h>
#include <tbb/task_group.h>

namespace ampi
{
  namespace _tbb_task_group_scheduler
  {
    template <typename R>
    struct _op;
    template <typename R>
    using op = _op<std::remove_cvref_t<R>>;

    struct _sender;
    class tbb_task_group_context;

    class tbb_task_group_scheduler {
    public:
      tbb_task_group_scheduler(tbb_task_group_context& context)
        : context_{&context} {}
      _sender schedule() const noexcept;

      friend auto operator<=>(
          const tbb_task_group_scheduler&,
          const tbb_task_group_scheduler&) = default;

    private:
      template <typename R>
      friend class _op;
      tbb_task_group_context* context_;
    };

    class tbb_task_group_context {
    public:
      tbb_task_group_context() = default;
      ~tbb_task_group_context() noexcept { wait(); }

      tbb_task_group_scheduler get_scheduler() noexcept { return *this; }

      tbb::task_group_status wait() noexcept {
        if (status == tbb::task_group_status::not_complete) {
          status = group.wait();
        }
        return status;
      }

    private:
      template <typename R>
      friend class _op;
      tbb::task_group group{};
      tbb::task_group_status status = tbb::task_group_status::not_complete;
    };

    template <typename R>
    struct _op {
      [[no_unique_address]] R receiver_;
      tbb_task_group_scheduler scheduler_;

      void start() & noexcept {
        if (scheduler_.context_) {
          scheduler_.context_->group.run(
              [r = &receiver_]() { unifex::set_value(std::move(*r)); });
        } else {
          unifex::set_done(std::move(receiver_));
        }
      }
    };

    struct _sender {
      template <
          template <typename...>
          class Variant,
          template <typename...>
          class Tuple>
      using value_types = Variant<Tuple<>>;

      template <
          template <typename...>
          class Variant,
          template <typename...>
          class Tuple>
      using next_types = Variant<Tuple<>>;

      template <template <typename...> class Variant>
      using error_types = Variant<std::exception_ptr>;

      static constexpr bool sends_done = true;

      template <typename R>
      op<R> connect(R&& receiver) const& noexcept {
        return {std::move(receiver), scheduler_};
      }

      tbb_task_group_scheduler scheduler_;
    };

    inline _sender tbb_task_group_scheduler::schedule() const noexcept {
      return _sender{*this};
    }

    static_assert(unifex::scheduler<tbb_task_group_scheduler>);

  }  // namespace _tbb_task_group_scheduler

  using _tbb_task_group_scheduler::tbb_task_group_context;
  using _tbb_task_group_scheduler::tbb_task_group_scheduler;

}  // namespace ampi