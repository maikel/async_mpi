#pragma once

#include <unifex/get_allocator.hpp>
#include <unifex/receiver_concepts.hpp>
#include <unifex/sender_concepts.hpp>
#include <unifex/scheduler_concepts.hpp>

#include <unifex/detail/atomic_intrusive_queue.hpp>

namespace ampi {
// Define the function object in its own namespace to prevent name clashes of
// helper classes.
namespace bulk_finally_ns {
template <typename Sender, typename Receiver>
using operation_t = decltype(unifex::connect(std::declval<Sender&&>(), std::declval<Receiver&&>()));

template <typename Allocator, typename T>
using rebind_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;

template <typename T, typename Allocator>
auto rebind_alloc(const Allocator& alloc) {
  return rebind_alloc_t<Allocator, T>(alloc);
}

template <typename Receiver>
using get_scheduler_t = decltype(unifex::get_scheduler(std::declval<Receiver&>()));

// Provide space for an operation state for each incoming sender of the
// source many sender.
template <typename SourceManySender, typename SenderFactory, typename ManyReceiver>
struct operation_state {
  struct operation_base {
    operation_base() = default;
    virtual ~operation_base() = default;
    operation_base* next{};
  };

  // This class receives next values from a ManySender source.
  // For each value we create new single senders using the SenderFactory.
  // Each result of the newly created senders will be forwarded to the
  // ManyReceiver.
  struct many_receiver_to_single_sender;

  // This class is a receiver for a single sender and forwards its received value to the
  // ManyReceiver using its set_next function.
  //
  // TODO: If set_error or set_done is called by a single sender this class calls std::terminate.
  struct single_sender_to_many_receiver;

  [[no_unique_address]] SenderFactory factory;
  [[no_unique_address]] ManyReceiver many_receiver;

  // We store the operation state of the parent in an optional which does not require a heap
  // allocation.
  unifex::connect_result_t<SourceManySender, many_receiver_to_single_sender> parent_operation;

  operation_state(SourceManySender&& source, SenderFactory&& f, ManyReceiver&& r);

  // This counts all active operations and will be increased by every created operation state, the
  // parent and all its continuations. It will also be decreased upon destruction of an operation
  // state. The operation that decreases this back to 0 is responsible for deallocating all
  // allocated operation states.
  std::atomic<std::size_t> n_operation_states{0};

  // Storage for each operation state of a continuation.
  // This will be dequeued and deallocated by the last active operation.
  unifex::atomic_intrusive_queue<operation_base, &operation_base::next>
      operation_states_of_children{};

  /// \brief Returns the allocator that is associated with the ManyReceiver of this operation.
  ///
  /// This allocator is used to allocate storage space for the operation state of the continuations.
  unifex::get_allocator_t<ManyReceiver> get_allocator() const noexcept {
    return unifex::get_allocator(many_receiver);
  }

  /// \brief Returns the scheduler that is associated with the ManyReceiver of this operation.
  ///
  /// This scheduler is used to queue up the continuations.
  get_scheduler_t<ManyReceiver> get_scheduler() const noexcept {
    return unifex::get_scheduler(many_receiver);
  }

  // Try to allocate, construct and start an operation state for the specified
  // continuation. This function also stores the operation state into the
  // op state queue of the main operation state.
  template <typename C>
  void enqueue_continuation(C&& continuation) noexcept;

  // Decrease operation state count. If the count hits zero after our operation we deallocate all
  // operation states using the allocator that is stored in the original many receiver.
  void decrease_operation_count_and_deallocate_all_children_if_last() noexcept {
    std::size_t count = n_operation_states.fetch_sub(1);
    if (count == 1) {
      using ChildBaseAllocator =
          rebind_alloc_t<unifex::get_allocator_t<ManyReceiver>, operation_base>;
      ChildBaseAllocator allocator = rebind_alloc<operation_base>(get_allocator());
      auto queue = operation_states_of_children.dequeue_all();
      while (!queue.empty()) {
        auto pointer = queue.pop_front();
        std::allocator_traits<ChildBaseAllocator>::destroy(allocator, pointer);
        std::allocator_traits<ChildBaseAllocator>::deallocate(allocator, pointer, 1);
      }
      unifex::set_value(std::move(many_receiver));
    }
  }

  void start() & noexcept {
    n_operation_states = 1;
    parent_operation.start();
  }
};

template <typename SourceManySender, typename SenderFactory, typename ManyReceiver>
struct operation_state<SourceManySender, SenderFactory, ManyReceiver>::
    single_sender_to_many_receiver {
  operation_state<SourceManySender, SenderFactory, ManyReceiver>* op;

  template <typename... Args>
  void set_value(Args&&... args) && noexcept {
    unifex::set_next(op->many_receiver, std::forward<Args>(args)...);
    op->decrease_operation_count_and_deallocate_all_children_if_last();
  }

  template <typename Error>
  [[noreturn]] void set_error(Error&&) && noexcept {
    std::terminate();
  }

  [[noreturn]] void set_done() && noexcept { std::terminate(); }
};

// This class receives next values from a ManySender source.
// For each value we create new single senders using the SenderFactory.
// Each result of the newly created senders will be forwarded to the
// ManyReceiver.
template <typename SourceManySender, typename SenderFactory, typename ManyReceiver>
struct operation_state<SourceManySender, SenderFactory, ManyReceiver>::
    many_receiver_to_single_sender {
  operation_state<SourceManySender, SenderFactory, ManyReceiver>* op;

  // For each value that we recieve through set_next we spawn one or more
  // operation states.
  template <typename... Args>
    requires(
        std::is_invocable_v<SenderFactory, Args...> &&
        (std::is_void_v<std::invoke_result_t<SenderFactory, Args...>> ||
         unifex::sender_to<
             std::invoke_result_t<SenderFactory, Args...>,
             single_sender_to_many_receiver>))
  void set_next(Args&&... args) const noexcept {
    if constexpr (!std::is_void_v<std::invoke_result_t<SenderFactory, Args...>>) {
      assert(op);
      std::apply(
          [&](auto&&... senders) { (op->enqueue_continuation(std::move(senders)), ...); },
          std::tuple{std::invoke(op->factory, std::forward<Args>(args)...)});
    }
  }

  void set_value() && noexcept {
    op->decrease_operation_count_and_deallocate_all_children_if_last();
  }

  [[noreturn]] void set_done() && noexcept {
    std::terminate();
    // op->decrease_operation_count_and_deallocate_all_children_if_last();
  }

  template <typename Error>
  [[noreturn]] void set_error(Error&&) && noexcept {
    std::terminate();
    // op->decrease_operation_count_and_deallocate_all_children_if_last();
  }
};

template <typename SourceManySender, typename SenderFactory, typename ManyReceiver>
template <typename C>
void operation_state<SourceManySender, SenderFactory, ManyReceiver>::enqueue_continuation(
    C&& continuation) noexcept {
  using Continuation = std::remove_cvref_t<C>;
  // Type erasure for the newly enqueued operation state.
  struct operation : operation_base {
    operation(Continuation&& s, single_sender_to_many_receiver&& r) noexcept
      : operation_base()
      , op{unifex::connect(std::move(s), std::move(r))} {}
    operation_t<Continuation, single_sender_to_many_receiver> op;
    void start() & noexcept { op.start(); }
  };
  using ChildAllocator = rebind_alloc_t<unifex::get_allocator_t<ManyReceiver>, operation>;
  ChildAllocator child_allocator = rebind_alloc<operation>(get_allocator());
  auto new_child = std::allocator_traits<ChildAllocator>::allocate(child_allocator, 1);
  n_operation_states.fetch_add(1);
  std::allocator_traits<ChildAllocator>::construct(
      child_allocator, new_child, std::move(continuation), single_sender_to_many_receiver{this});
  [[maybe_unused]] auto ignore_inactive = operation_states_of_children.enqueue(new_child);
  new_child->start();
}

template <typename SourceManySender, typename SenderFactory, typename ManyReceiver>
operation_state<SourceManySender, SenderFactory, ManyReceiver>::operation_state(
    SourceManySender&& source, SenderFactory&& f, ManyReceiver&& r)
  : factory(std::move(f))
  , many_receiver(std::move(r))
  , parent_operation(
        unifex::connect((SourceManySender &&) source, many_receiver_to_single_sender{this})) {
}

// A convenient typedef to strip cv qualifiers
template <typename Source, typename Completion, typename ManyReceiver>
using operation = operation_state<
    std::remove_cvref_t<Source>,
    std::remove_cvref_t<Completion>,
    std::remove_cvref_t<ManyReceiver>>;

template <typename SourceManySender, typename SenderFactory>
struct many_sender {
  template <template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant, template <typename...> class Tuple>
  using next_types = Variant<Tuple<>>;

  template <template <typename...> class Variant>
  using error_types = Variant<>;

  static constexpr bool sends_done = false;

  [[no_unique_address]] SourceManySender source;
  [[no_unique_address]] SenderFactory factory;

  template <typename ManyReceiver>
    requires unifex::receiver_of<ManyReceiver>
  auto connect(ManyReceiver&& receiver) && noexcept {
    return operation<SourceManySender, SenderFactory, ManyReceiver>{
        std::move(source), std::move(factory), std::move(receiver)};
  }
};

constexpr inline struct fn {
  template <typename ManySender, typename SenderFactory>
  many_sender<std::remove_cvref_t<ManySender>, std::remove_cvref_t<SenderFactory>>
  operator()(ManySender&& sender, SenderFactory&& factory) const noexcept {
    static_assert(
        unifex::detail::_has_sender_traits<
            many_sender<std::remove_cvref_t<ManySender>, std::remove_cvref_t<SenderFactory>>>);
    return {std::forward<ManySender>(sender), std::forward<SenderFactory>(factory)};
  }
} bulk_finally;

}  // namespace bulk_finally_ns

using bulk_finally_ns::bulk_finally;
}  // namespace ampi