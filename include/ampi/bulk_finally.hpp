#pragma once

#include <unifex/get_allocator.hpp>
#include <unifex/get_scheduler.hpp>
#include <unifex/receiver_concepts.hpp>
#include <unifex/sender_concepts.hpp>

namespace ampi
{
  // Define the function object in its own namespace to prevent name clashes of
  // helper classes.
  namespace bulk_finally_ns
  {
    // Provide space for an operation state for each incoming sender of the
    // source many sender.
    template <
        typename SourceManySender,
        typename SenderFactory,
        typename ManyReceiver>
    struct operation_state {
      struct operation_state_child_base {
        virtual ~operation_state_child_base() = default;
        operation_state_child* next;
      };

      [[no_unique_address]] SourceManySender source;
      [[no_unique_address]] SenderFactory factory;
      [[no_unique_address]] ManyReceiver many_receiver;

      std::atomic<std::size_t> n_operation_states{0};

      std::optional<
          unifex::operation_t<SourceManySender, many_receiver_to_single_sender>>
          parent_operation;

      atomic_intrusive_queue<
          operation_state_child_base*,
          &operation_state_child_base::next>
          operation_states_of_children{};

      unifex::get_allocator_t<ManyReceiver> get_allocator() const noexcept {
        return unifex::get_allocator(many_receiver);
      }

      unifex::get_scheduler_t<ManyReceiver> get_scheduler() const noexcept {
        return unifex::get_scheduler(many_receiver);
      }

      struct single_sender_to_many_receiver {
        template <typename... Args>
        void set_value(Args&&... args) && noexcept {
          unifex::set_next(op->receiver, std::forward<Args>(args)...);
        }

        // TODO: How to present errors ?

        // template <typename CPO, typename Recv, typename... Args>
        // friend decltype(auto)
        // tag_invoke(CPO&& cpo, Recv&& recv, Args&&... args) noexcept {
        //   return tag_invoke(
        //       std::forward<CPO>(cpo),
        //       std::move(op->receiver),
        //       std::forward<Args>(args)...);
        // }
      };

      // This class receives next values from a ManySender source.
      // For each value we create new single senders using the SenderFactory.
      // Each result of the newly created senders will be forwarded to the
      // ManyReceiver.
      struct many_receiver_to_single_sender {
        operation_state* op;

        // Try to allocate, construct and start an operation state for the
        // continuation. This function also stores the operation state into the
        // op state queue of the main operation state.
        template <typename C>
        void enqueue_continuation(C&& continuation) noexcept {
          using SingleContinuation = std::remove_cvref_t<C>;
          // Type erasure for the newly enqueued operation state.
          struct operation_state_child : operation_state_child_base {
            operation_state_child(
                SingleContinuation&& s,
                single_sender_to_many_receiver&& r) noexcept
              : operation_state_child_base{nullptr}
              , operation_state{unifex::connect(std::move(s), std::move(r))} {}
            operation_state_t<
                SingleContinuation,
                single_sender_to_many_receiver>
                operation_state;
            void start() & noexcept { operation_state.start(); }
          };
          auto child_allocator =
              rebind_alloc<operation_state_child>(op->get_allocator());
          auto new_child = std::allocator_traits<ChildAllocator>::allocate(
              child_allocator, 1);
          std::allocator_traits<ChildAllocator>::construct(
              child_allocator,
              new_child,
              std::move(sender),
              single_sender_to_many_receiver{parent_op});
          op->operation_states_of_children.enqueue(new_child);
          new_child->start();
        }

        // For each value that we recieve through set_next we spawn one or more
        // operation states.
        template <typename... Args>
        void set_next(Args&&... args) & noexcept {
          std::apply(
              [&](auto&&... senders) { (enqueue_continuation(senders), ...); },
              std::tuple{op->factory(std::forward<Args>(args)...)});
        }

        void set_value() && noexcept { op->parent_has_finished = true; }
      };

      void start() & noexcept {
        parent_operation = unifex::connect(
            std::move(source), many_receiver_to_single_sender{this});
        n_operation_states.fetch_add(1, std::memory_order_relaxed);
        parent_operation->start();
      }
    };

    // A convenient typedef to strip cv qualifiers
    template <typename Source, typename Completion, typename ManyReceiver>
    using operation = typename operation_state<
        std::remove_cvref_t<Source>,
        std::remove_cvref_t<Completion>,
        std::remove_cvref_t<ManyReceiver>>;

    template <typename SourceManySender, typename SenderFactory>
    struct many_sender {
      [[no_unique_address]] SourceManySender source;
      [[no_unique_address]] SenderFactory factory;

      template <typename ManyReceiver>
      requires unifex::receiver_to<ManyReceiver>
          operation<SourceManySender, SenderFactory, ManyReceiver>
          connect(Receiver&& receiver) && noexcept {
        return {std::move(source), std::move(factory), std::move(receiver)};
      }
    };

  }  // namespace bulk_finally_ns
}  // namespace ampi