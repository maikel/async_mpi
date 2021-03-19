#pragma once

#include <unifex/receiver_concepts.hpp>
#include <unifex/scheduler_concepts.hpp>
#include <unifex/sender_concepts.hpp>

#include <span>

#include <mpi.h>

namespace ampi {
namespace for_each_ns {
template <typename R>
struct operation {
public:
  operation(R&& receiver, MPI_Comm comm, std::vector<MPI_Request>&& requests, int tag)
    : receiver_{std::move(receiver)}
    , comm_{comm}
    , requests_{std::move(requests)}
    , tag_{tag} {}

  operation(const operation&) = delete;
  operation& operator=(const operation&) = delete;

  operation(operation&&) = delete;
  operation& operator=(operation&&) = delete;

  void start() & noexcept {
    const int size = static_cast<int>(requests_.size());
    int remaining = size;
    while (remaining > 0) {
      int index{-1};
      int errcode = MPI_Waitany(size, requests_.data(), &index, MPI_STATUS_IGNORE);
      if (errcode != MPI_SUCCESS) {
        unifex::set_error(std::move(receiver_), errcode);
        return;
      }
      unifex::set_next(receiver_, index);
      remaining -= 1;
    }
    unifex::set_value(std::move(receiver_));
  }

private:
  [[no_unique_address]] R receiver_;
  MPI_Comm comm_;
  std::vector<MPI_Request> requests_;
  int tag_;
};

class sender {
public:
  template <template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant, template <typename...> class Tuple>
  using next_types = Variant<Tuple<int>>;

  template <template <typename...> class Variant>
  using error_types = Variant<int>;

  static constexpr bool sends_done = true;

  sender(MPI_Comm comm, std::vector<MPI_Request> reqs, int tag)
    : comm_{comm}
    , requests_(std::move(reqs))
    , tag_{tag} {}

  template <typename Receiver>
  auto connect(Receiver&& receiver) && noexcept {
    return operation<std::remove_cvref_t<Receiver>>(
        std::forward<Receiver>(receiver), comm_, std::move(requests_), tag_);
  }

  constexpr std::integral_constant<unifex::blocking_kind, unifex::blocking_kind::always_inline>
  blocking() const noexcept {
    return {};
  }

private:
  MPI_Comm comm_;
  std::vector<MPI_Request> requests_;
  int tag_;
};

inline constexpr struct fn {
  auto operator()(std::vector<MPI_Request> requests, MPI_Comm comm, int tag) const noexcept {
    return sender(comm, std::move(requests), tag);
  }
} for_each{};

}  // namespace for_each_ns

using for_each_ns::for_each;
}  // namespace ampi
