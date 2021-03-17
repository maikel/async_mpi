#pragma once

#include <unifex/receiver_concepts.hpp>
#include <unifex/scheduler_concepts.hpp>
#include <unifex/sender_concepts.hpp>

#include <span>

#include <mpi.h>

namespace ampi {
namespace _for_each {
template <typename R>
struct operation {
  struct type;
};

template <typename R>
struct operation<R>::type {
public:
  type(R&& receiver, MPI_Comm comm, std::vector<MPI_Request>&& requests, int tag)
    : receiver_{std::move(receiver)}
    , comm_{comm}
    , requests_{std::move(requests)}
    , tag_{tag} {}

  type(const type&) = delete;
  type& operator=(const type&) = delete;

  type(type&&) = delete;
  type& operator=(type&&) = delete;

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

template <typename R>
using op = operation<std::remove_cvref_t<R>>::type;

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

private:
  MPI_Comm comm_;
  std::vector<MPI_Request> requests_;
  int tag_;

  template <typename R>
  friend auto tag_invoke(unifex::tag_t<unifex::connect>, sender sender, R&& recv) noexcept {
    return op<R>(
        std::move(recv),
        std::move(sender).comm_,
        std::move(sender).requests_,
        std::move(sender).tag_);
  }
};

inline constexpr struct fn {
  auto operator()(MPI_Comm comm, std::vector<MPI_Request> requests, int tag) const noexcept {
    return sender(comm, std::move(requests), tag);
  }
} for_each{};

}  // namespace _for_each

using _for_each::for_each;
}  // namespace ampi