#pragma once

#include "ampi/bulk_sequence.hpp"

namespace ampi {
namespace bulk_on_ {

inline constexpr struct fn {
  template <typename Scheduler, typename ManySender>
  auto operator()(Scheduler&& scheduler, ManySender&& many_sender) const noexcept {
    return bulk_sequence(unifex::schedule(scheduler), many_sender);
  }
} bulk_on;

}  // namespace bulk_on_

using bulk_on_::bulk_on;
}  // namespace ampi