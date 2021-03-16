#pragma once

namespace ampi {
namespace _bulk_finally {
template <typename Source, typename Completion, typename ManyReceiver>
struct _op { class type; };

template <typename Source, typename Completion, typename ManyReceiver>
using operation = typename _op<Source, Completion, ManyReceiver>::type;



}
}