#pragma once

#include <triqs/stat/accumulator.hpp>

#include <functional>
#include <utility>

namespace triqs_ctseg::measures {

  struct auto_correlation_time {

    auto_correlation_time(double &sign_act, double &order_act, std::function<int()> get_order)
       : sign_act(sign_act), order_act(order_act), get_order(std::move(get_order)) {}

    void accumulate(double s) {
      acc_sign << s;
      acc_order << get_order();
    }

    void collect_results(mpi::communicator const &comm) {
      auto [sign_errs, sign_count]   = acc_sign.log_bin_errors_all_reduce(comm);
      auto [order_errs, order_count] = acc_order.log_bin_errors_all_reduce(comm);
      auto num_blocks                = static_cast<int>(0.7 * static_cast<double>(sign_errs.size()));

      if (!sign_errs.empty()) {
        sign_act.get() = triqs::stat::tau_estimate_from_errors(sign_errs[num_blocks], sign_errs[0]);
      }

      if (!order_errs.empty()) {
        order_act.get() = triqs::stat::tau_estimate_from_errors(order_errs[num_blocks], order_errs[0]);
      }
    }

    std::reference_wrapper<double> sign_act;
    std::reference_wrapper<double> order_act;
    std::function<int()> get_order;
    triqs::stat::accumulator<double> acc_sign{0.0, -1, 0};
    triqs::stat::accumulator<double> acc_order{0.0, -1, 0};
  };

} // namespace triqs_ctseg::measures
