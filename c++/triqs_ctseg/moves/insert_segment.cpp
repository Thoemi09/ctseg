#include "insert_segment.hpp"
#include "../logs.hpp"
#include <cmath>

namespace moves {

  double insert_segment::attempt() {

    LOG("\n =================== ATTEMPT INSERT ================ \n");

    // ------------ Choice of segment --------------
    // Select insertion color
    color    = rng(config.n_color());
    auto &sl = config.seglists[color];
    LOG("Inserting at color {}", color);

    // Select insertion window [wtau_left,wtau_right]
    tau_t wtau_left = tau_t::beta(), wtau_right = tau_t::zero();

    if (not sl.empty()) {
      if (is_full_line(sl.back())) {
        LOG("Full line, cannot insert.");
        return 0;
      }
      // Randomly choose one existing segment
      long seg_idx         = rng(sl.size());
      wtau_left            = sl[seg_idx].tau_cdag; // wtau_left is cdag of this segment
      bool is_last_segment = seg_idx == sl.size() - 1;
      wtau_right = sl[is_last_segment ? 0 : seg_idx + 1].tau_c; // wtau_right is c of next segment, possibly cyclic
    }

    LOG("Insertion window is wtau_left = {}, wtau_right = {}", wtau_left, wtau_right);
    tau_t window_length = sl.empty() ? tau_t::beta() : wtau_left - wtau_right;

    // Choose two random times in insertion window
    auto dt1 = tau_t::random(rng, window_length);
    auto dt2 = tau_t::random(rng, window_length);
    if (dt1 == dt2) {
      LOG("Generated equal times");
      return 0;
    }
    if (dt1 > dt2 and !sl.empty()) std::swap(dt1, dt2); // if inserting into an empty line, two ways to insert
    prop_seg = segment_t{wtau_left - dt1, wtau_left - dt2};

    LOG("Inserting segment with c at {}, cdag at {}", prop_seg.tau_c, prop_seg.tau_cdag);

    // ------------  Trace ratio  -------------
    double ln_trace_ratio = wdata.mu(color) * prop_seg.length();
    for (auto c : range(config.n_color())) {
      if (c != color) { ln_trace_ratio += -wdata.U(color, c) * overlap(config.seglists[c], prop_seg); }
      if (wdata.has_Dt)
        ln_trace_ratio += K_overlap(config.seglists[c], prop_seg.tau_c, prop_seg.tau_cdag, wdata.K, color, c);
    }
    if (wdata.has_Dt)
      ln_trace_ratio +=
         -real(wdata.K(double(prop_seg.tau_c - prop_seg.tau_cdag))(color, color)); // Correct double counting
    double trace_ratio = std::exp(ln_trace_ratio);

    // ------------  Det ratio  ---------------
    auto &D             = wdata.dets[color];
    auto det_c_time     = [&](long i) { return D.get_y(i).first; };
    auto det_cdag_time  = [&](long i) { return D.get_x(i).first; };
    long det_index_c    = lower_bound(det_c_time, D.size(), prop_seg.tau_c);
    long det_index_cdag = lower_bound(det_cdag_time, D.size(), prop_seg.tau_cdag);
    // We insert tau_cdag as a line (first index) and tau_c as a column (second index).
    auto det_ratio = D.try_insert(det_index_cdag, det_index_c, {prop_seg.tau_cdag, 0}, {prop_seg.tau_c, 0});

    // ------------  Proposition ratio ------------

    double current_number_intervals = std::max(1.0, double(sl.size()));
    double future_number_segments   = double(sl.size()) + 1;
    double prop_ratio =
       (current_number_intervals * window_length * window_length / (sl.empty() ? 1 : 2)) / future_number_segments;
    // Account for absence of time swapping when inserting into empty line.

    LOG("trace_ratio  = {}, prop_ratio = {}, det_ratio = {}", trace_ratio, prop_ratio, det_ratio);

    double prod = trace_ratio * det_ratio * prop_ratio;
    det_sign    = (det_ratio > 0) ? 1.0 : -1.0;
    return (std::isfinite(prod) ? prod : det_sign);
  }

  //--------------------------------------------------

  double insert_segment::accept() {

    LOG("\n - - - - - ====> ACCEPT - - - - - - - - - - -\n");

    double initial_sign = config_sign(config, wdata.dets);
    LOG("Initial sign is {}. Initial configuration: {}", initial_sign, config);

    // Insert the times into the det
    wdata.dets[color].complete_operation();

    // Insert the segment in an ordered list
    auto &sl         = config.seglists[color];
    auto prop_seg_it = std::upper_bound(sl.begin(), sl.end(), prop_seg);
    sl.insert(prop_seg_it, prop_seg);

    // Check invariant
    if constexpr (check_invariants or ctseg_debug) check_invariant(config, wdata.dets);

    double final_sign = config_sign(config, wdata.dets);
    double sign_ratio = final_sign / initial_sign;
    LOG("Final sign is {}", final_sign);

    ALWAYS_EXPECTS((sign_ratio * det_sign == 1.0),
                   "Error: move has produced negative sign! Det sign is {} and additional sign is {}. Config: ",
                   det_sign, sign_ratio, config);
    LOG("Configuration is {}", config);

    return sign_ratio;
  }

  //--------------------------------------------------
  void insert_segment::reject() {
    LOG("\n - - - - - ====> REJECT - - - - - - - - - - -\n");
    wdata.dets[color].reject_last_try();
  }
}; // namespace moves
