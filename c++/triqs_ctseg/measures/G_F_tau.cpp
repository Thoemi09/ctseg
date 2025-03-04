// Copyright (c) 2022-2024 Simons Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You may obtain a copy of the License at
//     https://www.gnu.org/licenses/gpl-3.0.txt
//
// Authors: Nikita Kavokine, Olivier Parcollet, Nils Wentzell

#include "./G_F_tau.hpp"
#include "../logs.hpp"

namespace triqs_ctseg::measures {

  G_F_tau::G_F_tau(params_t const &p, work_data_t const &wdata, configuration_t const &config, results_t &results)
     : wdata{wdata}, config{config}, results{results} {

    beta          = p.beta;
    measure_F_tau = p.measure_F_tau and wdata.rot_inv;
    gf_struct     = p.gf_struct;

    G_tau   = block_gf<imtime>{triqs::mesh::imtime{beta, Fermion, p.n_tau_G}, p.gf_struct};
    F_tau   = block_gf<imtime>{triqs::mesh::imtime{beta, Fermion, p.n_tau_G}, p.gf_struct};
    G_tau() = 0;
    F_tau() = 0;
    Z       = 0;
  }

  // -------------------------------------

  void G_F_tau::accumulate(double s) {

    LOG("\n =================== MEASURE G(tau) ================ \n");

    Z += s;

    for (auto [bl_idx, det] : itertools::enumerate(wdata.dets)) {
      long N  = det.size();
      auto &g = G_tau[bl_idx];
      auto &f = F_tau[bl_idx];
      for (long id_y : range(N)) {
        auto y        = det.get_y(id_y);
        double f_fact = 0;
        if (measure_F_tau) f_fact = fprefactor(bl_idx, y);
        for (long id_x : range(N)) {
          auto x    = det.get_x(id_x);
          auto Minv = det.inverse_matrix(id_y, id_x);
          // beta-periodicity is implicit in the argument, just fix the sign properly
          auto val  = (y.first >= x.first ? s : -s) * Minv;
          auto dtau = double(y.first - x.first);
          g[closest_mesh_pt(dtau)](y.second, x.second) += val;
          if (measure_F_tau) f[closest_mesh_pt(dtau)](y.second, x.second) += val * f_fact;
        }
      }
    }
  }

  // -------------------------------------

  void G_F_tau::collect_results(mpi::communicator const &c) {

    Z = mpi::all_reduce(Z, c);

    G_tau = mpi::all_reduce(G_tau, c);
    G_tau = G_tau / (-beta * Z * G_tau[0].mesh().delta());

    // Fix the point at zero and beta, for each block
    for (auto &g : G_tau) {
      g[0] *= 2;
      g[g.mesh().size() - 1] *= 2;
    }
    // store the result (not reused later, hence we can move it).
    results.G_tau = std::move(G_tau);

    if (measure_F_tau) {
      F_tau = mpi::all_reduce(F_tau, c);
      F_tau = F_tau / (-beta * Z * F_tau[0].mesh().delta());

      for (auto &f : F_tau) {
        f[0] *= 2;
        f[f.mesh().size() - 1] *= 2;
      }
      results.F_tau = std::move(F_tau);
    }
  }

  // -------------------------------------

  double G_F_tau::fprefactor(long const &block, std::pair<tau_t, long> const &y) {
    int color    = wdata.block_to_color(block, y.second);
    double I_tau = 0;
    for (auto const &[c, sl] : itertools::enumerate(config.seglists)) {
      auto ntau = n_tau(y.first, sl); // Density to the right of y.first in sl
      if (c != color) I_tau += wdata.U(c, color) * ntau;
      if (wdata.has_Dt) {
        I_tau -= K_overlap(sl, y.first, false, wdata.Kprime, c, color);
        if (c == color) I_tau -= 2 * real(wdata.Kprime(0)(c, c));
      }
      if (wdata.has_Jperp) {
        I_tau -= 4 * real(wdata.Kprime_spin(0)(c, color)) * ntau;
        I_tau -= 2 * K_overlap(sl, y.first, false, wdata.Kprime_spin, c, color);
      }
    }
    return I_tau;
  }

} // namespace triqs_ctseg::measures
