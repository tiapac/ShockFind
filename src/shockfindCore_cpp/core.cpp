#include "core.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdio>

static constexpr double FOUR_PI = 4.0 * 3.141592653589793;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (used by flux_capacitor)
// ─────────────────────────────────────────────────────────────────────────────

static double vec_average(const std::vector<double>& v, int a, int b) {
    if (a >= b) return 0.0;
    double s = 0.0;
    for (int i = a; i < b; ++i) s += v[i];
    return s / (b - a);
}

static double vec_average_norm(const std::vector<double>& bx,
                               const std::vector<double>& by,
                               const std::vector<double>& bz,
                               int a, int b) {
    if (a >= b) return 0.0;
    double s = 0.0;
    for (int i = a; i < b; ++i)
        s += std::sqrt(bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i]);
    return s / (b - a);
}

// ─────────────────────────────────────────────────────────────────────────────
// shock_normal_average — uniform grid only (uses shape.nx/ny/nz + periodic wrap)
// ─────────────────────────────────────────────────────────────────────────────
Vec3 shock_normal_average(const FieldData& f, const CellIndex& idx,
                          int Rgrad, const bool periodic[3]) {
    const GridShape& sh = f.grid.shape;
    Vec3 sum{0, 0, 0};
    double w_sum = 0.0;

    for (int di = -Rgrad; di <= Rgrad; ++di) {
    for (int dj = -Rgrad; dj <= Rgrad; ++dj) {
    for (int dk = -Rgrad; dk <= Rgrad; ++dk) {
        double dist = std::sqrt(di*di + dj*dj + dk*dk);
        if (dist > Rgrad) continue;

        int ci = idx.i + di;
        int cj = idx.j + dj;
        int ck = idx.k + dk;

        if (periodic[0]) { ci = ((ci % sh.nx) + sh.nx) % sh.nx; }
        if (periodic[1]) { cj = ((cj % sh.ny) + sh.ny) % sh.ny; }
        if (periodic[2]) { ck = ((ck % sh.nz) + sh.nz) % sh.nz; }

        CellIndex c{ci, cj, ck};
        if (!f.grid.in_bounds(c)) continue;

        double gx = field_at(f.grad_x, f.grid, c);
        double gy = field_at(f.grad_y, f.grid, c);
        double gz = field_at(f.grad_z, f.grid, c);
        double w  = std::sqrt(gx*gx + gy*gy + gz*gz);

        sum += Vec3{gx, gy, gz} * w;
        w_sum += w;
    }}}

    if (w_sum == 0.0) return {0, 0, 0};
    Vec3 avg = sum * (1.0 / w_sum);
    return (avg * -1.0).normalized();
}

// ─────────────────────────────────────────────────────────────────────────────
// Transverse frame
// ─────────────────────────────────────────────────────────────────────────────
std::pair<Vec3,Vec3> transverse_point_field(const std::vector<Vec3>& B_line,
                                            int ref,
                                            const Vec3& ns) {
    Vec3 b = B_line[ref].normalized();
    Vec3 nt2 = ns.cross(b).normalized();
    Vec3 nt1 = nt2.cross(ns);
    return {nt1, nt2};
}

std::pair<Vec3,Vec3> transverse_average_field(const std::vector<Vec3>& B_line,
                                              int region_start,
                                              int region_end,
                                              const Vec3& ns) {
    Vec3 avg{0, 0, 0};
    int n = 0;
    for (int i = region_start; i < region_end; ++i) {
        avg += B_line[i];
        ++n;
    }
    if (n == 0) return {{0,0,1}, {0,1,0}};
    Vec3 b = avg.normalized();
    Vec3 nt2 = ns.cross(b).normalized();
    Vec3 nt1 = nt2.cross(ns);
    return {nt1, nt2};
}

// ─────────────────────────────────────────────────────────────────────────────
// flux_capacitor — shock classification from 1-D profiles
// ─────────────────────────────────────────────────────────────────────────────
ShockResult flux_capacitor(const LineProfile& prof,
                           double gamma,
                           double shock_ratio,
                           double shock_size) {
    ShockResult res;
    int L = static_cast<int>(prof.line.size());
    if (L == 0) { res.flag = 4; return res; }

    std::vector<double> p_mag(L);
    for (int i = 0; i < L; ++i) {
        double bsq = prof.bp[i]*prof.bp[i] + prof.bt1[i]*prof.bt1[i] + prof.bt2[i]*prof.bt2[i];
        p_mag[i] = bsq / (8.0 * 3.141592653589793);
    }

    std::vector<double> conv_pos;
    for (int i = 0; i < L; ++i)
        if (prof.conv[i] > 0.0) conv_pos.push_back(prof.conv[i]);
    if (conv_pos.empty()) { res.flag = 4; return res; }

    int centre_init = -1;
    for (int i = 0; i < L; ++i)
        if (prof.line[i] == 0.0) { centre_init = i; break; }
    if (centre_init < 0) {
        double best = 1e30;
        for (int i = 0; i < L; ++i)
            if (std::abs(prof.line[i]) < best) { best = std::abs(prof.line[i]); centre_init = i; }
    }
    if (centre_init <= 0 || centre_init >= L - 1) { res.flag = 2; return res; }

    int shock_peak = centre_init;
    res.peak_flag = (shock_peak == centre_init) ? 1 : 0;

    int sz = static_cast<int>(shock_size);
    int s1_lo = std::max(0, shock_peak - sz);
    int s1_hi = shock_peak;
    int s2_lo = std::min(shock_peak + 1, L);
    int s2_hi = std::min(shock_peak + sz + 1, L);

    if (s1_hi <= s1_lo || s2_hi <= s2_lo) { res.flag = 1; return res; }

    double rho1 = vec_average(prof.rho, s1_lo, s1_hi);
    double rho2 = vec_average(prof.rho, s2_lo, s2_hi);

    int state_pre_lo, state_pre_hi, state_pst_lo, state_pst_hi;
    double rho_pre, rho_pst;

    if (rho1 > shock_ratio * rho2) {
        state_pre_lo = s2_lo; state_pre_hi = s2_hi;
        state_pst_lo = s1_lo; state_pst_hi = s1_hi;
        rho_pre = rho2; rho_pst = rho1;
    } else if (rho2 > shock_ratio * rho1) {
        state_pre_lo = s1_lo; state_pre_hi = s1_hi;
        state_pst_lo = s2_lo; state_pst_hi = s2_hi;
        rho_pre = rho1; rho_pst = rho2;
    } else {
        res.flag = 1;
        res.r    = (rho2 > 0.0) ? rho1 / rho2 : 0.0;
        res.pmag_ratio = (vec_average(p_mag, s1_lo, s1_hi) > 0.0)
                       ?  vec_average(p_mag, s2_lo, s2_hi) / vec_average(p_mag, s1_lo, s1_hi)
                       : 0.0;
        res.vA    = vec_average_norm(prof.bp, prof.bt1, prof.bt2, s2_lo, s2_hi)
                  / std::sqrt(FOUR_PI * rho2);
        res.rho0  = rho2;
        res.B0    = vec_average_norm(prof.bp, prof.bt1, prof.bt2, s1_lo, s1_hi);
        return res;
    }

    res.r    = rho_pst / rho_pre;
    res.rho0 = rho_pre;
    res.B0   = vec_average_norm(prof.bp, prof.bt1, prof.bt2, state_pre_lo, state_pre_hi);

    double p_mag_pre = vec_average(p_mag, state_pre_lo, state_pre_hi);
    double p_mag_pst = vec_average(p_mag, state_pst_lo, state_pst_hi);
    res.pmag_ratio = (p_mag_pre > 0.0) ? p_mag_pst / p_mag_pre : 0.0;

    if      (p_mag_pst > p_mag_pre) res.family = 12;
    else if (p_mag_pst < p_mag_pre) res.family = 34;
    else                            res.family =  0;

    double u_pre = vec_average(prof.vp, state_pre_lo, state_pre_hi);
    double u_pst = vec_average(prof.vp, state_pst_lo, state_pst_hi);
    double denom = 1.0 - rho_pre / rho_pst;
    double vs = (denom != 0.0) ? (u_pre - u_pst) / denom : 0.0;
    res.vs = std::abs(vs);

    double b_pre = vec_average_norm(prof.bp, prof.bt1, prof.bt2, state_pre_lo, state_pre_hi);
    res.vA     = (rho_pre > 0.0) ? b_pre / std::sqrt(FOUR_PI * rho_pre) : 0.0;
    res.MachAlf = (res.vA > 0.0) ? res.vs / res.vA : 0.0;

    double p_pre  = vec_average(prof.pres, state_pre_lo, state_pre_hi);
    double csound = (rho_pre > 0.0) ? std::sqrt(gamma * p_pre / rho_pre) : 0.0;
    res.Mach = (csound > 0.0) ? res.vs / csound : 0.0;

    if      (res.family == 12 && res.MachAlf <= 1.0) res.flag = 3;
    else if (res.family == 34 && res.MachAlf >  1.0) res.flag = 3;
    else                                              res.flag = 0;

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// characterise_shocks — serial batch (uniform grid path)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<ShockResult> characterise_shocks(
    const std::vector<CellIndex>& candidates,
    const FieldData& fields,
    const ShockParams& params,
    bool quiet) {

    std::vector<ShockResult> results;
    results.reserve(candidates.size());

    int n = static_cast<int>(candidates.size());
    for (int i = 0; i < n; ++i) {
        ShockResult r = characterise_shock(candidates[i], fields, params);
        results.push_back(r);
        if (!quiet) {
            int print_every = std::max(1, n / 20);
            if (i % print_every == 0 || i == n - 1) {
                std::printf("  [rank %d] %d/%d  (%.0f%%)\n",
                            0, i + 1, n, 100.0 * (i + 1) / n);
                std::fflush(stdout);
            }
        }
    }
    return results;
}
