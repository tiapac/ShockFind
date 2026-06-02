#include "core.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdio>

// Physical constant: 4π (used in Alfvén speed formula)
static constexpr double FOUR_PI = 4.0 * 3.141592653589793;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static double vec_average(const std::vector<double>& v, int a, int b) {
    // Average v[a..b) inclusive — mirrors numpy average over index slices.
    if (a >= b) return 0.0;
    double s = 0.0;
    for (int i = a; i < b; ++i) s += v[i];
    return s / (b - a);
}

static double vec_average_norm(const std::vector<double>& bx,
                               const std::vector<double>& by,
                               const std::vector<double>& bz,
                               int a, int b) {
    // Average of |B| over index range [a, b)
    if (a >= b) return 0.0;
    double s = 0.0;
    for (int i = a; i < b; ++i)
        s += std::sqrt(bx[i]*bx[i] + by[i]*by[i] + bz[i]*bz[i]);
    return s / (b - a);
}

// ─────────────────────────────────────────────────────────────────────────────
// shock_normal_point
// ─────────────────────────────────────────────────────────────────────────────
Vec3 shock_normal_point(const FieldData& f, const CellIndex& idx) {
    double gx = field_at(f.grad_x, f.grid, idx);
    double gy = field_at(f.grad_y, f.grid, idx);
    double gz = field_at(f.grad_z, f.grid, idx);
    Vec3 g{gx, gy, gz};
    double mag = g.norm();
    if (mag == 0.0) return {0, 0, 0};
    return (g * (-1.0 / mag));  // n = -∇ρ / |∇ρ|
}

// ─────────────────────────────────────────────────────────────────────────────
// shock_normal_average
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

        // Periodic wrapping per axis
        if (periodic[0]) { ci = ((ci % sh.nx) + sh.nx) % sh.nx; }
        if (periodic[1]) { cj = ((cj % sh.ny) + sh.ny) % sh.ny; }
        if (periodic[2]) { ck = ((ck % sh.nz) + sh.nz) % sh.nz; }

        CellIndex c{ci, cj, ck};
        if (!f.grid.in_bounds(c)) continue;

        double gx = field_at(f.grad_x, f.grid, c);
        double gy = field_at(f.grad_y, f.grid, c);
        double gz = field_at(f.grad_z, f.grid, c);
        double w  = std::sqrt(gx*gx + gy*gy + gz*gz);  // weight = |∇ρ|

        sum += Vec3{gx, gy, gz} * w;
        w_sum += w;
    }}}

    if (w_sum == 0.0) return {0, 0, 0};
    Vec3 avg = sum * (1.0 / w_sum);
    return (avg * -1.0).normalized();
}

// ─────────────────────────────────────────────────────────────────────────────
// shock_normal dispatch
// ─────────────────────────────────────────────────────────────────────────────
Vec3 shock_normal(const FieldData& f, const CellIndex& idx,
                  const ShockParams& params) {
    if (params.method_norm == ShockParams::POINT_GRADIENT)
        return shock_normal_point(f, idx);
    else
        return shock_normal_average(f, idx, params.Rgrad, params.periodic);
}

// ─────────────────────────────────────────────────────────────────────────────
// transverse_point_field
// Computes the transverse frame (nt1, nt2) from the B-field at one reference
// point on the shock line.
//
// Convention (same as Python):
//   nt2 = ns × b     (where b is the normalised B at the reference point)
//   nt1 = nt2 × ns
// ─────────────────────────────────────────────────────────────────────────────
std::pair<Vec3,Vec3> transverse_point_field(const std::vector<Vec3>& B_line,
                                            int ref,
                                            const Vec3& ns) {
    Vec3 b = B_line[ref].normalized();
    Vec3 nt2 = ns.cross(b).normalized();
    Vec3 nt1 = nt2.cross(ns);   // already unit length if ns and nt2 are
    return {nt1, nt2};
}

// ─────────────────────────────────────────────────────────────────────────────
// transverse_average_field
// ─────────────────────────────────────────────────────────────────────────────
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
    if (n == 0) return {{0,0,1}, {0,1,0}};  // degenerate fallback
    Vec3 b = avg.normalized();
    Vec3 nt2 = ns.cross(b).normalized();
    Vec3 nt1 = nt2.cross(ns);
    return {nt1, nt2};
}

// ─────────────────────────────────────────────────────────────────────────────
// cylinder_average
// For each point on the shock line, average fields over a circle of radius
// Rcyl in the (nt1, nt2) plane.  Log-average density and pressure; linear
// average for velocity and field projections.
// Skips out-of-bounds cells — identical to the Python try-except behaviour.
// ─────────────────────────────────────────────────────────────────────────────
LineProfile cylinder_average(const FieldData& f,
                             const std::vector<CellIndex>& cells,
                             const std::vector<double>& line_coords,
                             const Vec3& ns,
                             const Vec3& nt1,
                             const Vec3& nt2,
                             int Rcyl) {
    int L = static_cast<int>(cells.size());
    LineProfile out;
    out.line = line_coords;
    out.rho .resize(L, 0.0);
    out.pres.resize(L, 0.0);
    out.vp  .resize(L, 0.0);
    out.vt1 .resize(L, 0.0);
    out.vt2 .resize(L, 0.0);
    out.bp  .resize(L, 0.0);
    out.bt1 .resize(L, 0.0);
    out.bt2 .resize(L, 0.0);
    out.conv.resize(L, 0.0);

    for (int li = 0; li < L; ++li) {
        const CellIndex& centre = cells[li];

        double log_rho_avg = 0.0, log_p_avg = 0.0;
        double vp_avg = 0.0, vt1_avg = 0.0, vt2_avg = 0.0;
        double bp_avg = 0.0, bt1_avg = 0.0, bt2_avg = 0.0;
        double conv_avg = 0.0;
        int N_avg = 0;

        for (int mu = -Rcyl; mu <= Rcyl; ++mu) {
        for (int mv = -Rcyl; mv <= Rcyl; ++mv) {
            // Check circular constraint (same as Python)
            Vec3 disp = nt2 * mu + nt1 * mv;
            if (disp.norm() > Rcyl) continue;

            CellIndex cp = f.grid.cyl_offset(centre, mu, mv, nt1, nt2);
            if (!f.grid.in_bounds(cp)) continue;

            double rho_v  = field_at(f.rho,  f.grid, cp);
            double pres_v = field_at(f.pres, f.grid, cp);
            double vx_v   = field_at(f.vx,   f.grid, cp);
            double vy_v   = field_at(f.vy,   f.grid, cp);
            double vz_v   = field_at(f.vz,   f.grid, cp);
            double bx_v   = field_at(f.bx,   f.grid, cp);
            double by_v   = field_at(f.by,   f.grid, cp);
            double bz_v   = field_at(f.bz,   f.grid, cp);
            double div_v  = field_at(f.div,  f.grid, cp);

            Vec3 vel  {vx_v, vy_v, vz_v};
            Vec3 Bvec {bx_v, by_v, bz_v};

            // Dynamic (online) mean — same formula as Python
            double nd = static_cast<double>(N_avg);
            log_rho_avg = (nd * log_rho_avg + std::log10(rho_v))  / (nd + 1.0);
            log_p_avg   = (nd * log_p_avg   + std::log10(pres_v)) / (nd + 1.0);
            vp_avg      = (nd * vp_avg      + ns.dot(vel))        / (nd + 1.0);
            vt1_avg     = (nd * vt1_avg     + nt1.dot(vel))       / (nd + 1.0);
            vt2_avg     = (nd * vt2_avg     + nt2.dot(vel))       / (nd + 1.0);
            bp_avg      = (nd * bp_avg      + ns.dot(Bvec))       / (nd + 1.0);
            bt1_avg     = (nd * bt1_avg     + nt1.dot(Bvec))      / (nd + 1.0);
            bt2_avg     = (nd * bt2_avg     + nt2.dot(Bvec))      / (nd + 1.0);
            conv_avg    = (nd * conv_avg    - div_v)               / (nd + 1.0);
            ++N_avg;
        }}

        // Convert log-averages back to linear scale
        out.rho [li] = (N_avg > 0) ? std::pow(10.0, log_rho_avg) : 0.0;
        out.pres[li] = (N_avg > 0) ? std::pow(10.0, log_p_avg)   : 0.0;
        out.vp  [li] = vp_avg;
        out.vt1 [li] = vt1_avg;
        out.vt2 [li] = vt2_avg;
        out.bp  [li] = bp_avg;
        out.bt1 [li] = bt1_avg;
        out.bt2 [li] = bt2_avg;
        out.conv[li] = conv_avg;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// flux_capacitor
// Classify a shock from 1-D line profiles.
// Faithfully mirrors the Python flux_capacitor logic.
// ─────────────────────────────────────────────────────────────────────────────
ShockResult flux_capacitor(const LineProfile& prof,
                           double gamma,
                           double shock_ratio,
                           double shock_size) {
    ShockResult res;
    int L = static_cast<int>(prof.line.size());
    if (L == 0) { res.flag = 4; return res; }

    // Magnetic pressure along the line
    std::vector<double> p_mag(L);
    for (int i = 0; i < L; ++i) {
        double bsq = prof.bp[i]*prof.bp[i] + prof.bt1[i]*prof.bt1[i] + prof.bt2[i]*prof.bt2[i];
        p_mag[i] = bsq / (8.0 * 3.141592653589793);
    }

    // --- Check that there is positive convergence ---
    std::vector<double> conv_pos;
    for (int i = 0; i < L; ++i)
        if (prof.conv[i] > 0.0) conv_pos.push_back(prof.conv[i]);

    if (conv_pos.empty()) { res.flag = 4; return res; }

    // --- Find convergence peak centre (index where line == 0) ---
    int centre_init = -1;
    for (int i = 0; i < L; ++i) {
        if (prof.line[i] == 0.0) { centre_init = i; break; }
    }
    // Fallback: index closest to 0
    if (centre_init < 0) {
        double best = 1e30;
        for (int i = 0; i < L; ++i) {
            if (std::abs(prof.line[i]) < best) { best = std::abs(prof.line[i]); centre_init = i; }
        }
    }
    if (centre_init <= 0 || centre_init >= L - 1) {
        res.flag = 2; return res;
    }

    int shock_peak = centre_init;   // Python: local walk — not needed for peak=0

    res.peak_flag = (shock_peak == centre_init) ? 1 : 0;

    // --- Define pre- and post-shock state regions ---
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
        // state1 is post-shock, state2 is pre-shock
        state_pre_lo = s2_lo; state_pre_hi = s2_hi;
        state_pst_lo = s1_lo; state_pst_hi = s1_hi;
        rho_pre = rho2; rho_pst = rho1;
    } else if (rho2 > shock_ratio * rho1) {
        // state2 is post-shock, state1 is pre-shock
        state_pre_lo = s1_lo; state_pre_hi = s1_hi;
        state_pst_lo = s2_lo; state_pst_hi = s2_hi;
        rho_pre = rho1; rho_pst = rho2;
    } else {
        // Density threshold not met
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

    // --- Shock speed and Mach numbers ---
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

    // Validation
    if      (res.family == 12 && res.MachAlf <= 1.0) res.flag = 3;
    else if (res.family == 34 && res.MachAlf >  1.0) res.flag = 3;
    else                                              res.flag = 0;

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// characterise_shock — single candidate
// ─────────────────────────────────────────────────────────────────────────────
ShockResult characterise_shock(const CellIndex& candidate,
                               const FieldData& fields,
                               const ShockParams& params) {
    ShockResult res;
    // Apply offset (subdomains pass local coords; offset corrects back to global)
    CellIndex idx{
        candidate.i - params.offset[0],
        candidate.j - params.offset[1],
        candidate.k - params.offset[2]
    };

    // 1. Shock normal
    Vec3 ns = shock_normal(fields, idx, params);
    if (ns.norm2() == 0.0) {
        res.flag = 4;
        res.loc_x = candidate.i; res.loc_y = candidate.j; res.loc_z = candidate.k;
        return res;
    }

    // 2. Build shock line
    std::vector<CellIndex>  line_cells;
    std::vector<double>     line_coords;
    for (int li = -params.line_range; li <= params.line_range; ++li) {
        CellIndex c = fields.grid.line_step(idx, static_cast<double>(li), ns);
        if (fields.grid.in_bounds(c)) {
            line_cells.push_back(c);
            line_coords.push_back(static_cast<double>(li));
        }
    }
    if (line_cells.empty()) {
        res.flag = 2;
        res.loc_x = candidate.i; res.loc_y = candidate.j; res.loc_z = candidate.k;
        return res;
    }

    // 3. Transverse frame
    // Build B along the line
    std::vector<Vec3> B_line;
    B_line.reserve(line_cells.size());
    for (const auto& c : line_cells) {
        B_line.push_back({
            field_at(fields.bx, fields.grid, c),
            field_at(fields.by, fields.grid, c),
            field_at(fields.bz, fields.grid, c)
        });
    }

    // Reference point: centre of line, ±field_ref offset
    int centre_pos = -1;
    for (int i = 0; i < (int)line_coords.size(); ++i) {
        if (line_coords[i] == 0.0) { centre_pos = i; break; }
    }
    if (centre_pos < 0) centre_pos = static_cast<int>(line_cells.size()) / 2;

    int ref = centre_pos - params.field_ref;
    if (ref < 0) ref = centre_pos + params.field_ref;
    ref = std::max(0, std::min(ref, static_cast<int>(B_line.size()) - 1));

    Vec3 nt1, nt2;
    if (params.method_plane == ShockParams::POINT_FIELD) {
        auto [t1, t2] = transverse_point_field(B_line, ref, ns);
        nt1 = t1; nt2 = t2;
    } else {
        int r_start = std::max(0, ref);
        int r_end   = std::min(centre_pos, static_cast<int>(B_line.size()));
        if (r_end <= r_start) r_end = r_start + 1;
        r_end = std::min(r_end, static_cast<int>(B_line.size()));
        auto [t1, t2] = transverse_average_field(B_line, r_start, r_end, ns);
        nt1 = t1; nt2 = t2;
    }

    // 4. Cylinder averaging
    LineProfile prof = cylinder_average(fields, line_cells, line_coords,
                                        ns, nt1, nt2, params.Rcylinder);

    // 5. Classify
    res = flux_capacitor(prof, params.gamma, params.shock_ratio, 3.0);

    // 6. Store location and direction (global coordinates)
    res.loc_x = idx.i + params.offset[0];
    res.loc_y = idx.j + params.offset[1];
    res.loc_z = idx.k + params.offset[2];
    res.dir_x = std::round(ns.x * 1000.0) / 1000.0;
    res.dir_y = std::round(ns.y * 1000.0) / 1000.0;
    res.dir_z = std::round(ns.z * 1000.0) / 1000.0;

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// characterise_shocks — serial batch
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
            int print_every = std::max(1, n / 200);  // ~200 progress lines max
            if (i % print_every == 0 || i == n - 1) {
                std::printf("[%d/%d] %.1f%%\r", i + 1, n, 100.0 * (i + 1) / n);
                std::fflush(stdout);
            }
        }
    }
    return results;
}
