#pragma once
#include "types.hpp"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <utility>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Non-templated helpers — do not touch the grid accessor, declared here,
// implemented in core.cpp.
// ─────────────────────────────────────────────────────────────────────────────

// Transverse frame helpers
std::pair<Vec3,Vec3> transverse_point_field(const std::vector<Vec3>& B_line,
                                            int ref,
                                            const Vec3& ns);
std::pair<Vec3,Vec3> transverse_average_field(const std::vector<Vec3>& B_line,
                                              int region_start,
                                              int region_end,
                                              const Vec3& ns);

// Flux capacitor — shock classification from 1-D profiles (no grid access)
ShockResult flux_capacitor(const LineProfile& prof,
                           double gamma,
                           double shock_ratio = 1.2,
                           double shock_size  = 3.0);

// Shock normal via weighted average over a ball — uniform grid only.
// Declared here so existing callers compile; the template shock_normal<G>
// below falls back to this for G=GridAccessor with AVERAGE_GRADIENT.
Vec3 shock_normal_average(const FieldData& f, const CellIndex& idx,
                          int Rgrad, const bool periodic[3]);

// Serial batch — uniform grid path (compatible with existing bindings).
std::vector<ShockResult> characterise_shocks(
    const std::vector<CellIndex>& candidates,
    const FieldData& fields,
    const ShockParams& params,
    bool quiet = false);

// ─────────────────────────────────────────────────────────────────────────────
// Template physics functions — work with any accessor G that provides:
//   bool        in_bounds(const CellIndex&) const
//   size_t      flat(const CellIndex&) const
//   CellIndex   line_step(const CellIndex&, double l, const Vec3& dir) const
//   CellIndex   cyl_offset(const CellIndex&, int mu, int mv,
//                           const Vec3& nt1, const Vec3& nt2) const
// ─────────────────────────────────────────────────────────────────────────────

// Normal from the density gradient at a single point: n = -∇ρ / |∇ρ|
template<typename G>
inline Vec3 shock_normal_point(const FieldDataT<G>& f, const CellIndex& idx) {
    double gx = field_at(f.grad_x, f.grid, idx);
    double gy = field_at(f.grad_y, f.grid, idx);
    double gz = field_at(f.grad_z, f.grid, idx);
    Vec3 g{gx, gy, gz};
    double mag = g.norm();
    if (mag == 0.0) return {0, 0, 0};
    return (g * (-1.0 / mag));
}

// Shock normal dispatch.
// For uniform grid (G=GridAccessor): honours params.method_norm.
// For all other G (e.g. OctaveGridAccessor): always uses point gradient.
template<typename G>
inline Vec3 shock_normal(const FieldDataT<G>& f, const CellIndex& idx,
                         const ShockParams& params) {
    if constexpr (std::is_same_v<G, GridAccessor>) {
        // FieldDataT<GridAccessor> == FieldData; safe direct call
        if (params.method_norm == ShockParams::AVERAGE_GRADIENT)
            return shock_normal_average(f, idx, params.Rgrad, params.periodic);
    }
    return shock_normal_point(f, idx);
}

// Cylinder average: for each point on the shock line, average fields over a
// circle of radius Rcyl in the (nt1, nt2) plane.
template<typename G>
inline LineProfile cylinder_average(const FieldDataT<G>& f,
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

// Per-cell worker — analyse one candidate.
// For AMR: loc_x/y/z store the leaf_id (not a grid cell index).
template<typename G>
inline ShockResult characterise_shock(const CellIndex& candidate,
                                      const FieldDataT<G>& fields,
                                      const ShockParams& params) {
    ShockResult res;

    // For uniform grid: apply offset into local coords.
    // For AMR: candidate already holds the physical centre (cx/cy/cz); use as-is.
    CellIndex idx;
    if constexpr (std::is_same_v<G, GridAccessor>) {
        idx = {
            candidate.i - params.offset[0],
            candidate.j - params.offset[1],
            candidate.k - params.offset[2]
        };
    } else {
        idx = candidate;  // AMR path: leaf_id + cx/cy/cz already set
    }

    // 1. Shock normal
    Vec3 ns = shock_normal(fields, idx, params);
    if (ns.norm2() == 0.0) {
        res.flag = 4;
        res.loc_x = candidate.i; res.loc_y = candidate.j; res.loc_z = candidate.k;
        return res;
    }

    // 2. Build shock line
    std::vector<CellIndex> line_cells;
    std::vector<double>    line_coords;
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

    // 3. Transverse frame from B along the line
    std::vector<Vec3> B_line;
    B_line.reserve(line_cells.size());
    for (const auto& c : line_cells) {
        B_line.push_back({
            field_at(fields.bx, fields.grid, c),
            field_at(fields.by, fields.grid, c),
            field_at(fields.bz, fields.grid, c)
        });
    }

    int centre_pos = -1;
    for (int i = 0; i < (int)line_coords.size(); ++i)
        if (line_coords[i] == 0.0) { centre_pos = i; break; }
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

    // 6. Store location and direction
    if constexpr (std::is_same_v<G, GridAccessor>) {
        res.loc_x = idx.i + params.offset[0];
        res.loc_y = idx.j + params.offset[1];
        res.loc_z = idx.k + params.offset[2];
    } else {
        // AMR: store leaf_id as loc_x; loc_y/z unused
        res.loc_x = candidate.leaf_id;
        res.loc_y = candidate.level;
        res.loc_z = 0;
    }
    res.dir_x = std::round(ns.x * 1000.0) / 1000.0;
    res.dir_y = std::round(ns.y * 1000.0) / 1000.0;
    res.dir_z = std::round(ns.z * 1000.0) / 1000.0;

    return res;
}
