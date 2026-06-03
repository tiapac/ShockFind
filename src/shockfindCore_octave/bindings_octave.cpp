#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Octave
#include "tree.hpp"
#include "octree_params.hpp"
#include "octave_shockfind_utils.hpp"

// ShockFind
#include "octave_accessor.hpp"
#include "../shockfindCore_cpp/core.hpp"

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// OctreeHandle — owns the MainTree<3> and all derived field arrays.
// Created by build_octree(), shared with find_candidates() and
// characterise_shocks_octree() via shared_ptr.
// ─────────────────────────────────────────────────────────────────────────────
struct OctreeHandle {
    std::unique_ptr<MainTree<3>> tree;
    // Field data indexed by Hilbert rank — parallel to tree->ordered_leaf_indices.
    // These are pointers into tree->leafFieldData (stable after build; no realloc).
    const double* rho    = nullptr;
    const double* pres   = nullptr;
    const double* vx     = nullptr;
    const double* vy     = nullptr;
    const double* vz     = nullptr;
    const double* bx     = nullptr;
    const double* by     = nullptr;
    const double* bz     = nullptr;
    // Derived (computed during build_octree)
    std::vector<double> div_v;    // velocity divergence per leaf (Hilbert order)
    std::vector<double> grad_x;   // ∇ρ x-component
    std::vector<double> grad_y;
    std::vector<double> grad_z;

    OctaveGridAccessor accessor() {
        return OctaveGridAccessor{tree.get()};
    }

    FieldDataT<OctaveGridAccessor> field_data() {
        FieldDataT<OctaveGridAccessor> f;
        f.grid  = accessor();
        f.rho   = rho;
        f.pres  = pres;
        f.vx    = vx;
        f.vy    = vy;
        f.vz    = vz;
        f.bx    = bx;
        f.by    = by;
        f.bz    = bz;
        f.div   = div_v.empty()  ? nullptr : div_v.data();
        f.grad_x = grad_x.empty() ? nullptr : grad_x.data();
        f.grad_y = grad_y.empty() ? nullptr : grad_y.data();
        f.grad_z = grad_z.empty() ? nullptr : grad_z.data();
        return f;
    }
};

// Resolve a field name to a pointer into tree->leafFieldData.
// Returns nullptr if the field is not registered.
static const double* resolve_field(MainTree<3>& tree, const std::string& name) {
    auto it = tree.leafFieldData.find(name);
    if (it == tree.leafFieldData.end() || it->second.empty()) return nullptr;
    return it->second.data();
}

// ─────────────────────────────────────────────────────────────────────────────
// build_octree
// Build a MainTree<3> from numpy position and attribute arrays.
//
// positions : (N, 3) float64 — normalised to [0,1]^3
// attrs     : (N, K) float64 — per-point attribute columns
// field_names : list[str] — names for each attribute column (length K)
//              Columns are assigned to fields rho/pres/vx/vy/vz/bx/by/bz
//              by matching this list of names.
//              Special reducers: sums for mass-like fields, means for everything else.
// max_depth, min_depth, max_members, max_nodes : tree parameters
// ─────────────────────────────────────────────────────────────────────────────
static std::shared_ptr<OctreeHandle> build_octree(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<double, py::array::c_style | py::array::forcecast> attrs,
    std::vector<std::string> field_names,
    int max_depth  = 14,
    int min_depth  = 3,
    int max_members = 1,
    int max_nodes  = 100000000)
{
    if (positions.ndim() != 2 || positions.shape(1) != 3)
        throw std::runtime_error("positions must be shape (N,3)");
    if (attrs.ndim() != 2 || attrs.shape(0) != positions.shape(0))
        throw std::runtime_error("attrs must be shape (N,K) matching positions N");
    if ((int)field_names.size() != attrs.shape(1))
        throw std::runtime_error("field_names length must equal attrs.shape(1)");

    auto handle = std::make_shared<OctreeHandle>();
    handle->tree = std::make_unique<MainTree<3>>(
        max_depth, min_depth, max_nodes,
        static_cast<size_t>(max_members),
        /*time_profile=*/false, /*copyOrderedLeaves=*/false,
        HilbertMode::Vector, FillingCurve::Hilbert);

    MainTree<3>& tree = *handle->tree;

    // Register all fields as mean (physically correct for intensive quantities)
    for (int k = 0; k < (int)field_names.size(); ++k)
        tree.registerField(field_names[k], k, FieldReducer::Mean);

    // Build points
    auto pos = positions.unchecked<2>();
    auto att = attrs.unchecked<2>();
    const ssize_t K = att.shape(1);
    std::vector<Point> pts;
    pts.reserve(static_cast<size_t>(pos.shape(0)));
    for (ssize_t i = 0; i < pos.shape(0); ++i) {
        double x = std::max(0.0, std::min(1.0, pos(i, 0)));
        double y = std::max(0.0, std::min(1.0, pos(i, 1)));
        double z = std::max(0.0, std::min(1.0, pos(i, 2)));
        Point pt(x, y, z);
        pt.attrs.resize(static_cast<size_t>(K));
        for (ssize_t k = 0; k < K; ++k)
            pt.attrs[static_cast<size_t>(k)] = att(i, k);
        pts.push_back(std::move(pt));
    }

    tree.buildFromPoints(std::move(pts));
    tree.computeOrderedLeaves();

    // Resolve field pointers by name
    auto res = [&](const std::vector<std::string>& candidates) -> const double* {
        for (const auto& n : candidates) {
            const double* p = resolve_field(tree, n);
            if (p) return p;
        }
        return nullptr;
    };
    handle->rho  = res({"density","rho","Density"});
    handle->pres = res({"pressure","pres","Pressure","p"});
    handle->vx   = res({"vx","velocity_x","Vx"});
    handle->vy   = res({"vy","velocity_y","Vy"});
    handle->vz   = res({"vz","velocity_z","Vz"});
    handle->bx   = res({"bx","Bx","magnetic_x","Bfield_x"});
    handle->by   = res({"by","By","magnetic_y","Bfield_y"});
    handle->bz   = res({"bz","Bz","magnetic_z","Bfield_z"});

    // Compute divergence of velocity (needed for candidate finding + cylinder_average)
    if (handle->vx && handle->vy && handle->vz) {
        // Get field names actually registered for vx/vy/vz
        std::string nvx, nvy, nvz;
        for (const auto& fn : field_names) {
            const double* p = resolve_field(tree, fn);
            if (p == handle->vx) nvx = fn;
            if (p == handle->vy) nvy = fn;
            if (p == handle->vz) nvz = fn;
        }
        if (!nvx.empty() && !nvy.empty() && !nvz.empty())
            handle->div_v = computeLeafDivergence(tree, nvx, nvy, nvz);
    }

    // Compute density gradient components (needed for shock_normal_point)
    if (handle->rho) {
        std::string nrho;
        for (const auto& fn : field_names) {
            const double* p = resolve_field(tree, fn);
            if (p == handle->rho) { nrho = fn; break; }
        }
        if (!nrho.empty()) {
            auto [gx, gy, gz] = computeLeafGradientComponents(tree, nrho);
            handle->grad_x = std::move(gx);
            handle->grad_y = std::move(gy);
            handle->grad_z = std::move(gz);
        }
    }

    return handle;
}

// ─────────────────────────────────────────────────────────────────────────────
// find_candidates_octree
// Returns leaf node indices where div(v) < div_threshold AND |∇ρ|/ρ > grad_threshold.
// Both thresholds are dimensionless (same convention as Octave's gradient indicator).
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int> find_candidates_octree(
    std::shared_ptr<OctreeHandle> handle,
    double div_threshold  = -0.1,
    double grad_threshold = 0.1)
{
    MainTree<3>& tree = *handle->tree;
    const size_t N = tree.ordered_leaf_indices.size();
    if (N == 0) return {};

    // Density gradient magnitude (dimensionless, from Octave's own method)
    const bool check_grad = !handle->grad_x.empty() && handle->rho != nullptr;
    const bool check_div  = !handle->div_v.empty();

    std::vector<int> result;
    for (size_t rank = 0; rank < N; ++rank) {
        const int nidx = static_cast<int>(tree.ordered_leaf_indices[rank]);

        bool ok = true;
        if (check_div) {
            if (handle->div_v[rank] >= div_threshold) ok = false;
        }
        if (ok && check_grad) {
            double gx = handle->grad_x[rank];
            double gy = handle->grad_y[rank];
            double gz = handle->grad_z[rank];
            double rho_v = handle->rho[rank];
            // |∇ρ| / ρ  — dimensionless density gradient
            double scale = std::max(std::abs(rho_v), 1e-30);
            double g = std::sqrt(gx*gx + gy*gy + gz*gz) / scale;
            if (g <= grad_threshold) ok = false;
        }
        if (ok) result.push_back(nidx);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// dict_to_params — same as in bindings.cpp
// ─────────────────────────────────────────────────────────────────────────────
static ShockParams dict_to_params(const py::dict& extra) {
    ShockParams p;
    auto get_int = [&](const char* k, int d) { return extra.contains(k) ? extra[k].cast<int>() : d; };
    auto get_dbl = [&](const char* k, double d) { return extra.contains(k) ? extra[k].cast<double>() : d; };
    auto get_str = [&](const char* k, const std::string& d) -> std::string {
        return extra.contains(k) ? extra[k].cast<std::string>() : d;
    };
    p.method_norm  = (get_str("method_norm","point_gradient") == "average_gradient")
                   ? ShockParams::AVERAGE_GRADIENT : ShockParams::POINT_GRADIENT;
    p.method_plane = (get_str("method_plane","point_field") == "average_field")
                   ? ShockParams::AVERAGE_FIELD : ShockParams::POINT_FIELD;
    if (extra.contains("periodic")) {
        auto per = extra["periodic"].cast<py::list>();
        for (int i = 0; i < 3; ++i) p.periodic[i] = per[i].cast<bool>();
    }
    p.Rgrad      = get_int("Rgrad",      3);
    p.Rcylinder  = get_int("Rcylinder",  3);
    p.line_range = get_int("line_range", 10);
    p.field_ref  = get_int("field_ref",  0);
    p.gamma      = get_dbl("gamma",      5.0/3.0);
    p.shock_ratio= get_dbl("shock_ratio",1.1);
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// results_to_python — same layout as bindings.cpp results_to_python
// ─────────────────────────────────────────────────────────────────────────────
static py::tuple results_to_python(const std::vector<ShockResult>& results) {
    int N = static_cast<int>(results.size());

    auto loc_x       = py::array_t<int>   (N);
    auto loc_y       = py::array_t<int>   (N);
    auto loc_z       = py::array_t<int>   (N);
    auto dir_x       = py::array_t<double>(N);
    auto dir_y       = py::array_t<double>(N);
    auto dir_z       = py::array_t<double>(N);
    auto families    = py::array_t<int>   (N);
    auto speeds      = py::array_t<double>(N);
    auto vA_arr      = py::array_t<double>(N);
    auto MachAlf_arr = py::array_t<double>(N);
    auto Mach_arr    = py::array_t<double>(N);
    auto r_arr       = py::array_t<double>(N);
    auto rho0_arr    = py::array_t<double>(N);
    auto B0_arr      = py::array_t<double>(N);
    auto pmag_arr    = py::array_t<double>(N);
    auto peak_arr    = py::array_t<int>   (N);
    auto flag_arr    = py::array_t<int>   (N);

    auto xb  = loc_x      .mutable_unchecked<1>();
    auto yb  = loc_y      .mutable_unchecked<1>();
    auto zb  = loc_z      .mutable_unchecked<1>();
    auto dxb = dir_x      .mutable_unchecked<1>();
    auto dyb = dir_y      .mutable_unchecked<1>();
    auto dzb = dir_z      .mutable_unchecked<1>();
    auto fb  = families   .mutable_unchecked<1>();
    auto sb  = speeds     .mutable_unchecked<1>();
    auto vAb = vA_arr     .mutable_unchecked<1>();
    auto mAb = MachAlf_arr.mutable_unchecked<1>();
    auto msb = Mach_arr   .mutable_unchecked<1>();
    auto rb  = r_arr      .mutable_unchecked<1>();
    auto r0b = rho0_arr   .mutable_unchecked<1>();
    auto B0b = B0_arr     .mutable_unchecked<1>();
    auto pmb = pmag_arr   .mutable_unchecked<1>();
    auto pkb = peak_arr   .mutable_unchecked<1>();
    auto flb = flag_arr   .mutable_unchecked<1>();

    for (int i = 0; i < N; ++i) {
        const ShockResult& r = results[i];
        xb [i] = r.loc_x;  yb [i] = r.loc_y;  zb [i] = r.loc_z;
        dxb[i] = r.dir_x;  dyb[i] = r.dir_y;  dzb[i] = r.dir_z;
        fb [i] = r.family; sb [i] = r.vs;
        vAb[i] = r.vA;     mAb[i] = r.MachAlf; msb[i] = r.Mach;
        rb [i] = r.r;      r0b[i] = r.rho0;    B0b[i] = r.B0;
        pmb[i] = r.pmag_ratio; pkb[i] = r.peak_flag; flb[i] = r.flag;
    }

    py::list data;
    data.append(loc_x);  data.append(loc_y);  data.append(loc_z);
    data.append(dir_x);  data.append(dir_y);  data.append(dir_z);
    data.append(families);
    data.append(speeds);  data.append(vA_arr);
    data.append(MachAlf_arr); data.append(Mach_arr);
    data.append(r_arr);   data.append(rho0_arr);
    data.append(B0_arr);  data.append(pmag_arr);
    data.append(peak_arr); data.append(flag_arr);

    py::list col_names;
    for (const char* h : {"x","y","z","nx","ny","nz","Family","vs","vA",
                          "MachAlf","Mach","r","rho0","B0","pmag_ratio","peak","FLAG"})
        col_names.append(h);

    return py::make_tuple(data, py::make_tuple(py::none(), col_names));
}

// ─────────────────────────────────────────────────────────────────────────────
// characterise_shocks_octree
// ─────────────────────────────────────────────────────────────────────────────
static py::tuple py_characterise_shocks_octree(
    std::shared_ptr<OctreeHandle> handle,
    std::vector<int> candidate_node_indices,
    py::dict extra,
    bool quiet = false)
{
    if (!handle || !handle->tree)
        throw std::runtime_error("Invalid octree handle");

    MainTree<3>& tree = *handle->tree;
    auto acc = OctaveGridAccessor{handle->tree.get()};
    FieldDataT<OctaveGridAccessor> fields = handle->field_data();

    ShockParams params = dict_to_params(extra);

    // Build CellIndex for each candidate from the node index
    std::vector<ShockResult> results;
    results.reserve(candidate_node_indices.size());
    int n = static_cast<int>(candidate_node_indices.size());

    const int max_depth = tree.maxDepth;

    for (int ci = 0; ci < n; ++ci) {
        int nidx = candidate_node_indices[static_cast<size_t>(ci)];
        if (nidx < 0 || nidx >= static_cast<int>(tree.nodes.size()) || !tree.nodes[nidx].isLeaf) {
            ShockResult bad; bad.flag = 4; bad.loc_x = nidx;
            results.push_back(bad);
            continue;
        }
        CellIndex candidate = acc.make_from_node(nidx);
        ShockResult r = characterise_shock(candidate, fields, params);

        // Overwrite position with finest-level integer grid coords so shock_finder's
        // shocks_data() / plot3D() work: loc * (1/2^max_depth) → physical position.
        {
            const auto& nd = tree.nodes[static_cast<size_t>(nidx)];
            const int shift = max_depth - nd.level;
            r.loc_x = nd.coord.x << shift;
            r.loc_y = nd.coord.y << shift;
            r.loc_z = nd.coord.z << shift;
        }
        results.push_back(r);
        if (!quiet) {
            int print_every = std::max(1, n / 20);
            if (ci % print_every == 0 || ci == n - 1) {
                std::printf("  [octave] %d/%d  (%.0f%%)\n", ci+1, n, 100.0*(ci+1)/n);
                std::fflush(stdout);
            }
        }
    }
    return results_to_python(results);
}

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(shockfindCore_octave, m) {
    m.doc() = "ShockFind backend using Octave's AMR octree instead of a uniform grid";

    py::class_<OctreeHandle, std::shared_ptr<OctreeHandle>>(m, "OctreeHandle")
        .def_property_readonly("num_leaves",
            [](const OctreeHandle& h) { return h.tree->numLeaves(); })
        .def_property_readonly("num_nodes",
            [](const OctreeHandle& h) { return h.tree->nodes.size(); })
        .def_property_readonly("max_depth",
            [](const OctreeHandle& h) { return h.tree->maxDepth; })
        .def_property_readonly("cell_size",
            // Finest-level cell size in [0,1]^3 — use as dx when calling shocks_data()
            [](const OctreeHandle& h) { return 1.0 / (1 << h.tree->maxDepth); });

    m.def("build_octree", &build_octree,
          py::arg("positions"),
          py::arg("attrs"),
          py::arg("field_names"),
          py::arg("max_depth")   = 14,
          py::arg("min_depth")   = 3,
          py::arg("max_members") = 1,
          py::arg("max_nodes")   = 100000000,
          R"(
Build an Octave MainTree<3> from position and attribute arrays.

Parameters
----------
positions  : (N,3) float64 — cell centres normalised to [0,1]^3
attrs      : (N,K) float64 — per-cell attribute columns
field_names: list[str]     — name for each column; recognised names include
             'density','pressure','vx','vy','vz','bx','by','bz'
max_depth, min_depth, max_members, max_nodes : tree construction parameters.
  Use max_members=1 to mirror the RAMSES AMR level structure exactly.

Returns
-------
OctreeHandle — opaque handle; pass to find_candidates / characterise_shocks_octree.
)");

    m.def("find_candidates", &find_candidates_octree,
          py::arg("handle"),
          py::arg("div_threshold")  = -0.1,
          py::arg("grad_threshold") = 0.1,
          R"(
Find shock candidate leaves.

Criteria: div(v) < div_threshold  AND  |∇ρ|/ρ > grad_threshold
Both thresholds are dimensionless.

Returns
-------
list[int] — node indices of candidate leaves in the tree.
)");

    m.def("characterise_shocks_octree", &py_characterise_shocks_octree,
          py::arg("handle"),
          py::arg("candidates"),
          py::arg("extra")  = py::dict(),
          py::arg("quiet")  = false,
          R"(
Characterise shock candidates on an Octave AMR octree.

Parameters
----------
handle     : OctreeHandle from build_octree()
candidates : list[int]   — node indices from find_candidates()
extra      : dict        — same ShockParams dict as characterise_shocks()
quiet      : bool        — suppress progress output

Returns
-------
(data, header) — same 17-array layout as shockfindCore_cpp.characterise_shocks().
  loc_x = leaf node index, loc_y = refinement level, loc_z = 0.
)");
}
