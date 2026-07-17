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
#ifdef _OPENMP
#include <omp.h>
#endif

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

    tree.buildFromPoints(std::move(pts));  // includes computeOrderedLeaves() internally

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
    auto level_arr   = py::array_t<int>   (N);
    auto volume_arr  = py::array_t<double>(N);

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
    auto lvb = level_arr  .mutable_unchecked<1>();
    auto vob = volume_arr .mutable_unchecked<1>();

    for (int i = 0; i < N; ++i) {
        const ShockResult& r = results[i];
        xb [i] = r.loc_x;  yb [i] = r.loc_y;  zb [i] = r.loc_z;
        dxb[i] = r.dir_x;  dyb[i] = r.dir_y;  dzb[i] = r.dir_z;
        fb [i] = r.family; sb [i] = r.vs;
        vAb[i] = r.vA;     mAb[i] = r.MachAlf; msb[i] = r.Mach;
        rb [i] = r.r;      r0b[i] = r.rho0;    B0b[i] = r.B0;
        pmb[i] = r.pmag_ratio; pkb[i] = r.peak_flag; flb[i] = r.flag;
        lvb[i] = r.level;  vob[i] = r.volume;
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
    data.append(level_arr); data.append(volume_arr);

    py::list col_names;
    for (const char* h : {"x","y","z","nx","ny","nz","Family","vs","vA",
                          "MachAlf","Mach","r","rho0","B0","pmag_ratio","peak","FLAG",
                          "level","volume"})
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

    const int n         = static_cast<int>(candidate_node_indices.size());
    const int max_depth = tree.maxDepth;

    // Pre-allocate so parallel writes land in separate slots (no push_back race).
    std::vector<ShockResult> results(static_cast<size_t>(n));

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int ci = 0; ci < n; ++ci) {
        int nidx = candidate_node_indices[static_cast<size_t>(ci)];
        ShockResult r{};
        if (nidx < 0 || nidx >= static_cast<int>(tree.nodes.size()) || !tree.nodes[nidx].isLeaf) {
            r.flag = 4; r.loc_x = nidx;
        } else {
            CellIndex candidate = acc.make_from_node(nidx);
            r = characterise_shock(candidate, fields, params);
            // Overwrite position with finest-level integer grid coords so shock_finder's
            // shocks_data() / plot3D() work: loc * (1/2^max_depth) → physical position.
            const auto& nd = tree.nodes[static_cast<size_t>(nidx)];
            const int shift = max_depth - nd.level;
            r.loc_x = nd.coord.x << shift;
            r.loc_y = nd.coord.y << shift;
            r.loc_z = nd.coord.z << shift;
            // Cell volume from the AMR level: a level-L cell has edge length
            // 1/2^L in normalised [0,1]^3 units, so volume = (1/2^L)^3.
            r.level = nd.level;
            const double dx_level = 1.0 / static_cast<double>(1u << nd.level);
            r.volume = dx_level * dx_level * dx_level;
        }
        results[static_cast<size_t>(ci)] = r;

#ifndef _OPENMP
        // Serial-only progress (ordering is meaningful; skip when multithreaded).
        if (!quiet) {
            const int print_every = std::max(1, n / 20);
            if (ci % print_every == 0 || ci == n - 1) {
                std::printf("  [octave] %d/%d  (%.0f%%)\n", ci+1, n, 100.0*(ci+1)/n);
                std::fflush(stdout);
            }
        }
#endif
    }

#ifdef _OPENMP
    if (!quiet) {
        std::printf("  [octave] %d/%d  (100%%) [%d OpenMP threads]\n",
                    n, n, omp_get_max_threads());
        std::fflush(stdout);
    }
#endif

    return results_to_python(results);
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_gradient_of
// Compute ∇f on the octree for an arbitrary per-leaf field array f.
// f must be a 1-D float64 array of length == handle.num_leaves (Hilbert order).
// Returns (gx, gy, gz) — same layout, same units as the pre-computed grad_rho_*.
// ─────────────────────────────────────────────────────────────────────────────
static py::tuple compute_gradient_of(
    std::shared_ptr<OctreeHandle> handle,
    py::array_t<double, py::array::c_style | py::array::forcecast> field_arr)
{
    MainTree<3>& tree = *handle->tree;
    const size_t n = tree.ordered_leaf_indices.size();
    if (static_cast<size_t>(field_arr.shape(0)) != n)
        throw std::runtime_error("field_arr length must equal handle.num_leaves");

    const std::string scratch = "__grad_tmp__";
    const double* raw = field_arr.data();
    tree.leafFieldData[scratch].assign(raw, raw + n);

    auto [gx, gy, gz] = computeLeafGradientComponents(tree, scratch);
    tree.leafFieldData.erase(scratch);

    auto to_arr = [](std::vector<double>& v) {
        auto a = py::array_t<double>(v.size());
        std::copy(v.begin(), v.end(), a.mutable_data());
        return a;
    };
    return py::make_tuple(to_arr(gx), to_arr(gy), to_arr(gz));
}

// ─────────────────────────────────────────────────────────────────────────────
// save_octree / load_octree — HDF5 persistence for the built OctreeHandle.
// save_octree writes all field data (AMR structure + per-leaf fields + adjacency).
// load_octree reads it back and recomputes div_v / grad_rho so the handle is
// immediately usable for candidate finding without a RAMSES re-load.
// Both require the shockfindCore_octave .so to have been built with USE_HDF5.
// ─────────────────────────────────────────────────────────────────────────────
static void py_save_octree(std::shared_ptr<OctreeHandle> handle, const std::string& filename)
{
#ifdef USE_HDF5
    if (!handle || !handle->tree)
        throw std::runtime_error("Invalid octree handle");
    std::vector<std::string> present;
    for (const char* f : {"density","pressure","vx","vy","vz","bx","by","bz"}) {
        auto it = handle->tree->leafFieldData.find(f);
        if (it != handle->tree->leafFieldData.end() && !it->second.empty())
            present.push_back(f);
    }
    handle->tree->exportAllHDF5(filename, present,
                                 /*include_rank=*/true,
                                 /*include_filling_curve=*/true,
                                 /*include_morton=*/false);
#else
    (void)handle; (void)filename;
    throw std::runtime_error("save_octree: HDF5 support not compiled in. Rebuild with USE_HDF5.");
#endif
}

static std::shared_ptr<OctreeHandle> py_load_octree(const std::string& filename)
{
#ifdef USE_HDF5
    auto handle = std::make_shared<OctreeHandle>();
    handle->tree = std::make_unique<MainTree<3>>(MainTree<3>::loadAllHDF5(filename));
    MainTree<3>& tree = *handle->tree;

    auto res = [&](std::initializer_list<const char*> names) -> const double* {
        for (const char* n : names) {
            auto it = tree.leafFieldData.find(n);
            if (it != tree.leafFieldData.end() && !it->second.empty())
                return it->second.data();
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

    // Find the registered name for a given pointer into leafFieldData
    auto find_name = [&](const double* ptr) -> std::string {
        for (const auto& kv : tree.leafFieldData)
            if (!kv.second.empty() && kv.second.data() == ptr) return kv.first;
        return "";
    };

    // div(v) and ∇ρ are derived — recompute after load
    if (handle->vx && handle->vy && handle->vz) {
        std::string nvx = find_name(handle->vx);
        std::string nvy = find_name(handle->vy);
        std::string nvz = find_name(handle->vz);
        if (!nvx.empty() && !nvy.empty() && !nvz.empty())
            handle->div_v = computeLeafDivergence(tree, nvx, nvy, nvz);
    }
    if (handle->rho) {
        std::string nrho = find_name(handle->rho);
        if (!nrho.empty()) {
            auto [gx, gy, gz] = computeLeafGradientComponents(tree, nrho);
            handle->grad_x = std::move(gx);
            handle->grad_y = std::move(gy);
            handle->grad_z = std::move(gz);
        }
    }
    return handle;
#else
    (void)filename;
    throw std::runtime_error("load_octree: HDF5 support not compiled in. Rebuild with USE_HDF5.");
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────

// Helper: copy a raw pointer array into a new numpy array (safe; avoids lifetime issues).
static py::array_t<double> ptr_to_arr(const double* ptr, size_t n) {
    if (!ptr || n == 0) return py::array_t<double>(0);
    auto a = py::array_t<double>(n);
    std::copy(ptr, ptr + n, a.mutable_data());
    return a;
}

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
            [](const OctreeHandle& h) { return 1.0 / (1 << h.tree->maxDepth); })
        // ── Per-leaf field arrays (Hilbert order, length == num_leaves) ──────
        .def_property_readonly("rho_arr",
            [](const OctreeHandle& h) {
                return ptr_to_arr(h.rho, h.tree->ordered_leaf_indices.size()); })
        .def_property_readonly("pres_arr",
            [](const OctreeHandle& h) {
                return ptr_to_arr(h.pres, h.tree->ordered_leaf_indices.size()); })
        .def_property_readonly("vx_arr",
            [](const OctreeHandle& h) {
                return ptr_to_arr(h.vx, h.tree->ordered_leaf_indices.size()); })
        .def_property_readonly("vy_arr",
            [](const OctreeHandle& h) {
                return ptr_to_arr(h.vy, h.tree->ordered_leaf_indices.size()); })
        .def_property_readonly("vz_arr",
            [](const OctreeHandle& h) {
                return ptr_to_arr(h.vz, h.tree->ordered_leaf_indices.size()); })
        .def_property_readonly("bx_arr",
            [](const OctreeHandle& h) {
                return ptr_to_arr(h.bx, h.tree->ordered_leaf_indices.size()); })
        .def_property_readonly("by_arr",
            [](const OctreeHandle& h) {
                return ptr_to_arr(h.by, h.tree->ordered_leaf_indices.size()); })
        .def_property_readonly("bz_arr",
            [](const OctreeHandle& h) {
                return ptr_to_arr(h.bz, h.tree->ordered_leaf_indices.size()); })
        // ── Pre-computed derived arrays ──────────────────────────────────────
        .def_property_readonly("div_v_arr",
            [](const OctreeHandle& h) {
                auto a = py::array_t<double>(h.div_v.size());
                std::copy(h.div_v.begin(), h.div_v.end(), a.mutable_data());
                return a; })
        .def_property_readonly("grad_rho_x",
            [](const OctreeHandle& h) {
                auto a = py::array_t<double>(h.grad_x.size());
                std::copy(h.grad_x.begin(), h.grad_x.end(), a.mutable_data());
                return a; })
        .def_property_readonly("grad_rho_y",
            [](const OctreeHandle& h) {
                auto a = py::array_t<double>(h.grad_y.size());
                std::copy(h.grad_y.begin(), h.grad_y.end(), a.mutable_data());
                return a; })
        .def_property_readonly("grad_rho_z",
            [](const OctreeHandle& h) {
                auto a = py::array_t<double>(h.grad_z.size());
                std::copy(h.grad_z.begin(), h.grad_z.end(), a.mutable_data());
                return a; })
        // ── Leaf topology ────────────────────────────────────────────────────
        .def_property_readonly("leaf_node_indices",
            // Node indices in Hilbert order (maps Hilbert rank → node index in tree.nodes).
            [](const OctreeHandle& h) {
                const auto& v = h.tree->ordered_leaf_indices;
                auto a = py::array_t<int>(v.size());
                auto buf = a.mutable_unchecked<1>();
                for (size_t i = 0; i < v.size(); ++i)
                    buf[static_cast<py::ssize_t>(i)] = static_cast<int>(v[i]);
                return a; })
        .def_property_readonly("leaf_levels",
            // AMR level of each leaf in Hilbert order.
            [](const OctreeHandle& h) {
                const auto& tree = *h.tree;
                const size_t n = tree.ordered_leaf_indices.size();
                auto a = py::array_t<int>(n);
                auto buf = a.mutable_unchecked<1>();
                for (size_t i = 0; i < n; ++i)
                    buf[static_cast<py::ssize_t>(i)] =
                        tree.nodes[tree.ordered_leaf_indices[i]].level;
                return a; });

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
Find shock candidate leaves (raw/normalised mode).

Criteria: div(v) < div_threshold  AND  |∇ρ|/ρ > grad_threshold
Both thresholds are dimensionless — suitable for unit-normalised fields.
For physical threshold computation from vshock_min/rhomean/mach_min use
the Python-level _find_candidates_full() in octree_shockfind_pipeline.

Returns
-------
list[int] — node indices of candidate leaves in the tree.
)");

    m.def("compute_gradient_of", &compute_gradient_of,
          py::arg("handle"),
          py::arg("field_arr"),
          R"(
Compute the AMR gradient of an arbitrary per-leaf scalar field.

field_arr : 1-D float64 array of length handle.num_leaves, in Hilbert order.
Returns (gx, gy, gz) — gradient components in normalised [0,1]^3 coordinates,
parallel to the handle's other derived arrays (div_v_arr, grad_rho_*).
)");

    m.def("save_octree", &py_save_octree,
          py::arg("handle"),
          py::arg("filename"),
          R"(
Save the built octree to an HDF5 file.

Persists: AMR node structure, per-leaf field data (density, pressure, vx/vy/vz, bx/by/bz),
leaf ordering, and face-neighbour adjacency.  div_v and grad_rho are NOT saved (they
are recomputed cheaply on load_octree()).  Requires USE_HDF5 at build time.

Parameters
----------
handle   : OctreeHandle from build_octree()
filename : output HDF5 path  (e.g. "run/octree.h5")
)");

    m.def("load_octree", &py_load_octree,
          py::arg("filename"),
          R"(
Load an octree from an HDF5 file previously written by save_octree().

Recomputes div_v and grad_rho after loading so the returned handle is ready for
_find_candidates_full() / characterise_shocks_octree() without any RAMSES re-load.
Requires USE_HDF5 at build time.

Returns
-------
OctreeHandle — identical to what build_octree() would have returned.
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
(data, header) — same 17-array layout as shockfindCore_cpp.characterise_shocks(),
  plus two AMR-only trailing columns:
    level  = AMR refinement level of the candidate's leaf cell
    volume = cell volume in normalised [0,1]^3 units, (1/2^level)^3
)");
}
