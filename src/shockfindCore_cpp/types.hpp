#pragma once
#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Vec3: minimal 3-D double vector used throughout the shock analysis
// ─────────────────────────────────────────────────────────────────────────────
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3 operator+(const Vec3& o)  const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o)  const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)       const { return {x*s,   y*s,   z*s  }; }
    Vec3 operator/(double s)       const { return {x/s,   y/s,   z/s  }; }
    Vec3& operator+=(const Vec3& o)      { x+=o.x; y+=o.y; z+=o.z; return *this; }

    double dot  (const Vec3& o)    const { return x*o.x + y*o.y + z*o.z; }
    Vec3   cross(const Vec3& o)    const {
        return { y*o.z - z*o.y,
                 z*o.x - x*o.z,
                 x*o.y - y*o.x };
    }
    double norm2() const { return x*x + y*y + z*z; }
    double norm()  const { return std::sqrt(norm2()); }
    Vec3 normalized() const {
        double n = norm();
        return (n > 0.0) ? Vec3{x/n, y/n, z/n} : Vec3{0,0,0};
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

// ─────────────────────────────────────────────────────────────────────────────
// CellIndex — extensible cell address.
//
// On a uniform Cartesian grid this is simply (i, j, k).
// AMR octree fields default to -1 so all existing GridAccessor code paths
// remain unaffected (leaf_id == -1 means uniform-grid mode).
// ─────────────────────────────────────────────────────────────────────────────
struct CellIndex {
    int i = 0, j = 0, k = 0;
    // AMR octree extension (leaf_id == -1 → uniform grid path)
    int    leaf_id = -1;
    int    level   = -1;
    double cx = 0.0, cy = 0.0, cz = 0.0;  // physical centre in [0,1]^3
};

// ─────────────────────────────────────────────────────────────────────────────
// GridAccessor — the single place that knows how cells map to memory.
//
// All field accesses, index validity checks, and geometric walks go through
// here.  Swapping this struct for an octree implementation is the only change
// needed to extend the algorithm to AMR/octree grids.
// ─────────────────────────────────────────────────────────────────────────────
struct GridShape { int nx = 0, ny = 0, nz = 0; };

struct GridAccessor {
    GridShape shape;

    bool in_bounds(const CellIndex& c) const {
        return c.i >= 0 && c.i < shape.nx
            && c.j >= 0 && c.j < shape.ny
            && c.k >= 0 && c.k < shape.nz;
    }

    // Row-major (C-order) flat index: matches numpy's default layout.
    size_t flat(const CellIndex& c) const {
        return static_cast<size_t>(c.i) * shape.ny * shape.nz
             + static_cast<size_t>(c.j) * shape.nz
             + static_cast<size_t>(c.k);
    }

    // Step l cells along direction dir from origin.
    // Rounds to nearest integer cell — same as the Python implementation.
    CellIndex line_step(const CellIndex& origin, double l, const Vec3& dir) const {
        return {
            static_cast<int>(std::round(origin.i + l * dir.x)),
            static_cast<int>(std::round(origin.j + l * dir.y)),
            static_cast<int>(std::round(origin.k + l * dir.z))
        };
    }

    // Cylinder cross-section offset.  mu steps along nt2, mv steps along nt1,
    // both in grid-cell units.  Matches the Python cylinder() loop:
    //   cpx = round(a + mux*nt2x + muy*nt1x)
    CellIndex cyl_offset(const CellIndex& centre,
                         int mu, int mv,
                         const Vec3& nt1, const Vec3& nt2) const {
        return {
            static_cast<int>(std::round(centre.i + mu*nt2.x + mv*nt1.x)),
            static_cast<int>(std::round(centre.j + mu*nt2.y + mv*nt1.y)),
            static_cast<int>(std::round(centre.k + mu*nt2.z + mv*nt1.z))
        };
    }
};

// Typed field access through the accessor — works with any GridAccessor-like type.
// The accessor must provide in_bounds(CellIndex) and flat(CellIndex).
template<typename T, typename G>
inline T field_at(const T* data, const G& g, const CellIndex& c) {
    if (!g.in_bounds(c)) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "field_at: cell out of bounds (leaf_id=%d, i=%d, j=%d, k=%d)",
            c.leaf_id, c.i, c.j, c.k);
        throw std::out_of_range(msg);
    }
    return data[g.flat(c)];
}

// ─────────────────────────────────────────────────────────────────────────────
// FieldDataT<G> — all simulation fields needed by the shock algorithm.
// Templated on the accessor type G so the same struct layout works for both
// the uniform-grid (G=GridAccessor) and AMR-octree (G=OctaveGridAccessor) paths.
// Pointers are non-owning (borrowed from the Python/NumPy arrays or tree data).
// ─────────────────────────────────────────────────────────────────────────────
template<typename G = GridAccessor>
struct FieldDataT {
    const double* rho    = nullptr;
    const double* pres   = nullptr;
    const double* vx     = nullptr;
    const double* vy     = nullptr;
    const double* vz     = nullptr;
    const double* bx     = nullptr;
    const double* by     = nullptr;
    const double* bz     = nullptr;
    const double* div    = nullptr;   // velocity divergence (pre-computed)
    const double* grad_x = nullptr;  // ∇ρ components
    const double* grad_y = nullptr;
    const double* grad_z = nullptr;
    G grid;
};

// Backward-compatible alias — all existing code that uses FieldData unchanged.
using FieldData = FieldDataT<GridAccessor>;

// ─────────────────────────────────────────────────────────────────────────────
// ShockParams — mirrors the Python `extra` dict passed to characterise_shocks.
// ─────────────────────────────────────────────────────────────────────────────
struct ShockParams {
    enum NormMethod  { POINT_GRADIENT  = 0, AVERAGE_GRADIENT = 1 };
    enum PlaneMethod { POINT_FIELD     = 0, AVERAGE_FIELD    = 1 };

    NormMethod  method_norm  = POINT_GRADIENT;
    PlaneMethod method_plane = POINT_FIELD;
    bool        periodic[3]  = {true, true, true};
    int         Rgrad        = 3;
    int         Rcylinder    = 3;
    double      gamma        = 5.0 / 3.0;
    int         line_range   = 10;
    int         field_ref    = 0;
    double      shock_ratio  = 1.1;
    int         offset[3]    = {0, 0, 0};
    bool        hydro_only   = false;  // no B field: use sonic-only classification
};

// ─────────────────────────────────────────────────────────────────────────────
// LineProfile — 1-D profiles produced by cylinder_average.
// Each vector has the same length L = number of valid points on the shock line.
// ─────────────────────────────────────────────────────────────────────────────
struct LineProfile {
    std::vector<double> line;   // fractional positions along line (−range … +range)
    std::vector<double> rho;    // log-averaged density
    std::vector<double> pres;   // log-averaged pressure
    std::vector<double> vp;     // velocity ∥ shock normal
    std::vector<double> vt1;    // velocity ⊥ (along nt1)
    std::vector<double> vt2;    // velocity ⊥ (along nt2)
    std::vector<double> bp;     // B ∥ shock normal
    std::vector<double> bt1;    // B ⊥ (along nt1)
    std::vector<double> bt2;    // B ⊥ (along nt2)
    std::vector<double> conv;   // convergence = −div(v)
};

// ─────────────────────────────────────────────────────────────────────────────
// ShockResult — all output quantities for one shock candidate.
// Matches the 17-column layout produced by the Python characterise_shocks.
// ─────────────────────────────────────────────────────────────────────────────
struct ShockResult {
    // Position (grid-cell indices, with offset applied)
    int    loc_x = 0, loc_y = 0, loc_z = 0;
    // Shock normal direction
    double dir_x = 0.0, dir_y = 0.0, dir_z = 0.0;
    // Classification
    int    family     = 0;    // 12 = fast, 34 = slow, 0 = unclassified
    // Physical quantities
    double vs         = 0.0;  // shock speed (cm/s)
    double vA         = 0.0;  // pre-shock Alfvén speed
    double MachAlf    = 0.0;  // Alfvénic Mach number
    double Mach       = 0.0;  // sonic Mach number
    double r          = 0.0;  // density compression ratio ρ_pst / ρ_pre
    double rho0       = 0.0;  // pre-shock density
    double B0         = 0.0;  // pre-shock |B|
    double pmag_ratio = 0.0;  // post/pre magnetic pressure
    // Quality flags
    int    peak_flag  = 0;    // 1 if convergence peak is centred
    int    flag       = 0;    // 0=ok, 1=density threshold, 2=edge, 3=B/Mach inconsistent, 4=no conv
};
