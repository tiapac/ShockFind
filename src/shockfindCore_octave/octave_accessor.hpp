#pragma once
// OctaveGridAccessor — drop-in GridAccessor replacement backed by Octave's MainTree<3>.
//
// This header is compiled into shockfindCore_octave.so, which includes tree.hpp
// from the Octave codebase.  It is NOT included by shockfindCore_cpp (uniform grid).
//
// Contract (same interface as GridAccessor):
//   bool      in_bounds(const CellIndex&) const
//   size_t    flat(const CellIndex&)      const
//   CellIndex line_step(const CellIndex&, double l, const Vec3&)        const
//   CellIndex cyl_offset(const CellIndex&, int mu, int mv,
//                         const Vec3& nt1, const Vec3& nt2)              const

#include <cmath>
#include <stdexcept>

// Octave headers (resolved via include path set in CMakeLists.txt)
#include "tree.hpp"

// ShockFind types (CellIndex, Vec3)
#include "../shockfindCore_cpp/types.hpp"

struct OctaveGridAccessor {
    MainTree<3>* tree = nullptr;  // borrowed; owned by the pybind11 binding

    // ── helpers ──────────────────────────────────────────────────────────────

    // Build a CellIndex from an Octave node index.
    CellIndex make_from_node(int node_idx) const {
        const auto& n  = tree->nodes[static_cast<size_t>(node_idx)];
        const Point p  = n.nodeCoord();
        CellIndex c;
        c.leaf_id = node_idx;
        c.level   = n.level;
        c.cx = p.x;
        c.cy = p.y;
        c.cz = p.z;
        return c;
    }

    // Query the tree at physical position (x,y,z) in [0,1]^3.
    // Returns an out-of-bounds CellIndex if the position is outside the domain
    // or the tree has no leaf there.
    CellIndex query_phys(double x, double y, double z) const {
        // Clamp to [0,1] to handle floating-point edge cases
        x = std::max(0.0, std::min(1.0, x));
        y = std::max(0.0, std::min(1.0, y));
        z = std::max(0.0, std::min(1.0, z));
        int idx = tree->queryLeaf(x, y, z);
        if (idx < 0) {
            CellIndex bad; bad.leaf_id = -1; bad.level = -1;
            return bad;
        }
        return make_from_node(idx);
    }

    // ── GridAccessor interface ────────────────────────────────────────────────

    bool in_bounds(const CellIndex& c) const {
        if (c.leaf_id < 0 || c.leaf_id >= static_cast<int>(tree->nodes.size()))
            return false;
        return tree->nodes[static_cast<size_t>(c.leaf_id)].isLeaf;
    }

    // Hilbert rank — index into leafFieldData arrays.
    size_t flat(const CellIndex& c) const {
        int nid = tree->nodes[static_cast<size_t>(c.leaf_id)].id;
        if (nid < 0 || nid >= static_cast<int>(tree->rank_by_id.size()))
            throw std::out_of_range("OctaveGridAccessor::flat: node id out of range");
        int r = tree->rank_by_id[static_cast<size_t>(nid)];
        if (r < 0)
            throw std::out_of_range("OctaveGridAccessor::flat: leaf has no Hilbert rank");
        return static_cast<size_t>(r);
    }

    // Step l "cells" along direction dir from origin.
    // The cell size used for the step is the local h at origin's level,
    // so the sampling density adapts to the local AMR resolution.
    // Periodic wrapping: wrap each coordinate modulo 1.
    CellIndex line_step(const CellIndex& origin, double l, const Vec3& dir) const {
        const double h = 1.0 / static_cast<double>(1 << origin.level);
        double x = origin.cx + l * h * dir.x;
        double y = origin.cy + l * h * dir.y;
        double z = origin.cz + l * h * dir.z;
        // Periodic wrap: fmod brings into (-1,1), adding 1 and fmod again → [0,1)
        auto wrap = [](double v) {
            v = std::fmod(v, 1.0);
            if (v < 0.0) v += 1.0;
            return v;
        };
        return query_phys(wrap(x), wrap(y), wrap(z));
    }

    // Cylinder cross-section offset.
    // mu steps along nt2, mv steps along nt1, both in units of the local h.
    // No periodic wrap: the cylinder is small and should not cross domain boundaries.
    CellIndex cyl_offset(const CellIndex& centre, int mu, int mv,
                         const Vec3& nt1, const Vec3& nt2) const {
        const double h = 1.0 / static_cast<double>(1 << centre.level);
        double x = centre.cx + (mu * nt2.x + mv * nt1.x) * h;
        double y = centre.cy + (mu * nt2.y + mv * nt1.y) * h;
        double z = centre.cz + (mu * nt2.z + mv * nt1.z) * h;
        return query_phys(x, y, z);
    }
};
