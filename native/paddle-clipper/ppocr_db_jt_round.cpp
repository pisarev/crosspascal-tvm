// Copyright (c) 2026 Yuriy Pisarev. MIT license; see LICENSE.
// Polygon-offset adapter for the PP-OCRv5 DB postprocessor.
// Clipper 6.4.2 is distributed under the Boost Software License.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include "clipper.hpp"

extern "C" __declspec(dllexport) int __cdecl ppocr_db_jt_round_v1(
    const int64_t* xy, int32_t count, double distance,
    int64_t* output_xy, int32_t capacity_points,
    int32_t* output_count, int32_t* output_paths) noexcept {
  if (output_count) *output_count = 0;
  if (output_paths) *output_paths = 0;
  if (!xy || !output_xy || !output_count || !output_paths ||
      count < 3 || count > 1000 || capacity_points < 3 ||
      capacity_points > 65536 || !std::isfinite(distance) ||
      distance <= 0.0 || distance > 4096.0) return 1;
  try {
    ClipperLib::Path polygon;
    polygon.reserve(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
      const int64_t x = xy[2 * i];
      const int64_t y = xy[2 * i + 1];
      if (x < -4096 || x > 4096 || y < -4096 || y > 4096) return 1;
      if (i > 0 && x == xy[2 * (i - 1)] && y == xy[2 * (i - 1) + 1])
        return 1;
      polygon.emplace_back(x, y);
    }
    if (polygon.front() == polygon.back()) return 1;
    if (std::abs(ClipperLib::Area(polygon)) < 0.5) return 1;
    ClipperLib::ClipperOffset offset;
    // EXACT vendor choices: PyclipperOffset.AddPath(box, JT_ROUND,
    // ET_CLOSEDPOLYGON). Default miter_limit=2.0 and arc_tolerance=0.25.
    offset.AddPath(polygon, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    ClipperLib::Paths expanded;
    offset.Execute(expanded, distance);
    *output_paths = static_cast<int32_t>(expanded.size());
    // The Paddle reference skips empty and multi-path offsets. No guessed
    // path union or bounding-box fallback may promote a malformed region.
    if (expanded.size() != 1 || expanded.front().size() < 3) return 2;
    const auto& path = expanded.front();
    if (path.size() > static_cast<size_t>(capacity_points)) return 3;
    for (size_t i = 0; i < path.size(); ++i) {
      if (path[i].X < -16384 || path[i].X > 16384 ||
          path[i].Y < -16384 || path[i].Y > 16384) return 4;
      output_xy[i * 2] = path[i].X;
      output_xy[i * 2 + 1] = path[i].Y;
    }
    *output_count = static_cast<int32_t>(path.size());
    return 0;
  } catch (...) {
    return 5;
  }
}
