// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_EMBREE_H
#define PBRT_CPU_EMBREE_H

#include <pbrt/pbrt.h>

#include <pbrt/cpu/primitive.h>
#include <pbrt/util/parallel.h>

#include <embree4/rtcore.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace pbrt {

// EmbreeAccelerationStructure - Uses Embree for ray tracing
class EmbreeAccelerationStructure {
  public:
    struct GeometryData {
        int primitiveIndex;
        Primitive primitive;
        RTCGeometry geometry = nullptr;
        unsigned int geomID = RTC_INVALID_GEOMETRY_ID;
        bool nativeTriangle = false;
    };

    EmbreeAccelerationStructure(const std::vector<Primitive> &prims);
    ~EmbreeAccelerationStructure();

    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    std::vector<GeometryData> geometries;
    std::unordered_map<unsigned int, size_t> geomIDToIndex;
    RTCDevice device = nullptr;
    RTCScene scene = nullptr;
};

// EmbreeAggregate Definition
class EmbreeAggregate {
  public:
    // EmbreeAggregate Public Methods
    EmbreeAggregate(std::vector<Primitive> prims);
    ~EmbreeAggregate();

    static EmbreeAggregate *Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters);

    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    // EmbreeAggregate Private Members
    std::vector<Primitive> primitives;
    std::unique_ptr<EmbreeAccelerationStructure> accelStructure;
    Bounds3f bounds;
};

}  // namespace pbrt

#endif  // PBRT_CPU_EMBREE_H
