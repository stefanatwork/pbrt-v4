// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cpu/embree.h>

#include <pbrt/cpu/primitive.h>
#include <pbrt/paramdict.h>
#include <pbrt/shapes.h>
#include <pbrt/util/error.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <embree4/rtcore_ray.h>
#include <embree4/rtcore_scene.h>

#include <cassert>
#include <cmath>

namespace pbrt {

STAT_COUNTER("Embree/Ray shape tests", rayShapeTests);

// Error callback for Embree
static void ErrorCallback(void *userPtr, enum RTCError error, const char *str) {
    if (error == RTC_ERROR_NONE)
        return;
    fprintf(stderr, "Embree error %d: %s\n", error, str);
}

static const Triangle *GetTriangleShape(const Primitive &prim) {
    if (prim.Is<GeometricPrimitive>()) {
        const GeometricPrimitive *gp = prim.Cast<GeometricPrimitive>();
        const Shape &shape = gp->GetShape();
        if (shape.Is<Triangle>())
            return shape.Cast<Triangle>();
    } else if (prim.Is<SimplePrimitive>()) {
        const SimplePrimitive *sp = prim.Cast<SimplePrimitive>();
        const Shape &shape = sp->GetShape();
        if (shape.Is<Triangle>())
            return shape.Cast<Triangle>();
    }
    return nullptr;
}

static const Curve *GetCurveShape(const Primitive &prim) {
    if (prim.Is<GeometricPrimitive>()) {
        const GeometricPrimitive *gp = prim.Cast<GeometricPrimitive>();
        const Shape &shape = gp->GetShape();
        if (shape.Is<Curve>())
            return shape.Cast<Curve>();
    } else if (prim.Is<SimplePrimitive>()) {
        const SimplePrimitive *sp = prim.Cast<SimplePrimitive>();
        const Shape &shape = sp->GetShape();
        if (shape.Is<Curve>())
            return shape.Cast<Curve>();
    }
    return nullptr;
}

static Ray RayFromRTCRay(const RTCRay &ray) {
    Ray r;
    r.o = Point3f(ray.org_x, ray.org_y, ray.org_z);
    r.d = Vector3f(ray.dir_x, ray.dir_y, ray.dir_z);
    r.time = ray.time;
    r.medium = nullptr;
    return r;
}

static bool TriangleHitPassesAlpha(const Primitive &prim, const RTCRay &rtcRay,
                                   const RTCHit &rtcHit) {
    if (!prim.Is<GeometricPrimitive>())
        return true;

    const GeometricPrimitive *gp = prim.Cast<GeometricPrimitive>();
    const Shape &shape = gp->GetShape();
    if (!shape.Is<Triangle>())
        return true;

    const Triangle *tri = shape.Cast<Triangle>();
    const TriangleMesh *mesh = tri->GetMeshForEmbree();
    if (!mesh)
        return true;

    int triIdx = tri->GetTriangleIndex();
    const int *v = &mesh->vertexIndices[3 * triIdx];
    Point3f p0 = mesh->p[v[0]];
    Point3f p1 = mesh->p[v[1]];
    Point3f p2 = mesh->p[v[2]];

    TriangleIntersection ti;
    ti.b1 = rtcHit.u;
    ti.b2 = rtcHit.v;
    ti.b0 = 1 - ti.b1 - ti.b2;
    Point3f pHit = ti.b0 * p0 + ti.b1 * p1 + ti.b2 * p2;

    Ray ray = RayFromRTCRay(rtcRay);
    Float d2 = LengthSquared(ray.d);
    Float tHit = (d2 > 0) ? Dot(pHit - ray.o, ray.d) / d2 : 0;
    ti.t = tHit;
    SurfaceInteraction intr =
        Triangle::InteractionFromIntersection(mesh, triIdx, ti, ray.time, -ray.d);
    ShapeIntersection si{intr, tHit};
    return gp->PassesAlphaTest(si, ray);
}

static void UserPrimitiveBoundsFunc(const RTCBoundsFunctionArguments *args) {
    const EmbreeAccelerationStructure::GeometryData *geom =
        static_cast<const EmbreeAccelerationStructure::GeometryData *>(args->geometryUserPtr);
    Bounds3f b = geom->primitive.Bounds();
    RTCBounds *bounds = args->bounds_o;
    bounds->lower_x = b.pMin.x;
    bounds->lower_y = b.pMin.y;
    bounds->lower_z = b.pMin.z;
    bounds->upper_x = b.pMax.x;
    bounds->upper_y = b.pMax.y;
    bounds->upper_z = b.pMax.z;
}

// User primitive intersection callback
static void UserPrimitiveIntersectFunc(const RTCIntersectFunctionNArguments *args) {
    assert(args->N == 1);
    if (!args->valid[0])
        return;

    RTCRayHit *rayhit = (RTCRayHit *)args->rayhit;
    const EmbreeAccelerationStructure::GeometryData *geom =
        static_cast<const EmbreeAccelerationStructure::GeometryData *>(args->geometryUserPtr);

    Ray r;
    r.o = Point3f(rayhit->ray.org_x, rayhit->ray.org_y, rayhit->ray.org_z);
    r.d = Vector3f(rayhit->ray.dir_x, rayhit->ray.dir_y, rayhit->ray.dir_z);
    r.time = rayhit->ray.time;
    r.medium = nullptr;

    pstd::optional<ShapeIntersection> si = geom->primitive.Intersect(r, rayhit->ray.tfar);
    if (!si)
        return;

    args->valid[0] = -1;
    rayhit->ray.tfar = si->tHit;
    rayhit->hit.geomID = args->geomID;
    rayhit->hit.primID = args->primID;
    rayhit->hit.Ng_x = si->intr.n.x;
    rayhit->hit.Ng_y = si->intr.n.y;
    rayhit->hit.Ng_z = si->intr.n.z;
}

// User primitive occlusion callback
static void UserPrimitiveOccludedFunc(const RTCOccludedFunctionNArguments *args) {
    assert(args->N == 1);
    if (!args->valid[0])
        return;

    RTCRay *ray = (RTCRay *)args->ray;
    const EmbreeAccelerationStructure::GeometryData *geom =
        static_cast<const EmbreeAccelerationStructure::GeometryData *>(args->geometryUserPtr);

    Ray r;
    r.o = Point3f(ray->org_x, ray->org_y, ray->org_z);
    r.d = Vector3f(ray->dir_x, ray->dir_y, ray->dir_z);
    r.time = ray->time;
    r.medium = nullptr;

    if (geom->primitive.IntersectP(r, ray->tfar)) {
        args->valid[0] = -1;
        ray->tfar = -Infinity;
    }
}

static void TriangleOccludedFilterFunc(const RTCFilterFunctionNArguments *args) {
    assert(args->N == 1);
    if (!args->valid[0])
        return;

    RTCRay ray = rtcGetRayFromRayN(args->ray, args->N, 0);
    RTCHit hit = rtcGetHitFromHitN(args->hit, args->N, 0);
    const EmbreeAccelerationStructure::GeometryData *geom =
        static_cast<const EmbreeAccelerationStructure::GeometryData *>(args->geometryUserPtr);
    if (!geom) {
        args->valid[0] = 0;
        return;
    }

    if (!TriangleHitPassesAlpha(geom->primitive, ray, hit))
        args->valid[0] = 0;
}

// EmbreeAccelerationStructure Implementation
EmbreeAccelerationStructure::EmbreeAccelerationStructure(
    const std::vector<Primitive> &prims) {
    if (prims.empty()) {
        device = nullptr;
        scene = nullptr;
        return;
    }

    // Create Embree device
    device = rtcNewDevice(nullptr);
    if (!device) {
        ErrorExit("Failed to create Embree device");
    }
    rtcSetDeviceErrorFunction(device, ErrorCallback, nullptr);

    // Create Embree scene
    scene = rtcNewScene(device);
    if (!scene) {
        ErrorExit("Failed to create Embree scene");
    }
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);

    // Add primitives to scene, using native Embree geometries where possible
    geometries.reserve(prims.size());

    for (size_t i = 0; i < prims.size(); ++i) {
        geometries.push_back({});
        GeometryData &geom = geometries.back();
        geom.primitiveIndex = int(i);
        geom.primitive = prims[i];

        RTCGeometry rtcGeom = nullptr;
        const Triangle *tri = GetTriangleShape(geom.primitive);
        const Curve *curve = GetCurveShape(geom.primitive);
        if (tri) {
            // Create native triangle geometry for Triangle shapes
            rtcGeom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
            if (!rtcGeom) {
                ErrorExit("Failed to create Embree triangle geometry");
            }

            const TriangleMesh *mesh = tri->GetMeshForEmbree();
            if (!mesh) {
                ErrorExit("Triangle has no mesh");
            }

            int triIdx = tri->GetTriangleIndex();
            const int *v = &mesh->vertexIndices[3 * triIdx];
            Point3f p0 = mesh->p[v[0]];
            Point3f p1 = mesh->p[v[1]];
            Point3f p2 = mesh->p[v[2]];

            float *vertices = (float *)rtcSetNewGeometryBuffer(
                rtcGeom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                3 * sizeof(float), 3);
            vertices[0] = p0.x;
            vertices[1] = p0.y;
            vertices[2] = p0.z;
            vertices[3] = p1.x;
            vertices[4] = p1.y;
            vertices[5] = p1.z;
            vertices[6] = p2.x;
            vertices[7] = p2.y;
            vertices[8] = p2.z;

            unsigned int *indices = (unsigned int *)rtcSetNewGeometryBuffer(
                rtcGeom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                3 * sizeof(unsigned int), 1);
            indices[0] = 0;
            indices[1] = 1;
            indices[2] = 2;

            rtcSetGeometryUserData(rtcGeom, (void *)&geom);
            rtcSetGeometryOccludedFilterFunction(rtcGeom, TriangleOccludedFilterFunc);

            geom.nativeTriangle = true;
        } else if (curve) {
            RTCGeometryType curveType = RTC_GEOMETRY_TYPE_ROUND_BEZIER_CURVE;
            switch (curve->GetType()) {
            case CurveType::Cylinder:
                // pbrt "cylinder" curves use a flat curve intersection and only
                // apply a cylindrical normal model for shading.
                curveType = RTC_GEOMETRY_TYPE_FLAT_BEZIER_CURVE;
                break;
            case CurveType::Flat:
                curveType = RTC_GEOMETRY_TYPE_FLAT_BEZIER_CURVE;
                break;
            case CurveType::Ribbon:
                curveType = RTC_GEOMETRY_TYPE_NORMAL_ORIENTED_BEZIER_CURVE;
                break;
            default:
                ErrorExit("Unhandled curve type");
            }

            rtcGeom = rtcNewGeometry(device, curveType);
            if (!rtcGeom) {
                ErrorExit("Failed to create Embree curve geometry");
            }

            Point3f cp[4];
            Float width[4];
            curve->GetEmbreeControlPoints(cp);
            curve->GetEmbreeWidths(width);

            float *vertices = (float *)rtcSetNewGeometryBuffer(
                rtcGeom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4,
                4 * sizeof(float), 4);
            for (int j = 0; j < 4; ++j) {
                vertices[4 * j + 0] = cp[j].x;
                vertices[4 * j + 1] = cp[j].y;
                vertices[4 * j + 2] = cp[j].z;
                vertices[4 * j + 3] = 0.5f * width[j];
            }

            unsigned int *indices = (unsigned int *)rtcSetNewGeometryBuffer(
                rtcGeom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT,
                sizeof(unsigned int), 1);
            indices[0] = 0;
            if (curve->GetType() == CurveType::Ribbon) {
                Normal3f n[4];
                curve->GetEmbreeNormals(n);
                float *normals = (float *)rtcSetNewGeometryBuffer(
                    rtcGeom, RTC_BUFFER_TYPE_NORMAL, 0, RTC_FORMAT_FLOAT3,
                    3 * sizeof(float), 4);
                for (int j = 0; j < 4; ++j) {
                    normals[3 * j + 0] = n[j].x;
                    normals[3 * j + 1] = n[j].y;
                    normals[3 * j + 2] = n[j].z;
                }
            }
        } else {
            // Create user-defined geometry for non-Triangle primitives
            rtcGeom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_USER);
            if (!rtcGeom) {
                ErrorExit("Failed to create Embree user geometry");
            }

            rtcSetGeometryUserPrimitiveCount(rtcGeom, 1);
            rtcSetGeometryUserData(rtcGeom, (void *)&geom);
            rtcSetGeometryBoundsFunction(rtcGeom, UserPrimitiveBoundsFunc, nullptr);
            rtcSetGeometryIntersectFunction(rtcGeom, UserPrimitiveIntersectFunc);
            rtcSetGeometryOccludedFunction(rtcGeom, UserPrimitiveOccludedFunc);
        }

        rtcCommitGeometry(rtcGeom);
        geom.geometry = rtcGeom;
        geom.geomID = rtcAttachGeometry(scene, rtcGeom);
        geomIDToIndex[geom.geomID] = geometries.size() - 1;
        rtcReleaseGeometry(rtcGeom);
    }

    rtcCommitScene(scene);
}

EmbreeAccelerationStructure::~EmbreeAccelerationStructure() {
    if (scene) {
        rtcReleaseScene(scene);
    }
    if (device) {
        rtcReleaseDevice(device);
    }
}

pstd::optional<ShapeIntersection> EmbreeAccelerationStructure::Intersect(
    const Ray &ray, Float tMax) const {
    if (!scene || geometries.empty()) {
        return {};
    }

    RTCRayHit rayhit;
    rayhit.ray.org_x = ray.o.x;
    rayhit.ray.org_y = ray.o.y;
    rayhit.ray.org_z = ray.o.z;
    rayhit.ray.dir_x = ray.d.x;
    rayhit.ray.dir_y = ray.d.y;
    rayhit.ray.dir_z = ray.d.z;
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = tMax;
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.ray.time = ray.time;
    rayhit.ray.id = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    rtcIntersect1(scene, &rayhit, &args);

    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
        return {};

    auto iter = geomIDToIndex.find(rayhit.hit.geomID);
    if (iter == geomIDToIndex.end())
        return {};

    const GeometryData &geom = geometries[iter->second];
    Float candidateTMax = std::min(
        tMax, Float(rayhit.ray.tfar) * (1 + 1e-4f) + 1e-6f);
    return geom.primitive.Intersect(ray, candidateTMax);
}

bool EmbreeAccelerationStructure::IntersectP(const Ray &ray, Float tMax) const {
    if (!scene || geometries.empty()) {
        return false;
    }

    RTCRay rtcRay;
    rtcRay.org_x = ray.o.x;
    rtcRay.org_y = ray.o.y;
    rtcRay.org_z = ray.o.z;
    rtcRay.dir_x = ray.d.x;
    rtcRay.dir_y = ray.d.y;
    rtcRay.dir_z = ray.d.z;
    rtcRay.tnear = ShadowEpsilon;
    rtcRay.tfar = tMax;
    rtcRay.mask = 0xFFFFFFFF;
    rtcRay.flags = 0;
    rtcRay.time = ray.time;
    rtcRay.id = 0;

    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);
    rtcOccluded1(scene, &rtcRay, &args);

    return rtcRay.tfar < 0;
}

// EmbreeAggregate Method Definitions

EmbreeAggregate::EmbreeAggregate(std::vector<Primitive> prims)
    : primitives(std::move(prims)) {
    if (primitives.empty()) {
        return;
    }

    // Initialize Embree acceleration structure
    accelStructure = std::make_unique<EmbreeAccelerationStructure>(primitives);

    // Compute bounds
    bounds = Bounds3f();
    for (const auto &prim : primitives) {
        bounds = Union(bounds, prim.Bounds());
    }
}

EmbreeAggregate::~EmbreeAggregate() = default;

EmbreeAggregate *EmbreeAggregate::Create(std::vector<Primitive> prims,
                                         const ParameterDictionary &parameters) {
    return new EmbreeAggregate(std::move(prims));
}

Bounds3f EmbreeAggregate::Bounds() const {
    return bounds;
}

pstd::optional<ShapeIntersection> EmbreeAggregate::Intersect(const Ray &ray,
                                                             Float tMax) const {
    ++rayShapeTests;

    if (!accelStructure || primitives.empty()) {
        return {};
    }

    return accelStructure->Intersect(ray, tMax);
}

bool EmbreeAggregate::IntersectP(const Ray &ray, Float tMax) const {
    ++rayShapeTests;

    if (!accelStructure || primitives.empty()) {
        return false;
    }

    return accelStructure->IntersectP(ray, tMax);
}

}  // namespace pbrt
