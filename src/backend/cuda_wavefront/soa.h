#pragma once

#include <core/core.h>
#include <core/material_handle.h>
#include <core/geometry.h>
#include <core/primitive.h>
#include <core/shape.h>
#include <vector>

struct WavefrontDeviceSceneView
{
    Primitive* primitives = nullptr;
    int primitiveCount = 0;
};

struct WavefrontSceneBuffers
{
    WavefrontDeviceSceneView Device{};
    std::vector<void*> MaterialAllocs;
    std::vector<void*> ShapeAllocs;

    void Free();
};

// Build device-ready SOA buffers from the scene primitives and upload them.
WavefrontSceneBuffers BuildWavefrontSceneBuffers(const std::vector<Primitive>& primitives);
