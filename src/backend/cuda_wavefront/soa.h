#pragma once

#include <core/core.h>
#include <core/material_handle.h>
#include <core/geometry.h>
#include <core/primitive.h>
#include <vector>

struct MaterialSOA
{
    glm::vec3* albedo = nullptr;
    glm::vec3* emission = nullptr;
    float* roughness = nullptr;
    float* ior = nullptr;
    MatType* type = nullptr;
    int count = 0;
};

struct CircleSOA
{
    float* radius = nullptr;
    int count = 0;
};

struct QuadSOA
{
    float* width = nullptr;
    float* height = nullptr;
    glm::vec3* normal = nullptr;
    int count = 0;
};

struct PrimitiveSOA
{
    uint8_t* shapeType = nullptr;   // ShapeType as uint8
    int* shapeIndex = nullptr;      // Index into the corresponding shape SOA
    int* materialIndex = nullptr;   // Index into MaterialSOA
    glm::mat4* transform = nullptr;
    glm::mat4* invTransform = nullptr;
    int count = 0;
};

struct WavefrontSceneBuffers
{
    MaterialSOA materials{};
    CircleSOA circles{};
    QuadSOA quads{};
    PrimitiveSOA primitives{};

    void Free();
};

// Build device-ready SOA buffers from the scene primitives and upload them.
WavefrontSceneBuffers BuildWavefrontSceneBuffers(const std::vector<Primitive>& primitives);
