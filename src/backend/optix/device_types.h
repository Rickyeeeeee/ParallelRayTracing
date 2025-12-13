#pragma once

// Device-side data structures for OptiX ray tracing
// This file defines GPU-compatible material and geometry types

#include "launch_params.h"

// Material type enumeration
enum class MaterialType : int
{
    Lambertian = 0,
    Metal      = 1,
    Dielectric = 2,
    Emissive   = 3
};

// GPU-side material structure (POD for device memory)
struct DeviceMaterial
{
    MaterialType type;
    float3       albedo;           // Color for Lambertian, Metal, Emissive
    float        roughness;        // For Metal
    float        refractionIndex;  // For Dielectric
    float3       emission;         // For Emissive materials
};

// Sphere/Circle geometry data
struct SphereData
{
    float3 center;
    float  radius;
    int    materialIndex;
};

// Quad geometry data (defined by corner and two edge vectors)
struct QuadData
{
    float3 corner;    // One corner of the quad
    float3 u;         // Edge vector 1
    float3 v;         // Edge vector 2
    float3 normal;    // Normal vector
    int    materialIndex;
};

// Triangle vertex data for custom intersection or built-in triangles
struct TriangleData
{
    float3 v0, v1, v2;    // Vertices
    float3 n0, n1, n2;    // Vertex normals
    float2 uv0, uv1, uv2; // Texture coordinates  
    int    materialIndex;
};

// SBT (Shader Binding Table) record data for hit groups
// Each geometry instance carries this in its SBT record
struct HitGroupData
{
    // Pointer to geometry-specific data (spheres, quads, or triangles)
    void* geometryData;
    
    // Pointer to material array
    DeviceMaterial* materials;
    
    // Geometry type identifier
    int geometryType;  // 0=sphere, 1=quad, 2=triangle
};

// Geometry type constants
constexpr int GEOMETRY_TYPE_SPHERE   = 0;
constexpr int GEOMETRY_TYPE_QUAD     = 1;
constexpr int GEOMETRY_TYPE_TRIANGLE = 2;

// Ray types
enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW   = 1,
    RAY_TYPE_COUNT    = 2
};

// Payload structure for radiance rays
struct RadiancePayload
{
    float3 color;       // Accumulated color
    float3 attenuation; // Current attenuation
    float3 origin;      // Next ray origin
    float3 direction;   // Next ray direction
    int    depth;       // Current recursion depth
    bool   done;        // Whether ray tracing is complete
    unsigned int seed;  // RNG seed
};

// Helper to pack/unpack payload pointer
static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

// Random number generation (PCG)
static __forceinline__ __device__ unsigned int pcg_hash(unsigned int input)
{
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static __forceinline__ __device__ float randomFloat(unsigned int& seed)
{
    seed = pcg_hash(seed);
    return (float)seed / (float)0xffffffffu;
}

static __forceinline__ __device__ float randomFloat(unsigned int& seed, float min, float max)
{
    return min + (max - min) * randomFloat(seed);
}

static __forceinline__ __device__ float3 randomInUnitSphere(unsigned int& seed)
{
    while (true)
    {
        float3 p = make_float3(
            randomFloat(seed, -1.0f, 1.0f),
            randomFloat(seed, -1.0f, 1.0f),
            randomFloat(seed, -1.0f, 1.0f)
        );
        if (lengthSquared(p) < 1.0f)
            return p;
    }
}

static __forceinline__ __device__ float3 randomUnitVector(unsigned int& seed)
{
    return normalize(randomInUnitSphere(seed));
}
