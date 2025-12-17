#pragma once

// Host/Device shared header for OptiX launch parameters
// This file is included by both CPU and GPU code

#include <cuda_runtime.h>
#include <optix_types.h>
#include <cstdint>

// CUDA already provides make_float3, make_float2, etc. in vector_functions.hpp
// We only need to add operators that CUDA doesn't provide

struct CameraData
{
    float3 position;
    float3 front;
    float3 right;
    float3 up;
    float  width;
    float  height;
    float  tanFovY;
    float  aspectRatio;
};

struct LaunchParams
{
    // Frame dimensions
    uint32_t width;
    uint32_t height;
    
    // Frame index for progressive accumulation and RNG seed
    uint32_t frameIndex;
    
    // Camera data
    CameraData camera;
    
    // Scene traversable handle (GAS or IAS)
    OptixTraversableHandle traversable;
    
    // Output buffers
    float3* colorBuffer;     // Display buffer (tone-mapped, for PBO)
    float3* accumBuffer;     // Accumulation buffer (linear space, running sum)
    float*  sampleBuffer;    // Per-frame linear sample buffer (RGBRGB...)
    
    // Maximum ray recursion depth
    int maxDepth;
    
    // Sky light color
    float3 skyLight;
};

// float3 operators (not provided by CUDA by default)
inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 operator*(float s, const float3& a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator/(const float3& a, float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}

inline __host__ __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 normalize(const float3& v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float3 cross(const float3& a, const float3& b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline __host__ __device__ float3 reflect(const float3& v, const float3& n)
{
    return v - 2.0f * dot(v, n) * n;
}

inline __host__ __device__ float length(const float3& v)
{
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float lengthSquared(const float3& v)
{
    return dot(v, v);
}
