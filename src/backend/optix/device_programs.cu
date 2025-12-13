// OptiX Device Programs (PTX Shaders)
// Compiled to PTX and loaded at runtime

#include <optix.h>
#include "launch_params.h"
#include "device_types.h"

// Global launch parameters (set by optixLaunch)
extern "C" {
    __constant__ LaunchParams params;
}


// Get radiance payload from optix registers
static __forceinline__ __device__ RadiancePayload* getPayload()
{
    unsigned int u0 = optixGetPayload_0();
    unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePayload*>(unpackPointer(u0, u1));
}

// Trace a ray through the scene
static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3 origin,
    float3 direction,
    float tmin,
    float tmax,
    RadiancePayload* payload)
{
    unsigned int u0, u1;
    packPointer(payload, u0, u1);
    
    optixTrace(
        handle,
        origin,
        direction,
        tmin,                    // tmin
        tmax,                    // tmax
        0.0f,                    // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,       // SBT offset
        RAY_TYPE_COUNT,          // SBT stride
        RAY_TYPE_RADIANCE,       // miss SBT index
        u0, u1);
}

// Schlick approximation for Fresnel reflectance
static __forceinline__ __device__ float schlickReflectance(float cosine, float refIdx)
{
    float r0 = (1.0f - refIdx) / (1.0f + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

// Refract vector
static __forceinline__ __device__ float3 refract(const float3& uv, const float3& n, float etaiOverEtat)
{
    float cosTheta = fminf(dot(make_float3(-uv.x, -uv.y, -uv.z), n), 1.0f);
    float3 rOutPerp = etaiOverEtat * (uv + cosTheta * n);
    float3 rOutParallel = -sqrtf(fabsf(1.0f - lengthSquared(rOutPerp))) * n;
    return rOutPerp + rOutParallel;
}



static __forceinline__ __device__ void ApplyMaterial(
    const DeviceMaterial& mat,
    const float3& hitPoint,
    const float3& geometryNormal,  // Raw geometric normal (outward)
    const float3& rayDir,
    RadiancePayload* payload)
{
    // Determine front face and compute shading normal
    bool frontFace = dot(rayDir, geometryNormal) < 0.0f;
    float3 faceNormal = frontFace ? geometryNormal : make_float3(-geometryNormal.x, -geometryNormal.y, -geometryNormal.z);

    switch (mat.type)
    {
        case MaterialType::Emissive:
        {
            payload->color = mat.emission;
            payload->attenuation = make_float3(0.0f, 0.0f, 0.0f);
            payload->done = true;
            break;
        }

        case MaterialType::Lambertian:
        {
            float3 scatterDir = faceNormal + randomUnitVector(payload->seed);
            // Handle degenerate scatter direction
            if (lengthSquared(scatterDir) < 1e-8f)
                scatterDir = faceNormal;
            scatterDir = normalize(scatterDir);
            
            payload->color = make_float3(0.0f, 0.0f, 0.0f);
            payload->attenuation = mat.albedo;
            // CPU REPLICA: No origin offset, relies on tMin = 0.001f
            payload->origin = hitPoint;
            payload->direction = scatterDir;
            break;
        }

        case MaterialType::Metal:
        {
            float3 reflected = reflect(rayDir, faceNormal);
            reflected = normalize(reflected + mat.roughness * randomUnitVector(payload->seed));
            
            if (dot(reflected, faceNormal) > 0.0f)
            {
                payload->color = make_float3(0.0f, 0.0f, 0.0f);
                payload->attenuation = mat.albedo;
                // CPU REPLICA: No origin offset
                payload->origin = hitPoint;
                payload->direction = reflected;
            }
            else
            {
                payload->done = true;
            }
            break;
        }

        case MaterialType::Dielectric:
        {
            float ri = frontFace ? (1.0f / mat.refractionIndex) : mat.refractionIndex;
            float3 unitDir = normalize(rayDir);
            
            float cosTheta = fminf(dot(make_float3(-unitDir.x, -unitDir.y, -unitDir.z), faceNormal), 1.0f);
            float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
            
            bool cannotRefract = ri * sinTheta > 1.0f;
            float3 direction;
            
            if (cannotRefract || schlickReflectance(cosTheta, ri) > randomFloat(payload->seed))
            {
                // Reflection
                direction = reflect(unitDir, faceNormal);
            }
            else
            {
                // Refraction
                direction = refract(unitDir, faceNormal, ri);
            }
            
            // CPU REPLICA: No origin offset, relies on tMin
            payload->origin = hitPoint;
            
            payload->color = make_float3(0.0f, 0.0f, 0.0f);
            payload->attenuation = make_float3(1.0f, 1.0f, 1.0f);
            payload->direction = direction;
            break;
        }
    }
}



extern "C" __global__ void __raygen__renderFrame()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    const unsigned int pixelIndex = idx.y * dim.x + idx.x;
    
    // Initialize RNG seed
    unsigned int seed = pcg_hash(pixelIndex ^ (params.frameIndex * 719393u));
    
    // Calculate ray direction with jittering for anti-aliasing
    float px = (float)idx.x + randomFloat(seed);
    float py = (float)idx.y + randomFloat(seed);
    
    // Normalize pixel coordinates to [-1, 1]
    float ndcX = (px / params.width) * 2.0f - 1.0f;
    float ndcY = 1.0f - (py / params.height) * 2.0f;  // Flip Y
    
    // Calculate ray direction in camera space then transform to world space
    // Camera looks along -Z in camera space (same as CPU renderer)
    float3 localDir = make_float3(
        ndcX * params.camera.aspectRatio * params.camera.tanFovY,
        ndcY * params.camera.tanFovY,
        -1.0f  // Looking along -Z in camera space
    );
    
    // Transform to world space using camera basis vectors (match CPU exactly)
    // CPU: rayDirCameraSpace.z * (-m_Front), so we use -front here
    float3 rayDir = normalize(
        localDir.x * params.camera.right +
        localDir.y * params.camera.up +
        localDir.z * make_float3(-params.camera.front.x, -params.camera.front.y, -params.camera.front.z)
    );
    
    // Initialize payload
    RadiancePayload payload;
    payload.color = make_float3(0.0f, 0.0f, 0.0f);
    payload.attenuation = make_float3(1.0f, 1.0f, 1.0f);
    payload.origin = params.camera.position;
    payload.direction = rayDir;
    payload.depth = params.maxDepth;
    payload.done = false;
    payload.seed = seed;
    
    // Path tracing loop
    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 rayOrigin = params.camera.position;
    float3 rayDirection = rayDir;
    
    for (int depth = 0; depth < params.maxDepth && !payload.done; ++depth)
    {
        payload.depth = params.maxDepth - depth;
        payload.done = false;
        
        traceRadiance(
            params.traversable,
            rayOrigin,
            rayDirection,
            0.001f,      // tmin (small offset to avoid self-intersection)
            1e16f,       // tmax
            &payload);
        
        finalColor = finalColor + throughput * payload.color;
        throughput = throughput * payload.attenuation;
        
        rayOrigin = payload.origin;
        rayDirection = payload.direction;
    }
    
    // Write the raw linear sample for Film accumulation (RGB interleaved).
    if (params.sampleBuffer)
    {
        const unsigned int base = pixelIndex * 3u;
        params.sampleBuffer[base + 0] = finalColor.x;
        params.sampleBuffer[base + 1] = finalColor.y;
        params.sampleBuffer[base + 2] = finalColor.z;
    }

    float3 accumColor = finalColor;
    
    if (params.frameIndex > 0)
    {
        // Add new sample to running sum
        float3 prevAccum = params.accumBuffer[pixelIndex];
        accumColor = prevAccum + finalColor;
    }
    
    // 2. Save accumulated sum for next frame
    params.accumBuffer[pixelIndex] = accumColor;
    
    // 3. Compute average (for display)
    float3 averagedColor = accumColor / (float)(params.frameIndex + 1);
    
    // 4. Tone Mapping & Gamma Correction on AVERAGED result
    // (Apply to converged value, not single noisy sample)
    
    // Reinhard Tone Mapping: HDR [0, inf) -> LDR [0, 1]
    averagedColor.x = averagedColor.x / (1.0f + averagedColor.x);
    averagedColor.y = averagedColor.y / (1.0f + averagedColor.y);
    averagedColor.z = averagedColor.z / (1.0f + averagedColor.z);
    
    // Gamma Correction: Linear -> sRGB (Gamma 2.2)
    const float invGamma = 1.0f / 2.2f;
    averagedColor.x = powf(averagedColor.x, invGamma);
    averagedColor.y = powf(averagedColor.y, invGamma);
    averagedColor.z = powf(averagedColor.z, invGamma);
    
    // Write display-ready result to output buffer (PBO)
    params.colorBuffer[pixelIndex] = averagedColor;
}



extern "C" __global__ void __miss__radiance()
{
    RadiancePayload* payload = getPayload();
    
    // Simple fixed color background (no gradient)
    // Color is set in updateLaunchParams on host side
    payload->color = params.skyLight;
    payload->done = true;
}



extern "C" __global__ void __closesthit__sphere()
{
    RadiancePayload* payload = getPayload();
    
    // Get SBT data
    const HitGroupData* sbtData = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const SphereData* spheres = reinterpret_cast<const SphereData*>(sbtData->geometryData);
    
    // Get primitive index and geometry
    const int primIdx = optixGetPrimitiveIndex();
    const SphereData& sphere = spheres[primIdx];
    
    // Calculate hit point and geometric normal
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDir = optixGetWorldRayDirection();
    const float t = optixGetRayTmax();
    float3 hitPoint = rayOrigin + t * rayDir;
    float3 geometryNormal = normalize(hitPoint - sphere.center);
    
    // Get material and apply
    const DeviceMaterial& mat = sbtData->materials[sphere.materialIndex];
    ApplyMaterial(mat, hitPoint, geometryNormal, rayDir, payload);
}



extern "C" __global__ void __intersection__sphere()
{
    const HitGroupData* sbtData = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const SphereData* spheres = reinterpret_cast<const SphereData*>(sbtData->geometryData);
    
    const int primIdx = optixGetPrimitiveIndex();
    const SphereData& sphere = spheres[primIdx];
    
    const float3 origin = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();
    
    float3 oc = origin - sphere.center;
    float a = dot(direction, direction);
    float halfB = dot(oc, direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = halfB * halfB - a * c;
    
    if (discriminant >= 0.0f)
    {
        float sqrtD = sqrtf(discriminant);
        float t1 = (-halfB - sqrtD) / a;
        float t2 = (-halfB + sqrtD) / a;
        
        float tmin = optixGetRayTmin();
        float tmax = optixGetRayTmax();
        
        float t = t1;
        if (t < tmin || t > tmax)
        {
            t = t2;
            if (t < tmin || t > tmax)
                return;
        }
        
        optixReportIntersection(t, 0);
    }
}



extern "C" __global__ void __closesthit__quad()
{
    RadiancePayload* payload = getPayload();
    
    const HitGroupData* sbtData = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const QuadData* quads = reinterpret_cast<const QuadData*>(sbtData->geometryData);
    
    // Get primitive index and geometry
    const int primIdx = optixGetPrimitiveIndex();
    const QuadData& quad = quads[primIdx];
    
    // Calculate hit point
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDir = optixGetWorldRayDirection();
    const float t = optixGetRayTmax();
    float3 hitPoint = rayOrigin + t * rayDir;
    
    // Get material and apply
    const DeviceMaterial& mat = sbtData->materials[quad.materialIndex];
    ApplyMaterial(mat, hitPoint, quad.normal, rayDir, payload);
}


extern "C" __global__ void __intersection__quad()
{
    const HitGroupData* sbtData = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const QuadData* quads = reinterpret_cast<const QuadData*>(sbtData->geometryData);
    
    const int primIdx = optixGetPrimitiveIndex();
    const QuadData& quad = quads[primIdx];
    
    const float3 origin = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();
    
    // Ray-plane intersection
    float denom = dot(quad.normal, direction);
    if (fabsf(denom) < 1e-8f) return;  // Parallel to plane
    
    float3 p0ToOrigin = origin - quad.corner;
    float t = -dot(quad.normal, p0ToOrigin) / denom;
    
    if (t < optixGetRayTmin() || t > optixGetRayTmax()) return;
    
    // Check if hit point is within quad bounds
    float3 hitPoint = origin + t * direction;
    float3 d = hitPoint - quad.corner;
    
    float uLen = lengthSquared(quad.u);
    float vLen = lengthSquared(quad.v);
    
    float u = dot(d, quad.u) / uLen;
    float v = dot(d, quad.v) / vLen;
    
    if (u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f)
    {
        optixReportIntersection(t, 0);
    }
}



extern "C" __global__ void __closesthit__triangle()
{
    RadiancePayload* payload = getPayload();
    
    const HitGroupData* sbtData = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const TriangleData* triangles = reinterpret_cast<const TriangleData*>(sbtData->geometryData);
    
    const int primIdx = optixGetPrimitiveIndex();
    const TriangleData& tri = triangles[primIdx];
    
    // Get barycentric coordinates
    const float2 bary = optixGetTriangleBarycentrics();
    const float u = bary.x;
    const float v = bary.y;
    const float w = 1.0f - u - v;
    
    // Interpolate normal
    float3 normal = normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
    
    const float t = optixGetRayTmax();
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDir = optixGetWorldRayDirection();
    float3 hitPoint = rayOrigin + t * rayDir;
    
    bool frontFace = dot(rayDir, normal) < 0.0f;
    if (!frontFace) normal = make_float3(-normal.x, -normal.y, -normal.z);
    
    const DeviceMaterial& mat = sbtData->materials[tri.materialIndex];
    
    // Same material handling as sphere/quad
    switch (mat.type)
    {
        case MaterialType::Emissive:
        {
            payload->color = mat.emission;
            payload->attenuation = make_float3(0.0f, 0.0f, 0.0f);
            payload->done = true;
            break;
        }
        case MaterialType::Lambertian:
        {
            float3 scatterDir = normal + randomUnitVector(payload->seed);
            if (lengthSquared(scatterDir) < 1e-8f)
                scatterDir = normal;
            scatterDir = normalize(scatterDir);
            
            payload->color = make_float3(0.0f, 0.0f, 0.0f);
            payload->attenuation = mat.albedo;
            payload->origin = hitPoint;
            payload->direction = scatterDir;
            break;
        }
        case MaterialType::Metal:
        {
            float3 reflected = reflect(rayDir, normal);
            reflected = normalize(reflected + mat.roughness * randomUnitVector(payload->seed));
            
            if (dot(reflected, normal) > 0.0f)
            {
                payload->color = make_float3(0.0f, 0.0f, 0.0f);
                payload->attenuation = mat.albedo;
                payload->origin = hitPoint;
                payload->direction = reflected;
            }
            else
            {
                payload->done = true;
            }
            break;
        }
        case MaterialType::Dielectric:
        {
            float ri = frontFace ? (1.0f / mat.refractionIndex) : mat.refractionIndex;
            float3 unitDir = normalize(rayDir);
            
            float cosTheta = fminf(dot(make_float3(-unitDir.x, -unitDir.y, -unitDir.z), normal), 1.0f);
            float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
            
            bool cannotRefract = ri * sinTheta > 1.0f;
            float3 direction;
            
            if (cannotRefract || schlickReflectance(cosTheta, ri) > randomFloat(payload->seed))
            {
                direction = reflect(unitDir, normal);
            }
            else
            {
                direction = refract(unitDir, normal, ri);
            }
            
            payload->color = make_float3(0.0f, 0.0f, 0.0f);
            payload->attenuation = make_float3(1.0f, 1.0f, 1.0f);
            payload->origin = hitPoint;
            payload->direction = direction;
            break;
        }
    }
}
