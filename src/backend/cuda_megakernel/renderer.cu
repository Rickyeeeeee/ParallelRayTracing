#include "backend/cuda_megakernel/renderer.h"
#include <vector>
#include <chrono>
#include <unordered_map>
#include <type_traits>
#include <core/math.h>

/*
TODO

- Optimize the cam buffer (call once in init(), and don't need to malloc and delete the buffer)
- Optimize camera in camera.h for GPU compatibility
- remove the hardcoded depth
- Optimize the Intersect Function
    - Not using CPU since too many vector and virtual, 
      waiting for other implementation before rebase
    - tMin hardcoded
*/

namespace
{
QUAL_GPU bool IntersectPrimitive(const Primitive& primitive, const Ray& worldRay, SurfaceInteraction* outSi)
{
    Ray rayLocal;
    rayLocal.Origin = TransformPoint(primitive.Transform.GetInvMat(), worldRay.Origin);
    rayLocal.Direction = TransformNormal(primitive.Transform.GetMat(), worldRay.Direction);

    SurfaceInteraction localSi;
    bool hit = primitive.Shape.Intersect(rayLocal, &localSi);

    if (!hit)
        return false;

    localSi.Position = TransformPoint(primitive.Transform.GetMat(), localSi.Position);
    localSi.Normal = TransformNormal(primitive.Transform.GetInvMat(), localSi.Normal);
    localSi.Material = primitive.Material;
    *outSi = localSi;
    return true;
}

QUAL_GPU bool IntersectSceneGPU(
    const Ray& ray,
    const Primitive* primitives, int numPrimitives,
    SurfaceInteraction* outSi,
    MaterialHandle* outMaterial)
{
    float minDistance2 = 1e30f;
    bool hitAny = false;
    SurfaceInteraction bestIntersection;
    MaterialHandle bestMaterial;

    for (int i = 0; i < numPrimitives; ++i)
    {
        SurfaceInteraction si;
        if (!IntersectPrimitive(primitives[i], ray, &si))
            continue;

        glm::vec3 d = ray.Origin - si.Position;
        float dist2 = glm::dot(d, d);
        if (dist2 < minDistance2)
        {
            minDistance2 = dist2;
            bestIntersection = si;
            bestMaterial = primitives[i].Material;
            hitAny = true;
        }
    }

    if (!hitAny)
    {
        outSi->HasIntersection = false;
        return false;
    }

    *outSi = bestIntersection;
    *outMaterial = bestMaterial;
    return true;
}

QUAL_GPU glm::vec3 TraceRayGPU(
    Ray ray,
    const Primitive* primitives, int numPrimitives,
    glm::vec3 skyLight,
    curandState* rngState,
    int maxDepth = 20)
{
    glm::vec3 L(0.0f);
    glm::vec3 throughput(1.0f);

    for (int depth = 0; depth < maxDepth; ++depth)
    {
        SurfaceInteraction si;
        MaterialHandle material;
        bool hit = IntersectSceneGPU(ray, primitives, numPrimitives, &si, &material);

        if (!hit || !si.HasIntersection)
        {
            L += throughput * skyLight;
            break;
        }

        glm::vec3 emitted(0.0f);
        si.Material.Emit(emitted);
        L += throughput * emitted;

        glm::vec3 attenuation(0.0f);
        Ray scattered;
        if (!si.Material.Scatter(ray, si, attenuation, scattered, rngState))
            break;

        throughput *= attenuation;
        ray = scattered;
        ray.Direction = glm::normalize(ray.Direction);
    }

    return L;
}

__global__ void InitRNGKernel(
    curandState* rngStates,
    uint32_t pixelCount,
    uint64_t seed)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixelCount) return;

    curand_init(seed, idx, 0, &rngStates[idx]);
}

__global__ void GPU_RayTracing(float* colors, Camera* cam, Primitive* primitives, int primitiveCount, curandState* rngStates)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixelCount = (uint32_t)cam->GetWidth() * (uint32_t)cam->GetHeight();
    if (idx >= pixelCount)
        return;

    const uint32_t x = idx % (uint32_t)cam->GetWidth();
    const uint32_t y = idx / cam->GetWidth();

    Ray ray = cam->GetCameraRay((float)x + 0.5f, (float)y + 0.5f);
    
    curandState localState;
    if (rngStates)
    {
        localState = rngStates[idx];
    }
    else
    {
        curand_init(0xC0FFEEULL, (unsigned long long)idx, 0, &localState);
    }

    // Trace Ray
    glm::vec3 raycolor = (primitiveCount > 0)
        ? TraceRayGPU(
            ray,
            primitives, primitiveCount,
            glm::vec3(0.4f, 0.3f, 0.6f),
            &localState)
        : glm::vec3(0.4f, 0.3f, 0.6f);

    if (rngStates)
    {
        rngStates[idx] = localState;
    }

    // Method 1: GLM Normalize
    const uint32_t base = idx * 3;
    colors[base + 0] = raycolor.r;
    colors[base + 1] = raycolor.g;
    colors[base + 2] = raycolor.b;
}

}


CudaMegakernelRenderer::~CudaMegakernelRenderer()
{
    if (m_RNGStates)
    {
        cudaFree(m_RNGStates);
        m_RNGStates = nullptr;
    }
    m_RNGCapacity = 0;
}

void CudaMegakernelRenderer::Init(Film& film, const Scene& scene, const Camera& camera)
{
    m_Film = &film;
    m_Scene = &scene;
    m_Camera = &camera;

    if (m_RNGStates)
    {
        cudaFree(m_RNGStates);
        m_RNGStates = nullptr;
    }
    m_RNGCapacity = 0;
    m_RNGSeed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

void CudaMegakernelRenderer::ProgressiveRender()
{
    if (!m_Film)
        return;

    const uint32_t width = m_Film->GetWidth();
    const uint32_t height = m_Film->GetHeight();
    const uint32_t pixelCount = width * height;

    if (pixelCount == 0)
        return;

    const size_t bufferSize = static_cast<size_t>(pixelCount) * 3 * sizeof(float);

    float* deviceBuffer = nullptr;
    cudaMalloc(&deviceBuffer, bufferSize);

    const uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;

    // Camera
    Camera* d_cam;
    cudaMalloc(&d_cam, sizeof(Camera));
    cudaMemcpy(d_cam, m_Camera, sizeof(Camera), cudaMemcpyHostToDevice);
    // Scene data
    DeviceSceneData deviceScene = UploadSceneData();

    if (m_RNGSeed == 0)
    {
        m_RNGSeed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    if (pixelCount > m_RNGCapacity)
    {
        if (m_RNGStates)
        {
            cudaFree(m_RNGStates);
            m_RNGStates = nullptr;
        }

        cudaMalloc(&m_RNGStates, sizeof(curandState) * static_cast<size_t>(pixelCount));
        m_RNGCapacity = pixelCount;

        InitRNGKernel<<<blocks, threadsPerBlock>>>(reinterpret_cast<curandState*>(m_RNGStates), pixelCount, m_RNGSeed);
    }


    GPU_RayTracing<<<blocks, threadsPerBlock>>>(
        deviceBuffer,
        d_cam,
        deviceScene.primitives,
        deviceScene.primitiveCount,
        reinterpret_cast<curandState*>(m_RNGStates));
    cudaDeviceSynchronize();
    
    
    // Render
    std::vector<float> hostBuffer(static_cast<size_t>(pixelCount) * 3);
    cudaMemcpy(hostBuffer.data(), deviceBuffer, bufferSize, cudaMemcpyDeviceToHost);

    m_Film->AddSampleBuffer(hostBuffer.data());

    cudaFree(deviceBuffer);
    cudaFree(d_cam);
    FreeDeviceScene(deviceScene);
}

CudaMegakernelRenderer::DeviceSceneData CudaMegakernelRenderer::UploadSceneData() const
{
    DeviceSceneData data{};
    if (!m_Scene)
        return data;

    std::unordered_map<const void*, const void*> materialRemap;
    std::unordered_map<const void*, const void*> shapeRemap;

    const auto uploadMaterial = [&](const MaterialHandle& handle) -> const void* {
        if (!handle.IsValid())
            return nullptr;

        auto it = materialRemap.find(handle.Ptr);
        if (it != materialRemap.end())
            return it->second;

        void* devicePtr = nullptr;
        handle.dispatch([&](const auto* material) {
            if (!material)
                return;
            using MatT = std::remove_cv_t<std::remove_reference_t<decltype(*material)>>;
            const size_t size = sizeof(MatT);
            cudaMalloc(&devicePtr, size);
            cudaMemcpy(devicePtr, material, size, cudaMemcpyHostToDevice);
        });

        if (!devicePtr)
            return nullptr;

        materialRemap[handle.Ptr] = devicePtr;
        data.materialAllocs.push_back(devicePtr);
        return devicePtr;
    };

    const auto uploadShape = [&](const ShapeHandle& handle) -> const void* {
        if (!handle.IsValid())
            return nullptr;

        auto it = shapeRemap.find(handle.Ptr);
        if (it != shapeRemap.end())
            return it->second;

        void* devicePtr = nullptr;
        handle.dispatch([&](const auto* shape) {
            if (!shape)
                return;
            using ShapeT = std::remove_cv_t<std::remove_reference_t<decltype(*shape)>>;
            const size_t size = sizeof(ShapeT);
            cudaMalloc(&devicePtr, size);
            cudaMemcpy(devicePtr, shape, size, cudaMemcpyHostToDevice);
        });

        if (!devicePtr)
            return nullptr;

        shapeRemap[handle.Ptr] = devicePtr;
        data.shapeAllocs.push_back(devicePtr);
        return devicePtr;
    };

    const auto& hostPrimitives = m_Scene->GetPrimitives();
    std::vector<Primitive> devicePrimitives(hostPrimitives.begin(), hostPrimitives.end());

    for (auto& primitive : devicePrimitives)
    {
        primitive.Material.Ptr = uploadMaterial(primitive.Material);
        primitive.Shape.Ptr = uploadShape(primitive.Shape);
    }

    data.primitiveCount = static_cast<int>(devicePrimitives.size());
    if (data.primitiveCount > 0)
    {
        const size_t primitivesSize = sizeof(Primitive) * data.primitiveCount;
        cudaMalloc(reinterpret_cast<void**>(&data.primitives), primitivesSize);
        cudaMemcpy(data.primitives, devicePrimitives.data(), primitivesSize, cudaMemcpyHostToDevice);
    }

    return data;
}

void CudaMegakernelRenderer::FreeDeviceScene(DeviceSceneData& data) const
{
    auto release = [](auto*& ptr) {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };

    release(data.primitives);
    data.primitiveCount = 0;

    for (auto* ptr : data.materialAllocs)
        cudaFree(ptr);
    data.materialAllocs.clear();

    for (auto* ptr : data.shapeAllocs)
        cudaFree(ptr);
    data.shapeAllocs.clear();
}
