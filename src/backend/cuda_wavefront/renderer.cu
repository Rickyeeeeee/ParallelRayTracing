#include "renderer.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <chrono>
#include <cfloat>
#include <core/camera.h>
#include <core/shape.h>
#include <core/material.h>
#include <core/surface_interaction.h>

QUAL_GPU inline PixelState PixelStateSOA::Load(uint32_t index) const
{
    PixelState state;
    state.Ray.Origin = RayOrigins[index];
    state.Ray.Direction = RayDirections[index];
    state.Throughput = Throughput[index];
    state.Radiance = Radiance[index];
    state.PathDepth = PathDepth[index];
    state.Alive = Alive[index];
    return state;
}

QUAL_GPU inline void PixelStateSOA::Store(uint32_t index, const PixelState& state) const
{
    RayOrigins[index] = state.Ray.Origin;
    RayDirections[index] = state.Ray.Direction;
    Throughput[index] = state.Throughput;
    Radiance[index] = state.Radiance;
    PathDepth[index] = state.PathDepth;
    Alive[index] = state.Alive;
}

QUAL_GPU inline Ray PixelStateSOA::GetRay(uint32_t index) const
{
    Ray ray;
    ray.Origin = RayOrigins[index];
    ray.Direction = RayDirections[index];
    return ray;
}

QUAL_GPU inline uint32_t RayQueueSOA::AllocateSlot() const
{
    if (!IsValid())
        return 0xFFFFFFFFu;

#if defined(__CUDA_ARCH__)
    // Warp-aggregated allocation: 1 atomicAdd per warp.
    const unsigned int mask = __activemask();
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int leader = __ffs(mask) - 1;
    const uint32_t warpCount = static_cast<uint32_t>(__popc(mask));

    uint32_t warpBase = 0;
    if (lane == leader)
        warpBase = atomicAdd(Count, warpCount);

    warpBase = __shfl_sync(mask, warpBase, leader);

    const unsigned int laneMask = (lane == 0) ? 0u : ((1u << lane) - 1u);
    const uint32_t laneRank = static_cast<uint32_t>(__popc(mask & laneMask));
    return warpBase + laneRank;
#else
    return atomicAdd(Count, 1u);
#endif
}

QUAL_GPU inline void RayQueueSOA::Push(uint32_t pixelIndex) const
{
    const uint32_t writeIdx = AllocateSlot();
    if (writeIdx >= Capacity)
        return;
    PixelIndices[writeIdx] = pixelIndex;
}

QUAL_GPU inline uint32_t HitQueueSOA::AllocateSlot() const
{
    if (!IsValid())
        return 0xFFFFFFFFu;

#if defined(__CUDA_ARCH__)
    const unsigned int mask = __activemask();
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int leader = __ffs(mask) - 1;
    const uint32_t warpCount = static_cast<uint32_t>(__popc(mask));

    uint32_t warpBase = 0;
    if (lane == leader)
        warpBase = atomicAdd(Count, warpCount);

    warpBase = __shfl_sync(mask, warpBase, leader);

    const unsigned int laneMask = (lane == 0) ? 0u : ((1u << lane) - 1u);
    const uint32_t laneRank = static_cast<uint32_t>(__popc(mask & laneMask));
    return warpBase + laneRank;
#else
    return atomicAdd(Count, 1u);
#endif
}

QUAL_GPU inline void HitQueueSOA::Push(uint32_t pixelIndex,
                                      const glm::vec3& hitPosition,
                                      const glm::vec3& hitNormal,
                                      const MaterialHandle& material,
                                      bool isFrontFace) const
{
    const uint32_t writeIdx = AllocateSlot();
    if (writeIdx >= Capacity)
        return;

    PixelIndices[writeIdx] = pixelIndex;
    HitPositions[writeIdx] = hitPosition;
    HitNormals[writeIdx] = hitNormal;
    Materials[writeIdx] = material;
    IsFrontFace[writeIdx] = isFrontFace ? 1 : 0;
}

void WavefrontQueues::ResetCounts(cudaStream_t stream) const
{
    auto zeroCount = [&](const auto& queue) {
        if (queue.Count)
            cudaMemsetAsync(queue.Count, 0, sizeof(uint32_t), stream);
    };
    
    zeroCount(RayQueue);
    zeroCount(HitQueue);
}

namespace
{

template<typename QueueT>
QUAL_GPU inline uint32_t QueueCount(const QueueT& queue)
{
    if (!queue.Count || queue.Capacity == 0)
        return 0;
    return *queue.Count;
}

QUAL_GPU bool IntersectPrimitiveDevice(const Primitive& primitive, const Ray& worldRay, SurfaceInteraction* outSi)
{
    Ray rayLocal;
    rayLocal.Origin = TransformPoint(primitive.Transform.GetInvMat(), worldRay.Origin);
    rayLocal.Direction = TransformNormal(primitive.Transform.GetMat(), worldRay.Direction);

    SurfaceInteraction localSi;
    const bool hit = primitive.Shape.Intersect(rayLocal, &localSi);
    if (!hit)
        return false;

    localSi.Position = TransformPoint(primitive.Transform.GetMat(), localSi.Position);
    localSi.Normal = TransformNormal(primitive.Transform.GetInvMat(), localSi.Normal);
    localSi.Material = primitive.Material;
    *outSi = localSi;
    return true;
}

QUAL_GPU inline void ResetPixelState(PixelStateSOA state, uint32_t pixelIndex)
{
    state.PathDepth[pixelIndex] = 0;
    state.Throughput[pixelIndex] = glm::vec3(1.0f);
    state.Radiance[pixelIndex] = glm::vec3(0.0f);
    state.Alive[pixelIndex] = 1;
}

__global__ void InitRNGStatesKernel(PixelStateSOA pixelStates, uint32_t pixelCount, uint64_t seed)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixelCount)
        return;

    if (!pixelStates.RNGStates)
        return;

    curand_init(seed, idx, 0, &pixelStates.RNGStates[idx]);
}

__global__ void GenerateCameraRaysKernel(PixelStateSOA pixelStates, WavefrontQueues queues, const Camera* cam, uint32_t width, uint32_t height)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixelCount = width * height;
    if (idx >= pixelCount)
        return;

    ResetPixelState(pixelStates, idx);

    const uint32_t x = idx % width;
    const uint32_t y = idx / width;

    Ray ray = cam->GetCameraRay((float)x + 0.5f, (float)y + 0.5f);
    ray.Normalize();
    pixelStates.RayOrigins[idx] = ray.Origin;
    pixelStates.RayDirections[idx] = ray.Direction;

    queues.PushRay(idx);
}

__global__ void IntersectClosestKernel(PixelStateSOA pixelStates, WavefrontQueues queues, WavefrontDeviceSceneView scene, glm::vec3 skyColor)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ uint32_t rayQueueCount;
    if (threadIdx.x == 0)
        rayQueueCount = QueueCount(queues.RayQueue);
    __syncthreads();
    if (idx >= rayQueueCount)
        return;

    if (!queues.RayQueue.PixelIndices || !queues.HitQueue.IsValid())
        return;
    if (!scene.primitives || scene.primitiveCount <= 0)
    {
        const uint32_t pixelIndex = queues.RayQueue.PixelIndices[idx];
        pixelStates.Radiance[pixelIndex] += pixelStates.Throughput[pixelIndex] * skyColor;
        pixelStates.Alive[pixelIndex] = 0;
        return;
    }

    const uint32_t pixelIndex = queues.RayQueue.PixelIndices[idx];
    const glm::vec3 rayOrigin = pixelStates.RayOrigins[pixelIndex];
    const glm::vec3 rayDirection = pixelStates.RayDirections[pixelIndex];

    float closestT = FLT_MAX;
    glm::vec3 bestNormal(0.0f);
    glm::vec3 bestPosition(0.0f);
    MaterialHandle bestMaterial{};
    bool bestFrontFace = true;
    bool hitSomething = false;

    const Primitive* primitives = scene.primitives;
    const int primitiveCount = scene.primitiveCount;

    Ray worldRay;
    worldRay.Origin = rayOrigin;
    worldRay.Direction = rayDirection;

    for (int primIdx = 0; primIdx < primitiveCount; ++primIdx)
    {
        SurfaceInteraction si{};
        if (!IntersectPrimitiveDevice(primitives[primIdx], worldRay, &si))
            continue;

        const float worldT = glm::length(si.Position - rayOrigin);

        if (worldT < closestT)
        {
            closestT = worldT;
            bestNormal = glm::normalize(si.Normal);
            bestPosition = si.Position;
            bestMaterial = si.Material;
            bestFrontFace = si.IsFrontFace;
            hitSomething = true;
        }
    }

    if (hitSomething)
    {
        queues.HitQueue.Push(pixelIndex, bestPosition, bestNormal, bestMaterial, bestFrontFace);
    }
    else
    {
        pixelStates.Radiance[pixelIndex] += pixelStates.Throughput[pixelIndex] * skyColor;
        pixelStates.Alive[pixelIndex] = 0;
    }
}

__global__ void ShadeHitsKernel(PixelStateSOA pixelStates, WavefrontQueues queues, uint32_t maxDepth)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ uint32_t hitCount;
    if (threadIdx.x == 0)
        hitCount = QueueCount(queues.HitQueue);
    __syncthreads();
    if (idx >= hitCount)
        return;

    if (!queues.HitQueue.IsValid() || !queues.RayQueue.PixelIndices)
        return;

    const uint32_t pixelIndex = queues.HitQueue.PixelIndices[idx];
    if (!pixelStates.Alive[pixelIndex])
        return;

    const uint32_t depth = pixelStates.PathDepth[pixelIndex];
    const glm::vec3 throughput = pixelStates.Throughput[pixelIndex];
    const MaterialHandle material = queues.HitQueue.Materials[idx];
    glm::vec3 emitted(0.0f);
    material.Emit(emitted);
    glm::vec3 attenuation(0.0f);
    Ray scatteredRay;
    curandState* rngState = pixelStates.RNGStates ? &pixelStates.RNGStates[pixelIndex] : nullptr;

    Ray inRay;
    inRay.Origin = pixelStates.RayOrigins[pixelIndex];
    inRay.Direction = pixelStates.RayDirections[pixelIndex];

    SurfaceInteraction si{};
    si.Position = queues.HitQueue.HitPositions[idx];
    si.Normal = glm::normalize(queues.HitQueue.HitNormals[idx]);
    si.HasIntersection = true;
    si.Material = material;
    si.IsFrontFace = queues.HitQueue.IsFrontFace[idx] != 0;

    glm::vec3 radiance = pixelStates.Radiance[pixelIndex];
    radiance += throughput * emitted;
    pixelStates.Radiance[pixelIndex] = radiance;

    if (!material.IsValid() || depth + 1 >= maxDepth)
    {
        pixelStates.Alive[pixelIndex] = 0;
        return;
    }

    const bool scattered = material.Scatter(inRay, si, attenuation, scatteredRay, rngState);
    if (!scattered)
    {
        pixelStates.Alive[pixelIndex] = 0;
        return;
    }

    scatteredRay.Normalize();

    pixelStates.PathDepth[pixelIndex] = depth + 1;
    pixelStates.Throughput[pixelIndex] = throughput * attenuation;
    pixelStates.RayOrigins[pixelIndex] = scatteredRay.Origin;
    pixelStates.RayDirections[pixelIndex] = scatteredRay.Direction;
    queues.PushRay(pixelIndex);
}

__global__ void BlitRadianceKernel(PixelStateSOA pixelStates, float* colors, uint32_t pixelCount)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixelCount)
        return;

    const glm::vec3 color = pixelStates.Radiance[idx];
    const uint32_t base = idx * 3;
    colors[base + 0] = color.r;
    colors[base + 1] = color.g;
    colors[base + 2] = color.b;
}
}

void CudaWavefrontRenderer::Init(Film& film, const Scene& scene, const Camera& camera)
{
    m_Film = &film;
    m_Scene = &scene;
    m_Camera = &camera;
    m_FrameIndex = 0;
    m_RNGSeed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // Rebuild the device-side SOA buffers whenever the renderer is re-initialized.
    m_SceneBuffers.Free();
    if (m_Scene)
    {
        m_SceneBuffers = BuildWavefrontSceneBuffers(m_Scene->GetPrimitives());
    }
}

void CudaWavefrontRenderer::ProgressiveRender()
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

    EnsureDeviceState(pixelCount);
    if (!m_PixelState.RayOrigins || !m_PixelState.RayDirections || !m_PixelState.Throughput || !m_PixelState.Radiance || !m_DeviceCamera
        || !m_Queues.RayQueue.PixelIndices || !m_Queues.RayQueue.Count
        || !m_Queues.HitQueue.IsValid())
        return;

    m_Queues.ResetCounts();
    cudaMemcpy(m_DeviceCamera, m_Camera, sizeof(Camera), cudaMemcpyHostToDevice);


    const uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    WavefrontQueues queueView = m_Queues;
    GenerateCameraRaysKernel<<<blocks, threadsPerBlock>>>(m_PixelState, queueView, m_DeviceCamera, width, height);

    const glm::vec3 skyColor(0.4f, 0.3f, 0.6f);
    const uint32_t shadeBlocks = blocks;
    const uint32_t activeRayBlocks = blocks;
    const uint32_t maxDepth = 20;

    for (uint32_t depth = 0; depth < maxDepth; ++depth)
    {
        cudaMemset(m_Queues.HitQueue.Count, 0, sizeof(uint32_t));

        IntersectClosestKernel<<<activeRayBlocks, threadsPerBlock>>>(m_PixelState, queueView, m_SceneBuffers.Device, skyColor);

        cudaMemset(m_Queues.RayQueue.Count, 0, sizeof(uint32_t));

        ShadeHitsKernel<<<shadeBlocks, threadsPerBlock>>>(m_PixelState, queueView, maxDepth);

        // uint32_t nextRayCount = 0;
        // cudaMemcpy(&nextRayCount, m_Queues.RayQueue.Count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // if (nextRayCount == 0)
        //     break;
    }

    BlitRadianceKernel<<<blocks, threadsPerBlock>>>(m_PixelState, deviceBuffer, pixelCount);
    std::vector<float> hostBuffer(static_cast<size_t>(pixelCount) * 3);
    cudaDeviceSynchronize();

    cudaMemcpy(hostBuffer.data(), deviceBuffer, bufferSize, cudaMemcpyDeviceToHost);

    m_Film->AddSampleBuffer(hostBuffer.data());

    cudaFree(deviceBuffer);

    ++m_FrameIndex;
}

CudaWavefrontRenderer::~CudaWavefrontRenderer()
{
    m_SceneBuffers.Free();
    ReleaseDeviceState();
}

void CudaWavefrontRenderer::EnsureDeviceState(uint32_t pixelCount)
{
    if (pixelCount == 0)
        return;

    auto reallocArray = [&](auto*& ptr, size_t elemSize) {
        if (ptr)
            cudaFree(ptr);
        cudaMalloc(&ptr, elemSize * pixelCount);
    };

    if (pixelCount > m_PixelState.Capacity)
    {
        reallocArray(m_PixelState.RayOrigins, sizeof(glm::vec3));
        reallocArray(m_PixelState.RayDirections, sizeof(glm::vec3));
        reallocArray(m_PixelState.Throughput, sizeof(glm::vec3));
        reallocArray(m_PixelState.Radiance, sizeof(glm::vec3));
        reallocArray(m_PixelState.PathDepth, sizeof(uint32_t));
        reallocArray(m_PixelState.RNGStates, sizeof(curandState));
        reallocArray(m_PixelState.Alive, sizeof(uint8_t));
        m_PixelState.Capacity = pixelCount;

        if (m_RNGSeed == 0)
        {
            m_RNGSeed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        }
        const uint32_t threadsPerBlock = 256;
        const uint32_t blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
        InitRNGStatesKernel<<<blocks, threadsPerBlock>>>(m_PixelState, pixelCount, m_RNGSeed);
    }

    auto ensureRayQueueCapacity = [&](RayQueueSOA& queue) {
        if (pixelCount > queue.Capacity)
        {
            reallocArray(queue.PixelIndices, sizeof(uint32_t));
            queue.Capacity = pixelCount;
        }
        if (!queue.Count)
            cudaMalloc(&queue.Count, sizeof(uint32_t));
    };

    auto ensureHitQueueCapacity = [&](HitQueueSOA& queue) {
        if (pixelCount > queue.Capacity)
        {
            reallocArray(queue.PixelIndices, sizeof(uint32_t));
            reallocArray(queue.HitPositions, sizeof(glm::vec3));
            reallocArray(queue.HitNormals, sizeof(glm::vec3));
            reallocArray(queue.Materials, sizeof(MaterialHandle));
            reallocArray(queue.IsFrontFace, sizeof(uint8_t));
            queue.Capacity = pixelCount;
        }
        if (!queue.Count)
            cudaMalloc(&queue.Count, sizeof(uint32_t));
    };

    ensureRayQueueCapacity(m_Queues.RayQueue);
    ensureHitQueueCapacity(m_Queues.HitQueue);

    if (!m_DeviceCamera)
    {
        cudaMalloc(&m_DeviceCamera, sizeof(Camera));
    }
}

void CudaWavefrontRenderer::ReleaseDeviceState()
{
    auto freePtr = [&](auto*& ptr) {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };

    freePtr(m_PixelState.RayOrigins);
    freePtr(m_PixelState.RayDirections);
    freePtr(m_PixelState.Throughput);
    freePtr(m_PixelState.Radiance);
    freePtr(m_PixelState.PathDepth);
    freePtr(m_PixelState.RNGStates);
    freePtr(m_PixelState.Alive);
    m_PixelState.Capacity = 0;

    auto freeRayQueue = [&](RayQueueSOA& queue) {
        freePtr(queue.PixelIndices);
        freePtr(queue.Count);
        queue.Capacity = 0;
    };

    auto freeHitQueue = [&](HitQueueSOA& queue) {
        freePtr(queue.PixelIndices);
        freePtr(queue.HitPositions);
        freePtr(queue.HitNormals);
        freePtr(queue.Materials);
        freePtr(queue.IsFrontFace);
        freePtr(queue.Count);
        queue.Capacity = 0;
    };

    freeRayQueue(m_Queues.RayQueue);
    freeHitQueue(m_Queues.HitQueue);

    freePtr(m_DeviceCamera);
}
