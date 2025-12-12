#include "renderer.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
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
    state.PendingEmission = PendingEmission[index];
    state.HitPosition = HitPositions[index];
    state.HitNormal = HitNormals[index];
    state.PixelIndex = PixelIndices[index];
    state.PathDepth = PathDepth[index];
    state.RNGSeed = RNGSeed[index];
    state.HitDistance = HitDistance[index];
    state.Material = Materials[index];
    state.Alive = Alive[index];
    state.HasHit = HasHit[index];
    return state;
}

QUAL_GPU inline void PixelStateSOA::Store(uint32_t index, const PixelState& state) const
{
    RayOrigins[index] = state.Ray.Origin;
    RayDirections[index] = state.Ray.Direction;
    Throughput[index] = state.Throughput;
    Radiance[index] = state.Radiance;
    PendingEmission[index] = state.PendingEmission;
    HitPositions[index] = state.HitPosition;
    HitNormals[index] = state.HitNormal;
    PixelIndices[index] = state.PixelIndex;
    PathDepth[index] = state.PathDepth;
    RNGSeed[index] = state.RNGSeed;
    HitDistance[index] = state.HitDistance;
    Materials[index] = state.Material;
    Alive[index] = state.Alive;
    HasHit[index] = state.HasHit;
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
    return atomicAdd(Count, 1u);
}

QUAL_GPU inline void RayQueueSOA::Push(uint32_t pixelIndex) const
{
    const uint32_t writeIdx = AllocateSlot();
    if (writeIdx >= Capacity)
        return;
    PixelIndices[writeIdx] = pixelIndex;
}

void WavefrontQueues::ResetCounts(cudaStream_t stream) const
{
    auto zeroCount = [&](const RayQueueSOA& queue) {
        if (queue.Count)
        {
            cudaMemsetAsync(queue.Count, 0, sizeof(uint32_t), stream);
        }
    };
    
    zeroCount(RayQueue);
    zeroCount(EscapeQueue);
    zeroCount(HitQueue);
}

namespace
{

QUAL_GPU inline uint32_t QueueCount(const RayQueueSOA& queue)
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

QUAL_GPU inline void ResetPixelState(PixelStateSOA state, uint32_t pixelIndex, uint32_t frameIndex)
{
    state.PixelIndices[pixelIndex] = pixelIndex;
    state.PathDepth[pixelIndex] = 0;
    state.Throughput[pixelIndex] = glm::vec3(1.0f);
    state.Radiance[pixelIndex] = glm::vec3(0.0f);
    state.PendingEmission[pixelIndex] = glm::vec3(0.0f);
    state.HitPositions[pixelIndex] = glm::vec3(0.0f);
    state.HitNormals[pixelIndex] = glm::vec3(0.0f);
    state.Materials[pixelIndex] = MaterialHandle{};
    state.Alive[pixelIndex] = 1;
    state.HitDistance[pixelIndex] = 0.0f;
    state.HasHit[pixelIndex] = 0;
    const uint32_t seed = frameIndex * 9781u + pixelIndex * 6271u + 17u;
    state.RNGSeed[pixelIndex] = seed;
    curand_init(seed, pixelIndex, 0, &state.RNGStates[pixelIndex]);
}

__global__ void GenerateCameraRaysKernel(PixelStateSOA pixelStates, WavefrontQueues queues, const Camera* cam, uint32_t width, uint32_t height, uint32_t frameIndex)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixelCount = width * height;
    if (idx >= pixelCount)
        return;

    ResetPixelState(pixelStates, idx, frameIndex);

    const uint32_t x = idx % width;
    const uint32_t y = idx / width;

    Ray ray = cam->GetCameraRay((float)x + 0.5f, (float)y + 0.5f);
    ray.Normalize();
    pixelStates.RayOrigins[idx] = ray.Origin;
    pixelStates.RayDirections[idx] = ray.Direction;

    queues.PushRay(idx);
}

__global__ void IntersectClosestKernel(PixelStateSOA pixelStates, WavefrontQueues queues, WavefrontDeviceSceneView scene)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rayQueueCount = QueueCount(queues.RayQueue);
    if (idx >= rayQueueCount)
        return;

    if (!queues.RayQueue.PixelIndices || !queues.HitQueue.PixelIndices || !queues.EscapeQueue.PixelIndices || !pixelStates.RNGStates)
        return;
    if (!scene.primitives || scene.primitiveCount <= 0)
    {
        const uint32_t pixelIndex = queues.RayQueue.PixelIndices[idx];
        pixelStates.HasHit[pixelIndex] = 0;
        queues.PushEscape(pixelIndex);
        return;
    }

    const uint32_t pixelIndex = queues.RayQueue.PixelIndices[idx];
    const glm::vec3 rayOrigin = pixelStates.RayOrigins[pixelIndex];
    const glm::vec3 rayDirection = glm::normalize(pixelStates.RayDirections[pixelIndex]);

    float closestT = FLT_MAX;
    glm::vec3 bestNormal(0.0f);
    glm::vec3 bestPosition(0.0f);
    MaterialHandle bestMaterial{};
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
            hitSomething = true;
        }
    }

    if (hitSomething)
    {
        pixelStates.HitPositions[pixelIndex] = bestPosition;
        pixelStates.HitNormals[pixelIndex] = bestNormal;
        pixelStates.HitDistance[pixelIndex] = closestT;
        pixelStates.Materials[pixelIndex] = bestMaterial;
        pixelStates.HasHit[pixelIndex] = 1;
        queues.PushHit(pixelIndex);
    }
    else
    {
        pixelStates.HasHit[pixelIndex] = 0;
        queues.PushEscape(pixelIndex);
    }
}

__global__ void ShadeHitsKernel(PixelStateSOA pixelStates, WavefrontQueues queues)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t hitCount = QueueCount(queues.HitQueue);
    if (idx >= hitCount)
        return;

    if (!queues.HitQueue.PixelIndices)
        return;

    const uint32_t pixelIndex = queues.HitQueue.PixelIndices[idx];
    if (!pixelStates.HasHit[pixelIndex])
        return;

    const glm::vec3 throughput = pixelStates.Throughput[pixelIndex];
    const MaterialHandle material = pixelStates.Materials[pixelIndex];
    glm::vec3 emitted(0.0f);
    material.Emit(emitted);
    glm::vec3 attenuation(0.0f);
    Ray scatteredRay;
    curandState* rngState = pixelStates.RNGStates ? &pixelStates.RNGStates[pixelIndex] : nullptr;

    Ray inRay;
    inRay.Origin = pixelStates.RayOrigins[pixelIndex];
    inRay.Direction = glm::normalize(pixelStates.RayDirections[pixelIndex]);

    SurfaceInteraction si{};
    si.Position = pixelStates.HitPositions[pixelIndex];
    si.Normal = glm::normalize(pixelStates.HitNormals[pixelIndex]);
    si.HasIntersection = true;
    si.Material = material;
    si.IsFrontFace = glm::dot(inRay.Direction, si.Normal) < 0.0f;
    if (!si.IsFrontFace)
        si.Normal = -si.Normal;

    glm::vec3 radiance = emitted;
    if (material.IsValid())
    {
        const bool scattered = material.Scatter(inRay, si, attenuation, scatteredRay, rngState);
        if (scattered)
            radiance += throughput * attenuation;
    }

    pixelStates.Radiance[pixelIndex] = radiance;
}

__global__ void ShadeMissKernel(PixelStateSOA pixelStates, WavefrontQueues queues, glm::vec3 skyColor)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t missCount = QueueCount(queues.EscapeQueue);
    if (idx >= missCount)
        return;

    if (!queues.EscapeQueue.PixelIndices)
        return;

    const uint32_t pixelIndex = queues.EscapeQueue.PixelIndices[idx];
    pixelStates.Radiance[pixelIndex] = skyColor;
    pixelStates.HasHit[pixelIndex] = 0;
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

    EnsureDeviceState(pixelCount);
    if (!m_PixelState.RayOrigins || !m_PixelState.RayDirections || !m_PixelState.Materials || !m_DeviceCamera
        || !m_Queues.RayQueue.PixelIndices || !m_Queues.RayQueue.Count
        || !m_Queues.EscapeQueue.PixelIndices || !m_Queues.EscapeQueue.Count
        || !m_Queues.HitQueue.PixelIndices || !m_Queues.HitQueue.Count)
        return;

    m_Queues.ResetCounts();
    cudaMemcpy(m_DeviceCamera, m_Camera, sizeof(Camera), cudaMemcpyHostToDevice);

    float* deviceBuffer = nullptr;
    cudaMalloc(&deviceBuffer, bufferSize);

    const uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    WavefrontQueues queueView = m_Queues;
    GenerateCameraRaysKernel<<<blocks, threadsPerBlock>>>(m_PixelState, queueView, m_DeviceCamera, width, height, m_FrameIndex);

    const uint32_t activeRayCount = pixelCount;
    if (activeRayCount > 0)
    {
        const uint32_t intersectBlocks = (activeRayCount + threadsPerBlock - 1) / threadsPerBlock;
        IntersectClosestKernel<<<intersectBlocks, threadsPerBlock>>>(m_PixelState, queueView, m_SceneBuffers.Device);
    }
    const uint32_t shadeBlocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    ShadeHitsKernel<<<shadeBlocks, threadsPerBlock>>>(m_PixelState, queueView);

    const glm::vec3 skyColor(0.5f, 0.7f, 1.0f);
    ShadeMissKernel<<<shadeBlocks, threadsPerBlock>>>(m_PixelState, queueView, skyColor);

    BlitRadianceKernel<<<blocks, threadsPerBlock>>>(m_PixelState, deviceBuffer, pixelCount);
    cudaDeviceSynchronize();

    std::vector<float> hostBuffer(static_cast<size_t>(pixelCount) * 3);
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
        reallocArray(m_PixelState.PendingEmission, sizeof(glm::vec3));
        reallocArray(m_PixelState.HitPositions, sizeof(glm::vec3));
        reallocArray(m_PixelState.HitNormals, sizeof(glm::vec3));
        reallocArray(m_PixelState.PixelIndices, sizeof(uint32_t));
        reallocArray(m_PixelState.PathDepth, sizeof(uint32_t));
        reallocArray(m_PixelState.RNGSeed, sizeof(uint32_t));
        reallocArray(m_PixelState.RNGStates, sizeof(curandState));
        reallocArray(m_PixelState.HitDistance, sizeof(float));
        reallocArray(m_PixelState.Materials, sizeof(MaterialHandle));
        reallocArray(m_PixelState.Alive, sizeof(uint8_t));
        reallocArray(m_PixelState.HasHit, sizeof(uint8_t));
        m_PixelState.Capacity = pixelCount;
    }

    auto ensureQueueCapacity = [&](RayQueueSOA& queue) {
        if (pixelCount > queue.Capacity)
        {
            reallocArray(queue.PixelIndices, sizeof(uint32_t));
            queue.Capacity = pixelCount;
        }
        if (!queue.Count)
        {
            cudaMalloc(&queue.Count, sizeof(uint32_t));
        }
    };

    ensureQueueCapacity(m_Queues.RayQueue);
    ensureQueueCapacity(m_Queues.EscapeQueue);
    ensureQueueCapacity(m_Queues.HitQueue);

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
    freePtr(m_PixelState.PendingEmission);
    freePtr(m_PixelState.HitPositions);
    freePtr(m_PixelState.HitNormals);
    freePtr(m_PixelState.PixelIndices);
    freePtr(m_PixelState.PathDepth);
    freePtr(m_PixelState.RNGSeed);
    freePtr(m_PixelState.RNGStates);
    freePtr(m_PixelState.HitDistance);
    freePtr(m_PixelState.Materials);
    freePtr(m_PixelState.Alive);
    freePtr(m_PixelState.HasHit);
    m_PixelState.Capacity = 0;

    auto freeQueue = [&](RayQueueSOA& queue) {
        freePtr(queue.PixelIndices);
        freePtr(queue.Count);
        queue.Capacity = 0;
    };

    freeQueue(m_Queues.RayQueue);
    freeQueue(m_Queues.EscapeQueue);
    freeQueue(m_Queues.HitQueue);

    freePtr(m_DeviceCamera);
}
