#pragma once

#include <core/core.h>
#include <core/geometry.h>
#include <core/material_handle.h>
#include <core/renderer.h>
#include "soa.h"

struct PixelState
{
    Ray Ray{};
    glm::vec3 Throughput{ 1.0f };
    glm::vec3 Radiance{ 0.0f };
    glm::vec3 PendingEmission{ 0.0f };
    glm::vec3 HitPosition{ 0.0f };
    glm::vec3 HitNormal{ 0.0f };
    uint32_t PixelIndex = 0;
    uint32_t PathDepth = 0;
    uint32_t RNGSeed = 0;
    float HitDistance = 0.0f;
    MaterialHandle Material{};
    uint8_t Alive = 1;
    uint8_t HasHit = 0;
};

struct PixelStateSOA
{
    glm::vec3* RayOrigins = nullptr;
    glm::vec3* RayDirections = nullptr;
    glm::vec3* Throughput = nullptr;
    glm::vec3* Radiance = nullptr;
    glm::vec3* PendingEmission = nullptr;
    glm::vec3* HitPositions = nullptr;
    glm::vec3* HitNormals = nullptr;
    uint32_t* PixelIndices = nullptr;
    uint32_t* PathDepth = nullptr;
    uint32_t* RNGSeed = nullptr;
    curandState* RNGStates = nullptr;
    float* HitDistance = nullptr;
    MaterialHandle* Materials = nullptr;
    uint8_t* Alive = nullptr;
    uint8_t* HasHit = nullptr;
    uint32_t Capacity = 0;

    QUAL_GPU inline PixelState Load(uint32_t index) const;
    QUAL_GPU inline void Store(uint32_t index, const PixelState& state) const;
    QUAL_GPU inline PixelState operator[](uint32_t index) const { return Load(index); }
    QUAL_GPU inline Ray GetRay(uint32_t index) const;
};

struct RayQueueSOA
{
    uint32_t* PixelIndices = nullptr;
    uint32_t* Count = nullptr;
    uint32_t Capacity = 0;

    QUAL_GPU inline bool IsValid() const { return PixelIndices && Count && Capacity > 0; }
    QUAL_GPU inline uint32_t AllocateSlot() const;
    QUAL_GPU inline void Push(uint32_t pixelIndex) const;
    QUAL_GPU inline uint32_t Size() const { return (Count && Capacity > 0) ? *Count : 0u; }
};

struct WavefrontQueues
{
    RayQueueSOA RayQueue{};
    RayQueueSOA EscapeQueue{};
    RayQueueSOA HitQueue{};

    QUAL_GPU inline void PushRay(uint32_t pixelIndex) const { RayQueue.Push(pixelIndex); }
    QUAL_GPU inline void PushEscape(uint32_t pixelIndex) const { EscapeQueue.Push(pixelIndex); }
    QUAL_GPU inline void PushHit(uint32_t pixelIndex) const { HitQueue.Push(pixelIndex); }

    void ResetCounts(cudaStream_t stream = nullptr) const;
};

// Placeholder CUDA wavefront renderer that simply fills the film.
class CudaWavefrontRenderer : public Renderer
{
public:
    CudaWavefrontRenderer() = default;
    ~CudaWavefrontRenderer() override;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    void ProgressiveRender() override;

private:
    Film* m_Film = nullptr;
    const Scene* m_Scene = nullptr;
    const Camera* m_Camera = nullptr;

    uint32_t m_FrameIndex = 0;
    WavefrontSceneBuffers m_SceneBuffers{};

    PixelStateSOA m_PixelState{};
    WavefrontQueues m_Queues{};

    Camera* m_DeviceCamera = nullptr;

    void EnsureDeviceState(uint32_t pixelCount);
    void ReleaseDeviceState();
};
