#pragma once

#include <core/core.h>
#include <core/renderer.h>
#include <core/primitive.h>
#include <core/material.h>
#include <core/shape.h>
#include <vector>

// Placeholder CUDA megakernel renderer that just writes test pixels.
class CudaMegakernelRenderer : public Renderer
{
public:
    CudaMegakernelRenderer() = default;
    ~CudaMegakernelRenderer() override;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    void ProgressiveRender() override;
    void SetCamera(const Camera& camera) override;

private:
    struct DeviceSceneData
    {
        Primitive* primitives = nullptr;
        int primitiveCount = 0;

        std::vector<void*> materialAllocs;
        std::vector<void*> shapeAllocs;
    };

    DeviceSceneData UploadSceneData() const;
    void FreeDeviceScene(DeviceSceneData& data) const;

 private:
    Film* m_Film = nullptr;
    const Scene* m_Scene = nullptr;
    const Camera* m_Camera = nullptr;

    void* m_RNGStates = nullptr;
    uint32_t m_RNGCapacity = 0;
    uint64_t m_RNGSeed = 0;

    float* deviceBuffer = nullptr;
    Camera* d_cam = nullptr;
    DeviceSceneData deviceScene;
};
