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
    ~CudaMegakernelRenderer() override = default;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    void ProgressiveRender() override;

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
};
