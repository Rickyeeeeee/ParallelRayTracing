#pragma once

#include <core/core.h>
#include <core/renderer.h>
#include "soa.h"

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
};
