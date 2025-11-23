#pragma once

#include <core/core.h>
#include <core/renderer.h>

#include <optix.h>

class OptixRenderer : public Renderer
{
public:
    OptixRenderer() = default;
    ~OptixRenderer() override = default;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    void ProgressiveRender() override;

private:
    Film* m_Film = nullptr;
    const Scene* m_Scene = nullptr;
    const Camera* m_Camera = nullptr;
};
