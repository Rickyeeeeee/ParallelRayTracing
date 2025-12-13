#pragma once

#include <core/core.h>
#include <core/renderer.h>

struct Tile
{
    uint32_t x0, y0, x1, y1;
};


class CPURenderer : public Renderer
{
public:
    CPURenderer() = default;
    virtual ~CPURenderer() = default;

    virtual void Init(Film& film, const Scene& scene, const Camera& camera) override;
    virtual void ProgressiveRender() override;
    void SetCamera(const Camera& camera) override;

private:

    glm::vec3 TraceRay(const Ray& ray, int depth);

private:
    Film* m_Film        = nullptr;
    const Scene* m_Scene      = nullptr;
    const Camera* m_Camera    = nullptr;

    glm::vec3 m_SkyLight{ 0.4f, 0.3f, 0.6f };
    glm::vec3 m_SkyLightDirection = glm::normalize(glm::vec3{ 1.0f, 1.0f, 1.0f });

    const int m_Depth = 20;
    const uint32_t m_tileSize = 16;
};
