#pragma once

#include <core/core.h>
#include <core/renderer.h>

struct GPUCamera {
    glm::vec3 position;
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;
    float  focal;
    uint32_t width;
    uint32_t height;

    QUAL_CPU_GPU 
    Ray GetCameraRay(float u, float v) const {
        glm::vec3 rayDir = glm::normalize(forward * focal + right * (u - (float)width * 0.5f) + up * (v - (float)height * 0.5f));
        Ray ray;
        ray.Direction = rayDir;
        ray.Origin = position;
        return ray;
    }
};

struct GPUSphere {
    Transform transform;
    float radius;
    int material;
};

struct GPUQuad {
    Transform transform;
    glm::vec3 normal;
    float width, height;
};

GPUCamera* CreateGPUCamera(const Camera* cam);

// Placeholder CUDA megakernel renderer that just writes test pixels.
class CudaMegakernelRenderer : public Renderer
{
public:
    CudaMegakernelRenderer() = default;
    ~CudaMegakernelRenderer() override = default;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    void ProgressiveRender() override;

    GPUSphere* ConvertCirclesToGPU();

private:
    Film* m_Film = nullptr;
    const Scene* m_Scene = nullptr;
    const Camera* m_Camera = nullptr;
};
