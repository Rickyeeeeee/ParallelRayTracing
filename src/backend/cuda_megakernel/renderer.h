#pragma once

#include <core/core.h>
#include <core/renderer.h>

struct GPUMaterial {
    MatType type;
    glm::vec3 color;       // Emissive, Metal, Lambertian
    float refractionIndex; // Dielectric
    float roughness;       // Metal
};

struct GPUSphere {
    Transform transform;
    float radius;
    int material;
    GPUMaterial matType;
};

struct GPUQuad {
    Transform transform;
    glm::vec3 normal;
    float width, height;
    GPUMaterial matType;
};

// Placeholder CUDA megakernel renderer that just writes test pixels.
class CudaMegakernelRenderer : public Renderer
{
public:
    CudaMegakernelRenderer() = default;
    ~CudaMegakernelRenderer() override = default;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    void ProgressiveRender() override;

    GPUSphere* ConvertCirclesToGPU();
    GPUQuad* ConvertQuadsToGPU();

private:
    Film* m_Film = nullptr;
    const Scene* m_Scene = nullptr;
    const Camera* m_Camera = nullptr;
};
