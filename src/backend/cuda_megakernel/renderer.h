#pragma once

#include <core/core.h>
#include <core/renderer.h>

#include <vector>

struct GPUMaterial {
    MatType type = MatType::NONE;
    glm::vec3 color{ 0.0f };  // Emissive, Metal, Lambertian
    float refractionIndex = 1.0f; // Dielectric
    float roughness = 0.0f;       // Metal
};

struct GPUPrimitive {
    ShapeType shapeType = ShapeType::CIRCLE;
    Transform transform;
    GPUMaterial material;
    float param0 = 0.0f;      // radius (circle) or width (quad)
    float param1 = 0.0f;      // height (quad)
    glm::vec3 normal{ 0.0f, 1.0f, 0.0f }; // quad normal
};

struct GPUPrimitiveBuffer {
    GPUPrimitive* devicePtr = nullptr;
    int count = 0;
};

// Placeholder CUDA megakernel renderer that just writes test pixels.
class CudaMegakernelRenderer : public Renderer
{
public:
    CudaMegakernelRenderer() = default;
    ~CudaMegakernelRenderer() override = default;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    void ProgressiveRender() override;

    GPUPrimitiveBuffer UploadPrimitives() const;
    GPUPrimitive ConvertPrimitive(const PrimitiveHandleView& view) const;
    GPUMaterial ConvertMaterial(const MaterialHandle& handle) const;

private:
    Film* m_Film = nullptr;
    const Scene* m_Scene = nullptr;
    const Camera* m_Camera = nullptr;
};
