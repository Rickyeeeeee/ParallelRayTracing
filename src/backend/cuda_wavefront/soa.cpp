#include "soa.h"
#include <core/primitive.h>
#include <core/material.h>
#include <core/shape.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

namespace
{
template<typename T>
void ReleaseDevicePtr(T*& ptr)
{
    if (ptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
    }
}

void WavefrontSceneBuffers::Free()
{
    ReleaseDevicePtr(materials.albedo);
    ReleaseDevicePtr(materials.emission);
    ReleaseDevicePtr(materials.roughness);
    ReleaseDevicePtr(materials.ior);
    ReleaseDevicePtr(materials.type);
    materials.count = 0;

    ReleaseDevicePtr(circles.radius);
    circles.count = 0;

    ReleaseDevicePtr(quads.width);
    ReleaseDevicePtr(quads.height);
    ReleaseDevicePtr(quads.normal);
    quads.count = 0;

    ReleaseDevicePtr(primitives.shapeType);
    ReleaseDevicePtr(primitives.shapeIndex);
    ReleaseDevicePtr(primitives.materialIndex);
    ReleaseDevicePtr(primitives.transform);
    ReleaseDevicePtr(primitives.invTransform);
    primitives.count = 0;
}

namespace
{
template<typename T>
T* AllocateAndCopy(const std::vector<T>& host)
{
    if (host.empty())
        return nullptr;
    T* devicePtr = nullptr;
    cudaMalloc(&devicePtr, sizeof(T) * host.size());
    cudaMemcpy(devicePtr, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    return devicePtr;
}
} // namespace

WavefrontSceneBuffers BuildWavefrontSceneBuffers(const std::vector<Primitive>& primitives)
{
    WavefrontSceneBuffers buffers{};

    std::unordered_map<const void*, int> materialIndex;
    std::unordered_map<const void*, int> circleIndex;
    std::unordered_map<const void*, int> quadIndex;

    std::vector<glm::vec3> matAlbedo;
    std::vector<glm::vec3> matEmission;
    std::vector<float> matRoughness;
    std::vector<float> matIor;
    std::vector<MatType> matType;

    std::vector<float> circleRadius;
    std::vector<float> quadWidth;
    std::vector<float> quadHeight;
    std::vector<glm::vec3> quadNormal;

    std::vector<uint8_t> primShapeType;
    std::vector<int> primShapeIndex;
    std::vector<int> primMaterialIndex;
    std::vector<glm::mat4> primTransform;
    std::vector<glm::mat4> primInvTransform;

    matAlbedo.reserve(primitives.size());
    matEmission.reserve(primitives.size());
    matRoughness.reserve(primitives.size());
    matIor.reserve(primitives.size());
    matType.reserve(primitives.size());

    primShapeType.reserve(primitives.size());
    primShapeIndex.reserve(primitives.size());
    primMaterialIndex.reserve(primitives.size());
    primTransform.reserve(primitives.size());
    primInvTransform.reserve(primitives.size());

    for (const auto& prim : primitives)
    {
        // Material
        int mIndex = -1;
        if (prim.Material.IsValid())
        {
            auto it = materialIndex.find(prim.Material.Ptr);
            if (it == materialIndex.end())
            {
                mIndex = static_cast<int>(matType.size());
                materialIndex[prim.Material.Ptr] = mIndex;

                glm::vec3 albedo(1.0f);
                glm::vec3 emission(0.0f);
                float rough = 0.0f;
                float ior = 1.0f;
                MatType mt = prim.Material.Tag();

                prim.Material.dispatch([&](const auto* material) {
                    using MatT = std::remove_cv_t<std::remove_reference_t<decltype(*material)>>;
                    if constexpr (std::is_same_v<MatT, LambertianMaterial>)
                    {
                        albedo = material->GetAlbedo();
                    }
                    else if constexpr (std::is_same_v<MatT, MetalMaterial>)
                    {
                        albedo = material->GetAlbedo();
                        rough = material->GetRoughness();
                    }
                    else if constexpr (std::is_same_v<MatT, DielectricMaterial>)
                    {
                        ior = material->GetRefractionIndex();
                    }
                    else if constexpr (std::is_same_v<MatT, EmissiveMaterial>)
                    {
                        emission = material->GetEmission();
                    }
                });

                matAlbedo.push_back(albedo);
                matEmission.push_back(emission);
                matRoughness.push_back(rough);
                matIor.push_back(ior);
                matType.push_back(mt);
            }
            else
            {
                mIndex = it->second;
            }
        }
        else
        {
            mIndex = -1;
        }

        // Shape
        const auto shapeTag = prim.Shape.Tag();
        int sIndex = -1;
        prim.Shape.dispatch([&](const auto* shape) {
            using ShapeT = std::remove_cv_t<std::remove_reference_t<decltype(*shape)>>;
            if (!shape)
                return;

            if constexpr (std::is_same_v<ShapeT, Circle>)
            {
                auto it = circleIndex.find(shape);
                if (it == circleIndex.end())
                {
                    sIndex = static_cast<int>(circleRadius.size());
                    circleIndex[shape] = sIndex;
                    circleRadius.push_back(shape->getRadius());
                }
                else
                {
                    sIndex = it->second;
                }
            }
            else if constexpr (std::is_same_v<ShapeT, Quad>)
            {
                auto it = quadIndex.find(shape);
                if (it == quadIndex.end())
                {
                    sIndex = static_cast<int>(quadWidth.size());
                    quadIndex[shape] = sIndex;
                    quadWidth.push_back(shape->GetWidth());
                    quadHeight.push_back(shape->GetHeight());
                    quadNormal.push_back(shape->GetNormal());
                }
                else
                {
                    sIndex = it->second;
                }
            }
        });

        primShapeType.push_back(static_cast<uint8_t>(shapeTag));
        primShapeIndex.push_back(sIndex);
        primMaterialIndex.push_back(mIndex);
        primTransform.push_back(prim.Transform.GetMat());
        primInvTransform.push_back(prim.Transform.GetInvMat());
    }

    // Upload materials
    buffers.materials.albedo = AllocateAndCopy(matAlbedo);
    buffers.materials.emission = AllocateAndCopy(matEmission);
    buffers.materials.roughness = AllocateAndCopy(matRoughness);
    buffers.materials.ior = AllocateAndCopy(matIor);
    buffers.materials.type = AllocateAndCopy(matType);
    buffers.materials.count = static_cast<int>(matType.size());

    // Upload shapes
    buffers.circles.radius = AllocateAndCopy(circleRadius);
    buffers.circles.count = static_cast<int>(circleRadius.size());

    buffers.quads.width = AllocateAndCopy(quadWidth);
    buffers.quads.height = AllocateAndCopy(quadHeight);
    buffers.quads.normal = AllocateAndCopy(quadNormal);
    buffers.quads.count = static_cast<int>(quadWidth.size());

    // Upload primitives
    buffers.primitives.shapeType = AllocateAndCopy(primShapeType);
    buffers.primitives.shapeIndex = AllocateAndCopy(primShapeIndex);
    buffers.primitives.materialIndex = AllocateAndCopy(primMaterialIndex);
    buffers.primitives.transform = AllocateAndCopy(primTransform);
    buffers.primitives.invTransform = AllocateAndCopy(primInvTransform);
    buffers.primitives.count = static_cast<int>(primShapeType.size());

    return buffers;
}
