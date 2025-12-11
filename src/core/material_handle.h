#pragma once

#include "core/tagged_pointer.h"

struct Ray;
struct SurfaceInteraction;

struct LambertianMaterial;
struct MetalMaterial;
struct DielectricMaterial;
struct EmissiveMaterial;

enum MatType : uint8_t {
    NONE = 0,
    LAMBERTIAN = 1,
    METAL = 2,
    DIELECTRIC = 3,
    EMISSIVE = 4
};

using MaterialHandleBase = TaggedPointer<
    MatType,
    TaggedType<MatType, MatType::LAMBERTIAN, LambertianMaterial>,
    TaggedType<MatType, MatType::METAL, MetalMaterial>,
    TaggedType<MatType, MatType::DIELECTRIC, DielectricMaterial>,
    TaggedType<MatType, MatType::EMISSIVE, EmissiveMaterial>>;

struct MaterialHandle : public MaterialHandleBase
{
    using MaterialHandleBase::MaterialHandleBase;

    QUAL_CPU_GPU void Emit(glm::vec3& emittedColor) const;
    QUAL_CPU_GPU bool Scatter(const Ray& inRay, const SurfaceInteraction& interaction, glm::vec3& attenuation, Ray& outRay, curandState* rngState = nullptr) const;
};

MaterialHandle MakeMaterialHandle(MatType type, const void* material);
