#pragma once

#include "geometry.h"
#include "surface_interaction.h"
#include "math.h"
#include <cassert>
#include <deque>

enum MatType : uint8_t {
    NONE = 0,
    LAMBERTIAN = 1,
    METAL = 2,
    DIELECTRIC = 3,
    EMISSIVE = 4
};

struct LambertianMaterial
{
    LambertianMaterial() = default;
    explicit LambertianMaterial(const glm::vec3& albedo) : m_Albedo(albedo) {}

    QUAL_CPU_GPU bool Scatter(const Ray& inRay, const SurfaceInteraction& interaction, glm::vec3& attenuation, Ray& outRay, curandState* rngState = nullptr) const
    {
        auto scatterDirection = interaction.Normal + RandomUnitVector(rngState);

        auto& e = scatterDirection;
        auto s = 1e-8;
        if ((glm::abs(e[0]) < s) && (glm::abs(e[1]) < s) && (glm::abs(e[2]) < s))
        {
            scatterDirection = interaction.Normal;
        }

        outRay.Origin = interaction.Position;
        outRay.Direction = glm::normalize(scatterDirection);
        attenuation = m_Albedo;
        return true;
    }

    QUAL_CPU_GPU void Emit(glm::vec3& emittedColor) const
    {
        emittedColor = glm::vec3(0.0f);
    }

    glm::vec3 GetAlbedo() const { return m_Albedo; }

    glm::vec3 m_Albedo{ 0.0f };
};

struct MetalMaterial
{
    MetalMaterial() = default;
    MetalMaterial(const glm::vec3& albedo, float metallic=0.0f) : m_Albedo(albedo), m_Roughness(metallic) {}

    QUAL_CPU_GPU bool Scatter(const Ray& inRay, const SurfaceInteraction& interaction, glm::vec3& attenuation, Ray& outRay, curandState* rngState = nullptr) const
    {
        auto reflectedDir = glm::reflect(inRay.Direction, interaction.Normal);
        reflectedDir = glm::normalize(reflectedDir) + m_Roughness * RandomUnitVector(rngState);

        outRay.Origin = interaction.Position;
        outRay.Direction = glm::normalize(reflectedDir);
        attenuation = m_Albedo;
        return glm::dot(outRay.Direction, interaction.Normal) > 0.0f;
    }

    QUAL_CPU_GPU void Emit(glm::vec3& emittedColor) const
    {
        emittedColor = glm::vec3(0.0f);
    }

    glm::vec3 GetAlbedo() const { return m_Albedo; }
    float GetRoughness() const { return m_Roughness; }

    glm::vec3 m_Albedo{ 1.0f };
    float m_Roughness = 0.0f;
};

struct DielectricMaterial
{
    DielectricMaterial() = default;
    explicit DielectricMaterial(float refractionIndex) : m_RefractionIndex(refractionIndex) {}

    QUAL_CPU_GPU bool Scatter(const Ray& inRay, const SurfaceInteraction& interaction, glm::vec3& attenuation, Ray& outRay, curandState* rngState = nullptr) const
    {
        attenuation = glm::vec3 { 1.0f, 1.0f, 1.0f };
        float ri = interaction.IsFrontFace ? (1.0f / m_RefractionIndex) : m_RefractionIndex;

        glm::vec3 unit_direction = inRay.Direction;
        float cos_theta = glm::min(glm::dot(-unit_direction, interaction.Normal), 1.0f);
        float sin_theta = glm::sqrt(1.0f - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        glm::vec3 direction;

        if (cannot_refract || fresnelReflectance(cos_theta, ri) > Random(rngState))
            direction = glm::reflect(unit_direction, interaction.Normal);
        else
            direction = Reflect(unit_direction, interaction.Normal, ri);

        outRay = Ray{ interaction.Position, direction };
        return true;
    }

    QUAL_CPU_GPU void Emit(glm::vec3& emittedColor) const
    {
        emittedColor = glm::vec3(0.0f);
    }

    float GetRefractionIndex() const { return m_RefractionIndex; }

private:
    QUAL_CPU_GPU static float fresnelReflectance(float cosine, float refractionIndex) {
        auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
        r0 = r0*r0;
        return r0 + (1-r0)*glm::pow((1 - cosine),5);
    }

    float m_RefractionIndex = 1.0f;
};

struct EmissiveMaterial
{
    EmissiveMaterial() = default;
    explicit EmissiveMaterial(const glm::vec3& color) : m_Color(color) {}

    QUAL_CPU_GPU bool Scatter(const Ray&, const SurfaceInteraction&, glm::vec3&, Ray&, curandState* = nullptr) const
    {
        return false;
    }

    QUAL_CPU_GPU void Emit(glm::vec3& emittedColor) const
    {
        emittedColor = m_Color;
    }

    glm::vec3 GetEmission() const { return m_Color; }

    glm::vec3 m_Color{ 0.0f };
};

inline MaterialHandle MakeMaterialHandle(MatType type, const void* material)
{
    MaterialHandle handle;
    handle.Type = static_cast<uint8_t>(type);
    handle.Ptr = material;
    return handle;
}

template<typename F>
QUAL_CPU_GPU decltype(auto) MaterialHandle::dispatch(F&& func) const
{
    switch (static_cast<MatType>(Type))
    {
    case MatType::LAMBERTIAN:
        return func(static_cast<const LambertianMaterial*>(Ptr));
    case MatType::METAL:
        return func(static_cast<const MetalMaterial*>(Ptr));
    case MatType::DIELECTRIC:
        return func(static_cast<const DielectricMaterial*>(Ptr));
    case MatType::EMISSIVE:
        return func(static_cast<const EmissiveMaterial*>(Ptr));
    default:
#ifndef __CUDA_ARCH__
        assert(false && "Invalid MaterialHandle dispatch");
#endif
        return func(static_cast<const LambertianMaterial*>(nullptr));
    }
}

struct MaterialPool
{
    std::deque<LambertianMaterial> LambertianMaterials;
    std::deque<MetalMaterial> MetalMaterials;
    std::deque<DielectricMaterial> DielectricMaterials;
    std::deque<EmissiveMaterial> EmissiveMaterials;

    MaterialHandle AddLambertian(const glm::vec3& albedo)
    {
        LambertianMaterials.emplace_back(albedo);
        return MakeMaterialHandle(MatType::LAMBERTIAN, &LambertianMaterials.back());
    }

    MaterialHandle AddMetal(const glm::vec3& albedo, float roughness)
    {
        MetalMaterials.emplace_back(albedo, roughness);
        return MakeMaterialHandle(MatType::METAL, &MetalMaterials.back());
    }

    MaterialHandle AddDielectric(float ri)
    {
        DielectricMaterials.emplace_back(ri);
        return MakeMaterialHandle(MatType::DIELECTRIC, &DielectricMaterials.back());
    }

    MaterialHandle AddEmissive(const glm::vec3& emission)
    {
        EmissiveMaterials.emplace_back(emission);
        return MakeMaterialHandle(MatType::EMISSIVE, &EmissiveMaterials.back());
    }

    template<typename F>
    void Dispatch(const MaterialHandle& handle, F&& func) const
    {
        handle.dispatch(std::forward<F>(func));
    }

};
