#pragma once

#include "geometry.h"
#include "surface_interaction.h"
#include "math.h"
#include <cassert>

enum MatType : uint8_t {
    NONE = 0,
    LAMBERTIAN = 1,
    METAL = 2,
    DIELECTRIC = 3,
    EMISSIVE = 4
};

class Material
{
public:
    Material() = default;
    virtual ~Material() = default;

    virtual bool Scatter(const Ray& inRay, const SurfaceInteraction& intersection, glm::vec3& attenuation, Ray& outRay) const
    {
        return false;
    }

    virtual void Emit(glm::vec3& emittedColor) const
    {
        emittedColor = glm::vec3(0.0f); // Default to no emission
    }
};

// outRay = reflect ray + subsurface scattering ray
// reflect ray = 0.0f
// subsurface scattering ray = 1.0f (Randomness because of subsurface scattering)
class LambertianMaterial : public Material
{
public:
    LambertianMaterial(const glm::vec3& albedo) : m_Albedo(albedo) {};

    virtual bool Scatter(const Ray& inRay, const SurfaceInteraction& interaction, glm::vec3& attenuation, Ray& outRay) const override
    {
        auto scatterDirection = interaction.Normal + RandomUnitVector();

        auto& e = scatterDirection;
        auto s = 1e-8;
        if ((std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s))
        {
            scatterDirection = interaction.Normal;
        }

        outRay.Origin = interaction.Position;
        outRay.Direction = glm::normalize(scatterDirection);

        attenuation = m_Albedo;

        return true;
    }

    glm::vec3 GetAlbedo() const { return m_Albedo; }

private:

    glm::vec3 m_Albedo;
};

// outRay = reflect ray + subsurface scattering ray
// reflect ray = 1.0f (randomness because of PBR roughness)
// subsurface scattering ray = 0.0f
class MetalMaterial : public Material
{
public:
    MetalMaterial(const glm::vec3& albedo, float metallic=0.0f) : m_Albedo(albedo), m_Roughness(metallic) {}

    virtual bool Scatter(const Ray& inRay, const SurfaceInteraction& interaction, glm::vec3& attenuation, Ray& outRay) const override
    {
        auto reflectedDir = glm::reflect(inRay.Direction, interaction.Normal);
        reflectedDir = glm::normalize(reflectedDir) + m_Roughness * RandomUnitVector();
        // reflectedDir = glm::normalize(reflectedDir);

        outRay.Origin = interaction.Position;
        outRay.Direction = glm::normalize(reflectedDir);

        attenuation = m_Albedo;

        return glm::dot(outRay.Direction, interaction.Normal) > 0.0f;
    }

    glm::vec3 GetAlbedo() const { return m_Albedo; }
    float GetRoughness() const { return m_Roughness; }

private:
    glm::vec3 m_Albedo;
    float m_Roughness; // Fresnel reflectance
};

class DielectricMaterial : public Material
{
public:
    DielectricMaterial(float refractionIndex) : m_RefractionIndex(refractionIndex) {};

    virtual bool Scatter(const Ray& inRay, const SurfaceInteraction& interaction, glm::vec3& attenuation, Ray& outRay) const override
    {
        attenuation = glm::vec3 { 1.0f, 1.0f, 1.0f };
        float ri = interaction.IsFrontFace ? (1.0f / m_RefractionIndex) : m_RefractionIndex;

        glm::vec3 unit_direction = inRay.Direction;
        float cos_theta = std::fmin(glm::dot(-unit_direction, interaction.Normal), 1.0f);
        float sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        glm::vec3 direction;

        if (cannot_refract || this->fresnelReflectance(cos_theta, ri) > Random())
            direction = glm::reflect(unit_direction, interaction.Normal);
        else
            direction = Reflect(unit_direction, interaction.Normal, ri);

        outRay = Ray{ interaction.Position, direction };

        return true;
    }

    float GetRefractionIndex() const { return m_RefractionIndex; }

private:
    float m_RefractionIndex;

    static float fresnelReflectance(float cosine, float refractionIndex) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};

class EmissiveMaterial : public Material
{
public:
    EmissiveMaterial(const glm::vec3& color) : m_Color(color) {}

    virtual void Emit(glm::vec3& emittedColor) const override
    {
        emittedColor = m_Color;
    }
    
    glm::vec3 GetEmission() const { return m_Color; }

private:
    glm::vec3 m_Color;
};

inline MatType DeduceMaterialType(const Material* material)
{
    if (dynamic_cast<const LambertianMaterial*>(material))
        return MatType::LAMBERTIAN;
    if (dynamic_cast<const MetalMaterial*>(material))
        return MatType::METAL;
    if (dynamic_cast<const DielectricMaterial*>(material))
        return MatType::DIELECTRIC;
    if (dynamic_cast<const EmissiveMaterial*>(material))
        return MatType::EMISSIVE;
    return MatType::NONE;
}

inline MaterialHandle MakeMaterialHandle(const Material* material, MatType hint)
{
    MaterialHandle handle;
    if (!material)
        return handle;

    MatType type = hint;
    if (type == MatType::NONE)
        type = DeduceMaterialType(material);

    handle.type = static_cast<uint8_t>(type);
    switch (type)
    {
    case MatType::LAMBERTIAN:
        handle.ptr = static_cast<const LambertianMaterial*>(material);
        break;
    case MatType::METAL:
        handle.ptr = static_cast<const MetalMaterial*>(material);
        break;
    case MatType::DIELECTRIC:
        handle.ptr = static_cast<const DielectricMaterial*>(material);
        break;
    case MatType::EMISSIVE:
        handle.ptr = static_cast<const EmissiveMaterial*>(material);
        break;
    default:
        handle.Reset();
        break;
    }

    return handle;
}

template<typename F>
QUAL_CPU_GPU decltype(auto) MaterialHandle::dispatch(F&& func) const
{
    switch (type)
    {
    case static_cast<uint8_t>(MatType::LAMBERTIAN):
        return func(static_cast<const LambertianMaterial*>(ptr));
    case static_cast<uint8_t>(MatType::METAL):
        return func(static_cast<const MetalMaterial*>(ptr));
    case static_cast<uint8_t>(MatType::DIELECTRIC):
        return func(static_cast<const DielectricMaterial*>(ptr));
    case static_cast<uint8_t>(MatType::EMISSIVE):
        return func(static_cast<const EmissiveMaterial*>(ptr));
    default:
#ifndef __CUDA_ARCH__
        assert(false && "Invalid MaterialHandle dispatch");
#endif
        return func(static_cast<const LambertianMaterial*>(nullptr));
    }
}
