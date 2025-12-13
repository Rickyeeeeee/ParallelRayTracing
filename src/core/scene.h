#pragma once

#include "primitive.h"
#include <vector>

enum class ScenePreset
{
    DEFAULT,
    LIGHT_TEST,
    MATERIAL_TEST,
    CORNELL,
    RANDOM_BALLS
};

class Scene
{
public:
    explicit Scene(ScenePreset preset = ScenePreset::CORNELL);

    void Intersect(const Ray& ray, SurfaceInteraction* intersect) const
    {
        m_Primitives.Intersect(ray, intersect);
    }

    double GetSkyLightIntensity() const { return m_SkyLightIntensity; }

    const std::vector<Primitive>& GetPrimitives() const
    {
        return m_Primitives.GetPrimitives();
    }

private:
    // helpers
    Transform MakeTransform(
        const glm::vec3& scale,
        const glm::vec3& eulerAnglesDeg,
        const glm::vec3& translation);

    void AddPrimitive(
        const ShapeHandle& shape,
        const MaterialHandle& material,
        const Transform& transform,
        ShapeType shapeType);

    // scene variants
    void InitDefault();
    void InitLightTest();
    void InitMaterialTest();
    void InitCornell();
    void InitRandomBalls();

private:
    MaterialPool  m_Materials;
    ShapePool     m_Shapes;
    PrimitiveList m_Primitives;
    double        m_SkyLightIntensity = 1.0;
};
