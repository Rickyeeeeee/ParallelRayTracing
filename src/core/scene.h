#pragma once

#include "primitive.h"
#include <vector>

class Scene
{
public:
    Scene();

    void Intersect(const Ray& ray, SurfaceInteraction* intersect) const
    {
        m_Primitives.Intersect(ray, intersect);
    }

    double GetSkyLightIntensity() const { return m_SkyLightIntensity; }

    const std::vector<Primitive>& GetPrimitives() const { return m_Primitives.GetPrimitives(); }

private:
    MaterialPool m_Materials;
    ShapePool m_Shapes;
    PrimitiveList m_Primitives;
    double m_SkyLightIntensity = 1.0;
};
