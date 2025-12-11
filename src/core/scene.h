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

    std::vector<PrimitiveHandleView> getCircleViews() const { return m_Primitives.getCircleViews(); }
    std::vector<PrimitiveHandleView> getQuadViews() const { return m_Primitives.getQuadViews(); }
    const std::vector<PrimitiveHandleView>& getPrimitiveViews() const { return m_Primitives.getPrimitiveViews(); }

private:
    MaterialPool m_Materials;
    ShapePool m_Shapes;
    PrimitiveList m_Primitives;
    double m_SkyLightIntensity = 1.0;
};
