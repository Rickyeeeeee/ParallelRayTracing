#pragma once

#include "shape.h"
#include "material.h"
#include <vector>

struct PrimitiveHandleView
{
    ShapeHandle shape;
    MaterialHandle material;
    Transform transform;
};

class PrimitiveList
{
public:
    PrimitiveList() = default;

    void AddPrimitive(const PrimitiveHandleView& primitive);
    void AddCircle(const PrimitiveHandleView& primitive);
    void AddQuad(const PrimitiveHandleView& primitive);

    void Intersect(const Ray& ray, SurfaceInteraction* intersect) const;

    std::vector<PrimitiveHandleView> getCircleViews() const;
    std::vector<PrimitiveHandleView> getQuadViews() const;
    const std::vector<PrimitiveHandleView>& getPrimitiveViews() const;

private:
    std::vector<PrimitiveHandleView> m_Primitives;
    std::vector<PrimitiveHandleView> m_Circles;
    std::vector<PrimitiveHandleView> m_Quads;
};
