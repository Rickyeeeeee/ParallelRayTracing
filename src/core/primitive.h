#pragma once

#include "shape.h"
#include "material.h"
#include <vector>

struct Primitive
{
    ShapeHandle Shape;
    MaterialHandle Material;
    Transform Transform;
};

class PrimitiveList
{
public:
    PrimitiveList() = default;

    void AddPrimitive(const Primitive& primitive);
    void AddCircle(const Primitive& primitive);
    void AddQuad(const Primitive& primitive);

    void Intersect(const Ray& ray, SurfaceInteraction* intersect) const;

    const std::vector<Primitive>& GetPrimitives() const;

private:
    std::vector<Primitive> m_Primitives;
    std::vector<Primitive> m_Circles;
    std::vector<Primitive> m_Quads;
};
