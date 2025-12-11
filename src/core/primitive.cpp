#include "primitive.h"
#include <limits>

void PrimitiveList::AddPrimitive(const Primitive& primitive)
{
    m_Primitives.push_back(primitive);
}

void PrimitiveList::AddCircle(const Primitive& primitive)
{
    AddPrimitive(primitive);
    m_Circles.push_back(primitive);
}

void PrimitiveList::AddQuad(const Primitive& primitive)
{
    AddPrimitive(primitive);
    m_Quads.push_back(primitive);
}

void PrimitiveList::Intersect(const Ray& ray, SurfaceInteraction* intersect) const
{
    float minDistance = std::numeric_limits<float>::max();
    SurfaceInteraction bestIntersection;

    for (const auto& primitive : m_Primitives)
    {
        Ray rayLocal;
        rayLocal.Origin = TransformPoint(primitive.Transform.GetInvMat(), ray.Origin);
        rayLocal.Direction = TransformNormal(primitive.Transform.GetMat(), ray.Direction);

        SurfaceInteraction si;
        const bool hit = primitive.Shape.dispatch([&](const auto* shape) {
            if (!shape)
                return false;
            shape->Intersect(rayLocal, &si);
            return si.HasIntersection;
        });

        if (!hit || !si.HasIntersection)
            continue;

        si.Position = TransformPoint(primitive.Transform.GetMat(), si.Position);
        si.Normal = TransformNormal(primitive.Transform.GetInvMat(), si.Normal);
        si.Material = primitive.Material;

        auto disVec = ray.Origin - si.Position;
        auto distance2 = glm::dot(disVec, disVec);
        if (distance2 < minDistance)
        {
            minDistance = distance2;
            bestIntersection = si;
        }
    }

    if (minDistance < std::numeric_limits<float>::max())
    {
        *intersect = bestIntersection;
    }
    else
    {
        intersect->HasIntersection = false;
    }
}

const std::vector<Primitive>& PrimitiveList::getPrimitiveViews() const
{
    return m_Primitives;
}
