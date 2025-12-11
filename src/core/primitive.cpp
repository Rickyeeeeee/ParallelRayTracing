#include "primitive.h"
#include <limits>

void PrimitiveList::AddPrimitive(const PrimitiveHandleView& primitive)
{
    m_Primitives.push_back(primitive);
}

void PrimitiveList::AddCircle(const PrimitiveHandleView& primitive)
{
    AddPrimitive(primitive);
    m_Circles.push_back(primitive);
}

void PrimitiveList::AddQuad(const PrimitiveHandleView& primitive)
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
        rayLocal.Origin = TransformPoint(primitive.transform.GetInvMat(), ray.Origin);
        rayLocal.Direction = TransformNormal(primitive.transform.GetMat(), ray.Direction);

        SurfaceInteraction si;
        const bool hit = primitive.shape.dispatch([&](const auto* shape) {
            if (!shape)
                return false;
            shape->Intersect(rayLocal, &si);
            return si.HasIntersection;
        });

        if (!hit || !si.HasIntersection)
            continue;

        si.Position = TransformPoint(primitive.transform.GetMat(), si.Position);
        si.Normal = TransformNormal(primitive.transform.GetInvMat(), si.Normal);
        si.Material = primitive.material;

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

std::vector<PrimitiveHandleView> PrimitiveList::getCircleViews() const
{
    return m_Circles;
}

std::vector<PrimitiveHandleView> PrimitiveList::getQuadViews() const
{
    return m_Quads;
}

const std::vector<PrimitiveHandleView>& PrimitiveList::getPrimitiveViews() const
{
    return m_Primitives;
}
