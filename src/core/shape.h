#pragma once

#include "geometry.h"
#include "surface_interaction.h"
#include <cassert>
#include <deque>
#include <cmath>
#include "tagged_pointer.h"

enum class ShapeType : uint8_t
{
    CIRCLE = 0,
    QUAD,
    TRIANGLE
};

class Circle
{
public:
    explicit Circle(float radius=1.0f) : m_Radius(radius) {}

    QUAL_CPU_GPU void Intersect(const Ray& ray, SurfaceInteraction* intersect) const;
    QUAL_CPU_GPU AABB GetAABB(Transform* transform) const;

    float getRadius() const { return m_Radius; }

private:
    float m_Radius{ 1.0f };
};

class Quad
{
public:
    Quad(float width=1.0f, float height=1.0f) : m_Width(width), m_Height(height) {}

    QUAL_CPU_GPU void Intersect(const Ray& ray, SurfaceInteraction* intersect) const;
    QUAL_CPU_GPU AABB GetAABB(Transform* transform) const;

    float GetWidth() const { return m_Width; }
    float GetHeight() const { return m_Height; }
    glm::vec3 GetNormal() const { return m_Normal; }

private:
    float m_Width;
    float m_Height;
    const glm::vec3 m_Normal = glm::vec3(0.0f, 1.0f, 0.0f);
};

class Triangle
{
public:
    Triangle() = default;
    Triangle(
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2,
        const glm::vec3& n0,
        const glm::vec3& n1,
        const glm::vec3& n2,
        const glm::vec2& uv0,
        const glm::vec2& uv1,
        const glm::vec2& uv2)
    {
        m_Vertices[0] = v0;
        m_Vertices[1] = v1;
        m_Vertices[2] = v2;
        m_Normals[0] = n0;
        m_Normals[1] = n1;
        m_Normals[2] = n2;
        m_UVs[0] = uv0;
        m_UVs[1] = uv1;
        m_UVs[2] = uv2;
    }

    QUAL_CPU_GPU void Intersect(const Ray& ray, SurfaceInteraction* intersect) const;
    QUAL_CPU_GPU AABB GetAABB(Transform* transform) const;

private:
    glm::vec3 m_Vertices[3]{};
    glm::vec3 m_Normals[3]{};
    glm::vec2 m_UVs[3]{};
};


using ShapeHandleBase = TaggedPointer<
    ShapeType,
    TaggedType<ShapeType, ShapeType::CIRCLE, Circle>,
    TaggedType<ShapeType, ShapeType::QUAD, Quad>,
    TaggedType<ShapeType, ShapeType::TRIANGLE, Triangle>>;

struct ShapeHandle : public ShapeHandleBase
{
    using ShapeHandleBase::ShapeHandleBase;

    QUAL_CPU_GPU bool Intersect(const Ray& ray, SurfaceInteraction* intersect) const;
    QUAL_CPU_GPU AABB GetAABB(Transform* transform) const;
};

ShapeHandle MakeShapeHandle(ShapeType type, const void* shape);


inline ShapeHandle MakeShapeHandle(ShapeType type, const void* shape)
{
    return ShapeHandle{ type, shape };
}

struct ShapePool
{
    std::deque<Circle> Circles;
    std::deque<Quad> Quads;

    ShapeHandle AddCircle(float radius)
    {
        Circles.emplace_back(radius);
        return MakeShapeHandle(ShapeType::CIRCLE, &Circles.back());
    }

    ShapeHandle AddQuad(float width, float height)
    {
        Quads.emplace_back(width, height);
        return MakeShapeHandle(ShapeType::QUAD, &Quads.back());
    }

    const std::deque<Circle>& GetCircles() const { return Circles; }
    const std::deque<Quad>& GetQuads() const { return Quads; }
};

constexpr float kShapeRayTMin = 0.001f;

QUAL_CPU_GPU inline bool ShapeHandle::Intersect(const Ray& ray, SurfaceInteraction* intersect) const
{
    if (!IsValid() || !intersect)
    {
        if (intersect)
            intersect->HasIntersection = false;
        return false;
    }
    return dispatch([&](const auto* shape) {
        if (!shape || !intersect)
            return false;
        shape->Intersect(ray, intersect);
        return intersect->HasIntersection;
    });
}

QUAL_CPU_GPU inline AABB ShapeHandle::GetAABB(Transform* transform) const
{
    if (!IsValid())
        return AABB{};
    return dispatch([&](const auto* shape) {
        if (!shape)
            return AABB{};
        return shape->GetAABB(transform);
    });
}

QUAL_CPU_GPU inline void Circle::Intersect(const Ray &ray, SurfaceInteraction* intersect) const
{
    auto l = ray.Origin;
    float a = glm::dot(ray.Direction, ray.Direction);
    float b = 2.0f * glm::dot(l, ray.Direction);
    float c = glm::dot(l,l) - m_Radius * m_Radius;

    float discriminant = b * b - 4.0f * a * c;

    if (discriminant >= 0.0f)
    {
        float t1 = (-b + sqrtf(discriminant)) / (2 * a);
        float t2 = (-b - sqrtf(discriminant)) / (2 * a);
        float t = 0.0f;

        intersect->HasIntersection = true;
        if (t1 >= kShapeRayTMin && t2 >= kShapeRayTMin)
        {
            t = t1 < t2 ? t1 : t2;
            intersect->IsFrontFace = true;
        }
        else if (t1 >= kShapeRayTMin)
        {
            t = t1;
            intersect->IsFrontFace = false;
        }
        else if (t2 >= kShapeRayTMin)
        {
            t = t2;
            intersect->IsFrontFace = false;
        }
        else
        {
            intersect->HasIntersection = false;
        }
        auto intersectPoint = ray.Origin + ray.Direction * t;
        auto normal = glm::normalize(intersectPoint);
        if (!intersect->IsFrontFace)
            normal *= -1.0f;
        intersect->Position = intersectPoint;
        intersect->Normal = normal;
    }
    else
    {
        intersect->HasIntersection = false;
    }
}

QUAL_CPU_GPU inline AABB Circle::GetAABB(Transform* transform) const
{
    return AABB{
        glm::vec3{ -m_Radius },
        glm::vec3{  m_Radius }
    }.TransformAndBound(transform);
}

QUAL_CPU_GPU inline void Quad::Intersect(const Ray& ray, SurfaceInteraction* intersect) const
{
    if (fabs(ray.Direction.y) < 1e-8f)
    {
        intersect->HasIntersection = false;
        return;
    }

    auto t = -ray.Origin.y / ray.Direction.y;

    auto p = ray.Origin + ray.Direction * t;

    float halfWidth = m_Width / 2.0f;
    float halfHeight = m_Height / 2.0f;

    if (t > kShapeRayTMin && (p.x * p.x < halfWidth * halfWidth) && (p.z * p.z < halfHeight * halfHeight))
    {
        intersect->HasIntersection = true;
        intersect->Position = p;
        intersect->IsFrontFace = ray.Origin.y > 0.0f;
        intersect->Normal = intersect->IsFrontFace ? m_Normal : -m_Normal;
    }
    else
    {
        intersect->HasIntersection = false;
    }
}

QUAL_CPU_GPU inline AABB Quad::GetAABB(Transform* transform) const
{
    glm::vec3 v[4] = {
        TransformPoint(transform->GetMat(), glm::vec3{  m_Width / 2.0f, 0.0f,  m_Height / 2.0f }),
        TransformPoint(transform->GetMat(), glm::vec3{  m_Width / 2.0f, 0.0f, -m_Height / 2.0f }),
        TransformPoint(transform->GetMat(), glm::vec3{ -m_Width / 2.0f, 0.0f,  m_Height / 2.0f }),
        TransformPoint(transform->GetMat(), glm::vec3{ -m_Width / 2.0f, 0.0f, -m_Height / 2.0f }),
    };

    glm::vec3 min = v[0];
    glm::vec3 max = v[0];

    for (int i = 1; i < 4; i++)
    {
        min = glm::min(min, v[i]);
        max = glm::max(max, v[i]);
    }

    return AABB{ min, max };
}

QUAL_CPU_GPU inline void Triangle::Intersect(const Ray& ray, SurfaceInteraction* intersect) const
{
    const auto& P0 = m_Vertices[0];
    const auto& P1 = m_Vertices[1];
    const auto& P2 = m_Vertices[2];
    const auto& N0 = m_Normals[0];
    const auto& N1 = m_Normals[1];
    const auto& N2 = m_Normals[2];
    auto S = ray.Origin - P0;
    auto E1 = P1 - P0;
    auto E2 = P2 - P0;
    auto S1 = glm::cross(ray.Direction, E2);
    auto S2 = glm::cross(S, E1);

    auto divisor = glm::dot(S1, E1);
    if (divisor == 0)
    {
        return;
    }

    auto t = glm::dot(S2, E2) / divisor;
    auto b1 = glm::dot(S1, S) / divisor;
    auto b2 = glm::dot(S2, ray.Direction) / divisor;

    if (t < kShapeRayTMin || b1 < 0.0f || b2 < 0.0f || b1 + b2 > 1.0f)
    {
        return;
    }

    intersect->Position = (1 - b1 - b2) * P0 + b1 * P1 + b2 * P2;
    intersect->Normal = (1 - b1 - b2) * N0 + b1 * N1 + b2 * N2;
    intersect->HasIntersection = true;
    if (glm::dot(intersect->Normal, ray.Direction) > 0.0f)
    {
        intersect->Normal *= -1.0f;
        intersect->IsFrontFace = false;
    }
    else
    {
        intersect->IsFrontFace = true;
    }
}

QUAL_CPU_GPU inline AABB Triangle::GetAABB(Transform* transform) const
{
    AABB bound;

    glm::vec3 v[3] = {
        TransformPoint(transform->GetMat(), m_Vertices[0]),
        TransformPoint(transform->GetMat(), m_Vertices[1]),
        TransformPoint(transform->GetMat(), m_Vertices[2]),
    };

    for (int i = 0; i < 3; i++)
    {
        bound.Min = glm::min(bound.Min, v[i]);
        bound.Max = glm::max(bound.Max, v[i]);
    }

    return bound;
}
