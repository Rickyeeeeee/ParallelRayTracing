#pragma once

#include "geometry.h"
#include "surface_interaction.h"
#include <memory>
#include <cassert>

enum class ShapeType : uint8_t
{
    CIRCLE = 0,
    QUAD,
    TRIANGLE
};

class Shape
{
public:
    Shape() {};
    virtual void Intersect(const Ray& ray, SurfaceInteraction* intersect) const = 0;
    // TODO: Add Transform parameter
    virtual AABB GetAABB(Transform* transform) const = 0;
};

class Circle : public Shape
{
public:
    Circle(float radius=1.0f) : m_Radius(radius), Shape() {};

    virtual void Intersect(const Ray& ray, SurfaceInteraction* intersect) const override;
    virtual AABB GetAABB(Transform* transform) const override;

    inline float getRadius() const { return m_Radius; }
    
    private:
    float m_Radius{ 1.0f };
};

class Quad : public Shape
{
    public:
    Quad(float width=1.0f, float height=1.0f) : m_Width(width), m_Height(height) {}
    
    virtual void Intersect(const Ray& ray, SurfaceInteraction* intersect) const override;
    virtual AABB GetAABB(Transform* transform) const override;

    inline float GetWidth() const { return m_Width; }
    inline float GetHeight() const { return m_Height; }
    inline glm::vec3 GetNormal() const { return m_Normal; }

private:
    float m_Width;
    float m_Height;

    const glm::vec3 m_Normal = glm::vec3(0.0f, 1.0f, 0.0f);
};

class Triangle : public Shape
{
public:
    Triangle(
        const glm::vec3& v0, 
        const glm::vec3& v1, 
        const glm::vec3& v2,
        const glm::vec3& n0, 
        const glm::vec3& n1, 
        const glm::vec3& n2,
        const glm::vec2& uv0, 
        const glm::vec2& uv1, 
        const glm::vec2& uv2
        )
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

    virtual void Intersect(const Ray& ray, SurfaceInteraction* intersect) const override;
    virtual AABB GetAABB(Transform* transform) const override;

private:
    glm::vec3 m_Vertices[3];
    glm::vec3 m_Normals[3];
    glm::vec2 m_UVs[3];
};

template<typename F>
QUAL_CPU_GPU decltype(auto) ShapeHandle::dispatch(F&& func) const
{
    switch (type)
    {
    case static_cast<uint8_t>(ShapeType::CIRCLE):
        return func(static_cast<const Circle*>(ptr));
    case static_cast<uint8_t>(ShapeType::QUAD):
        return func(static_cast<const Quad*>(ptr));
    case static_cast<uint8_t>(ShapeType::TRIANGLE):
        return func(static_cast<const Triangle*>(ptr));
    default:
#ifndef __CUDA_ARCH__
        assert(false && "Invalid ShapeHandle dispatch");
#endif
        return func(static_cast<const Circle*>(nullptr));
    }
}

inline ShapeHandle MakeShapeHandle(const Shape* shape)
{
    ShapeHandle handle;
    if (!shape)
        return handle;

    if (auto circle = dynamic_cast<const Circle*>(shape))
    {
        handle.type = static_cast<uint8_t>(ShapeType::CIRCLE);
        handle.ptr = circle;
        return handle;
    }

    if (auto quad = dynamic_cast<const Quad*>(shape))
    {
        handle.type = static_cast<uint8_t>(ShapeType::QUAD);
        handle.ptr = quad;
        return handle;
    }

    if (auto triangle = dynamic_cast<const Triangle*>(shape))
    {
        handle.type = static_cast<uint8_t>(ShapeType::TRIANGLE);
        handle.ptr = triangle;
        return handle;
    }

    return handle;
}
