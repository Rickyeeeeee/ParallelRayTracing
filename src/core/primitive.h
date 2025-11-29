#pragma once

#include "shape.h"
#include "material.h"
#include <initializer_list>
#include "core/mesh.h"

enum MatType {
    NONE = 0,
    LAMBERTIAN = 1,
    METAL = 2,
    DIELECTRIC = 3,
    EMISSIVE = 4
};

class Primitive
{
public:
    Primitive() = default;
    virtual void Intersect(const Ray& ray, SurfaceInteraction* intersect) const = 0;
};


class SimplePrimitive : public Primitive
{
public:

    SimplePrimitive(std::shared_ptr<Shape> shape,
                    std::shared_ptr<Material> material,
                    MatType type = MatType::NONE,
                    Transform transform = Transform())
        : m_Shape(std::move(shape))
        , m_Transform(transform)
        , m_Material(std::move(material))
    { m_Type = type; }

    // Overload: allow constructing with a Transform as the third parameter
    SimplePrimitive(std::shared_ptr<Shape> shape,
                    std::shared_ptr<Material> material,
                    Transform transform)
        : m_Shape(std::move(shape))
        , m_Transform(transform)
        , m_Material(std::move(material))
    { }

    // main virtual
    void Intersect(const Ray& ray, SurfaceInteraction* intersect) const override;

    // accessors
    Shape& GetShape()                     { return *m_Shape; }
    Material& GetMaterial()               { return *m_Material; }

    Transform GetTransform() const        { return m_Transform; }
    void SetTransform(const Transform& t) { m_Transform = t; }

    void SetTransform(const glm::vec3& scale,
                      const glm::vec3& eulerAngles,
                      const glm::vec3& translation)
    {
        m_Transform = Transform();
        m_Transform.Set(scale, glm::radians(eulerAngles), translation);
    }

    AABB GetAABB()
    {
        return m_Shape->GetAABB(&m_Transform);
    }

    MatType GetType() const                  { return m_Type; }

private:
    std::shared_ptr<Shape>    m_Shape;
    Transform                 m_Transform;
    std::shared_ptr<Material> m_Material;
    MatType                   m_Type { MatType::NONE };
};

class PrimitiveList : public Primitive
{
public:
    PrimitiveList() = default;

    void AddItem(std::shared_ptr<Primitive> shapePrimitive)
    {
        m_List.push_back(shapePrimitive);
    }

    void AddCircle(std::shared_ptr<Primitive> circle) 
    {
        m_clist.push_back(circle);
        AddItem(circle);
    }

    void AddQuad(std::shared_ptr<Primitive> quad) 
    {
        m_qlist.push_back(quad);
        AddItem(quad);
    }

    virtual void Intersect(const Ray& ray, SurfaceInteraction* intersect) const override;
    
    const std::vector<std::shared_ptr<Primitive>>& getCircles() const { return m_clist; }
    const std::vector<std::shared_ptr<Primitive>>& getQuads() const { return m_qlist; }

private:
    std::vector<std::shared_ptr<Primitive>> m_List;

    std::vector<std::shared_ptr<Primitive>> m_clist;
    std::vector<std::shared_ptr<Primitive>> m_qlist;
};

class TriangleList : public Primitive
{
public:
    TriangleList(const Mesh& mesh, std::shared_ptr<Material> material);
    virtual void Intersect(const Ray& ray, SurfaceInteraction* intersect) const override;
    Transform GetTransform() const              { return m_Transform; }
    std::vector<std::shared_ptr<SimplePrimitive>> GetPrimitives() const;
    void SetTransform(const Transform& trans)   { m_Transform = trans; }
    void SetTransform(
        const glm::vec3& scale, 
        const glm::vec3& eulerAngles, 
        const glm::vec3& translation)   
    { 
        m_Transform = Transform(); 
        m_Transform.Set(scale, glm::radians(eulerAngles), translation);
    }
private:
    std::vector<Triangle>       m_TriangleList;
    std::shared_ptr<Material>   m_Material;
    Transform                   m_Transform;
};