#pragma once

#include "shape.h"
#include "material.h"
#include <initializer_list>
#include "core/mesh.h"

struct PrimitiveHandleView
{
    ShapeHandle shape;
    MaterialHandle material;
    Transform transform;
    MatType type { MatType::NONE };
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
    {
        m_Type = type;
        this->RefreshHandles();
    }

    // Overload: allow constructing with a Transform as the third parameter
    SimplePrimitive(std::shared_ptr<Shape> shape,
                    std::shared_ptr<Material> material,
                    Transform transform)
        : m_Shape(std::move(shape))
        , m_Transform(transform)
        , m_Material(std::move(material))
    {
        this->RefreshHandles();
    }

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
    PrimitiveHandleView GetHandleView() const;

private:
    void RefreshHandles();

    std::shared_ptr<Shape>    m_Shape;
    Transform                 m_Transform;
    std::shared_ptr<Material> m_Material;
    MatType                   m_Type { MatType::NONE };
    ShapeHandle               m_ShapeHandle;
    MaterialHandle            m_MaterialHandle;
};

class PrimitiveList : public Primitive
{
public:
    PrimitiveList() = default;

    void AddItem(std::shared_ptr<Primitive> shapePrimitive)
    {
        m_List.push_back(shapePrimitive);
    }

    void AddCircle(const std::shared_ptr<SimplePrimitive>& circle) 
    {
        m_clist.push_back(circle);
        AddItem(circle);
    }

    void AddQuad(const std::shared_ptr<SimplePrimitive>& quad) 
    {
        m_qlist.push_back(quad);
        AddItem(quad);
    }

    virtual void Intersect(const Ray& ray, SurfaceInteraction* intersect) const override;
    
    const std::vector<std::shared_ptr<SimplePrimitive>>& getCircles() const { return m_clist; }
    const std::vector<std::shared_ptr<SimplePrimitive>>& getQuads() const { return m_qlist; }
    std::vector<PrimitiveHandleView> getCircleViews() const;
    std::vector<PrimitiveHandleView> getQuadViews() const;

private:
    std::vector<std::shared_ptr<Primitive>> m_List;

    std::vector<std::shared_ptr<SimplePrimitive>> m_clist;
    std::vector<std::shared_ptr<SimplePrimitive>> m_qlist;
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
    MaterialHandle              m_MaterialHandle;
};
