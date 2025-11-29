#pragma once

#include "primitive.h"
#include <vector>
#include <core/core.h>
#include "core/mesh.h"
// #include "BVH.h"

class Scene
{
public:
    Scene() 
    {
        auto circle = std::make_shared<SimplePrimitive>(
            std::make_shared<Circle>(1.0f),
            std::make_unique<EmissiveMaterial>(glm::vec3{ 10.0f, 5.0f, 5.0f })
        );
        circle->SetTransform(
            glm::vec3{ 2.0f },
            glm::vec3{ 0.0f },
            glm::vec3{ 5.0f, 6.0f, 0.0f }
        );

        auto lightQuad = std::make_shared<SimplePrimitive>(
            std::make_shared<Quad>(8.0f, 8.0f),
            std::make_unique<EmissiveMaterial>(glm::vec3{ 3.0f, 4.0f, 2.0f })
        );

        lightQuad->SetTransform(
            glm::vec3{ 1.0f },
            glm::vec3{ 50.0f, 0.0f, 0.0f },
            glm::vec3{ -4.0f, 7.0f, 7.0f }
        );
        auto lightQuad2 = std::make_shared<SimplePrimitive>(
            std::make_shared<Quad>(8.0f, 8.0f),
            std::make_unique<EmissiveMaterial>(glm::vec3{ 3.0f, 2.0f, 1.0f })
        );
        lightQuad2->SetTransform(
            glm::vec3{ 1.0f },
            glm::vec3{ 50.0f, 0.0f, 0.0f },
            glm::vec3{ 4.0f, 7.0f, 7.0f }
        );
        auto circle2 = std::make_shared<SimplePrimitive>(
            std::make_shared<Circle>(1.0f),
            std::make_unique<LambertianMaterial>(glm::vec3{ 0.2f, 1.0f, 0.2f })
        );
        circle2->SetTransform(
            glm::vec3{ 1.0f },
            glm::vec3{ 0.0f },
            glm::vec3{ 4.0f, 1.0f, 0.0f }
        );
        auto circle3 = std::make_shared<SimplePrimitive>(
            std::make_shared<Circle>(1.0f),
            std::make_unique<LambertianMaterial>(glm::vec3{ 1.0f, 0.2f, 0.2f })
        );
        circle3->SetTransform(
            glm::vec3{ 1.0f },
            glm::vec3{ 0.0f },
            glm::vec3{ -4.0f, 1.0f, 0.0f }
        );
        auto circle4 = std::make_shared<SimplePrimitive>(
            std::make_shared<Circle>(1.0f),
            std::make_shared<DielectricMaterial>(0.9f)
        );
        circle4->SetTransform(
            glm::vec3{ 1.0f },
            glm::vec3{ 0.0f },
            glm::vec3{ 0.0f, 1.0f, 4.0f }
        );
        auto circle5 = std::make_shared<SimplePrimitive>(
            std::make_shared<Circle>(1.0f),
            std::make_unique<MetalMaterial>(glm::vec3{ 1.0f, 0.7, 0.8f }, 0.01f)
        );
        circle5->SetTransform(
            glm::vec3{ 1.0f },
            glm::vec3{ 0.0f },
            glm::vec3{ 0.0f, 1.0f, -4.0f }
        );
        auto quad = std::make_shared<SimplePrimitive>(
            std::make_shared<Quad>(20.0f, 20.0f),
            std::make_unique<LambertianMaterial>(glm::vec3{ 0.7f, 0.7f, 0.4f})
        );
        quad->SetTransform(
            glm::vec3{ 1.0f },
            glm::vec3{ 0.0f, 0.0f, 0.0f },
            glm::vec3{ 0.0f, 0.0f, 0.0f }
        );

        // Circles
        m_Primitives.AddCircle(circle2);
        m_Primitives.AddCircle(circle3);
        m_Primitives.AddCircle(circle4);
        m_Primitives.AddCircle(circle5);

        // Quad
        m_Primitives.AddQuad(quad);
        m_Primitives.AddQuad(lightQuad);
        m_Primitives.AddQuad(lightQuad2);
    }

    void Intersect(const Ray& ray, SurfaceInteraction* intersect) const
    {
        m_Primitives.Intersect(ray, intersect);
    }

   const auto getCircles() const { return m_Primitives.getCircles(); }

private:
    PrimitiveList m_Primitives;
};