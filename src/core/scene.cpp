#include "scene.h"

Scene::Scene()
{
    auto makeTransform = [](const glm::vec3& scale, const glm::vec3& eulerAnglesDeg, const glm::vec3& translation) {
        Transform transform;
        transform.Set(scale, glm::radians(eulerAnglesDeg), translation);
        return transform;
    };

    auto addPrimitive = [&](const ShapeHandle& shape, const MaterialHandle& material, const Transform& transform, MatType type, ShapeType shapeType) {
        Primitive primitive;
        primitive.Shape = shape;
        primitive.Material = material;
        primitive.Transform = transform;
        if (shapeType == ShapeType::CIRCLE)
            m_Primitives.AddCircle(primitive);
        else if (shapeType == ShapeType::QUAD)
            m_Primitives.AddQuad(primitive);
        else
            m_Primitives.AddPrimitive(primitive);
    };

    // Emissive circle
    auto emissiveMat = m_Materials.AddEmissive(glm::vec3{ 10.0f, 5.0f, 5.0f });
    auto emissiveCircle = m_Shapes.AddCircle(1.0f);
    addPrimitive(emissiveCircle, emissiveMat, makeTransform(glm::vec3{ 2.0f }, glm::vec3{ 0.0f }, glm::vec3{ 5.0f, 6.0f, 0.0f }), MatType::EMISSIVE, ShapeType::CIRCLE);

    auto quadEmissive = m_Materials.AddEmissive(glm::vec3{ 3.0f, 4.0f, 2.0f });
    auto lightQuad = m_Shapes.AddQuad(8.0f, 8.0f);
    addPrimitive(lightQuad, quadEmissive, makeTransform(glm::vec3{ 1.0f }, glm::vec3{ 50.0f, 0.0f, 0.0f }, glm::vec3{ -4.0f, 7.0f, 7.0f }), MatType::EMISSIVE, ShapeType::QUAD);

    auto quadEmissive2 = m_Materials.AddEmissive(glm::vec3{ 3.0f, 2.0f, 1.0f });
    auto lightQuad2 = m_Shapes.AddQuad(8.0f, 8.0f);
    addPrimitive(lightQuad2, quadEmissive2, makeTransform(glm::vec3{ 1.0f }, glm::vec3{ 50.0f, 0.0f, 0.0f }, glm::vec3{ 4.0f, 7.0f, 7.0f }), MatType::EMISSIVE, ShapeType::QUAD);

    auto lambertianGreen = m_Materials.AddLambertian(glm::vec3{ 0.2f, 1.0f, 0.2f });
    auto circleShape = m_Shapes.AddCircle(1.0f);
    addPrimitive(circleShape, lambertianGreen, makeTransform(glm::vec3{ 1.0f }, glm::vec3{ 0.0f }, glm::vec3{ 4.0f, 1.0f, 0.0f }), MatType::LAMBERTIAN, ShapeType::CIRCLE);

    auto lambertianRed = m_Materials.AddLambertian(glm::vec3{ 1.0f, 0.2f, 0.2f });
    auto circleShape2 = m_Shapes.AddCircle(1.0f);
    addPrimitive(circleShape2, lambertianRed, makeTransform(glm::vec3{ 1.0f }, glm::vec3{ 0.0f }, glm::vec3{ -4.0f, 1.0f, 0.0f }), MatType::LAMBERTIAN, ShapeType::CIRCLE);

    auto dielectricMat = m_Materials.AddDielectric(0.9f);
    auto circleShape3 = m_Shapes.AddCircle(1.0f);
    addPrimitive(circleShape3, dielectricMat, makeTransform(glm::vec3{ 1.0f }, glm::vec3{ 0.0f }, glm::vec3{ 0.0f, 1.0f, 4.0f }), MatType::DIELECTRIC, ShapeType::CIRCLE);

    auto metalMat = m_Materials.AddMetal(glm::vec3{ 1.0f, 0.7f, 0.8f }, 0.01f);
    auto circleShape4 = m_Shapes.AddCircle(1.0f);
    addPrimitive(circleShape4, metalMat, makeTransform(glm::vec3{ 1.0f }, glm::vec3{ 0.0f }, glm::vec3{ 0.0f, 1.0f, -4.0f }), MatType::METAL, ShapeType::CIRCLE);

    auto lambertianGround = m_Materials.AddLambertian(glm::vec3{ 0.7f, 0.7f, 0.4f });
    auto quadShape = m_Shapes.AddQuad(20.0f, 20.0f);
    addPrimitive(quadShape, lambertianGround, makeTransform(glm::vec3{ 1.0f }, glm::vec3{ 0.0f }, glm::vec3{ 0.0f }), MatType::LAMBERTIAN, ShapeType::QUAD);
}
