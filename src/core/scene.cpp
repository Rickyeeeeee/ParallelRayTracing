#include "scene.h"
#include <glm/gtc/matrix_transform.hpp>
#include <random>

// ------------------------------------------------------------
// Helpers (members, so private data is safe)
// ------------------------------------------------------------

Transform Scene::MakeTransform(
    const glm::vec3& scale,
    const glm::vec3& eulerAnglesDeg,
    const glm::vec3& translation)
{
    Transform t;
    t.Set(scale, glm::radians(eulerAnglesDeg), translation);
    return t;
}

void Scene::AddPrimitive(
    const ShapeHandle& shape,
    const MaterialHandle& material,
    const Transform& transform,
    ShapeType shapeType)
{
    Primitive p;
    p.Shape     = shape;
    p.Material  = material;
    p.Transform = transform;

    if (shapeType == ShapeType::CIRCLE)
        m_Primitives.AddCircle(p);
    else if (shapeType == ShapeType::QUAD)
        m_Primitives.AddQuad(p);
    else
        m_Primitives.AddPrimitive(p);
}

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

Scene::Scene(ScenePreset preset)
{
    switch (preset)
    {
    case ScenePreset::LIGHT_TEST:    InitLightTest();    break;
    case ScenePreset::MATERIAL_TEST: InitMaterialTest(); break;
    case ScenePreset::CORNELL:       InitCornell();      break;
    case ScenePreset::RANDOM_BALLS:  InitRandomBalls();  break;
    case ScenePreset::DEFAULT:
    default:                         InitDefault();      break;
    }
}


// ------------------------------------------------------------
// Scene variants
// ------------------------------------------------------------

void Scene::InitRandomBalls()
{
    // Strong test for BVH & traversal
    m_SkyLightIntensity = 1.0;

    // -----------------------------
    // Ground
    // -----------------------------
    auto groundMat = m_Materials.AddLambertian(glm::vec3(0.5f));
    auto ground = m_Shapes.AddQuad(200.0f, 200.0f);

    AddPrimitive(
        ground,
        groundMat,
        MakeTransform(
            glm::vec3(1, 1, 1),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 0, 0)),
        ShapeType::QUAD
    );

    // -----------------------------
    // Random generator (deterministic)
    // -----------------------------
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> distPos(-40.0f, 40.0f);
    std::uniform_real_distribution<float> distRadius(0.2f, 0.6f);

    constexpr int BALL_COUNT = 800;   // crank this to 2k–10k if you want pain

    // -----------------------------
    // Balls
    // -----------------------------
    for (int i = 0; i < BALL_COUNT; ++i)
    {
        float radius = distRadius(rng);
        glm::vec3 pos(
            distPos(rng),
            radius,
            distPos(rng)
        );

        ShapeHandle sphere = m_Shapes.AddCircle(radius);

        float m = dist01(rng);
        MaterialHandle mat;

        if (m < 0.65f)
        {
            // Lambertian
            mat = m_Materials.AddLambertian(glm::vec3(
                dist01(rng),
                dist01(rng),
                dist01(rng)
            ));
        }
        else if (m < 0.9f)
        {
            // Metal
            mat = m_Materials.AddMetal(
                glm::vec3(0.7f + 0.3f * dist01(rng)),
                0.05f * dist01(rng)
            );
        }
        else
        {
            // Dielectric
            mat = m_Materials.AddDielectric(1.3f + 0.4f * dist01(rng));
        }

        AddPrimitive(
            sphere,
            mat,
            MakeTransform(
                glm::vec3(1, 1, 1),
                glm::vec3(0, 0, 0),
                pos),
            ShapeType::CIRCLE
        );
    }

    // -----------------------------
    // A few emissive balls (lighting chaos)
    // -----------------------------
    for (int i = 0; i < 8; ++i)
    {
        float radius = 1.5f;
        glm::vec3 pos(
            distPos(rng),
            8.0f,
            distPos(rng)
        );

        auto lightMat = m_Materials.AddEmissive(
            glm::vec3(10.0f + 10.0f * dist01(rng))
        );

        auto lightSphere = m_Shapes.AddCircle(radius);

        AddPrimitive(
            lightSphere,
            lightMat,
            MakeTransform(
                glm::vec3(1,1,1),
                glm::vec3(0,0,0),
                pos),
            ShapeType::CIRCLE
        );
    }
}


void Scene::InitDefault()
{
    // Emissive circle
    auto emissiveMat = m_Materials.AddEmissive(glm::vec3(10, 5, 5));
    auto emissiveCircle = m_Shapes.AddCircle(1.0f);
    AddPrimitive(
        emissiveCircle,
        emissiveMat,
        MakeTransform(
            glm::vec3(2.0f, 2.0f, 2.0f),
            glm::vec3(0, 0, 0),
            glm::vec3(5, 6, 0)),
        ShapeType::CIRCLE);

    auto quadEmissive = m_Materials.AddEmissive(glm::vec3(3, 4, 2));
    auto lightQuad = m_Shapes.AddQuad(8, 8);
    AddPrimitive(
        lightQuad,
        quadEmissive,
        MakeTransform(
            glm::vec3(1, 1, 1),
            glm::vec3(50, 0, 0),
            glm::vec3(-4, 7, 7)),
        ShapeType::QUAD);

    auto quadEmissive2 = m_Materials.AddEmissive(glm::vec3(3, 2, 1));
    auto lightQuad2 = m_Shapes.AddQuad(8, 8);
    AddPrimitive(
        lightQuad2,
        quadEmissive2,
        MakeTransform(
            glm::vec3(1, 1, 1),
            glm::vec3(50, 0, 0),
            glm::vec3(4, 7, 7)),
        ShapeType::QUAD);

    auto lambertianGreen = m_Materials.AddLambertian(glm::vec3(0.2f, 1.0f, 0.2f));
    auto circleShape = m_Shapes.AddCircle(1.0f);
    AddPrimitive(
        circleShape,
        lambertianGreen,
        MakeTransform(
            glm::vec3(1, 1, 1),
            glm::vec3(0, 0, 0),
            glm::vec3(4, 1, 0)),
        ShapeType::CIRCLE);

    auto lambertianRed = m_Materials.AddLambertian(glm::vec3(1.0f, 0.2f, 0.2f));
    auto circleShape2 = m_Shapes.AddCircle(1.0f);
    AddPrimitive(
        circleShape2,
        lambertianRed,
        MakeTransform(
            glm::vec3(1, 1, 1),
            glm::vec3(0, 0, 0),
            glm::vec3(-4, 1, 0)),
        ShapeType::CIRCLE);

    auto dielectricMat = m_Materials.AddDielectric(0.9f);
    auto circleShape3 = m_Shapes.AddCircle(1.0f);
    AddPrimitive(
        circleShape3,
        dielectricMat,
        MakeTransform(
            glm::vec3(1, 1, 1),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 1, 4)),
        ShapeType::CIRCLE);

    auto metalMat = m_Materials.AddMetal(glm::vec3(1, 0.7f, 0.8f), 0.01f);
    auto circleShape4 = m_Shapes.AddCircle(1.0f);
    AddPrimitive(
        circleShape4,
        metalMat,
        MakeTransform(
            glm::vec3(1, 1, 1),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 1, -4)),
        ShapeType::CIRCLE);

    auto groundMat = m_Materials.AddLambertian(glm::vec3(0.7f, 0.7f, 0.4f));
    auto quadShape = m_Shapes.AddQuad(20, 20);
    AddPrimitive(
        quadShape,
        groundMat,
        MakeTransform(
            glm::vec3(1, 1, 1),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 0, 0)),
        ShapeType::QUAD);
}

void Scene::InitLightTest()
{
    m_SkyLightIntensity = 0.0;

    auto groundMat = m_Materials.AddLambertian(glm::vec3(0.6f));
    auto ground = m_Shapes.AddQuad(30, 30);
    AddPrimitive(
        ground,
        groundMat,
        MakeTransform(glm::vec3(1,1,1), glm::vec3(0), glm::vec3(0)),
        ShapeType::QUAD);

    auto lightShape = m_Shapes.AddCircle(0.5f);
    for (int i = -5; i <= 5; ++i)
    {
        auto lightMat = m_Materials.AddEmissive(glm::vec3(4));
        AddPrimitive(
            lightShape,
            lightMat,
            MakeTransform(
                glm::vec3(1,1,1),
                glm::vec3(0),
                glm::vec3(float(i * 2), 6, 0)),
            ShapeType::CIRCLE);
    }
}

void Scene::InitMaterialTest()
{
    auto groundMat = m_Materials.AddLambertian(glm::vec3(0.8f));
    auto ground = m_Shapes.AddQuad(25, 25);
    AddPrimitive(
        ground,
        groundMat,
        MakeTransform(glm::vec3(1), glm::vec3(0), glm::vec3(0)),
        ShapeType::QUAD);

    auto sphere = m_Shapes.AddCircle(1.0f);

    AddPrimitive(sphere, m_Materials.AddLambertian(glm::vec3(1,0,0)),
        MakeTransform(glm::vec3(1), glm::vec3(0), glm::vec3(-4,1,0)),
        ShapeType::CIRCLE);

    AddPrimitive(sphere, m_Materials.AddMetal(glm::vec3(0.9f), 0.0f),
        MakeTransform(glm::vec3(1), glm::vec3(0), glm::vec3(0,1,0)),
        ShapeType::CIRCLE);

    AddPrimitive(sphere, m_Materials.AddDielectric(1.5f),
        MakeTransform(glm::vec3(1), glm::vec3(0), glm::vec3(4,1,0)),
        ShapeType::CIRCLE);
}

void Scene::InitCornell()
{
    m_SkyLightIntensity = 0.0;

    auto red   = m_Materials.AddLambertian(glm::vec3(0.75f,0.1f,0.1f));
    auto green = m_Materials.AddLambertian(glm::vec3(0.1f,0.75f,0.1f));
    auto white = m_Materials.AddLambertian(glm::vec3(0.8f));

    auto quad = m_Shapes.AddQuad(10, 10);

    AddPrimitive(quad, white, MakeTransform(glm::vec3(1), glm::vec3(0), glm::vec3(0)), ShapeType::QUAD);
    AddPrimitive(quad, red,   MakeTransform(glm::vec3(1), glm::vec3(90,0,0), glm::vec3(-5,5,0)), ShapeType::QUAD);
    AddPrimitive(quad, green, MakeTransform(glm::vec3(1), glm::vec3(90,0,0), glm::vec3(5,5,0)), ShapeType::QUAD);

    auto lightMat = m_Materials.AddEmissive(glm::vec3(15));
    AddPrimitive(quad, lightMat,
        MakeTransform(glm::vec3(1), glm::vec3(90,0,0), glm::vec3(0,9,0)),
        ShapeType::QUAD);
}
