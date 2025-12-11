#include "renderer.h"
#include <vector>
#include <future>

void CPURenderer::Init(Film& film, const Scene& scene, const Camera& camera)
{
    film.Clear();
    this->m_Film = &film;
    this->m_Scene = &scene;
    this->m_Camera = &camera;
}

void CPURenderer::ProgressiveRender()
{
    std::vector<Tile> tiles;

    auto width = (uint32_t)m_Camera->GetWidth();
    auto height = (uint32_t)m_Camera->GetHeight();

    for (uint32_t i = 0; i < width; i += m_tileSize)
    {
        for (uint32_t j = 0; j < height; j += m_tileSize)
        {
            tiles.push_back(Tile{
                i, j, std::min(i + m_tileSize, width), std::min(j + m_tileSize, height)
            });
        }
    }

    std::vector<std::future<void>> futures;
    
    for (const auto& tile : tiles)
    {
        futures.push_back(std::async(std::launch::async, [&]() {

            for (uint32_t i = tile.x0; i < tile.x1; i++)
            {
                for (uint32_t j = tile.y0; j < tile.y1; j++)
                {
                    auto ray = m_Camera->GetCameraRay((float)i + 0.5f, (float)j + 0.5f);
                    
                    auto L = TraceRay(ray, m_Depth);

                    this->m_Film->AddSample(i, j, L.r, L.g, L.b);
                }
            }
        }));
    }

    for (auto& fut : futures)
        fut.get();
}

glm::vec3 CPURenderer::TraceRay(const Ray& ray, int depth)
{
    if (depth <= 0)
    {
        return glm::vec3{ 0.0f };
    }

    glm::vec3 L{ 0.0f };
    SurfaceInteraction intersect;
    m_Scene->Intersect(ray, &intersect);

    if (intersect.HasIntersection)
    {
        // Emitted Lighting
        glm::vec3 emittedColor{ 0.0f };
        intersect.Material.Emit(emittedColor);
        L += emittedColor;
        glm::vec3 attenuation;
        
        // Scattered Lighting
        Ray scatteredRay;
        bool scattered = intersect.Material.Scatter(ray, intersect, attenuation, scatteredRay);

        if (scattered)
        {
            scatteredRay.Normalize();

            // Direct Lighting
            // SurfaceInteraction visiblity;
            // m_Scene->Intersect(Ray{ intersect.Position, m_SkyLightDirection }, &visiblity);
            // if (!visiblity.HasIntersection)
            //     L += attenuation * glm::vec3{ 0.1f } * glm::clamp(glm::dot(intersect.Normal, m_SkyLightDirection), 0.0f, 1.0f);

            // Indirect Lighting
            L += attenuation * TraceRay(scatteredRay, depth-1);

        }
    }
    else
    {
        L += m_SkyLight;
    }
    
    return L;
}
