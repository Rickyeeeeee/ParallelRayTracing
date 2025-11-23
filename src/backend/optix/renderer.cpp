#include "renderer.h"
// OptiX
#include <optix.h>
#include <optix_stubs.h>
#include <iostream>
#include <vector>

void OptixRenderer::Init(Film& film, const Scene& scene, const Camera& camera)
{
    m_Film = &film;
    m_Scene = &scene;
    m_Camera = &camera;

    OptixResult initResult = optixInit();
    const char* errorName = optixGetErrorName(initResult);
    std::cout << "[OptixRenderer] optixInit result: "
              << (errorName ? errorName : "Unknown") << "\n";
}

void OptixRenderer::ProgressiveRender()
{
    if (!m_Film)
        return;

    const uint32_t width = m_Film->GetWidth();
    const uint32_t height = m_Film->GetHeight();
    const uint32_t pixelCount = width * height;

    if (pixelCount == 0)
        return;

    std::vector<float> white(static_cast<size_t>(pixelCount) * 3, 1.0f);

    m_Film->Clear();
    m_Film->AddSampleBuffer(white.data());
}
