#include "renderer.h"

#include <cuda_runtime.h>
#include <vector>
#include <core/camera.h>

namespace
{
__global__ void GenerateCameraRaysKernel(float* colors, const Camera* cam, uint32_t width, uint32_t height)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixelCount = width * height;
    if (idx >= pixelCount)
        return;

    const uint32_t x = idx % width;
    const uint32_t y = idx / width;

    Ray ray = cam->GetCameraRay((float)x + 0.5f, (float)y + 0.5f);
    ray.Normalize();
    const glm::vec3 dir = glm::normalize(ray.Direction);

    // Simple sky gradient based on ray direction.
    const float t = 0.5f * (dir.y + 1.0f);
    const glm::vec3 color = (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);

    const uint32_t base = idx * 3;
    colors[base + 0] = color.r;
    colors[base + 1] = color.g;
    colors[base + 2] = color.b;
}
}

void CudaWavefrontRenderer::Init(Film& film, const Scene& scene, const Camera& camera)
{
    m_Film = &film;
    m_Scene = &scene;
    m_Camera = &camera;
    m_FrameIndex = 0;
}

void CudaWavefrontRenderer::ProgressiveRender()
{
    if (!m_Film)
        return;

    const uint32_t width = m_Film->GetWidth();
    const uint32_t height = m_Film->GetHeight();
    const uint32_t pixelCount = width * height;

    if (pixelCount == 0)
        return;

    const size_t bufferSize = static_cast<size_t>(pixelCount) * 3 * sizeof(float);

    float* deviceBuffer = nullptr;
    cudaMalloc(&deviceBuffer, bufferSize);

    Camera* d_cam = nullptr;
    cudaMalloc(&d_cam, sizeof(Camera));
    cudaMemcpy(d_cam, m_Camera, sizeof(Camera), cudaMemcpyHostToDevice);

    const uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    GenerateCameraRaysKernel<<<blocks, threadsPerBlock>>>(deviceBuffer, d_cam, width, height);
    cudaDeviceSynchronize();

    std::vector<float> hostBuffer(static_cast<size_t>(pixelCount) * 3);
    cudaMemcpy(hostBuffer.data(), deviceBuffer, bufferSize, cudaMemcpyDeviceToHost);

    m_Film->AddSampleBuffer(hostBuffer.data());

    cudaFree(deviceBuffer);
    cudaFree(d_cam);

    ++m_FrameIndex;
}

CudaWavefrontRenderer::~CudaWavefrontRenderer()
{
    m_SceneBuffers.Free();
}

