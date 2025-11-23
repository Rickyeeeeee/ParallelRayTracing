#include "renderer.h"

#include <cuda_runtime.h>
#include <vector>

namespace
{
__global__ void WavefrontPatternKernel(float* colors, uint32_t width, uint32_t height, float time)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixelCount = width * height;
    if (idx >= pixelCount)
        return;

    const uint32_t x = idx % width;
    const uint32_t y = idx / width;

    const float u = (width > 1) ? static_cast<float>(x) / static_cast<float>(width - 1) : 0.0f;
    const float v = (height > 1) ? static_cast<float>(y) / static_cast<float>(height - 1) : 0.0f;

    const float twoPi = 6.2831853f;
    const float r = 0.5f + 0.5f * __sinf(twoPi * (u + time));
    const float g = 0.5f + 0.5f * __sinf(twoPi * (v + time));
    const float b = 0.2f + 0.3f * __sinf(twoPi * (u + v + time));

    const uint32_t base = idx * 3;
    colors[base + 0] = r;
    colors[base + 1] = g;
    colors[base + 2] = b;
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

    const float time = static_cast<float>(m_FrameIndex) * 0.01f;
    const size_t bufferSize = static_cast<size_t>(pixelCount) * 3 * sizeof(float);

    float* deviceBuffer = nullptr;
    cudaMalloc(&deviceBuffer, bufferSize);

    const uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    WavefrontPatternKernel<<<blocks, threadsPerBlock>>>(deviceBuffer, width, height, time);
    cudaDeviceSynchronize();

    std::vector<float> hostBuffer(static_cast<size_t>(pixelCount) * 3);
    cudaMemcpy(hostBuffer.data(), deviceBuffer, bufferSize, cudaMemcpyDeviceToHost);

    m_Film->AddSampleBuffer(hostBuffer.data());

    cudaFree(deviceBuffer);

    ++m_FrameIndex;
}
