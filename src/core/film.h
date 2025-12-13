#pragma once

#include <core/core.h>

#include <vector>
#include <cmath>
#include <cuda_runtime.h>


class Film
{
public:
    Film(uint32_t width, uint32_t height);

    void Resize(uint32_t width, uint32_t height);

    // Clear accumulated samples and display buffer
    void Clear();

    // Add a single sample to pixel (x, y).
    // color is in linear RGB (0..inf), weight usually = 1.0f
    void AddSample(uint32_t x, uint32_t y,
                   float r, float g, float b,
                   float weight = 1.0f);

    // Add one sample per pixel from a full-frame buffer:
    // rgb has size = width * height * 3 (interleaved RGB, row-major).
    void AddSampleBuffer(const float* rgb, float weight = 1.0f);
    void AddSampleBufferGPU(const float* d_rgb, float weight = 1.0f);

    // Convert accumulated linear RGB to 8-bit RGBA for display.
    // Simple tonemapping + gamma.
    void UpdateDisplay(float exposure = 1.0f, float gamma = 2.2f);
    void UpdateDisplayGPU(float exposure = 1.0f, float gamma = 2.2f);

    // Pointer suitable for OpenGLTexture::SetData (RGBA8)
    const uint8_t* GetDisplayData() const { return m_Display.data(); }
    const uint8_t* GetDisplayDataGPU() const { return d_Display; }

    uint32_t GetWidth() const  { return m_Width;  }
    uint32_t GetHeight() const { return m_Height; }
    uint32_t GetSampleCount() const { return m_Samples; }

private:
    uint32_t m_Width  = 0;
    uint32_t m_Height = 0;

    // Accumulated color (linear RGB), per pixel:
    // m_Accum[3 * (y * width + x) + c]
    std::vector<float>   m_Accum;
    float* d_Accum;

    // Per-pixel total weight (number of samples, or sum of weights)
    std::vector<float>   m_Weights;
    float* d_Weights;

    // Display buffer in RGBA8
    std::vector<uint8_t> m_Display;
    uint8_t* d_Display;

    uint32_t m_Samples = 0; // global sample counter (for UI etc.)

    static float Tonemap(float v, float exposure)
    {
        float x = v * exposure;
        // Simple Reinhard tonemap: x / (1 + x)
        x = x / (1.0f + x);
        return x;
    }

    static uint8_t ToByte(float v)
    {
        v = std::fmax(0.0f, std::fmin(1.0f, v));
        return static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
};
