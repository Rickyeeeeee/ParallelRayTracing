#include "film.h"

Film::Film(uint32_t width, uint32_t height)
{
    Resize(width, height);
    Clear();
}

void Film::Resize(uint32_t width, uint32_t height)
{
    m_Width  = width;
    m_Height = height;

    m_Accum.resize(static_cast<size_t>(width) * height * 3);
    m_Weights.resize(static_cast<size_t>(width) * height);
    m_Display.resize(static_cast<size_t>(width) * height * 4);
}

void Film::Clear()
{
    std::fill(m_Accum.begin(),   m_Accum.end(),   0.0f);
    std::fill(m_Weights.begin(), m_Weights.end(), 0.0f);
    std::fill(m_Display.begin(), m_Display.end(), 0u);
    m_Samples = 0;
}

void Film::AddSample(uint32_t x, uint32_t y,
                     float r, float g, float b,
                     float weight)
{
    if (x >= m_Width || y >= m_Height)
        return;

    const uint32_t idx = y * m_Width + x;

    const uint32_t c0 = 3 * idx + 0;
    const uint32_t c1 = 3 * idx + 1;
    const uint32_t c2 = 3 * idx + 2;

    m_Accum[c0] += r * weight;
    m_Accum[c1] += g * weight;
    m_Accum[c2] += b * weight;

    m_Weights[idx] += weight;
}

void Film::AddSampleBuffer(const float* rgb, float weight)
{
    if (!rgb) return;

    const uint32_t pixelCount = m_Width * m_Height;

    for (uint32_t i = 0; i < pixelCount; ++i)
    {
        const float r = rgb[3 * i + 0];
        const float g = rgb[3 * i + 1];
        const float b = rgb[3 * i + 2];

        m_Accum[3 * i + 0] += r * weight;
        m_Accum[3 * i + 1] += g * weight;
        m_Accum[3 * i + 2] += b * weight;

        m_Weights[i] += weight;
    }

    ++m_Samples;
}

void Film::UpdateDisplay(float exposure, float gamma)
{
    const uint32_t pixelCount = m_Width * m_Height;
    const float invGamma = 1.0f / gamma;

    for (uint32_t i = 0; i < pixelCount; ++i)
    {
        const float w = m_Weights[i];
        float r = 0.0f, g = 0.0f, b = 0.0f;

        if (w > 0.0f)
        {
            const float invW = 1.0f / w;
            r = m_Accum[3 * i + 0] * invW;
            g = m_Accum[3 * i + 1] * invW;
            b = m_Accum[3 * i + 2] * invW;

            // Tonemap
            r = Tonemap(r, exposure);
            g = Tonemap(g, exposure);
            b = Tonemap(b, exposure);

            // Gamma
            r = std::pow(r, invGamma);
            g = std::pow(g, invGamma);
            b = std::pow(b, invGamma);
        }

        const uint32_t idxRGBA = 4 * i;
        m_Display[idxRGBA + 0] = ToByte(r);
        m_Display[idxRGBA + 1] = ToByte(g);
        m_Display[idxRGBA + 2] = ToByte(b);
        m_Display[idxRGBA + 3] = 255;
    }
}
