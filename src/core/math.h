#pragma once

#include "core/core.h"
#include <cmath>
#include <curand_kernel.h>
#include <glm/glm.hpp>

// Device overloads that accept a curand state pointer.
QUAL_CPU_GPU
inline float Random(curandState* state = nullptr)
{
#if defined(__CUDA_ARCH__)
    return curand_uniform(state);
#else
    return std::rand() / (RAND_MAX + 1.0f);
#endif
}

QUAL_CPU_GPU
inline float Random(float min, float max, curandState* state = nullptr)
{
    return min + (max - min) * Random(state);
}

QUAL_CPU_GPU
inline glm::vec3 RandomUnitVector(curandState* state = nullptr)
{
    constexpr float epsilon = 1e-8f;
    while (true)
    {
        glm::vec3 p = { Random(-1, 1, state), Random(-1, 1, state), Random(-1, 1, state) };
        auto lensq = glm::dot(p, p);
        if (epsilon < lensq && lensq <= 1.0f)
            return p / sqrt(lensq);
    }
}

QUAL_CPU_GPU
inline float LengthSquared(const glm::vec3& v)
{
    return glm::dot(v, v);
}

QUAL_CPU_GPU
inline glm::vec3 Reflect(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat) {
    auto cos_theta = glm::min(glm::dot(-uv, n), 1.0f);
    glm::vec3 r_out_perp =  etai_over_etat * (uv + cos_theta * n);
    glm::vec3 r_out_parallel = -glm::sqrt(glm::abs(1.0f - LengthSquared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}