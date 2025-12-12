#pragma once

#include "core/geometry.h"
#include "core/material_handle.h"

struct SurfaceInteraction
{
    glm::vec3 Position{ 0.0f };
    glm::vec3 Normal{ 0.0f, 0.0f, 0.0f };
    bool HasIntersection = false;
    bool IsFrontFace = false;
    MaterialHandle Material{};
};
