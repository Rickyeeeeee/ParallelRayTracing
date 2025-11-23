#pragma once

#include <core/core.h>
#include <core/film.h>
#include <core/scene.h>
#include <core/camera.h>

class Renderer
{
public:
    virtual ~Renderer() = 0;

    virtual void Init(Film& film, const Scene& scene, const Camera& camera) = 0;
    virtual void ProgressiveRender() = 0;
};