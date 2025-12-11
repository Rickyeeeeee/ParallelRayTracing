#pragma once

#include <core/core.h>
#include <core/material.h>

struct MaterialSOA 
{
    glm::vec3* color;
    MatType* type;
    float* refractionIndex;
    float* roughness;
    int count;
};

struct SphereSOA 
{
    glm::vec3* center;
    float* radius;
    int* materialId;
    int count;
};

struct QuadSOA 
{
    glm::vec3* position;
    glm::vec3* normal;
    float* width;
    float* height;
    int* materialId;
    int count;
};
