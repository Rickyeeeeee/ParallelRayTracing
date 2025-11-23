#pragma once

#include <stdint.h>
#include <string>

#ifndef ASSETS_DIR
#define ASSETS_DIR "."
#endif

#define QUAL_CPU_GPU __host__ __device__
#define QUAL_GPU __device__


const std::string assetRoot = ASSETS_DIR;

// #define GLM_FORCE_INTRINSICS
// #define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>