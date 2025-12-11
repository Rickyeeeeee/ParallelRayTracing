#pragma once

#include "core/core.h"
#include <stdint.h>

enum MatType : uint8_t;
class Material;

enum class ShapeType : uint8_t;
class Shape;

struct MaterialHandle
{
    static constexpr uint8_t kInvalid = 0xFF;

    uint8_t type = kInvalid;
    const void* ptr = nullptr;

    QUAL_CPU_GPU bool IsValid() const { return type != kInvalid && ptr != nullptr; }
    QUAL_CPU_GPU void Reset()
    {
        type = kInvalid;
        ptr = nullptr;
    }

    template<typename F>
    QUAL_CPU_GPU decltype(auto) dispatch(F&& func) const;
};

struct ShapeHandle
{
    static constexpr uint8_t kInvalid = 0xFF;

    uint8_t type = kInvalid;
    const void* ptr = nullptr;

    QUAL_CPU_GPU bool IsValid() const { return type != kInvalid && ptr != nullptr; }
    QUAL_CPU_GPU void Reset()
    {
        type = kInvalid;
        ptr = nullptr;
    }

    template<typename F>
    QUAL_CPU_GPU decltype(auto) dispatch(F&& func) const;
};

MaterialHandle MakeMaterialHandle(const Material* material, MatType hint = (MatType)0);
ShapeHandle MakeShapeHandle(const Shape* shape);
