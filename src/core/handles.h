#pragma once

#include "core/core.h"
#include <stdint.h>

enum MatType : uint8_t;
enum class ShapeType : uint8_t;

struct MaterialHandle
{
    static constexpr uint8_t kInvalid = 0xFF;

    uint8_t Type = kInvalid;
    const void* Ptr = nullptr;

    QUAL_CPU_GPU bool IsValid() const { return Type != kInvalid && Ptr != nullptr; }
    QUAL_CPU_GPU void Reset()
    {
        Type = kInvalid;
        Ptr = nullptr;
    }

    template<typename F>
    QUAL_CPU_GPU decltype(auto) dispatch(F&& func) const;
};

struct ShapeHandle
{
    static constexpr uint8_t kInvalid = 0xFF;

    uint8_t Type = kInvalid;
    const void* Ptr = nullptr;

    QUAL_CPU_GPU bool IsValid() const { return Type != kInvalid && Ptr != nullptr; }
    QUAL_CPU_GPU void Reset()
    {
        Type = kInvalid;
        Ptr = nullptr;
    }

    template<typename F>
    QUAL_CPU_GPU decltype(auto) dispatch(F&& func) const;
};

MaterialHandle MakeMaterialHandle(MatType type, const void* material);
ShapeHandle MakeShapeHandle(ShapeType type, const void* shape);
