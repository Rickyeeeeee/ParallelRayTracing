#include "soa.h"
#include <core/primitive.h>
#include <core/material.h>
#include <core/shape.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <type_traits>

namespace
{
template<typename T>
void ReleaseDevicePtr(T*& ptr)
{
    if (ptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
}
} // namespace

void WavefrontSceneBuffers::Free()
{
    ReleaseDevicePtr(Device.primitives);
    Device.primitiveCount = 0;

    for (void* ptr : MaterialAllocs)
        cudaFree(ptr);
    MaterialAllocs.clear();

    for (void* ptr : ShapeAllocs)
        cudaFree(ptr);
    ShapeAllocs.clear();
}

WavefrontSceneBuffers BuildWavefrontSceneBuffers(const std::vector<Primitive>& primitives)
{
    WavefrontSceneBuffers buffers{};
    if (primitives.empty())
        return buffers;

    std::unordered_map<const void*, const void*> materialRemap;
    std::unordered_map<const void*, const void*> shapeRemap;

    const auto uploadMaterial = [&](const MaterialHandle& handle) -> const void* {
        if (!handle.IsValid())
            return nullptr;

        auto it = materialRemap.find(handle.Ptr);
        if (it != materialRemap.end())
            return it->second;

        void* devicePtr = nullptr;
        handle.dispatch([&](const auto* material) {
            if (!material)
                return;
            using MatT = std::remove_cv_t<std::remove_reference_t<decltype(*material)>>;
            const size_t size = sizeof(MatT);
            cudaMalloc(&devicePtr, size);
            cudaMemcpy(devicePtr, material, size, cudaMemcpyHostToDevice);
        });

        if (!devicePtr)
            return nullptr;

        materialRemap[handle.Ptr] = devicePtr;
        buffers.MaterialAllocs.push_back(devicePtr);
        return devicePtr;
    };

    const auto uploadShape = [&](const ShapeHandle& handle) -> const void* {
        if (!handle.IsValid())
            return nullptr;

        auto it = shapeRemap.find(handle.Ptr);
        if (it != shapeRemap.end())
            return it->second;

        void* devicePtr = nullptr;
        handle.dispatch([&](const auto* shape) {
            if (!shape)
                return;
            using ShapeT = std::remove_cv_t<std::remove_reference_t<decltype(*shape)>>;
            const size_t size = sizeof(ShapeT);
            cudaMalloc(&devicePtr, size);
            cudaMemcpy(devicePtr, shape, size, cudaMemcpyHostToDevice);
        });

        if (!devicePtr)
            return nullptr;

        shapeRemap[handle.Ptr] = devicePtr;
        buffers.ShapeAllocs.push_back(devicePtr);
        return devicePtr;
    };

    std::vector<Primitive> devicePrimitives(primitives.begin(), primitives.end());
    for (auto& primitive : devicePrimitives)
    {
        primitive.Material.Ptr = uploadMaterial(primitive.Material);
        primitive.Shape.Ptr = uploadShape(primitive.Shape);
    }

    buffers.Device.primitiveCount = static_cast<int>(devicePrimitives.size());
    if (buffers.Device.primitiveCount > 0)
    {
        const size_t bytes = sizeof(Primitive) * devicePrimitives.size();
        cudaMalloc(reinterpret_cast<void**>(&buffers.Device.primitives), bytes);
        cudaMemcpy(buffers.Device.primitives, devicePrimitives.data(), bytes, cudaMemcpyHostToDevice);
    }

    return buffers;
}
