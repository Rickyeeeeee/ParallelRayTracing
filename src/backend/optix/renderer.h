#pragma once

#include <vector>  // For std::vector<MeshGAS>
#include <core/core.h>
#include <core/renderer.h>
#include <core/scene.h>

// OptiX
#include <optix.h>

// CUDA
#include <cuda_runtime.h>

// Forward declarations
struct LaunchParams;

class OptixRenderer : public Renderer
{
public:
    OptixRenderer() = default;
    ~OptixRenderer() override;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    void ProgressiveRender() override;

private:
    // Initialization helpers
    void initCUDA();
    void initOptix();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void buildSBT();
    void uploadSceneData();
    void buildAccelerationStructure();
    void updateLaunchParams();
    void cleanup();

private:
    // Core references
    Film* m_Film = nullptr;
    const Scene* m_Scene = nullptr;
    const Camera* m_Camera = nullptr;

    // CUDA
    cudaStream_t m_Stream = nullptr;

    // OptiX context and pipeline
    OptixDeviceContext m_OptixContext = nullptr;
    OptixModule m_Module = nullptr;
    OptixPipeline m_Pipeline = nullptr;
    OptixPipelineCompileOptions m_PipelineCompileOptions = {};

    // Program groups
    OptixProgramGroup m_RaygenPG = nullptr;
    OptixProgramGroup m_MissPG = nullptr;
    OptixProgramGroup m_HitSpherePG = nullptr;
    OptixProgramGroup m_HitQuadPG = nullptr;
    OptixProgramGroup m_HitTrianglePG = nullptr;

    // Shader Binding Table
    OptixShaderBindingTable m_SBT = {};
    CUdeviceptr m_RaygenRecord = 0;
    CUdeviceptr m_MissRecord = 0;
    CUdeviceptr m_HitgroupRecord = 0;

    // Acceleration structure (using IAS for proper SBT mapping)
    OptixTraversableHandle m_TraversableHandle = 0;  // IAS handle
    CUdeviceptr m_IasBuffer = 0;
    
    // Per-geometry GAS
    OptixTraversableHandle m_SphereGasHandle = 0;
    CUdeviceptr m_SphereGasBuffer = 0;
    OptixTraversableHandle m_QuadGasHandle = 0;
    CUdeviceptr m_QuadGasBuffer = 0;

    // Scene data on device
    CUdeviceptr m_d_SphereData = 0;
    CUdeviceptr m_d_QuadData = 0;
    CUdeviceptr m_d_TriangleData = 0;
    CUdeviceptr m_d_Materials = 0;
    size_t m_NumSpheres = 0;
    size_t m_NumQuads = 0;
    size_t m_NumTriangles = 0;
    size_t m_NumMaterials = 0;

    // Output buffer
    CUdeviceptr m_d_ColorBuffer = 0;

    // Launch parameters
    CUdeviceptr m_d_LaunchParams = 0;

    // Rendering state
    uint32_t m_FrameIndex = 0;
    bool m_Initialized = false;
    
    // Settings
    int m_MaxDepth = 10;
    float m_SkyLight = 1.0f;
};
