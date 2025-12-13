#pragma once

#include <vector>  // For std::vector<MeshGAS>
#include <core/core.h>
#include <core/renderer.h>
#include <core/scene.h>

// OptiX
#include <optix.h>

// CUDA
#include <cuda_runtime.h>

// OpenGL & CUDA Interop for Zero-Copy rendering
#include <glad/glad.h>
#include <cuda_gl_interop.h>

// Forward declarations
struct LaunchParams;
struct SphereData;
struct QuadData;
struct TriangleData;
struct DeviceMaterial;
class OpenGLTexture;

class OptixRenderer : public Renderer
{
public:
    OptixRenderer() = default;
    ~OptixRenderer() override;

    void Init(Film& film, const Scene& scene, const Camera& camera) override;
    
    // Zero-Copy rendering with OpenGL Interop
    void ProgressiveRender(OpenGLTexture* targetTexture);
    
    // Legacy CPU path (for compatibility)
    void ProgressiveRender() override;

private:
    // Initialization helpers
    void initCUDA();
    void initOptix();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void buildSBT();
    void uploadSceneData(std::vector<SphereData>& outputSpheres, std::vector<QuadData>& outputQuads);
    void buildAccelerationStructure(const std::vector<SphereData>& spheres, const std::vector<QuadData>& quads);
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
    SphereData* m_d_SphereData = nullptr;
    QuadData* m_d_QuadData = nullptr;
    TriangleData* m_d_TriangleData = nullptr;
    DeviceMaterial* m_d_Materials = nullptr;
    size_t m_NumSpheres = 0;
    size_t m_NumQuads = 0;
    size_t m_NumTriangles = 0;
    size_t m_NumMaterials = 0;

    // Output buffer
    float3* m_d_ColorBuffer = nullptr;   // Display buffer (tone-mapped)
    float3* m_d_AccumBuffer = nullptr;   // Accumulation buffer (linear, running sum)
    float* m_d_SampleBuffer = nullptr;   // Per-frame linear sample buffer (RGBRGB...)
    
    // OpenGL Interop for Zero-Copy rendering
    GLuint m_PBO = 0;  // Pixel Buffer Object
    cudaGraphicsResource* m_CudaGraphicsResource = nullptr;  // CUDA-registered GL resource

    // Launch parameters
    LaunchParams* m_d_LaunchParams = nullptr;

    // Rendering state
    uint32_t m_FrameIndex = 0;
    bool m_Initialized = false;
    
    // Settings
    int m_MaxDepth = 10;
    float m_SkyLight = 1.0f;
};
