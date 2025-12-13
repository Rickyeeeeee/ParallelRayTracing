#include "renderer.h"
#include "device_types.h"
#include "launch_params.h"

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// OptiX
#include <optix.h>
#include <optix_stubs.h>
// Note: optix_function_table_definition.h is included ONLY in main.cpp

// Standard
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <glm/gtc/type_ptr.hpp>  // For glm::value_ptr>
#include <stdexcept>
#include <algorithm>

// Core includes
#include <core/scene.h>
#include <core/camera.h>
#include <core/film.h>

// OpenGL includes for Zero-Copy interop
#include <opengl/opengl_utils.h>

//------------------------------------------------------------------------------
// Rendering Constants (extracted from CPU renderer source)
//------------------------------------------------------------------------------

namespace {

// Sky light color from CPU renderer.h:29
constexpr float SKY_R = 0.4f;
constexpr float SKY_G = 0.3f;
constexpr float SKY_B = 0.6f;

// Material constants
constexpr float GLASS_REFRACTION_INDEX = 0.9f;  // From CPU scene.h
constexpr float METAL_ROUGHNESS = 0.01f;        // Mirror-like metal

} // anonymous namespace

//------------------------------------------------------------------------------
// Macros for error checking
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t rc = call;                                                 \
        if (rc != cudaSuccess) {                                               \
            std::stringstream ss;                                              \
            ss << "CUDA Error: " << cudaGetErrorString(rc)                     \
               << " (" << __FILE__ << ":" << __LINE__ << ")";                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)

#define CU_CHECK(call)                                                         \
    do {                                                                       \
        CUresult rc = call;                                                    \
        if (rc != CUDA_SUCCESS) {                                              \
            const char* errStr;                                                \
            cuGetErrorString(rc, &errStr);                                     \
            std::stringstream ss;                                              \
            ss << "CUDA Driver Error: " << errStr                              \
               << " (" << __FILE__ << ":" << __LINE__ << ")";                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult rc = call;                                                 \
        if (rc != OPTIX_SUCCESS) {                                             \
            std::stringstream ss;                                              \
            ss << "OptiX Error: " << optixGetErrorName(rc)                     \
               << " - " << optixGetErrorString(rc)                             \
               << " (" << __FILE__ << ":" << __LINE__ << ")";                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)

//------------------------------------------------------------------------------
// SBT Record Templates
//------------------------------------------------------------------------------

template<typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RaygenRecord   = SbtRecord<int>;
using MissRecord     = SbtRecord<int>;
using HitGroupRecord = SbtRecord<HitGroupData>;

//------------------------------------------------------------------------------
// Helper: Load PTX from file
//------------------------------------------------------------------------------

static std::string loadPTXFromFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open PTX file: " + filename);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

//------------------------------------------------------------------------------
// Helper Functions for Dynamic Scene Loading
//------------------------------------------------------------------------------

namespace {

// Convert glm::mat4 (Column-major) to OptiX transform (Row-major 3x4)
static void ToOptixTransform(const glm::mat4& mat, float transform[12])
{
    const float* s = glm::value_ptr(mat); // GLM is Column-Major
    
    // OptiX expects Row-Major 3x4:
    // [0  1  2  3 ]  (Row 0)
    // [4  5  6  7 ]  (Row 1)
    // [8  9 10 11]  (Row 2)
    
    // GLM memory layout (Column-major):
    // s[0-3]: Column 0, s[4-7]: Column 1, s[8-11]: Column 2, s[12-15]: Column 3
    
    // Transpose while collecting only first 3 rows
    transform[0]  = s[0];  transform[1]  = s[4];  transform[2]  = s[8];   transform[3]  = s[12];
    transform[4]  = s[1];  transform[5]  = s[5];  transform[6]  = s[9];   transform[7]  = s[13];
    transform[8]  = s[2];  transform[9]  = s[6];  transform[10] = s[10];  transform[11] = s[14];
}

} // anonymous namespace

//------------------------------------------------------------------------------
// OptixRenderer Implementation
//------------------------------------------------------------------------------

OptixRenderer::~OptixRenderer()
{
    cleanup();
}

void OptixRenderer::cleanup()
{
    if (!m_Initialized) return;
    
    // Free device buffers
    if (m_d_ColorBuffer)   cudaFree(reinterpret_cast<void*>(m_d_ColorBuffer));
    if (m_d_AccumBuffer)   cudaFree(reinterpret_cast<void*>(m_d_AccumBuffer));
    if (m_d_LaunchParams)  cudaFree(reinterpret_cast<void*>(m_d_LaunchParams));
    if (m_d_SphereData)    cudaFree(reinterpret_cast<void*>(m_d_SphereData));
    if (m_d_QuadData)      cudaFree(reinterpret_cast<void*>(m_d_QuadData));
    if (m_d_TriangleData)  cudaFree(reinterpret_cast<void*>(m_d_TriangleData));
    if (m_d_Materials)     cudaFree(reinterpret_cast<void*>(m_d_Materials));
    
    // Free OpenGL Interop resources
    if (m_CudaGraphicsResource) {
        cudaGraphicsUnregisterResource(m_CudaGraphicsResource);
        m_CudaGraphicsResource = nullptr;
    }
    if (m_PBO) {
        glDeleteBuffers(1, &m_PBO);
        m_PBO = 0;
    }
    
    // Free acceleration structures (IAS and per-geometry GAS)
    if (m_IasBuffer)       cudaFree(reinterpret_cast<void*>(m_IasBuffer));
    if (m_SphereGasBuffer) cudaFree(reinterpret_cast<void*>(m_SphereGasBuffer));
    if (m_QuadGasBuffer)   cudaFree(reinterpret_cast<void*>(m_QuadGasBuffer));
    
    // Free SBT records
    if (m_RaygenRecord)    cudaFree(reinterpret_cast<void*>(m_RaygenRecord));
    if (m_MissRecord)      cudaFree(reinterpret_cast<void*>(m_MissRecord));
    if (m_HitgroupRecord)  cudaFree(reinterpret_cast<void*>(m_HitgroupRecord));
    
    // Destroy OptiX objects
    if (m_Pipeline)        optixPipelineDestroy(m_Pipeline);
    if (m_RaygenPG)        optixProgramGroupDestroy(m_RaygenPG);
    if (m_MissPG)          optixProgramGroupDestroy(m_MissPG);
    if (m_HitSpherePG)     optixProgramGroupDestroy(m_HitSpherePG);
    if (m_HitQuadPG)       optixProgramGroupDestroy(m_HitQuadPG);
    if (m_HitTrianglePG)   optixProgramGroupDestroy(m_HitTrianglePG);
    if (m_Module)          optixModuleDestroy(m_Module);
    if (m_OptixContext)    optixDeviceContextDestroy(m_OptixContext);
    
    // Destroy CUDA stream
    if (m_Stream)          cudaStreamDestroy(m_Stream);
    
    m_Initialized = false;
}

void OptixRenderer::initCUDA()
{
    CUDA_CHECK(cudaFree(0));  // Force CUDA initialization
    
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA-capable device found");
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&m_Stream));
    
    std::cout << "[OptixRenderer] CUDA initialized\n";
}

void OptixRenderer::initOptix()
{
    OPTIX_CHECK(optixInit());
    
    CUcontext cuContext;
    CU_CHECK(cuCtxGetCurrent(&cuContext));
    
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackLevel = 4;
    
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &m_OptixContext));
    
    std::cout << "[OptixRenderer] OptiX context created\n";
}

void OptixRenderer::createModule()
{
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    
    m_PipelineCompileOptions = {};
    m_PipelineCompileOptions.usesMotionBlur = false;
    m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_PipelineCompileOptions.numPayloadValues = 2;
    m_PipelineCompileOptions.numAttributeValues = 2;
    m_PipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_PipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    m_PipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    
    // Load PTX
    std::string ptxCode;
#ifdef PTX_DIR
    std::string ptxPath = std::string(PTX_DIR) + "/device_programs.ptx";
#else
    std::string ptxPath = "device_programs.ptx";
#endif
    
    ptxCode = loadPTXFromFile(ptxPath);
    std::cout << "[OptixRenderer] Loaded PTX from: " << ptxPath << "\n";
    
    char log[2048];
    size_t logSize = sizeof(log);
    
    OPTIX_CHECK(optixModuleCreate(
        m_OptixContext,
        &moduleCompileOptions,
        &m_PipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log, &logSize,
        &m_Module));
    
    if (logSize > 1) {
        std::cout << "[OptixRenderer] Module log: " << log << "\n";
    }
    
    std::cout << "[OptixRenderer] Module created\n";
}

void OptixRenderer::createProgramGroups()
{
    OptixProgramGroupOptions pgOptions = {};
    char log[2048];
    size_t logSize;
    
    // Raygen
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = m_Module;
        desc.raygen.entryFunctionName = "__raygen__renderFrame";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            m_OptixContext, &desc, 1, &pgOptions, log, &logSize, &m_RaygenPG));
    }
    
    // Miss
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = m_Module;
        desc.miss.entryFunctionName = "__miss__radiance";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            m_OptixContext, &desc, 1, &pgOptions, log, &logSize, &m_MissPG));
    }
    
    // Hit group for spheres
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = m_Module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
        desc.hitgroup.moduleIS = m_Module;
        desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            m_OptixContext, &desc, 1, &pgOptions, log, &logSize, &m_HitSpherePG));
    }
    
    // Hit group for quads
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = m_Module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__quad";
        desc.hitgroup.moduleIS = m_Module;
        desc.hitgroup.entryFunctionNameIS = "__intersection__quad";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            m_OptixContext, &desc, 1, &pgOptions, log, &logSize, &m_HitQuadPG));
    }
    
    std::cout << "[OptixRenderer] Program groups created\n";
}

void OptixRenderer::createPipeline()
{
    OptixProgramGroup programGroups[] = {
        m_RaygenPG,
        m_MissPG,
        m_HitSpherePG,
        m_HitQuadPG
    };
    
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = m_MaxDepth;
    
    char log[2048];
    size_t logSize = sizeof(log);
    
    OPTIX_CHECK(optixPipelineCreate(
        m_OptixContext,
        &m_PipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(programGroups[0]),
        log, &logSize,
        &m_Pipeline));
    
    if (logSize > 1) {
        std::cout << "[OptixRenderer] Pipeline log: " << log << "\n";
    }
    
    std::cout << "[OptixRenderer] Pipeline created\n";
}

void OptixRenderer::buildSBT()
{
    // Raygen record
    {
        RaygenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_RaygenPG, &rec));
        
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_RaygenRecord), sizeof(RaygenRecord)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_RaygenRecord), &rec, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    }
    
    // Miss record
    {
        MissRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_MissPG, &rec));
        
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_MissRecord), sizeof(MissRecord)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_MissRecord), &rec, sizeof(MissRecord), cudaMemcpyHostToDevice));
    }
    
    // Hit group records - one for spheres, one for quads
    {
        std::vector<HitGroupRecord> hitRecords;
        
        // Sphere hit record
        {
            HitGroupRecord rec = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(m_HitSpherePG, &rec));
            rec.data.geometryData = reinterpret_cast<void*>(m_d_SphereData);
            rec.data.materials = reinterpret_cast<DeviceMaterial*>(m_d_Materials);
            rec.data.geometryType = GEOMETRY_TYPE_SPHERE;
            hitRecords.push_back(rec);
        }
        
        // Quad hit record
        {
            HitGroupRecord rec = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(m_HitQuadPG, &rec));
            rec.data.geometryData = reinterpret_cast<void*>(m_d_QuadData);
            rec.data.materials = reinterpret_cast<DeviceMaterial*>(m_d_Materials);
            rec.data.geometryType = GEOMETRY_TYPE_QUAD;
            hitRecords.push_back(rec);
        }
        
        size_t hitRecordsSize = sizeof(HitGroupRecord) * hitRecords.size();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_HitgroupRecord), hitRecordsSize));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_HitgroupRecord), hitRecords.data(), hitRecordsSize, cudaMemcpyHostToDevice));
        
        m_SBT.hitgroupRecordBase = m_HitgroupRecord;
        m_SBT.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        m_SBT.hitgroupRecordCount = static_cast<unsigned int>(hitRecords.size());
    }
    
    m_SBT.raygenRecord = m_RaygenRecord;
    m_SBT.missRecordBase = m_MissRecord;
    m_SBT.missRecordStrideInBytes = sizeof(MissRecord);
    m_SBT.missRecordCount = 1;
    
    std::cout << "[OptixRenderer] SBT built\n";
}

void OptixRenderer::uploadSceneData()
{
    // Create test scene matching CPU renderer style
    std::vector<SphereData> spheres;
    std::vector<QuadData> quads;
    std::vector<DeviceMaterial> materials;
    
    // Ground - material 0
    {
        DeviceMaterial mat;
        mat.type = MaterialType::Lambertian;
        mat.albedo = make_float3(0.7f, 0.7f, 0.4f);
        mat.roughness = 0.0f;
        mat.refractionIndex = 1.0f;
        mat.emission = make_float3(0.0f, 0.0f, 0.0f);
        materials.push_back(mat);
        
        QuadData quad;
        quad.corner = make_float3(-10.0f, 0.0f, -10.0f);
        quad.u = make_float3(20.0f, 0.0f, 0.0f);
        quad.v = make_float3(0.0f, 0.0f, 20.0f);
        quad.normal = make_float3(0.0f, 1.0f, 0.0f);
        quad.materialIndex = 0;
        quads.push_back(quad);
    }
    
    // Green sphere - material 1
    {
        DeviceMaterial mat;
        mat.type = MaterialType::Lambertian;
        mat.albedo = make_float3(0.2f, 1.0f, 0.2f);
        materials.push_back(mat);
        
        SphereData sphere;
        sphere.center = make_float3(4.0f, 1.0f, 0.0f);
        sphere.radius = 1.0f;
        sphere.materialIndex = 1;
        spheres.push_back(sphere);
    }
    
    // Red sphere - material 2
    {
        DeviceMaterial mat;
        mat.type = MaterialType::Lambertian;
        mat.albedo = make_float3(1.0f, 0.2f, 0.2f);
        materials.push_back(mat);
        
        SphereData sphere;
        sphere.center = make_float3(-4.0f, 1.0f, 0.0f);
        sphere.radius = 1.0f;
        sphere.materialIndex = 2;
        spheres.push_back(sphere);
    }
    
    // Glass sphere - material 3
    {
        DeviceMaterial mat;
        mat.type = MaterialType::Dielectric;
        mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
        mat.refractionIndex = GLASS_REFRACTION_INDEX;  // From CPU scene.h
        materials.push_back(mat);
        
        SphereData sphere;
        sphere.center = make_float3(0.0f, 1.0f, 4.0f);
        sphere.radius = 1.0f;
        sphere.materialIndex = 3;
        spheres.push_back(sphere);
    }
    
    // Metal sphere - material 4
    {
        DeviceMaterial mat;
        mat.type = MaterialType::Metal;
        mat.albedo = make_float3(1.0f, 0.7f, 0.8f);
        mat.roughness = METAL_ROUGHNESS;  // Mirror-like surface
        materials.push_back(mat);
        
        SphereData sphere;
        sphere.center = make_float3(0.0f, 1.0f, -4.0f);
        sphere.radius = 1.0f;
        sphere.materialIndex = 4;
        spheres.push_back(sphere);
    }
    
    // Light quads - materials 5 and 6
    {
        DeviceMaterial mat;
        mat.type = MaterialType::Emissive;
        // Match CPU scene.h quad1: EmissiveMaterial(glm::vec3{ 3.0f, 4.0f, 2.0f })
        mat.emission = make_float3(3.0f, 4.0f, 2.0f);
        materials.push_back(mat);
        
        QuadData quad;
        quad.corner = make_float3(-8.0f, 7.0f, 3.0f);
        quad.u = make_float3(8.0f, 0.0f, 0.0f);
        quad.v = make_float3(0.0f, 0.0f, 8.0f);
        quad.normal = make_float3(0.0f, -1.0f, 0.0f);
        quad.materialIndex = 5;
        quads.push_back(quad);
    }
    {
        DeviceMaterial mat;
        mat.type = MaterialType::Emissive;
        mat.emission = make_float3(3.0f, 2.0f, 1.0f);
        materials.push_back(mat);
        
        QuadData quad;
        quad.corner = make_float3(0.0f, 7.0f, 3.0f);
        quad.u = make_float3(8.0f, 0.0f, 0.0f);
        quad.v = make_float3(0.0f, 0.0f, 8.0f);
        quad.normal = make_float3(0.0f, -1.0f, 0.0f);
        quad.materialIndex = 6;
        quads.push_back(quad);
    }
    
    // Upload to GPU
    m_NumSpheres = spheres.size();
    m_NumQuads = quads.size();
    m_NumMaterials = materials.size();
    
    if (m_NumSpheres > 0) {
        size_t size = sizeof(SphereData) * m_NumSpheres;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_SphereData), size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_SphereData), spheres.data(), size, cudaMemcpyHostToDevice));
    }
    
    if (m_NumQuads > 0) {
        size_t size = sizeof(QuadData) * m_NumQuads;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_QuadData), size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_QuadData), quads.data(), size, cudaMemcpyHostToDevice));
    }
    
    if (m_NumMaterials > 0) {
        size_t size = sizeof(DeviceMaterial) * m_NumMaterials;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_Materials), size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_d_Materials), materials.data(), size, cudaMemcpyHostToDevice));
    }
    
    std::cout << "[OptixRenderer] Scene: " << m_NumSpheres << " spheres, " << m_NumQuads << " quads\n";
}

void OptixRenderer::buildAccelerationStructure()
{
    std::vector<OptixInstance> instances;
    unsigned int instanceId = 0;
    
    // ==== Build Sphere GAS ====
    if (m_NumSpheres > 0) {
        std::vector<OptixAabb> aabbs(m_NumSpheres);
        float3 centers[] = {
            {4.0f, 1.0f, 0.0f},
            {-4.0f, 1.0f, 0.0f},
            {0.0f, 1.0f, 4.0f},
            {0.0f, 1.0f, -4.0f}
        };
        
        for (size_t i = 0; i < m_NumSpheres; ++i) {
            float r = 1.0f;
            aabbs[i].minX = centers[i].x - r;
            aabbs[i].minY = centers[i].y - r;
            aabbs[i].minZ = centers[i].z - r;
            aabbs[i].maxX = centers[i].x + r;
            aabbs[i].maxY = centers[i].y + r;
            aabbs[i].maxZ = centers[i].z + r;
        }
        
        CUdeviceptr d_aabbBuffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabbBuffer), sizeof(OptixAabb) * m_NumSpheres));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabbBuffer), aabbs.data(), sizeof(OptixAabb) * m_NumSpheres, cudaMemcpyHostToDevice));
        
        uint32_t sphereFlags = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput sphereInput = {};
        sphereInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        sphereInput.customPrimitiveArray.aabbBuffers = &d_aabbBuffer;
        sphereInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(m_NumSpheres);
        sphereInput.customPrimitiveArray.flags = &sphereFlags;
        sphereInput.customPrimitiveArray.numSbtRecords = 1;
        
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        
        OptixAccelBufferSizes bufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(m_OptixContext, &accelOptions, &sphereInput, 1, &bufferSizes));
        
        CUdeviceptr d_tempBuffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), bufferSizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_SphereGasBuffer), bufferSizes.outputSizeInBytes));
        
        OPTIX_CHECK(optixAccelBuild(
            m_OptixContext, m_Stream, &accelOptions,
            &sphereInput, 1,
            d_tempBuffer, bufferSizes.tempSizeInBytes,
            m_SphereGasBuffer, bufferSizes.outputSizeInBytes,
            &m_SphereGasHandle, nullptr, 0));
        
        CUDA_CHECK(cudaStreamSynchronize(m_Stream));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabbBuffer)));
        
        // Create instance for spheres - SBT offset 0
        OptixInstance sphereInstance = {};
        sphereInstance.transform[0] = 1.0f; sphereInstance.transform[5] = 1.0f; sphereInstance.transform[10] = 1.0f;
        sphereInstance.instanceId = instanceId++;
        sphereInstance.sbtOffset = 0;  // First hit group record (HitSpherePG)
        sphereInstance.visibilityMask = 255;
        sphereInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        sphereInstance.traversableHandle = m_SphereGasHandle;
        instances.push_back(sphereInstance);
    }
    
    // ==== Build Quad GAS ====
    if (m_NumQuads > 0) {
        std::vector<OptixAabb> aabbs(m_NumQuads);
        
        struct QuadInfo { float3 corner, u, v; };
        QuadInfo quadInfos[] = {
            {{-10.0f, 0.0f, -10.0f}, {20.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 20.0f}},
            {{-8.0f, 7.0f, 3.0f}, {8.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 8.0f}},
            {{0.0f, 7.0f, 3.0f}, {8.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 8.0f}}
        };
        
        for (size_t i = 0; i < m_NumQuads; ++i) {
            float3 c = quadInfos[i].corner;
            float3 u = quadInfos[i].u;
            float3 v = quadInfos[i].v;
            
            aabbs[i].minX = fminf(fminf(c.x, c.x + u.x), fminf(c.x + v.x, c.x + u.x + v.x)) - 0.01f;
            aabbs[i].minY = fminf(fminf(c.y, c.y + u.y), fminf(c.y + v.y, c.y + u.y + v.y)) - 0.01f;
            aabbs[i].minZ = fminf(fminf(c.z, c.z + u.z), fminf(c.z + v.z, c.z + u.z + v.z)) - 0.01f;
            aabbs[i].maxX = fmaxf(fmaxf(c.x, c.x + u.x), fmaxf(c.x + v.x, c.x + u.x + v.x)) + 0.01f;
            aabbs[i].maxY = fmaxf(fmaxf(c.y, c.y + u.y), fmaxf(c.y + v.y, c.y + u.y + v.y)) + 0.01f;
            aabbs[i].maxZ = fmaxf(fmaxf(c.z, c.z + u.z), fmaxf(c.z + v.z, c.z + u.z + v.z)) + 0.01f;
        }
        
        CUdeviceptr d_aabbBuffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabbBuffer), sizeof(OptixAabb) * m_NumQuads));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabbBuffer), aabbs.data(), sizeof(OptixAabb) * m_NumQuads, cudaMemcpyHostToDevice));
        
        uint32_t quadFlags = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput quadInput = {};
        quadInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        quadInput.customPrimitiveArray.aabbBuffers = &d_aabbBuffer;
        quadInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(m_NumQuads);
        quadInput.customPrimitiveArray.flags = &quadFlags;
        quadInput.customPrimitiveArray.numSbtRecords = 1;
        
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        
        OptixAccelBufferSizes bufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(m_OptixContext, &accelOptions, &quadInput, 1, &bufferSizes));
        
        CUdeviceptr d_tempBuffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), bufferSizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_QuadGasBuffer), bufferSizes.outputSizeInBytes));
        
        OPTIX_CHECK(optixAccelBuild(
            m_OptixContext, m_Stream, &accelOptions,
            &quadInput, 1,
            d_tempBuffer, bufferSizes.tempSizeInBytes,
            m_QuadGasBuffer, bufferSizes.outputSizeInBytes,
            &m_QuadGasHandle, nullptr, 0));
        
        CUDA_CHECK(cudaStreamSynchronize(m_Stream));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabbBuffer)));
        
        // Create instance for quads - SBT offset 1
        OptixInstance quadInstance = {};
        quadInstance.transform[0] = 1.0f; quadInstance.transform[5] = 1.0f; quadInstance.transform[10] = 1.0f;
        quadInstance.instanceId = instanceId++;
        quadInstance.sbtOffset = 1;  // Second hit group record (HitQuadPG)
        quadInstance.visibilityMask = 255;
        quadInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        quadInstance.traversableHandle = m_QuadGasHandle;
        instances.push_back(quadInstance);
    }
    
    if (instances.empty()) {
        std::cout << "[OptixRenderer] Warning: No geometry instances\n";
        return;
    }
    
    // ==== Build IAS ====
    CUdeviceptr d_instances;
    size_t instancesSize = sizeof(OptixInstance) * instances.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instancesSize));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances), instances.data(), instancesSize, cudaMemcpyHostToDevice));
    
    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = d_instances;
    iasInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());
    
    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    iasOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_OptixContext, &iasOptions, &iasInput, 1, &iasBufferSizes));
    
    CUdeviceptr d_tempBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), iasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_IasBuffer), iasBufferSizes.outputSizeInBytes));
    
    OPTIX_CHECK(optixAccelBuild(
        m_OptixContext, m_Stream, &iasOptions,
        &iasInput, 1,
        d_tempBuffer, iasBufferSizes.tempSizeInBytes,
        m_IasBuffer, iasBufferSizes.outputSizeInBytes,
        &m_TraversableHandle, nullptr, 0));
    
    CUDA_CHECK(cudaStreamSynchronize(m_Stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
    
    std::cout << "[OptixRenderer] IAS built with " << instances.size() << " instances\n";
}

void OptixRenderer::updateLaunchParams()
{
    // Camera movement detection for accumulation reset
    static glm::vec3 lastPos = glm::vec3(0.0f);
    static glm::vec3 lastDir = glm::vec3(0.0f);
    
    glm::vec3 currentPos = m_Camera->GetPosition();
    glm::vec3 currentDir = m_Camera->GetViewDir();
    
    // Reset frame index if camera moved
    if (currentPos != lastPos || currentDir != lastDir) {
        m_FrameIndex = 0;
        // Clear accumulation buffer
        if (m_d_AccumBuffer && m_Film) {
            size_t bufferSize = sizeof(float3) * m_Film->GetWidth() * m_Film->GetHeight();
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_d_AccumBuffer), 0, bufferSize));
        }
        lastPos = currentPos;
        lastDir = currentDir;
    }
    
    LaunchParams launchParams = {};
    
    // Camera
    launchParams.camera.position = make_float3(
        m_Camera->GetPosition().x,
        m_Camera->GetPosition().y,
        m_Camera->GetPosition().z);
    
    glm::vec3 front = m_Camera->GetViewDir();
    glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 up = glm::normalize(glm::cross(right, front));
    
    launchParams.camera.front = make_float3(front.x, front.y, front.z);
    launchParams.camera.right = make_float3(right.x, right.y, right.z);
    launchParams.camera.up = make_float3(up.x, up.y, up.z);
    launchParams.camera.width = m_Camera->GetWidth();
    launchParams.camera.height = m_Camera->GetHeight();
    launchParams.camera.tanFovY = tanf(0.5f);
    launchParams.camera.aspectRatio = m_Camera->GetWidth() / m_Camera->GetHeight();
    
    // Debug output on first frame
    if (m_FrameIndex == 0) {
        std::cout << "[OptixRenderer] Camera Debug:\n";
        std::cout << "  Position: (" << m_Camera->GetPosition().x << ", " 
                  << m_Camera->GetPosition().y << ", " << m_Camera->GetPosition().z << ")\n";
        std::cout << "  Front: (" << front.x << ", " << front.y << ", " << front.z << ")\n";
        std::cout << "  Right: (" << right.x << ", " << right.y << ", " << right.z << ")\n";
        std::cout << "  Up: (" << up.x << ", " << up.y << ", " << up.z << ")\n";
        std::cout << "  Film: " << m_Film->GetWidth() << "x" << m_Film->GetHeight() << "\n";
        std::cout << "  AspectRatio: " << launchParams.camera.aspectRatio << ", tanFovY: " << launchParams.camera.tanFovY << "\n";
    }
    
    launchParams.width = m_Film->GetWidth();
    launchParams.height = m_Film->GetHeight();
    launchParams.frameIndex = m_FrameIndex;
    launchParams.traversable = m_TraversableHandle;
    launchParams.maxDepth = m_MaxDepth;
    
    // Sky light from constants (CPU renderer.h:29)
    launchParams.skyLight = make_float3(SKY_R, SKY_G, SKY_B);
    launchParams.colorBuffer = reinterpret_cast<float3*>(m_d_ColorBuffer);
    launchParams.accumBuffer = reinterpret_cast<float3*>(m_d_AccumBuffer);
    
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_d_LaunchParams),
        &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

void OptixRenderer::Init(Film& film, const Scene& scene, const Camera& camera)
{
    m_Film = &film;
    m_Scene = &scene;
    m_Camera = &camera;
    
    film.Clear();
    
    try {
        initCUDA();
        initOptix();
        uploadSceneData();
        createModule();
        createProgramGroups();
        createPipeline();
        buildAccelerationStructure();
        buildSBT();
        
        // Allocate output buffers
        size_t bufferSize = sizeof(float3) * film.GetWidth() * film.GetHeight();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_ColorBuffer), bufferSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_AccumBuffer), bufferSize));
        
        // Initialize accumulation buffer to zero
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(m_d_AccumBuffer), 0, bufferSize));
        
        // Create OpenGL PBO for Zero-Copy rendering
        glGenBuffers(1, &m_PBO);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        
        // Register PBO with CUDA for interop
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
            &m_CudaGraphicsResource,
            m_PBO,
            cudaGraphicsMapFlagsWriteDiscard));
        
        std::cout << "[OptixRenderer] OpenGL PBO created and registered with CUDA\n";
        
        // Allocate launch params
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_LaunchParams), sizeof(LaunchParams)));
        
        m_FrameIndex = 0;
        m_Initialized = true;
        
        std::cout << "[OptixRenderer] Initialization complete!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "[OptixRenderer] Init failed: " << e.what() << "\n";
        cleanup();
    }
}

void OptixRenderer::ProgressiveRender()
{
    if (!m_Initialized || !m_Film) return;
    
    const uint32_t width = m_Film->GetWidth();
    const uint32_t height = m_Film->GetHeight();
    
    if (width == 0 || height == 0) return;
    
    updateLaunchParams();
    
    OPTIX_CHECK(optixLaunch(
        m_Pipeline,
        m_Stream,
        m_d_LaunchParams,
        sizeof(LaunchParams),
        &m_SBT,
        width, height, 1));
    
    CUDA_CHECK(cudaStreamSynchronize(m_Stream));
    
    // Copy results back
    std::vector<float3> colorBuffer(width * height);
    CUDA_CHECK(cudaMemcpy(
        colorBuffer.data(),
        reinterpret_cast<void*>(m_d_ColorBuffer),
        sizeof(float3) * width * height,
        cudaMemcpyDeviceToHost));
    
    // Convert to RGB
    std::vector<float> rgb(width * height * 3);
    for (size_t i = 0; i < colorBuffer.size(); ++i) {
        rgb[i * 3 + 0] = colorBuffer[i].x;
        rgb[i * 3 + 1] = colorBuffer[i].y;
        rgb[i * 3 + 2] = colorBuffer[i].z;
    }
    
    m_Film->AddSampleBuffer(rgb.data());
    m_FrameIndex++;
}

// Zero-Copy rendering with OpenGL Interop
void OptixRenderer::ProgressiveRender(OpenGLTexture* targetTexture)
{
    if (!m_Initialized || !m_Film) return;
    
    const uint32_t width = m_Film->GetWidth();
    const uint32_t height = m_Film->GetHeight();
    
    if (width == 0 || height == 0) return;
    
    updateLaunchParams();
    
    // Launch OptiX rendering
    OPTIX_CHECK(optixLaunch(
        m_Pipeline,
        m_Stream,
        m_d_LaunchParams,
        sizeof(LaunchParams),
        &m_SBT,
        width, height, 1));
    
    if (targetTexture)
    {
        // Zero-Copy path: Direct GPU-to-GPU transfer
        
        // 1. Map PBO to get CUDA pointer
        CUDA_CHECK(cudaGraphicsMapResources(1, &m_CudaGraphicsResource, m_Stream));
        
        void* pbo_d_ptr = nullptr;
        size_t num_bytes = 0;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&pbo_d_ptr, &num_bytes, m_CudaGraphicsResource));
        
        // 2. Device-to-Device copy (OptiX buffer -> PBO)
        // This is extremely fast as data stays in VRAM
        CUDA_CHECK(cudaMemcpyAsync(
            pbo_d_ptr,
            reinterpret_cast<void*>(m_d_ColorBuffer),
            width * height * sizeof(float3),
            cudaMemcpyDeviceToDevice,
            m_Stream));
        
        // 3. Unmap PBO
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_CudaGraphicsResource, m_Stream));
        
        // 4. Synchronize to ensure copy is complete
        CUDA_CHECK(cudaStreamSynchronize(m_Stream));
        
        // 5. Update OpenGL texture from PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
        glBindTexture(GL_TEXTURE_2D, targetTexture->GetTextureID());
        
        // Transfer from PBO to texture (last param is 0 because PBO is bound)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, 0);
        
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        // Fallback to slow CPU path if no texture provided
        CUDA_CHECK(cudaStreamSynchronize(m_Stream));
        
        std::vector<float3> colorBuffer(width * height);
        CUDA_CHECK(cudaMemcpy(
            colorBuffer.data(),
            reinterpret_cast<void*>(m_d_ColorBuffer),
            sizeof(float3) * width * height,
            cudaMemcpyDeviceToHost));
        
        std::vector<float> rgb(width * height * 3);
        for (size_t i = 0; i < colorBuffer.size(); ++i) {
            rgb[i * 3 + 0] = colorBuffer[i].x;
            rgb[i * 3 + 1] = colorBuffer[i].y;
            rgb[i * 3 + 2] = colorBuffer[i].z;
        }
        
        m_Film->AddSampleBuffer(rgb.data());
    }
    
    m_FrameIndex++;
}
