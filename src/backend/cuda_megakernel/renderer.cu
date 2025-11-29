#include "backend/cuda_megakernel/renderer.h"
#include <vector>

/*
TODO

- Optimize the cam buffer (call once in init(), and don't need to malloc and delete the buffer)
- remove the hardcoded depth
- Optimize the Intersect Function
    - tMin hardcoded
    - if else thing
*/

namespace
{
QUAL_GPU inline float  IntersectGPUSphere(const GPUSphere& sphere, Ray &pray, SurfaceInteraction* intersect) {
    Ray ray;
    ray.Origin = TransformPoint(sphere.transform.GetInvMat(), pray.Origin);
    ray.Direction = TransformNormal(sphere.transform.GetMat(), pray.Direction);
    
    auto l = ray.Origin;
    float a = glm::dot(ray.Direction, ray.Direction);
    float b = 2.0f * glm::dot(l, ray.Direction);
    float c = glm::dot(l,l) - sphere.radius * sphere.radius;

    constexpr static float tMin = 0.001f;

    float discriminant = b * b - 4.0f * a * c;

    if (discriminant >= 0.0f)
    {
        float t1 = (-b + sqrtf(discriminant)) /( 2 * a);
        float t2 = (-b - sqrtf(discriminant)) /( 2 * a);
        float t = 0.0f;

        intersect->HasIntersection = true;
        if (t1 >= tMin && t2 >= tMin)
        {
            t = t1 < t2 ? t1 : t2;
            intersect->IsFrontFace = true;
        }
        else if (t1 >= tMin)
        {
            t = t1;
            intersect->IsFrontFace = false;
        }
        else if (t2 >= tMin)
        {
            t = t2;
            intersect->IsFrontFace = false;
        }
        else
        {
            intersect->HasIntersection = false;
        }
        auto intersectPoint = ray.Origin + ray.Direction * t;
        auto normal = glm::normalize(intersectPoint);
        if (!intersect->IsFrontFace)
            normal *= -1.0f;
        intersect->Position = intersectPoint;
        intersect->Normal = normal;
    }
    else
    {
        intersect->HasIntersection = false;
    }

    return discriminant;
}
__global__ void GPU_RayTracing(float* colors, GPUCamera * cam, GPUSphere* spheres)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixelCount = cam->width * cam->height;
    if (idx >= pixelCount)
        return;

    const uint32_t x = idx % cam->width;
    const uint32_t y = idx / cam->width;

    Ray ray = cam->GetCameraRay((float)x + 0.5f, (float)y + 0.5f);
    
    // // Trace Ray
    // const uint32_t max_depth = 20;
    // for (uint32_t depth = 0; depth < max_depth; ++depth) {
    //     glm::vec3 L{ 0.0f };
    //     SurfaceInteraction intersect;
    // }
    float colora = 0.0f;
    float colorb = 0.0f;
    for (int i=0; i<4; ++i) {
        SurfaceInteraction intersect;
        float det = IntersectGPUSphere(spheres[i], ray, &intersect);
        if (intersect.HasIntersection) {
            colora = 255.0f;
        }
        else {
            colorb = -det / 1000.0f;
        }
    }

    // Method 1: GLM Normalize
    const uint32_t base = idx * 3;
    glm::vec3 pos = glm::normalize(cam->position);
    colors[base + 0] = pos.x;
    colors[base + 1] = pos.y;
    colors[base + 2] = pos.z;

    // Method 2: Manual Normalize
    // in device code
    // float px = cam->position.x;
    // float py = cam->position.y;
    // float pz = cam->position.z;

    // float lenSq = px * px + py * py + pz * pz;

    // // avoid divide-by-zero
    // if (lenSq > 0.0f) {
    //     float invLen = rsqrtf(lenSq);   // 1 / sqrt(lenSq), CUDA intrinsic

    //     float nx = px * invLen;
    //     float ny = py * invLen;
    //     float nz = pz * invLen;

    //     colors[base + 0] = nx;
    //     colors[base + 1] = ny;
    //     colors[base + 2] = nz;
    // } else {
    //     // fallback if cam->position == (0,0,0)
    //     colors[base + 0] = 0.0f;
    //     colors[base + 1] = 0.0f;
    //     colors[base + 2] = 0.0f;
    // }
}

}

void CudaMegakernelRenderer::Init(Film& film, const Scene& scene, const Camera& camera)
{
    m_Film = &film;
    m_Scene = &scene;
    m_Camera = &camera;
}

void CudaMegakernelRenderer::ProgressiveRender()
{
    if (!m_Film)
        return;

    const uint32_t width = m_Film->GetWidth();
    const uint32_t height = m_Film->GetHeight();
    const uint32_t pixelCount = width * height;

    if (pixelCount == 0)
        return;

    const size_t bufferSize = static_cast<size_t>(pixelCount) * 3 * sizeof(float);

    float* deviceBuffer = nullptr;
    cudaMalloc(&deviceBuffer, bufferSize);

    const uint32_t threadsPerBlock = 256;
    const uint32_t blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;

    // Data Parameters
    GPUCamera* d_cam = CreateGPUCamera(m_Camera);
    GPUSphere* spheres = ConvertCirclesToGPU();

    GPU_RayTracing<<<blocks, threadsPerBlock>>>(deviceBuffer, d_cam, spheres);
    cudaDeviceSynchronize();
    
    
    // Render
    std::vector<float> hostBuffer(static_cast<size_t>(pixelCount) * 3);
    cudaMemcpy(hostBuffer.data(), deviceBuffer, bufferSize, cudaMemcpyDeviceToHost);

    m_Film->AddSampleBuffer(hostBuffer.data());

    cudaFree(deviceBuffer);
    cudaFree(d_cam);
    cudaFree(spheres);
}

GPUSphere* CudaMegakernelRenderer::ConvertCirclesToGPU() {
    std::vector<GPUSphere> gs_list;

    const auto clist = m_Scene->getCircles();
    for (auto circle: clist) {
        auto _c = std::static_pointer_cast<SimplePrimitive>(circle);
        Shape& _s = _c->GetShape();
        Circle& c = dynamic_cast<Circle&>(_s);

        GPUSphere gs;
        gs.transform = _c->GetTransform();
        gs.radius = c.getRadius();

        gs_list.push_back(gs);
    }

    GPUSphere* gs_ptr;
    size_t size = sizeof(GPUSphere) * gs_list.size();
    cudaMalloc(&gs_ptr, size);
    cudaMemcpy(gs_ptr, gs_list.data(), size, cudaMemcpyHostToDevice);

    return gs_ptr;
}

GPUCamera* CreateGPUCamera(const Camera* cam) {
    GPUCamera gc{};

    gc.position = cam->GetPosition();
    gc.forward  = cam->GetViewDir();

    glm::vec3 forward = cam->GetViewDir();
    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
    glm::vec3 up = glm::normalize(glm::cross(right, forward));

    gc.right = right;
    gc.up = up;

    gc.focal = cam->GetFocal();
    gc.width  = cam->GetWidth();
    gc.height = cam->GetHeight();
    gc.focal  = cam->GetFocal();

    GPUCamera* d_cam;
    cudaMalloc(&d_cam, sizeof(GPUCamera));
    cudaMemcpy(d_cam, &gc, sizeof(GPUCamera), cudaMemcpyHostToDevice);
    return d_cam;
}