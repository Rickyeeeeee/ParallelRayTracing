#include "backend/cuda_megakernel/renderer.h"
#include <vector>

/*
TODO

- Optimize the cam buffer (call once in init(), and don't need to malloc and delete the buffer)
- Optimize camera in camera.h for GPU compatibility
- remove the hardcoded depth
- Optimize the Intersect Function
    - Not using CPU since too many vector and virtual, 
      waiting for other implementation before rebase
    - tMin hardcoded
*/

namespace
{
QUAL_GPU inline void IntersectGPUSphere(const GPUSphere& sphere, const Ray &pray, SurfaceInteraction* intersect) {
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
        float t1 = (-b + sqrtf(discriminant)) / 2 * a;
        float t2 = (-b - sqrtf(discriminant)) / 2 * a;
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

    intersect->Position = TransformPoint(sphere.transform.GetMat(), intersect->Position);
    intersect->Normal = TransformNormal(sphere.transform.GetInvMat(), intersect->Normal);
}

QUAL_GPU inline void IntersectGPUQuad(const GPUQuad& quad, const Ray &pray, SurfaceInteraction* intersect) {
    Ray ray;
    ray.Origin = TransformPoint(quad.transform.GetInvMat(), pray.Origin);
    ray.Direction = TransformNormal(quad.transform.GetMat(), pray.Direction);

    if (fabs(ray.Direction.y) < 1e-8f)
    {
        intersect->HasIntersection = false;
        return;
    }

    auto t = -ray.Origin.y / ray.Direction.y;

    auto p = ray.Origin + ray.Direction * t;

    float halfWidth = quad.width / 2.0f;
    float halfHeight = quad.height / 2.0f;

    constexpr static float tMin = 0.001f;

    if (t > tMin && (p.x * p.x < halfWidth * halfWidth) && (p.z * p.z < halfHeight * halfHeight))
    {
        intersect->HasIntersection = true;
        intersect->Position = p;
        intersect->IsFrontFace = ray.Origin.y > 0.0f;
        intersect->Normal = intersect->IsFrontFace ? quad.normal : -quad.normal;
    }
    else
    {
        intersect->HasIntersection = false;
    }
    
    intersect->Position = TransformPoint(quad.transform.GetMat(), intersect->Position);
    intersect->Normal = TransformNormal(quad.transform.GetInvMat(), intersect->Normal);
}

__global__ void GPU_RayTracing(float* colors, Camera * cam, GPUSphere* spheres, GPUQuad* quads)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixelCount = (uint32_t)cam->GetWidth() * (uint32_t)cam->GetHeight();
    if (idx >= pixelCount)
        return;

    const uint32_t x = idx % (uint32_t)cam->GetWidth();
    const uint32_t y = idx / cam->GetWidth();

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
        IntersectGPUSphere(spheres[i], ray, &intersect);
        if (intersect.HasIntersection) {
            colora = 255.0f;
        }
    }
    for (int i=0; i<3; i++) {
        SurfaceInteraction intersect;
        IntersectGPUQuad(quads[i], ray, &intersect);
        if (intersect.HasIntersection) {
            colorb = 255.0f;
        }
    }

    // Method 1: GLM Normalize
    const uint32_t base = idx * 3;
    colors[base + 0] = colora;
    colors[base + 1] = colorb;
    colors[base + 2] = 0.0f;
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

    // Camera
    Camera* d_cam;
    cudaMalloc(&d_cam, sizeof(Camera));
    cudaMemcpy(d_cam, m_Camera, sizeof(Camera), cudaMemcpyHostToDevice);
    // Objects
    GPUSphere* spheres = ConvertCirclesToGPU();
    GPUQuad* quads = ConvertQuadsToGPU();

    GPU_RayTracing<<<blocks, threadsPerBlock>>>(deviceBuffer, d_cam, spheres, quads);
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

GPUQuad* CudaMegakernelRenderer::ConvertQuadsToGPU() {
    std::vector<GPUQuad> gq_list;

    const auto qlist = m_Scene->getQuads();
    for (auto quad: qlist) {
        auto _q = std::static_pointer_cast<SimplePrimitive>(quad);
        Shape& _s = _q->GetShape();
        Quad& q = dynamic_cast<Quad&>(_s);

        GPUQuad gq;
        gq.transform = _q->GetTransform();
        gq.width = q.GetWidth();
        gq.height = q.GetHeight();
        gq.normal = q.GetNormal();

        gq_list.push_back(gq);
    }

    GPUQuad* gq_ptr;
    size_t size = sizeof(GPUQuad) * gq_list.size();
    cudaMalloc(&gq_ptr, size);
    cudaMemcpy(gq_ptr, gq_list.data(), size, cudaMemcpyHostToDevice);

    return gq_ptr;
}