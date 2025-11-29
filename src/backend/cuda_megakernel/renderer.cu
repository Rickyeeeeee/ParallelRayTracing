#include "backend/cuda_megakernel/renderer.h"
#include <vector>
#include <chrono>
#include <core/math.h>

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
QUAL_GPU inline void GPUEmit(GPUMaterial mat, glm::vec3& color) {
    color = (mat.type == MatType::EMISSIVE) ? mat.color : glm::vec3{ 0.0f };
}
QUAL_GPU inline bool GPUScatter(const GPUMaterial mat, const Ray& inRay, const SurfaceInteraction& intersection, glm::vec3& attenuation, Ray& outRay, curandState* rngState) {
    if (mat.type == MatType::LAMBERTIAN) {
        glm::vec3 scatterDirection = intersection.Normal + RandomUnitVector(rngState);

        auto& e = scatterDirection;
        auto s = 1e-8;
        if ((glm::abs(e[0]) < s) && (glm::abs(e[1]) < s) && (glm::abs(e[2]) < s))
        {
            scatterDirection = intersection.Normal;
        }

        outRay.Origin = intersection.Position;
        outRay.Direction = glm::normalize(scatterDirection);

        attenuation = mat.color;
    }
    else if (mat.type == MatType::METAL) {
        auto reflectedDir = glm::reflect(inRay.Direction, intersection.Normal);
        reflectedDir = glm::normalize(reflectedDir) + mat.roughness * RandomUnitVector(rngState);
        // reflectedDir = glm::normalize(reflectedDir);

        outRay.Origin = intersection.Position;
        outRay.Direction = glm::normalize(reflectedDir);

        attenuation = mat.color;

        return glm::dot(outRay.Direction, intersection.Normal) > 0.0f;
    }
    else if (mat.type == MatType::DIELECTRIC) {
        auto fresnelReflectance = [](float cosine, float refractionIndex) {
            // Use Schlick's approximation for reflectance.
            auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
            r0 = r0*r0;
            return r0 + (1-r0)*glm::pow((1 - cosine),5);
        };

        attenuation = glm::vec3 { 1.0f, 1.0f, 1.0f };
        float ri = intersection.IsFrontFace ? (1.0f / mat.refractionIndex) : mat.refractionIndex;

        glm::vec3 unit_direction = inRay.Direction;
        float cos_theta = glm::min(glm::dot(-unit_direction, intersection.Normal), 1.0f);
        float sin_theta = glm::sqrt(1.0f - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        glm::vec3 direction;

        if (cannot_refract || fresnelReflectance(cos_theta, ri) > Random(rngState))
            direction = glm::reflect(unit_direction, intersection.Normal);
        else
            direction = Reflect(unit_direction, intersection.Normal, ri);

        outRay = Ray{ intersection.Position, direction };
        return true;
    }
    else {
        return false;
    }
    return true;
}

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

QUAL_GPU GPUMaterial IntersectSceneGPU(
    const Ray& ray,
    const GPUSphere* spheres, int numSpheres,
    const GPUQuad*   quads,   int numQuads,
    SurfaceInteraction* outSi)
{
    float minDistance2 = 1e30f;
    GPUMaterial hit = {MatType::NONE};

    SurfaceInteraction si;

    // --- Spheres ---
    for (int i = 0; i < numSpheres; ++i) {
        IntersectGPUSphere(spheres[i], ray, &si);
        if (si.HasIntersection) {
            glm::vec3 d = ray.Origin - si.Position;
            float dist2 = glm::dot(d, d);
            if (dist2 < minDistance2) {
                minDistance2 = dist2;
                *outSi = si;
                hit = spheres[i].matType;
            }
        }
    }

    // --- Quads ---
    for (int i = 0; i < numQuads; ++i) {
        IntersectGPUQuad(quads[i], ray, &si);
        if (si.HasIntersection) {
            glm::vec3 d = ray.Origin - si.Position;
            float dist2 = glm::dot(d, d);
            if (dist2 < minDistance2) {
                minDistance2 = dist2;
                *outSi = si;
                hit = quads[i].matType;
            }
        }
    }

    if (hit.type == MatType::NONE) {
        outSi->HasIntersection = false;
    }

    return hit;
}

QUAL_GPU glm::vec3 TraceRayGPU(
    Ray ray,
    const GPUSphere* spheres, int numSpheres,
    const GPUQuad*   quads,   int numQuads,
    const GPUMaterial* materials,
    glm::vec3 skyLight,
    curandState* rngState,
    int maxDepth = 20)
{
    glm::vec3 L(0.0f);          // accumulated radiance
    glm::vec3 throughput(1.0f); // path throughput

    for (int depth = 0; depth < maxDepth; ++depth) {
        SurfaceInteraction si;
        GPUMaterial hit = IntersectSceneGPU(ray, spheres, numSpheres, quads, numQuads, &si);

        if (hit.type == MatType::NONE || !si.HasIntersection) {
            L += throughput * skyLight;
            break;
        }

        // Emission
        glm::vec3 emitted(0.0f);
        GPUEmit(hit, emitted);
        L += throughput * emitted;

        // Scatter
        glm::vec3 attenuation(0.0f);
        Ray scattered;
        if (!GPUScatter(hit, ray, si, attenuation, scattered, rngState)) {
            // Absorbed or no further bounce
            break;
        }

        // Update throughput and continue with new ray
        throughput *= attenuation;
        ray = scattered;
        // normalize direction if needed:
        ray.Direction = glm::normalize(ray.Direction);
    }

    return L;
}

__global__ void GPU_RayTracing(float* colors, Camera * cam, GPUSphere* spheres, GPUQuad* quads, unsigned long long timeSeed)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixelCount = (uint32_t)cam->GetWidth() * (uint32_t)cam->GetHeight();
    if (idx >= pixelCount)
        return;

    const uint32_t x = idx % (uint32_t)cam->GetWidth();
    const uint32_t y = idx / cam->GetWidth();

    Ray ray = cam->GetCameraRay((float)x + 0.5f, (float)y + 0.5f);
    
    // initialize per-thread RNG state
    curandState localState;
    unsigned long long seed = timeSeed ^ (0x9E3779B97F4A7C15ULL + (unsigned long long)idx * 0xBF58476D1CE4E5B9ULL);
    curand_init(
        seed,
        (unsigned long long)idx,
        0,
        &localState
    );

    // Trace Ray
    glm::vec3 raycolor = TraceRayGPU(
        ray,
        spheres, 4,
        quads, 3,
        nullptr,
        glm::vec3(0.4f, 0.3f, 0.6f),
        &localState
    );


    // Method 1: GLM Normalize
    const uint32_t base = idx * 3;
    colors[base + 0] = raycolor.r;
    colors[base + 1] = raycolor.g;
    colors[base + 2] = raycolor.b;
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
    // Random Seed
    uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());


    GPU_RayTracing<<<blocks, threadsPerBlock>>>(deviceBuffer, d_cam, spheres, quads, seed);
    cudaDeviceSynchronize();
    
    
    // Render
    std::vector<float> hostBuffer(static_cast<size_t>(pixelCount) * 3);
    cudaMemcpy(hostBuffer.data(), deviceBuffer, bufferSize, cudaMemcpyDeviceToHost);

    m_Film->AddSampleBuffer(hostBuffer.data());

    cudaFree(deviceBuffer);
    cudaFree(d_cam);
    cudaFree(spheres);
}

GPUMaterial createGPUMaterial(const std::shared_ptr<SimplePrimitive> primitive) {
    MatType type = primitive->GetType();
    GPUMaterial gmat;

    if (type == MatType::LAMBERTIAN) {
        auto& mat = dynamic_cast<LambertianMaterial&>(primitive->GetMaterial());
        gmat.type = MatType::LAMBERTIAN;
        gmat.color = mat.GetAlbedo();
    }
    else if (type == MatType::METAL) {
        auto& mat = dynamic_cast<MetalMaterial&>(primitive->GetMaterial());
        gmat.type = MatType::METAL;
        gmat.color = mat.GetAlbedo();
        gmat.roughness = mat.GetRoughness();
    }
    else if (type == MatType::DIELECTRIC) {
        auto& mat = dynamic_cast<DielectricMaterial&>(primitive->GetMaterial());
        gmat.type = MatType::DIELECTRIC;
        gmat.refractionIndex = mat.GetRefractionIndex();
    }
    else if (type == MatType::EMISSIVE) {
        auto& mat = dynamic_cast<EmissiveMaterial&>(primitive->GetMaterial());
        gmat.type = MatType::EMISSIVE;
        gmat.color = mat.GetEmission();
    }
    else {
        gmat.type = MatType::NONE;
    }

    return gmat;
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
        gs.matType = createGPUMaterial(_c);

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
        gq.matType = createGPUMaterial(_q);

        gq_list.push_back(gq);
    }

    GPUQuad* gq_ptr;
    size_t size = sizeof(GPUQuad) * gq_list.size();
    cudaMalloc(&gq_ptr, size);
    cudaMemcpy(gq_ptr, gq_list.data(), size, cudaMemcpyHostToDevice);

    return gq_ptr;
}