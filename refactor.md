# Renderer Core Handle Refactor

## Stage Tracker
| Stage | Description | Status |
| --- | --- | --- |
| 0 | Document scope and plan in `refactor.md` | Done |
| 1 | Introduce tagged handle infrastructure for shapes/materials (headers, helpers, tests) | Done |
| 2 | Use handles on CPU primitives & renderer (SimplePrimitive, TriangleList, CPURenderer) | Done |
| 3 | Refactor `Scene`/`PrimitiveList` to expose handle-based primitive views (ownership unchanged) | Done |
| 4 | Integrate handles into CUDA megakernel (host/device data upload) | Done |
| 5 | Replace CPU ownership with POD pools & remove virtual inheritance | Done |
| 6 | Build handle-driven SOA packers for wavefront backend | Done |
| 7 | Cleanup & Validation | Pending |

---

## Detailed Plan

### 1. Handle Infrastructure (Done)
- Files: `src/core/material.h`, `src/core/shape.h`, `src/core/handles.h`.
- Implemented tagged pointer handles (`{ uint8_t type; void* ptr; }`) with a constexpr type index for materials and shapes plus templated `dispatch` helpers usable on host/device builds.
- `SurfaceInteraction` and primitive records now hold `MaterialHandle`/`ShapeHandle`, removing raw pointer/virtual dependencies.
- Follow-up considerations: extend the type lists if/when new material/shape structs (e.g., textured or mesh-backed) are introduced.

### 2. CPU Renderer & Primitive Usage (Done)
- Files: `src/core/primitive.cpp`, `src/backend/cpu/renderer.cpp`.
- Primitive traversal now emits `PrimitiveHandleView` snapshots whose `ShapeHandle`/`MaterialHandle` populate `SurfaceInteraction`, so the CPU intersection path has zero virtual dispatch.
- `CPURenderer::TraceRay` shades solely through `MaterialHandle::dispatch`, preserving the previous behaviour.

### 3. Scene & PrimitiveList Refactor (Done)
- Files: `src/core/scene.h`, `src/core/primitive.h/.cpp`.
- `PrimitiveList` now caches raw `PrimitiveHandleView` vectors (plus circle/quad subsets) and intersects rays by dispatching `ShapeHandle` instead of touching any shared ownership.
- `Scene` wires the pools into `PrimitiveList`, exposes `getCircleViews()` / `getQuadViews()` snapshots, and forwards handle-based intersections to both CPU and GPU backends.

### 4. CUDA Megakernel Integration (Done)
- Files: `src/backend/cuda_megakernel/renderer.{h,cu}`, `src/core/scene.h`, `src/core/primitive.*`.
- During `ProgressiveRender` the renderer copies the core `MaterialPool`/`ShapePool` contents to device memory, remaps every `PrimitiveHandleView` to point at those device buffers, and uploads the remapped views as-is.
- The CUDA kernel now iterates the shared `PrimitiveHandleView` array, dispatches `ShapeHandle` to intersect circles/quads, and shades via `MaterialHandle::dispatch` calling the exact CPU material structs (GPU-specific structs were deleted).
- `Circle`/`Quad`/`Triangle` intersectors were moved inline with `QUAL_CPU_GPU` so both CPU and CUDA reuse the identical implementation (no GPU-only math paths remain).
- Scene pools remain private; the upload code just walks `Scene::getPrimitiveViews()`, remaps unique material/shape handles, and mirrors only the referenced structs.

### 5. CPU Ownership Pools & Virtual Removal (Done)
- Files: `src/core/scene.*`, `src/core/primitive.*`, `src/core/material.h`, `src/core/shape.h`.
- Introduced `MaterialPool`/`ShapePool` (per-type `std::deque`s) plus simple factory helpers that return stable handles for every material/shape instance.
- `Scene` now owns every primitive via these pools, builds `PrimitiveHandleView` records (shape handle + material handle + transform), and `PrimitiveList` intersects directly against those handles.
- All virtual base classes and `std::shared_ptr` ownership for primitives/materials were removed; renderers and GPU upload paths only deal with handle views.
- Scene no longer exposes bespoke material dispatchers; renderers access the shared `MaterialPool` directly, keeping the handle logic in one place.

### 6. CUDA Wavefront SOA Pack
- Files: `src/backend/cuda_wavefront/*`.
- Actions:
  1. Mirror the scene primitives to device memory (remapping every material/shape handle like the megakernel uploader) so kernels can call the shared `ShapeHandle::Intersect` / `MaterialHandle::dispatch` paths.
  2. Keep wavefront-only data (`PixelState` arrays, ray/hit/escape queues) in SOA form for coalesced access while avoiding duplicate scene-specific SOAs.
  3. Upload the contiguous `Primitive` array during `CudaWavefrontRenderer::Init` and reuse it for all kernel launches.
- Notes:
  - Wavefront now iterates the same primitive data the megakernel uses, so intersection/shading logic remains unified across backends.
  - Planned kernels for wavefront (minimal set):
    - `GenerateCameraRays` ??fill ray/state queues for the first bounce.
    - `IntersectPrimitives` ??traverse the device primitive array, write hit info + material handles.
    - `ShadeScatter` ??read hits, fetch material data from `MaterialHandle`, spawn next-bounce rays and accumulate emission/throughput.
    - `AccumulateFrame` ??write accumulated radiance to the film buffer.
- Status:
  - `CudaWavefrontRenderer::Init` now calls `BuildWavefrontSceneBuffers`, which uploads the primitive array plus remapped handles so device data always mirrors the CPU scene.

#### Wavefront PixelState & Queue Plan
- **PixelState layout**: POD struct per pixel with `Ray ray`, `glm::vec3 throughput`, `glm::vec3 radiance`, `glm::vec3 pendingEmission`, `glm::vec3 hitPosition`, `glm::vec3 hitNormal`, `uint32_t pixelIndex`, `uint32_t depth`, RNG state, `MaterialHandle material`, and `bool alive` so kernels only exchange pixel indices.
- **Device queues**:
  - `RayQueue`: pixel indices whose `PixelState.ray` must be intersected (seeded by `GenerateCameraRays`, appended in `ShadeScatter`).
  - `PrimitiveWorkQueue`: `{ pixelIndex, primitiveIndex }` entries emitted after bucketing so shape-specialized kernels can stay coherent (future optimization).
  - `HitQueue`: pixel indices that hit; shading kernels fetch the associated `PixelState` to continue the path.
  - `AccumulatorQueue`: pixel indices that missed or terminated; `AccumulateFrame` drains it and writes `PixelState.radiance` into the film buffer.
- **Kernel flow**:
  1. `GenerateCameraRays` resets every `PixelState` (throughput=1, radiance=0, depth=0), writes primary rays, pushes all indices into `RayQueue`.
  2. `RayQueueCompact` (future) drains `RayQueue`, bins work per primitive/shape and fills `PrimitiveWorkQueue` to keep intersection kernels coherent.
  3. `IntersectPrimitives` reads the primitive array, updates best hits in `HitQueue`, and pushes misses into `AccumulatorQueue`.
  4. `ShadeScatter` drains `HitQueue`, uses `MaterialHandle` to accumulate emission (`radiance += throughput * emitted`), updates throughput/depth, and either re-enqueues the pixel into `RayQueue` (new ray) or into `AccumulatorQueue` (finished).
  5. `AccumulateFrame` drains `AccumulatorQueue`, writes radiance to the film accumulation buffer, and optionally resets completed PixelStates for the next frame.
- **Lifetime**: Device primitives only change when `Init` runs; `PixelState`/queues persist across progressive frames so RNG seeds and accumulated radiance remain stable.

### 7. Cleanup & Validation
- Files: `src/core/material.h`, `src/core/shape.h`, `src/core/primitive.*`, renderers.
- Actions:
  - Delete obsolete virtual base classes and any `MatType` plumbing that the new handles supersede.
  - Run CPU/GPU regression renders (small scenes) to verify parity.
  - Update documentation (`README.md`) with the new architecture summary.

---

## Notes
- Keep `refactor.md` updated as each stage completes.
- Prefer incremental commits per stage to simplify review/rollback.
