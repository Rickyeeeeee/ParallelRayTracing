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
| 6 | Build handle-driven SOA packers for wavefront backend | In Progress (next) |
| 7 | Implement simple wavefront renderer using the new SOA data | Pending |
| 8 | Cleanup legacy virtual code and validate with renders | Pending |

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
  1. Reuse the primitive/material pools to fill `MaterialSOA`, `SphereSOA`, `QuadSOA` buffers based on handle tags.
  2. Ensure each SOA entry captures the correct attributes (albedo, emission, roughness, etc.).
  3. Upload SOA buffers during `CudaWavefrontRenderer::Init`, eliminating RTTI/dynamic_cast usage.
- Notes:
  - Wavefront can now piggyback on the megakernel uploader to obtain contiguous vectors per material/shape type before transposing into SOA form.

### 7. Simple Wavefront Renderer
- Files: `src/backend/cuda_wavefront/renderer.{h,cu}`.
- Actions:
  - Implement a minimal wavefront renderer that consumes the new SOA data end-to-end, validating the shared packing pipeline and setting the stage for future optimizations.

### 8. Cleanup & Validation
- Files: `src/core/material.h`, `src/core/shape.h`, `src/core/primitive.*`, renderers.
- Actions:
  - Delete obsolete virtual base classes and any `MatType` plumbing that the new handles supersede.
  - Run CPU/GPU regression renders (small scenes) to verify parity.
  - Update documentation (`README.md`) with the new architecture summary.

---

## Notes
- Keep `refactor.md` updated as each stage completes.
- Prefer incremental commits per stage to simplify review/rollback.
