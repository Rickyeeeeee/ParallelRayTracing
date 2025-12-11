# Renderer Core Handle Refactor

## Stage Tracker
| Stage | Description | Status |
| --- | --- | --- |
| 0 | Document scope and plan in `refactor.md` | ✅ |
| 1 | Introduce tagged handle infrastructure for shapes/materials (headers, helpers, tests) | ✅ |
| 2 | Migrate CPU-side owners (`SimplePrimitive`, `PrimitiveList`, `TriangleList`, `Scene`) to handles | ☐ |
| 3 | Update CPU renderer path (`CPURenderer::TraceRay`) to dispatch via handles | ☐ |
| 4 | Port CUDA backends to share the new data layout (`cuda_wavefront`, `cuda_megakernel`) | ☐ |
| 5 | Cleanup legacy virtual code and validate (render tests) | ☐ |

---

## Detailed Plan

### 1. Handle Infrastructure
- Files: `src/core/material.h`, `src/core/shape.h`, new helper header (`src/core/handle.h`?).
- Design (decided 2025-12-11):
  - Handles are tagged pointers: `{ uint8_t type; void* ptr; }`. `type` indexes a compile-time type list (`MaterialTypes`, `ShapeTypes`), and `ptr` references CPU storage (host address space only).
  - Add a templated `dispatch(handle, lambda)` helper that matches the tag, casts the pointer, and invokes `lambda(static_cast<T*>(handle.ptr))`. Provide both host/device qualifiers so GPU copies can reuse it after upload.
  - `SurfaceInteraction` uses `MaterialHandle` instead of raw `Material*`; primitives store `ShapeHandle` + `MaterialHandle`.
- Actions:
  - Implement `handle.h` with tag enums, helper templates, and compile-time assertions.
  - Convert material/shape classes into POD structs with `QUAL_CPU_GPU` methods or free utility functions.
  - Update `SurfaceInteraction` definition accordingly.

### 2. CPU Renderer & Primitive Usage
- Files: `src/core/primitive.cpp`, `src/backend/cpu/renderer.cpp`.
- Status: ✅ (handles used for intersections and shading).
- Actions done:
  - `SimplePrimitive` and `TriangleList` refresh handles when constructed.
  - `SimplePrimitive::Intersect` writes handles into `SurfaceInteraction`.
  - `CPURenderer::TraceRay` emits/scatters via `MaterialHandle::dispatch`.

### 3. Scene & PrimitiveList Refactor (defer ownership swap)
- Files: `src/core/scene.h`, `src/core/primitive.h/.cpp`.
- Actions (next):
  - Keep `std::shared_ptr` ownership for now but remove other virtual dependencies by iterating over concrete `SimplePrimitive` lists.
  - Ensure `Scene::Intersect` / `PrimitiveList::Intersect` rely on handles only (no `dynamic_cast`).
  - Provide lightweight views over circles/quads so GPU backends can pull data without re-casting.

### 4. CUDA Backends
- Files: `src/backend/cuda_megakernel/*`, `src/backend/cuda_wavefront/*`.
- Actions:
  - Remove duplicated `GPUMaterial`, `GPUSphere`, `GPUQuad` structs; instead upload the shared shape/material storage.
  - Replace `dynamic_cast` usage in conversion helpers with handle-aware code.
  - Ensure handle dispatch works with device pointers (may require unified memory or explicit GPU allocations for pools).

### 5. CPU Ownership Pools (postpone until handles work everywhere)
- Files: `src/core/scene.*`, new pool helpers.
- Actions:
  - Replace `shared_ptr` graphs with contiguous POD pools (per-type vectors or arena) to avoid heap fragmentation and to make GPU uploads trivial.
  - Manage lifetime at `Scene` scope; expose spans to renderers.
  - Ensure transforms/material references remain valid across uploads.

### 6. Cleanup & Validation
- Files: `src/core/material.h`, `src/core/shape.h`, `src/core/primitive.*`, renderers.
- Actions:
  - Delete obsolete virtual base classes and any `MatType` plumbing that the new handles supersede.
  - Run CPU/GPU regression renders (small scenes) to verify parity.
  - Update documentation (`README.md`) with the new architecture summary.

---

## Notes
- Keep `refactor.md` updated as each stage completes.
- Prefer incremental commits per stage to simplify review/rollback.
