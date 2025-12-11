# Renderer Core Handle Refactor

## Stage Tracker
| Stage | Description | Status |
| --- | --- | --- |
| 0 | Document scope and plan in `refactor.md` | ✅ |
| 1 | Introduce tagged handle infrastructure for shapes/materials (headers, helpers, tests) | ✅ |
| 2 | Use handles on CPU primitives & renderer (SimplePrimitive, TriangleList, CPURenderer) | ✅ |
| 3 | Refactor `Scene`/`PrimitiveList` to expose handle-based primitive views (ownership unchanged) | ✅ |
| 4 | Integrate handles into CUDA megakernel (host/device data upload) | ✅ |
| 5 | Build handle-driven SOA packers for wavefront backend | ☐ |
| 6 | Implement simple wavefront renderer using the new SOA data | ☐ |
| 7 | Replace CPU ownership with POD pools (Scene-level storage) | ☐ |
| 8 | Cleanup legacy virtual code and validate with renders | ☐ |

---

## Detailed Plan

### 1. Handle Infrastructure
- Files: `src/core/material.h`, `src/core/shape.h`, new helper header (`src/core/handles.h`).
- Design (decided 2025-12-11):
  - Handles are tagged pointers: `{ uint8_t type; void* ptr; }`. `type` indexes a compile-time type list (`MaterialTypes`, `ShapeTypes`), and `ptr` references CPU storage (host address space only).
  - Add a templated `dispatch(handle, lambda)` helper that matches the tag, casts the pointer, and invokes `lambda(static_cast<T*>(handle.ptr))`. Provide both host/device qualifiers so GPU copies can reuse it after upload.
  - `SurfaceInteraction` uses `MaterialHandle` instead of raw `Material*`; primitives store `ShapeHandle` + `MaterialHandle`.
- Actions:
  - Implement `handles.h` with tag enums, helper templates, and compile-time assertions.
  - Convert material/shape classes into POD-like structs with `QUAL_CPU_GPU` methods or free utility functions.
  - Update `SurfaceInteraction` definition accordingly.

### 2. CPU Renderer & Primitive Usage
- Files: `src/core/primitive.cpp`, `src/backend/cpu/renderer.cpp`.
- Status: ✅ (handles used for intersections and shading).
- Actions done:
  - `SimplePrimitive` and `TriangleList` refresh handles when constructed.
  - `SimplePrimitive::Intersect` writes handles into `SurfaceInteraction`.
  - `CPURenderer::TraceRay` emits/scatters via `MaterialHandle::dispatch`.

### 3. Scene & PrimitiveList Refactor (ownership deferred)
- Files: `src/core/scene.h`, `src/core/primitive.h/.cpp`.
- Status: ✅ (primitive handle views exposed; ownership unchanged).
- Summary:
  - `SimplePrimitive` now yields `PrimitiveHandleView` snapshots.
  - `PrimitiveList` caches circle/quad views and exposes them via getters.
  - `Scene` forwards both shared_ptr lists and handle views to downstream consumers.

### 4. CUDA Megakernel Integration
- Files: `src/backend/cuda_megakernel/renderer.{h,cu}`, `src/core/scene.h`, `src/core/primitive.*`.
- Status: ✅
- Summary:
  - Added `GPUPrimitive`/`GPUMaterial` POD structs plus a buffer upload path fed by `PrimitiveHandleView` data from `Scene`.
  - Removed legacy `GPUSphere`/`GPUQuad` structs and all `dynamic_cast`-based conversion helpers.
  - Kernel now consumes a single primitive buffer, branching on `ShapeType` to intersect circles/quads, fully driven by handle data.

### 5. CUDA Wavefront SOA Pack
- Files: `src/backend/cuda_wavefront/*`.
- Actions:
  1. Reuse the primitive/material views to fill `MaterialSOA`, `SphereSOA`, `QuadSOA` buffers based on handle tags.
  2. Ensure each SOA entry captures the correct attributes (albedo, emission, roughness, etc.).
  3. Upload SOA buffers during `CudaWavefrontRenderer::Init`, eliminating RTTI/dynamic_cast usage.

### 6. Simple Wavefront Renderer
- Files: `src/backend/cuda_wavefront/renderer.{h,cu}`.
- Actions:
  - Implement a minimal wavefront renderer that consumes the new SOA data end-to-end, validating the shared packing pipeline and setting the stage for future optimizations.

### 7. CPU Ownership Pools (postpone until handles work everywhere)
- Files: `src/core/scene.*`, new pool helpers.
- Actions:
  - Replace `shared_ptr` graphs with contiguous POD pools (per-type vectors or arena) to avoid heap fragmentation and make GPU uploads trivial.
  - Manage lifetime at `Scene` scope; expose spans to renderers.
  - Ensure transforms/material references remain valid across uploads.

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
