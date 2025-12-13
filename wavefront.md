# CUDA Wavefront Renderer Performance Roadmap

This document tracks concrete optimization work for the CUDA wavefront backend (`src/backend/cuda_wavefront/*`), focused on throughput (samples/sec) and minimizing frame-to-frame overhead.

## Profiling Results (So Far)

- **Bottleneck**: `atomicAdd` contention on queue counters dominates runtime (queue allocation in `RayQueueSOA::AllocateSlot`).

## Optimization 1: Warp-Aggregated Queue Allocation

Replace per-thread queue counter increments with a warp-aggregated scheme:

- Use `__activemask()` and `__popc()` to compute how many lanes in the warp are enqueuing.
- One “leader” lane performs `atomicAdd(counter, warpCount)`.
- Broadcast the base index via `__shfl_sync`, and each lane computes a local rank to get its unique slot.

Status:
- Implemented for `RayQueueSOA::AllocateSlot` in `src/backend/cuda_wavefront/renderer.cu:58` (1 atomic per warp).

## Optimization 2: Reduce Global Memory Traffic (Shrink PixelState)

Move hit/miss-specific data out of `PixelStateSOA` into queue item SOAs to reduce per-pixel state size and avoid reading/writing large structures when only a subset of fields is needed.

Status:
- `HitQueueSOA` now stores hit items in SOA form (`pixelIndex`, hit position/normal, material handle, distance, front-face) and `EscapeQueueSOA` stores `pixelIndex` only (`src/backend/cuda_wavefront/renderer.h:48`).
- `PixelStateSOA` no longer stores hit-specific arrays; it keeps only per-path state (ray, throughput, radiance, depth, RNG, alive) (`src/backend/cuda_wavefront/renderer.h:26`).

## Goals & Metrics

- **Primary**: maximize samples/sec at fixed image quality.
- **Secondary**: reduce per-frame CPU↔GPU synchronization and allocations.
- **Measure**:
  - GPU time per frame and per bounce (Nsight Systems timeline).
  - Kernel time breakdown + achieved occupancy (Nsight Compute).
  - Rays processed per bounce; queue sizes over depth.
  - Global memory throughput and L2 hit rate.

## Current Hotspots (Expected)

- Per-bounce queue management using device memsets and any CPU-side reads of device counters.
- Intersection kernel doing a full primitive loop per ray (no acceleration structure).
- Multiple full-screen kernels per bounce (work-inefficient when queues shrink).
- Global memory traffic from large per-pixel state reads/writes each stage.

## Priority 0: Remove Frame Overhead (Quick Wins)

1. **No per-bounce device→host queue reads**
   - Keep the depth loop fully GPU-driven, or accept running fixed `maxDepth` iterations and let kernels early-out on empty queues (no host sync inside the bounce loop).
   - If early break is required, use a single device-side flag written by a tiny kernel and read once per frame (or use mapped/pinned memory with async copies).

2. **Use `cudaMemsetAsync` + a dedicated stream**
   - Reset queue counters with `cudaMemsetAsync` and overlap with other work where possible.

3. **Persistent allocations**
   - Ensure all per-frame allocations are removed (pixel state arrays, counters, queues should be allocated once and reused).

4. **CUDA Graphs for steady-state frames**
   - Capture the fixed launch sequence for a frame into a CUDA Graph and replay each frame to reduce CPU launch overhead.

## Priority 1: Make Work Proportional to Active Rays

1. **Launch kernels based on queue size**
   - Replace full-screen `shadeBlocks = blocks` launches with `blocks = ceil(queueCount / blockDim)` for hit/miss/shade stages.
   - Maintain separate counters for each queue and use those counts to size launches.

2. **Queue compaction**
   - Add a compact step so queues contain only active pixels; avoid iterating dead pixels.
   - For large queues, implement block-level compaction (prefix sums in shared memory) into a global queue.

3. **Split miss handling**
   - A miss queue should contain only misses; avoid scanning all pixels to find misses.

## Priority 2: Reduce Memory Traffic (SOA + Packing)

1. **Minimize PixelState writes**
   - Only write what the next stage needs. For example:
     - Intersection writes `{hitPos, hitNormal, matHandle, frontFace}`.
     - Shade writes `{rayOrigin, rayDir, throughput, depth, alive}`.

2. **Use `float3`/`half` where acceptable**
   - Consider storing throughput/radiance in `float3` (or `half` for some fields) to reduce bandwidth, keeping accumulation in `float`.

3. **Avoid repeated normalizations**
   - Normalize rays at generation + after scatter; avoid normalizing again in intersection unless required.

## Priority 3: Improve Intersection Throughput

1. **Acceleration structure**
   - Biggest win: add a simple BVH (even a basic SAH-less BVH or LBVH) to avoid O(numPrimitives) per ray.
   - If a BVH is too large a step, add a coarse uniform grid or bounding volume culling first.

2. **Shape bucketing / specialization**
   - Add a work queue of `{pixelIndex, primitiveIndex}` grouped by shape type, enabling specialized intersection kernels per shape (as suggested in `refactor.md`).

3. **Memory layout for primitives**
   - Ensure device primitive array is tightly packed and aligned; consider splitting transforms/material handles into SoA for better cache behavior.

## Priority 4: Path Tracing Quality/Speed Tradeoffs

1. **Russian roulette termination**
   - After a few bounces, probabilistically terminate low-throughput paths to reduce work with minimal bias.

2. **Clamp fireflies**
   - Optional: clamp throughput or radiance contributions to stabilize convergence and reduce variance spikes.

## Profiling Checklist

- Capture an Nsight Systems trace:
  - Confirm no CPU stalls in the bounce loop.
  - Confirm kernel launch cadence and stream usage.
- For the intersection and shade kernels (Nsight Compute):
  - Check occupancy limiters (registers, shared memory).
  - Check global load/store efficiency and branch divergence.
  - Check instruction mix (math vs memory bound).

## Suggested Next Changes (Concrete)

- Add queue-size based launches for hit/miss/shade kernels.
- Remove per-bounce CPU queue count reads; accept fixed-depth loop first.
- Add compaction so later bounces launch fewer threads.
