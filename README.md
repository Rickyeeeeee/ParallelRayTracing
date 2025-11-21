# Build Instructions (only test on Windows)
- Install Nvidia CUDA Toolkit (Only tested with 12.4).
- Install Nvidia Optix SDK (Only tested with 8.1.0).
- Install cmake.
- Install git.
- Install Visual Studio 2022.
```bash
git clone --recursive https://github.com/Rickyeeeeee/ParallelRayTracing.git
cd ParallelRayTracing
cmake -B build -DOPTIX_ROOT="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.1.0"
cmake --build build
./bin/Debug/viewer.exe
```
