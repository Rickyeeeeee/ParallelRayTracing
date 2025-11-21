#Build Instruction
```bash
git clone --recursive https://github.com/Rickyeeeeee/ParallelRayTracing.git
cd ParallelRayTracing
cmake -B build -DOPTIX_ROOT="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.1.0"
cmake --build build
./bin/Debug/viewer.exe
```
