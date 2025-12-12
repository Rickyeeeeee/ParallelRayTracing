cmake_minimum_required(VERSION 3.20)

project(ParallelRayTracing
    VERSION 0.1
    LANGUAGES CXX CUDA)

# ---- Options ----
option(BUILD_RENDERER_CPU          "Build CPU std::threads renderer"        ON)
option(BUILD_RENDERER_CUDA_MEGA    "Build CUDA megakernel renderer"         ON)
option(BUILD_RENDERER_CUDA_WF      "Build CUDA wavefront renderer"          ON)
option(BUILD_RENDERER_OPTIX        "Build OptiX renderer"                   ON)
option(BUILD_VIEWER_APP            "Build OpenGL viewer application"        ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# For CUDA (adjust arch for your GPU)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)  # 89 for Ada; change according to your GPU

# Optional: treat warnings as errors during development
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- External dependencies ----
if (BUILD_RENDERER_CUDA_MEGA)
    find_package(CUDAToolkit REQUIRED)
endif()

# OpenGL + GLFW
if (BUILD_VIEWER_APP)
    find_package(OpenGL REQUIRED)
endif()

# If you vendor GLAD:
# add_subdirectory(extern/glad)

if (BUILD_RENDERER_OPTIX)
    # Let the user specify the OptiX SDK install dir
    set(OPTIX_ROOT "" CACHE PATH "Path to the OptiX SDK root directory (contains 'include')")

    if (NOT OPTIX_ROOT)
        message(FATAL_ERROR
            "OPTIX_ROOT is not set.\n"
            "Set it with -DOPTIX_ROOT=\"C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/SDK\" (example)."
        )
    endif()

    message(STATUS "Using OPTIX_ROOT = ${OPTIX_ROOT}")

    find_path(OPTIX_INCLUDE_DIR optix.h
        HINTS "${OPTIX_ROOT}/include")

    if (NOT OPTIX_INCLUDE_DIR)
        message(FATAL_ERROR
            "Could not find optix.h under OPTIX_ROOT = ${OPTIX_ROOT}\n"
            "Expected at: ${OPTIX_ROOT}/include/optix.h")
    endif()
endif()


# ---- Subdirectories ----

add_subdirectory(external)
add_subdirectory(src)

# if (BUILD_RENDERER_CPU)
#     add_subdirectory(src/renderers/cpu)
# endif()

# if (BUILD_RENDERER_CUDA_MEGA)
#     add_subdirectory(src/renderers/cuda_mega)
# endif()

# if (BUILD_RENDERER_CUDA_WF)
#     add_subdirectory(src/renderers/cuda_wavefront)
# endif()

# if (BUILD_RENDERER_OPTIX)
#     add_subdirectory(src/renderers/optix)
# endif()

# if (BUILD_VIEWER_APP)
#     add_subdirectory(src/app)
# endif()

# Fix IntelliSense in Visual Studio
if (MSVC)
    target_compile_options(Viewer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --extended-lambda
            -lineinfo
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            /EHsc
            /Zc:__cplusplus
        >
    )
endif()