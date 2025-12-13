cmake_minimum_required(VERSION 3.5)

file(GLOB CPP_SOURCE 
    *.cpp
    *.cu
    core/*.cpp
    core/*.cu
    backend/*.cpp
    backend/cpu/*.cpp
    backend/cuda_megakernel/*.cpp
    backend/cuda_wavefront/*.cpp
    backend/optix/*.cpp
    backend/*.cu
    backend/cpu/*.cu
    backend/cuda_megakernel/*.cu
    backend/cuda_wavefront/*.cu
    # NOTE: backend/optix/*.cu is NOT included - OptiX device programs must be compiled as PTX only
    imgui_impl/*.cpp
    opengl/*.cpp
    )
    
file (GLOB HEADER_SOURCE 
    *.h
    core/*.h
    backend/*.h
    backend/cpu/*.h
    backend/cuda_megakernel/*.h
    backend/cuda_wavefront/*.h
    backend/optix/*.h
    imgui_impl/*.h
    opengl/*.h
)


add_executable(
    Viewer
    ${CPP_SOURCE}
    ${HEADER_SOURCE}
)

# Normalize path so it uses forward slashes (safe in C/C++ string literals)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" PROJECT_ROOT_DIR)

target_compile_definitions(Viewer PRIVATE
    PROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}"
    ASSETS_DIR="${PROJECT_ROOT_DIR}/assets"
    PTX_DIR="${CMAKE_BINARY_DIR}/src"
)

# set_target_properties(Viewer PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
# )

target_link_libraries(Viewer
    glfw
    glm::glm
    glad
    imgui
    tinyply
    CUDA::cuda_driver
    CUDA::cudart
)

target_include_directories(Viewer PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/external/glfw/include
    ${PROJECT_SOURCE_DIR}/external/glm
    ${PROJECT_SOURCE_DIR}/external/imgui
    ${PROJECT_SOURCE_DIR}/external/tinyply/source
    ${OPTIX_INCLUDE_DIR}
)

#------------------------------------------------------------------------------
# OptiX PTX Compilation
#------------------------------------------------------------------------------
find_program(CUDA_NVCC_EXECUTABLE nvcc)

if(CUDA_NVCC_EXECUTABLE)
    set(OPTIX_PTX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/backend/optix/device_programs.cu")
    set(OPTIX_PTX_OUTPUT "${CMAKE_BINARY_DIR}/src/device_programs.ptx")
    
    add_custom_command(
        OUTPUT ${OPTIX_PTX_OUTPUT}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            -ptx
            -O3
            --use_fast_math
            -arch=sm_89
            -I${OPTIX_INCLUDE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/backend/optix
            -I${CUDAToolkit_INCLUDE_DIRS}
            -o ${OPTIX_PTX_OUTPUT}
            ${OPTIX_PTX_SOURCE}
        DEPENDS ${OPTIX_PTX_SOURCE}
        COMMENT "Compiling OptiX PTX: device_programs.cu -> device_programs.ptx"
        VERBATIM
    )
    
    add_custom_target(OptiXPTX ALL DEPENDS ${OPTIX_PTX_OUTPUT})
    add_dependencies(Viewer OptiXPTX)
    
    message(STATUS "OptiX PTX compilation enabled: ${OPTIX_PTX_OUTPUT}")
endif()

