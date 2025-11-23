cmake_minimum_required(VERSION 3.5)

file(GLOB CPP_SOURCE 
    *.cpp
    *.cu
    core/*.cpp
    backend/*.cpp
    backend/cpu/*.cpp
    backend/cuda_megakernel/*.cpp
    backend/cuda_wavefront/*.cpp
    backend/optix/*.cpp
    backend/*.cu
    backend/cpu/*.cu
    backend/cuda_megakernel/*.cu
    backend/cuda_wavefront/*.cu
    backend/optix/*.cu
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
