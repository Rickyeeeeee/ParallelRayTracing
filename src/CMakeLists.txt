cmake_minimum_required(VERSION 3.5)

file(GLOB CPP_SOURCE 
    *.cpp
    # core/*.cpp
    # RayTracing/*.cpp
    # RasterEngine/*.cpp
)

file (GLOB HEADER_SOURCE 
    *.h
    # core/*.h
    # RayTracing/*.h
    # RasterEngine/*.h
)

add_executable(
    Viewer
    ${CPP_SOURCE}
    ${HEADER_SOURCE}
)
set_target_properties(Viewer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

target_link_libraries(Viewer
    # glfw
    # glm::glm
    # imgui
)

target_include_directories(Viewer PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    # ${PROJECT_SOURCE_DIR}/external/glfw/include
    # ${PROJECT_SOURCE_DIR}/external/nvrhi/include
    # ${PROJECT_SOURCE_DIR}/external/glm
    # ${PROJECT_SOURCE_DIR}/external/imgui
    # ${PROJECT_SOURCE_DIR}/external/tinyply/source
)