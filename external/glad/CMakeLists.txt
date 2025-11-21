cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME_VAR glad)
project(${PROJECT_NAME_VAR} VERSION 1.0.0 LANGUAGES C CXX)
# C++ Version

add_library(${PROJECT_NAME_VAR} src/glad.c)

target_include_directories(${PROJECT_NAME_VAR}
    PUBLIC include)