cmake_minimum_required(VERSION 3.14)
project(letter_box_neon)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
pybind11_add_module(letter_box_neon letter_box.cpp)

# 处理不同架构的编译选项
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build architectures")
    
    target_compile_options(letter_box_neon PRIVATE
        $<$<CONFIG:Release>:-O3>
        -mcpu=apple-a14
        -mtune=apple-a14
        -fPIC
        -Wno-unused-variable
        -Wno-unused-function
    )
else()
    target_compile_options(letter_box_neon PRIVATE
        $<$<CONFIG:Release>:-O3>
        -march=armv8-a+simd
        -mtune=cortex-a76
        -fPIC
        -Wno-unused-variable
        -Wno-unused-function
    )
endif()

target_include_directories(letter_box_neon PRIVATE
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

set_target_properties(letter_box_neon PROPERTIES 
    PREFIX ""
    SUFFIX ".so"
)

if(APPLE)
    set_target_properties(letter_box_neon PROPERTIES 
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()