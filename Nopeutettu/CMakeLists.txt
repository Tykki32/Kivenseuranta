cmake_minimum_required(VERSION 3.20)

project(mode_engine LANGUAGES CXX)
set(CMAKE_TOOLCHAIN_FILE
    "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING ""
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# K‰ytet‰‰n CMake:n uutta Python-tunnistusta.
set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(pybind11_DIR "C:/Python312/Lib/site-packages/pybind11/share/cmake/pybind11")

find_package(pybind11 CONFIG REQUIRED)
set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)

pybind11_add_module(
    mode_engine
    mode_engine.cpp
)

target_link_libraries(
    mode_engine
    PRIVATE
    ${OpenCV_LIBS}
)

target_include_directories(
    mode_engine
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

if(MSVC)
    target_compile_options(
        mode_engine
        PRIVATE
        /O2
        /EHsc
    )
endif()

# Laitetaan valmis .pyd suoraan projektikansioon.
foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)

    string(
        TOUPPER
        "${CONFIG}"
        CONFIG_UPPER
    )

    set_target_properties(
        mode_engine
        PROPERTIES

        RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}
        "${CMAKE_SOURCE_DIR}"

        LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER}
        "${CMAKE_SOURCE_DIR}"
    )

endforeach()