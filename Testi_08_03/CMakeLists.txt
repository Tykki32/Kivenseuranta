cmake_minimum_required(VERSION 3.20)

# C++-laajennusmoduulit mode_engine ja stone_tracker (lahdekoodi kansiossa cpp/). Valmiit .pyd-tiedostot
# kopioidaan projektikansioon (main.py:n viereen).
#
# Windows (vcpkg + Python 3.12):
#   cmake -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
# Linux (testaus):
#   g++ -O2 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) $(pkg-config --cflags opencv4) \
#       cpp/stone_tracker.cpp -o stone_tracker$(python3-config --extension-suffix) $(pkg-config --libs opencv4) -lpthread -ldl
#   (mode_engine samoin: cpp/mode_engine.cpp)

# vcpkg:n toolchain on asetettava ENNEN project()-kutsua (muuten mm. OpenCV:n riippuvuudet eivat loydy)
set(CMAKE_TOOLCHAIN_FILE
    "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING ""
)

project(kivenseuranta LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sama Python, jolla main.py ajetaan (muuten esim. Anacondan Python voi loytya PATH:sta ensin)
set(PYBIND11_FINDPYTHON ON)
set(Python_EXECUTABLE "C:/Python312/python.exe")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(pybind11_DIR "C:/Python312/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)

pybind11_add_module(mode_engine cpp/mode_engine.cpp)
pybind11_add_module(stone_tracker cpp/stone_tracker.cpp)

foreach(TARGET_NAME mode_engine stone_tracker)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBS})
    target_include_directories(${TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /O2 /EHsc)
    endif()
    foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${CONFIG}" CONFIG_UPPER)
        set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_SOURCE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_SOURCE_DIR}")
    endforeach()
endforeach()
