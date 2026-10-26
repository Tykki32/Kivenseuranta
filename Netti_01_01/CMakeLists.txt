cmake_minimum_required(VERSION 3.20)

# C++-laajennusmoduulit mode_engine, stone_tracker ja netti_fit (lahdekoodi kansiossa cpp/). Valmiit .pyd-tiedostot
# kopioidaan projektikansioon (main.py:n ja netti_live.py:n viereen).
# netti_fit = nettivideon kivenseurannan tarkka sovitus (B-taso): netti_live.py / netti_seuranta.py kayttavat sita automaattisesti, jos
# moduuli loytyy (noin 6 x nopeampi kuin Python-toteutus); ilman sita ajetaan Python-toteutus. Vain netti_fit: cmake --build build --config Release --target netti_fit
#
# Windows (vcpkg + Python 3.12):
#   cmake -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
# HUOM Python-versio: .pyd toimii vain sen Pythonin versiolla, jolle se on kaannetty (netti_fit.cp312-win_amd64.pyd = Python 3.12). Oletus on C:/Python312;
# jos ohjelmaa ajetaan toisella Pythonilla (python --version), kaanna sille (poista ensin vanha build-kansio, koska CMake muistaa edellisen Pythonin):
#   cmake -B build -G "Visual Studio 17 2022" -A x64 -DPython_EXECUTABLE="C:/polku/python.exe"
#   (pybind11 loytyy talle Pythonille komennolla: python -m pip install pybind11)
# Myos vcpkg ja OpenCV voi ohittaa: -DCMAKE_TOOLCHAIN_FILE=... -DOpenCV_DIR=... -Dpybind11_DIR=...
# Linux (testaus):
#   g++ -O3 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) $(pkg-config --cflags opencv4) \
#       cpp/stone_tracker.cpp -o stone_tracker$(python3-config --extension-suffix) $(pkg-config --libs opencv4) -lpthread -ldl
#   (mode_engine ja netti_fit samoin: cpp/mode_engine.cpp, cpp/netti_fit.cpp)

# vcpkg:n toolchain on asetettava ENNEN project()-kutsua (muuten mm. OpenCV:n riippuvuudet eivat loydy). Oletus C:/vcpkg, jos se on olemassa.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE
        "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING ""
    )
endif()

project(kivenseuranta LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sama Python, jolla ohjelmat ajetaan (muuten esim. Anacondan Python voi loytya PATH:sta ensin). Oletus C:/Python312, jos se on olemassa; muuten CMake etsii Pythonin
# itse tai kaytetaan komentorivin -DPython_EXECUTABLE=...
set(PYBIND11_FINDPYTHON ON)
if(NOT DEFINED Python_EXECUTABLE AND EXISTS "C:/Python312/python.exe")
    set(Python_EXECUTABLE "C:/Python312/python.exe")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python ${Python_VERSION} (${Python_EXECUTABLE}): moduulien paate .${Python_SOABI}. Kayta ohjelmia TALLA Pythonilla (python --version).")

# pybind11 haetaan valitulta Pythonilta (python -m pybind11 --cmakedir); varana 3.12:n oletuspolku
if(NOT DEFINED pybind11_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE _pybind11_cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _pybind11_tulos ERROR_QUIET)
    if(_pybind11_tulos EQUAL 0 AND _pybind11_cmakedir)
        file(TO_CMAKE_PATH "${_pybind11_cmakedir}" _pybind11_cmakedir)
        set(pybind11_DIR "${_pybind11_cmakedir}")
    elseif(EXISTS "C:/Python312/Lib/site-packages/pybind11/share/cmake/pybind11")
        set(pybind11_DIR "C:/Python312/Lib/site-packages/pybind11/share/cmake/pybind11")
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

if(NOT DEFINED OpenCV_DIR AND EXISTS "C:/vcpkg/installed/x64-windows/share/opencv4")
    set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv4")
endif()
find_package(OpenCV REQUIRED)

pybind11_add_module(mode_engine cpp/mode_engine.cpp)
pybind11_add_module(stone_tracker cpp/stone_tracker.cpp)
pybind11_add_module(netti_fit cpp/netti_fit.cpp)

foreach(TARGET_NAME mode_engine stone_tracker netti_fit)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBS})
    target_include_directories(${TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /O2 /EHsc)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -O3)    # -O3: silmukat vektoroituvat (netti_fit ~15 % nopeampi kuin -O2); liukulukutulokset ovat samat
    endif()
    foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${CONFIG}" CONFIG_UPPER)
        set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_SOURCE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_SOURCE_DIR}")
    endforeach()
endforeach()

# Kaannon jalkeen: latautuuko netti_fit kaannokseen kaytetylla Pythonilla (nayttaa myos puuttuvat DLL:t); ei kaada kaannosta
add_custom_command(TARGET netti_fit POST_BUILD
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cpp/tarkista_moduuli.py" netti_fit
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Tarkistetaan, etta netti_fit latautuu"
    VERBATIM)
