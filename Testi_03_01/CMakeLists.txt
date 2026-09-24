cmake_minimum_required(VERSION 3.20)

# HUOM: CMAKE_TOOLCHAIN_FILE ON ASETETTAVA ENNEN project()-kutsua -
# CMake lukee/soveltaa toolchain-tiedoston VAIN project()-kutsun
# yhteydessa, joten taman jalkeen asetettuna (kuten aiemmin tassa
# tiedostossa) vcpkg:n toolchain EI koskaan oikeasti aktivoidu, minka
# vuoksi esim. Protobuf (OpenCV:n rippuvuus vcpkg:ssa) ei loydy vaikka
# se on asennettu - vain suoraan annetut polut (kuten OpenCV_DIR alla)
# toimivat silloin, koska ne eivat riipu vcpkg:n automaattisesta
# CMAKE_PREFIX_PATH-taydennyksesta.
set(CMAKE_TOOLCHAIN_FILE
    "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING ""
)

project(mode_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# K�ytet��n CMake:n uutta Python-tunnistusta.
set(PYBIND11_FINDPYTHON ON)

# PAKOTETTU Python 3.12 (sama kuin pybind11_DIR alla ja se "python"
# jolla main.py oikeasti ajetaan) - ilman tata find_package(Python)
# saattaa loytaa VAARAN Python-asennuksen (esim. Anacondan) PATH:in
# hakujarjestyksesta riippuen, jolloin kaannetyt .pyd-tiedostot eivat
# ole tuotavissa sailla Pythonilla jota oikeasti kaytetaan.
set(Python_EXECUTABLE "C:/Python312/python.exe")

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
    # /arch:AVX2 (Testi_03_01, kayttajan pyynnosta "aggressiivisempi"
    # optimointi): koskee VAIN taman .cpp-tiedoston OMAA, kasin
    # kirjoitettua koodia (ei OpenCV:n omia, jo valmiiksi SIMD-
    # dispatchattuja kirjastofunktioita) - kaikki Intel UHD Graphics
    # -sarjan kannettavat (7. sukupolvi/Kaby Lake, 2016+) tukevat
    # AVX2:ta, joten tama on turvallinen oletus talle laitteistoluokalle.
    # EI /arch:AVX512 - moni uudempikaan Intel-mobiilisuoritin ei tue
    # sita (BIOS/mikroarkkitehtuurikohtaista), liian iso riski
    # "illegal instruction" -kaatumiselle oletuksena.
    target_compile_options(
        mode_engine
        PRIVATE
        /O2
        /EHsc
        /arch:AVX2
    )
endif()

# stone_tracker.cpp: SEURANTA+HAKU-vaiheiden C++-porttaus (Task 5+6) -
# oma, erillinen pybind11-moduuli (EI kosketa mode_engine:a), sama
# OpenCV/pybind11-asetus kuin ylla.
pybind11_add_module(
    stone_tracker
    stone_tracker.cpp
)

target_link_libraries(
    stone_tracker
    PRIVATE
    ${OpenCV_LIBS}
)

target_include_directories(
    stone_tracker
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

if(MSVC)
    # /arch:AVX2 - ks. mode_engine:n vastaavan kohdan kommentti ylla
    # (sama perustelu - erityisen relevantti TASSA, koska stone_tracker.
    # cpp:n LM-sovitus/ristikkohaku on juuri sita liukulukuraskasta
    # silmukkakoodia jota kaantaja voi auto-vektoroida AVX2:lla).
    target_compile_options(
        stone_tracker
        PRIVATE
        /O2
        /EHsc
        /arch:AVX2
    )
endif()

# Laitetaan valmiit .pyd-tiedostot suoraan projektikansioon.
foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)

    string(
        TOUPPER
        "${CONFIG}"
        CONFIG_UPPER
    )

    foreach(TARGET_NAME mode_engine stone_tracker)

        set_target_properties(
            ${TARGET_NAME}
            PROPERTIES

            RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}
            "${CMAKE_SOURCE_DIR}"

            LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER}
            "${CMAKE_SOURCE_DIR}"
        )

    endforeach()

endforeach()