cmake_minimum_required(VERSION 3.20)
project(GravityLab VERSION 0.1.0 LANGUAGES CXX)

option(GRAVITY_LAB_BUILD_DESKTOP "Build the optional SDL2 desktop game" ON)
option(GRAVITY_LAB_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(gravity_lab_core STATIC
    src/environment.cpp
    src/map.cpp
)
target_include_directories(gravity_lab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gravity_lab_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(gravity_lab_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(gravity_lab_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(gravity_lab_c SHARED src/c_api.cpp)
target_link_libraries(gravity_lab_c PRIVATE gravity_lab_core)
target_include_directories(gravity_lab_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(gravity_lab_c PROPERTIES OUTPUT_NAME gravity_lab)

add_executable(gravity_lab_headless apps/headless.cpp)
target_link_libraries(gravity_lab_headless PRIVATE gravity_lab_core)

if(GRAVITY_LAB_BUILD_DESKTOP)
    find_package(SDL2 QUIET CONFIG)
    if(NOT SDL2_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(SDL2 QUIET IMPORTED_TARGET sdl2)
        endif()
    endif()

    if(TARGET SDL2::SDL2)
        add_executable(gravity_lab_play apps/desktop.cpp)
        target_link_libraries(gravity_lab_play PRIVATE gravity_lab_core SDL2::SDL2)
        if(TARGET SDL2::SDL2main)
            target_link_libraries(gravity_lab_play PRIVATE SDL2::SDL2main)
        endif()
    elseif(TARGET PkgConfig::SDL2)
        add_executable(gravity_lab_play apps/desktop.cpp)
        target_link_libraries(gravity_lab_play PRIVATE gravity_lab_core PkgConfig::SDL2)
    else()
        message(STATUS "SDL2 not found: desktop frontend skipped; headless and Python APIs remain available")
    endif()
endif()

if(GRAVITY_LAB_BUILD_TESTS)
    enable_testing()
    add_executable(gravity_lab_tests tests/cpp/test_environment.cpp)
    target_link_libraries(gravity_lab_tests PRIVATE gravity_lab_core)
    add_test(NAME gravity_lab_cpp COMMAND gravity_lab_tests ${CMAKE_CURRENT_SOURCE_DIR}/maps/training.gdmap)
endif()

install(TARGETS gravity_lab_core gravity_lab_c gravity_lab_headless)
if(TARGET gravity_lab_play)
    install(TARGETS gravity_lab_play)
endif()
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY maps/ DESTINATION share/gravity-lab/maps)
