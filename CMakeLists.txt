cmake_minimum_required(VERSION 3.20)
project(GravityLab VERSION 0.2.0 LANGUAGES CXX)

option(GRAVITY_LAB_BUILD_DESKTOP "Build the lightweight RL-sandbox SDL2 viewer" ON)
option(GRAVITY_LAB_BUILD_CLASSIC "Build the vendored faithful classic game (needs SDL2_image and SDL2_ttf)" OFF)
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
        add_executable(gravity_lab_sandbox apps/desktop.cpp)
        target_link_libraries(gravity_lab_sandbox PRIVATE gravity_lab_core SDL2::SDL2)
        if(TARGET SDL2::SDL2main)
            target_link_libraries(gravity_lab_sandbox PRIVATE SDL2::SDL2main)
        endif()
    elseif(TARGET PkgConfig::SDL2)
        add_executable(gravity_lab_sandbox apps/desktop.cpp)
        target_link_libraries(gravity_lab_sandbox PRIVATE gravity_lab_core PkgConfig::SDL2)
    else()
        message(STATUS "SDL2 not found: desktop frontend skipped; headless and Python APIs remain available")
    endif()
endif()

if(GRAVITY_LAB_BUILD_CLASSIC)
    add_subdirectory(classic)

    add_library(gravity_lab_classic_core STATIC src/classic_environment.cpp)
    target_include_directories(gravity_lab_classic_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gravity_lab_classic_core PUBLIC GravityDefiedEngine)
    target_compile_features(gravity_lab_classic_core PUBLIC cxx_std_20)

    add_library(gravity_lab_classic SHARED src/classic_c_api.cpp)
    target_include_directories(gravity_lab_classic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gravity_lab_classic PRIVATE gravity_lab_classic_core)
    set_target_properties(gravity_lab_classic PROPERTIES OUTPUT_NAME gravity_lab_classic)

    add_executable(gravity_lab_classic_headless apps/classic_headless.cpp)
    target_link_libraries(gravity_lab_classic_headless PRIVATE gravity_lab_classic_core)

    add_executable(gravity_lab_classic_q apps/classic_q_learning.cpp)
    target_link_libraries(gravity_lab_classic_q PRIVATE gravity_lab_classic_core)
endif()

if(GRAVITY_LAB_BUILD_TESTS)
    enable_testing()
    add_executable(gravity_lab_tests tests/cpp/test_environment.cpp)
    target_link_libraries(gravity_lab_tests PRIVATE gravity_lab_core)
    add_test(NAME gravity_lab_cpp COMMAND gravity_lab_tests ${CMAKE_CURRENT_SOURCE_DIR}/maps/training.gdmap)
    if(GRAVITY_LAB_BUILD_CLASSIC)
        add_executable(gravity_lab_classic_tests tests/cpp/test_classic_environment.cpp)
        target_link_libraries(gravity_lab_classic_tests PRIVATE gravity_lab_classic_core)
        add_test(NAME gravity_lab_classic_cpp
                 COMMAND gravity_lab_classic_tests ${CMAKE_CURRENT_SOURCE_DIR}/classic/assets/levels.mrg)
        add_test(
            NAME gravity_lab_classic_q_train
            COMMAND gravity_lab_classic_q --episodes 2 --eval-episodes 1 --max-steps 3
                    --checkpoint ${CMAKE_CURRENT_BINARY_DIR}/classic_q_test.tsv
        )
        set_tests_properties(gravity_lab_classic_q_train PROPERTIES
            FIXTURES_SETUP classic_q_checkpoint
            FAIL_REGULAR_EXPRESSION "epsilon=(inf|nan)"
        )
        add_test(
            NAME gravity_lab_classic_q_load
            COMMAND gravity_lab_classic_q --eval-only --eval-episodes 1 --max-steps 3
                    --checkpoint ${CMAKE_CURRENT_BINARY_DIR}/classic_q_test.tsv
        )
        set_tests_properties(gravity_lab_classic_q_load PROPERTIES FIXTURES_REQUIRED classic_q_checkpoint)
    endif()
endif()

install(TARGETS gravity_lab_core gravity_lab_c gravity_lab_headless)
if(TARGET gravity_lab_sandbox)
    install(TARGETS gravity_lab_sandbox)
endif()
if(TARGET gravity_lab_classic)
    install(TARGETS GravityDefied GravityDefiedEngine Assets gravity_lab_classic_core
                    gravity_lab_classic gravity_lab_classic_headless gravity_lab_classic_q)
endif()
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY maps/ DESTINATION share/gravity-lab/maps)
