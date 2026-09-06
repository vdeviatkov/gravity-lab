cmake_minimum_required(VERSION 3.20)
project(GravityLab VERSION 0.4.0 LANGUAGES CXX)

option(GRAVITY_LAB_BUILD_CLASSIC "Build the vendored faithful classic game (needs SDL2_image and SDL2_ttf)" OFF)
option(GRAVITY_LAB_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(gravity_lab_policy STATIC src/dense_policy.cpp)
target_include_directories(gravity_lab_policy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gravity_lab_policy PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(gravity_lab_policy PRIVATE /W4 /permissive-)
else()
    target_compile_options(gravity_lab_policy PRIVATE -Wall -Wextra -Wpedantic)
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
    set_target_properties(gravity_lab_classic PROPERTIES OUTPUT_NAME gravity_lab_classic PREFIX "")

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRAVITY_LAB_SDL2_IMAGE REQUIRED IMPORTED_TARGET SDL2_image)

    add_library(gravity_lab_classic_renderer STATIC src/classic_renderer.cpp)
    target_include_directories(gravity_lab_classic_renderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gravity_lab_classic_renderer PUBLIC gravity_lab_classic_core
                          PkgConfig::GRAVITY_LAB_SDL2_IMAGE)
    target_compile_features(gravity_lab_classic_renderer PUBLIC cxx_std_20)

    add_executable(gravity_lab_classic_headless apps/classic_headless.cpp)
    target_link_libraries(gravity_lab_classic_headless PRIVATE gravity_lab_classic_core)

    add_executable(gravity_lab_classic_viewer apps/classic_policy_viewer.cpp)
    target_link_libraries(gravity_lab_classic_viewer PRIVATE gravity_lab_classic_renderer gravity_lab_policy)
endif()

if(GRAVITY_LAB_BUILD_TESTS)
    enable_testing()
    add_executable(gravity_lab_policy_tests tests/cpp/test_dense_policy.cpp)
    target_link_libraries(gravity_lab_policy_tests PRIVATE gravity_lab_policy)
    add_test(NAME gravity_lab_policy_cpp
             COMMAND gravity_lab_policy_tests
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dense_policy_fixture.gdp
                     ${CMAKE_CURRENT_BINARY_DIR}/dense_policy_roundtrip.gdp)
    if(GRAVITY_LAB_BUILD_CLASSIC)
        add_executable(gravity_lab_classic_tests tests/cpp/test_classic_environment.cpp)
        target_link_libraries(gravity_lab_classic_tests PRIVATE gravity_lab_classic_core)
        add_test(NAME gravity_lab_classic_cpp
                 COMMAND gravity_lab_classic_tests ${CMAKE_CURRENT_SOURCE_DIR}/classic/assets/levels.mrg)
        add_test(NAME gravity_lab_classic_viewer_validate
                 COMMAND gravity_lab_classic_viewer --validate-only
                         --policy ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/classic_zero_policy.gdp)
        add_test(NAME gravity_lab_classic_viewer_smoke
                 COMMAND gravity_lab_classic_viewer --policy
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/classic_zero_policy.gdp
                         --episodes 1 --max-steps 1 --fps 0 --hold-ms 0)
        set_tests_properties(gravity_lab_classic_viewer_smoke PROPERTIES
            ENVIRONMENT "SDL_VIDEODRIVER=dummy;SDL_AUDIODRIVER=dummy")
    endif()
endif()

install(TARGETS gravity_lab_policy)
if(TARGET gravity_lab_classic)
    install(TARGETS GravityDefied GravityDefiedEngine Assets gravity_lab_classic_core
                    gravity_lab_classic_renderer gravity_lab_classic gravity_lab_classic_headless
                    gravity_lab_classic_viewer)
endif()
install(DIRECTORY include/ DESTINATION include)
