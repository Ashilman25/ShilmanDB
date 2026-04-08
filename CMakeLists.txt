cmake_minimum_required(VERSION 3.24)
project(shilmandb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options 
option(SHILMANDB_ENABLE_ML "Build with LibTorch for learned components" OFF)
option(SHILMANDB_BUILD_TESTS "Build unit and integration tests" ON)
option(SHILMANDB_BUILD_BENCH "Build benchmark tools" ON)

# LibTorch 
if(SHILMANDB_ENABLE_ML)
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    add_compile_definitions(SHILMANDB_HAS_LIBTORCH)
endif()

# Core library 
file(GLOB_RECURSE CORE_SOURCES
    src/common/*.cpp
    src/storage/*.cpp
    src/buffer/buffer_pool_manager.cpp
    src/buffer/lru_eviction_policy.cpp
    src/index/*.cpp
    src/catalog/*.cpp
    src/types/*.cpp
    src/parser/*.cpp
    src/planner/planner.cpp
    src/planner/join_order_optimizer.cpp
    src/executor/*.cpp
    src/engine/*.cpp
)

add_library(shilmandb_core STATIC ${CORE_SOURCES})
target_include_directories(shilmandb_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

if(SHILMANDB_ENABLE_ML)
    # ML source files
    target_sources(shilmandb_core PRIVATE
        src/planner/learned_join_optimizer.cpp
        src/buffer/learned_eviction_policy.cpp
    )
    target_link_libraries(shilmandb_core PUBLIC "${TORCH_LIBRARIES}")
    target_include_directories(shilmandb_core PUBLIC "${TORCH_INCLUDE_DIRS}")
endif()


if(SHILMANDB_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif()

# Benchmark stuff
if(SHILMANDB_BUILD_BENCH)
    add_subdirectory(bench)
endif()