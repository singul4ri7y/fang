cmake_minimum_required(VERSION 3.30)
project(Fang VERSION 0.0.1 LANGUAGES C)

# Build type
if(NOT BUILD_TYPE)
    set(BUILD_TYPE debug CACHE STRING "Fang build type" FORCE)
endif()

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-pedantic -Wall -Wextra -Wformat-security -Wundef
        -Wno-unused)
endif()

# Build type specific flags
if(BUILD_TYPE STREQUAL "debug")
    add_definitions(-DFANG_BUILD_DEBUG)

    if(MSVC)
        add_compile_options(/Od /Zi /RTC1 /MDd)
    else()
        add_compile_options(-g -Og)
    endif()
elseif(BUILD_TYPE STREQUAL "release")
    add_definitions(-DFANG_BUILD_RELEASE)

    if(MSVC)
        add_compile_options(/O2 /Oi /GL /MD)
    else()
        add_compile_options(-O3 -s)
    endif()
endif()

# Detect platform type
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PLATFORM "linux")
    message(STATUS "Using Fang platform: linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PLATFORM "windows")
    message(STATUS "Using Fang platform: win")
else()
    mesasge(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}!")
endif()

# Check for AVX512 and AVX2 support
include(CheckCSourceRuns)
set(CMAKE_REQUIRED_FLAGS -mavx512f -mavx2)

check_c_source_runs("
    #include <immintrin.h>
    int main() {
        __m512 a = _mm512_set1_ps(1.0f);
        __m512 b = _mm512_set1_ps(2.0f);
        __m512 c = _mm512_add_ps(a, b);
        return 0;
    }
" HAVE_AVX512)

# Fallback to AVX2 if AVX512 is not available
if(NOT HAVE_AVX512)
    check_c_source_runs("
        #include <immintrin.h>
        int main() {
            __m256 a = _mm256_set1_ps(1.0f);
            __m256 b = _mm256_set1_ps(2.0f);
            __m256 c = _mm256_add_ps(a, b);
            return 0;
        }
    " HAVE_AVX2)
endif()

if(HAVE_AVX512)
    add_compile_options(-mavx512f)
    add_definitions(-DFANG_USE_AVX512)
    message(STATUS "AVX512 support enabled")
elseif(HAVE_AVX2)
    add_compile_options(-mavx2)
    add_definitions(-DFANG_USE_AVX2)
    message(STATUS "AVX2 support enabled")
endif()

# Building Fang as a shared library
add_definitions(-DFANG_LIB)

# Source files
file(GLOB_RECURSE SOURCE_FILES "src/fang/*.c")
list(FILTER SOURCE_FILES EXCLUDE REGEX "src/fang/platform/.*\\.c$")

# Platform specific source files
file(GLOB_RECURSE PLATFORM_SOURCE_FILES "src/fang/platform/${PLATFORM}/*.c")
list(APPEND SOURCE_FILES ${PLATFORM_SOURCE_FILES})

# Create the library
add_library(fang SHARED ${SOURCE_FILES})

# Includes
target_include_directories(fang PRIVATE
    "src/include"
    "src/include/platform/${PLATFORM}/"
)

# Set version
target_compile_definitions(fang PRIVATE
    PROJECT_VERSION="${PROJECT_VERSION}"
)

# Set library properties
set_target_properties(fang PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Add project custom module directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/.cmake/module/")

# Testing
enable_testing()
find_package(Criterion REQUIRED)
find_program(VALGRIND "valgrind")

if(VALGRIND)
    message(STATUS "Found Valgrind: tests will run against Valgrind")
endif()

# Run tests
file(GLOB_RECURSE TEST_FILES "test/*.c")
foreach(TEST_SOURCE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE "src/include")
    target_link_libraries(${TEST_NAME} PRIVATE fang ${CRITERION_LIBRARIES})

    if(VALGRIND)
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${CMAKE_COMMAND} -E env ${VALGRIND} --track-origins=yes
            --leak-check=full --error-exitcode=1 $<TARGET_FILE:${TEST_NAME}>
            --verbose
        )
    else()
        add_test(
            NAME ${TEST_NAME}
            COMMAND $<TARGET_FILE:${TEST_NAME}> --verbose
        )
    endif()
endforeach()
