cmake_minimum_required(VERSION 3.30)
project(Fang VERSION 0.0.1 LANGUAGES C)

# Build type
if(NOT BUILD_TYPE)
    set(BUILD_TYPE debug CACHE STRING "Fang build type" FORCE)
endif()

if(BUILD_TYPE STREQUAL "debug")
    message(STATUS "Configuring Debug mode")
elseif(BUILD_TYPE STREQUAL "release")
    message(STATUS "Configuring Release mode")
else()
    message(FATAL_ERROR "BUILD_TYPE unrecognized, available: debug, release")
endif()

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-pedantic -Wall -Wextra -Wformat-security -Wundef
        -Wuninitialized -Wunused)
endif()

# Build type specific flags
if(BUILD_TYPE STREQUAL "debug")
    add_definitions(-DFANG_BUILD_DEBUG)

    if(MSVC)
        add_compile_options(/Od /Zi /RTC1 /MDd)
    else()
        add_compile_options(-g -O0)
    endif()
else()
    add_definitions(-DFANG_BUILD_RELEASE)

    if(MSVC)
        add_compile_options(/O2 /Oi /GL /MD)
    else()
        add_compile_options(-s -O3)
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

# Add custom module directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/.cmake/")
find_package(CMocka REQUIRED)

# Add a testing unit
function(add_fang_test TEST_SOURCE TEST_OUTPUT_DIR)
    # Extract the relative path
    file(RELATIVE_PATH REL_SOURCE_PATH ${CMAKE_SOURCE_DIR}/test ${TEST_SOURCE})

    # Get filename w/o extension and directory
    get_filename_component(TEST_NAME ${REL_SOURCE_PATH} NAME_WE)

    # Get relative directory w/o filename
    get_filename_component(TEST_DIR ${REL_SOURCE_PATH} DIRECTORY)

    # Create test binary
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # Set includes
    target_include_directories(${TEST_NAME} PRIVATE "src/include")

    # Link to Fang and CMocka
    target_link_libraries(${TEST_NAME} PRIVATE fang ${CMOCKA_LIBRARIES})

    # Set test output directory
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIR}")

    # Enable Address Sanitizer to detect memory errors
    if(MSVC)
        # Remove complaints about "unused parameter `void **state`" with
        # `/wd4100` flag
        target_compile_options(${TEST_NAME} PRIVATE /fsanitize=address)
    else()
        target_compile_options(${TEST_NAME} PRIVATE -fsanitize=address
            -Wno-unused-parameter)
        target_link_options(${TEST_NAME} PRIVATE -fsanitize=address)
    endif()

    add_test(
        NAME "${TEST_DIR}/${TEST_NAME}"
        COMMAND "${TEST_OUTPUT_DIR}/${TEST_NAME}"
    )
endfunction()

# Check dependency for testing
function(check_test_dependency)
    if(NOT TEST_DEPENDENCY_CHECKED)
        find_package(CMocka REQUIRED)
        set(TEST_DEPENDENCY_CHECKED TRUE)
    endif()
endfunction()

# If unit testing is desired
if(UNIT_TESTING)
    enable_testing()
    check_test_dependency()

    # Create directory
    set(TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/tests/unit")
    file(MAKE_DIRECTORY ${TEST_OUTPUT_DIR})

    # Unit test files
    set(TEST_FILES memory.c util/buffer.c)

    foreach(TEST_SOURCE ${TEST_FILES})
        add_fang_test("${CMAKE_SOURCE_DIR}/test/unit/${TEST_SOURCE}"
            ${TEST_OUTPUT_DIR})
    endforeach()
endif()

# If full testing is desired
if(FULL_TESTING)
    enable_testing()
    check_test_dependency()

    # Create directory
    set(TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/tests/full")
    file(MAKE_DIRECTORY ${TEST_OUTPUT_DIR})

    # Unit test files
    set(TEST_FILES)

    foreach(TEST_SOURCE ${TEST_FILES})
        add_fang_test("${CMAKE_SOURCE_DIR}/test/full/${TEST_SOURCE}"
            ${TEST_OUTPUT_DIR})
    endforeach()
endif()
