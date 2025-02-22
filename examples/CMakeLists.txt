cmake_minimum_required(VERSION 3.10)
project(FangTensorExample C)

set(CMAKE_C_STANDARD 11)

# Include directories 
# copy the src/include items to your /usr/local/include directory
include_directories(/usr/local/include)

# Link directories
# directory of libfang.so
link_directories(../)

# Add the executable
add_executable(declerations declerations.c)
add_executable(testy testy.c)

# Link against the fang library
target_link_libraries(declerations PRIVATE fang)
target_link_libraries(testy PRIVATE fang)

# Set rpath to the directory of libfang.so library
set_target_properties(declerations PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" RPATH ../)
set_target_properties(testy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" RPATH ../)