cmake_minimum_required(VERSION 3.10)

project(rbj_eq_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single test executable from the .cpp in this directory
add_executable(test_rbj
    test_rbj.cpp
)

# We don't actually need any include dirs if rbj_eq.hpp is in the
# same directory as test_rbj.cpp, because #include "rbj_eq.hpp"
# will search that directory automatically. So no -I is required.

# Copy the golden file into the build directory so test_rbj can
# open "rbj_golden.txt" in its working directory.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/rbj_golden.txt
    ${CMAKE_CURRENT_BINARY_DIR}/rbj_golden.txt
    COPYONLY
)

# Target to run tests from the build directory
add_custom_target(run_tests
    COMMAND test_rbj
    DEPENDS test_rbj
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
