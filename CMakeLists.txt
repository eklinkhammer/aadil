cmake_minimum_required(VERSION 2.8)
project(multirover)

include_directories(include)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -I ~/resources/eigen")
add_subdirectory(include)

set( LIBS_TO_LINK Utilities Learning Domains Agents POMDPs)

## Create executable
add_executable(generateMultiRoverExperts generateMultiRoverExperts.cpp)
target_link_libraries( generateMultiRoverExperts ${LIBS_TO_LINK} )

## Create library
add_library(lib_name SHARED dummy.cpp)
target_link_libraries(lib_name ${LIBS_TO_LINK})

## Google Test Setup
configure_file(CMakeLists.txt.in
               googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
 
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build)

## Create Test Executable
enable_testing()
add_executable(test_exec test/dummy_test.cpp)
target_link_libraries(test_exec gtest gtest_main lib_name)
add_test(NAME gtest-lib_name COMMAND test_exec)
