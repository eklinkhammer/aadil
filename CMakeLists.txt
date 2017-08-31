cmake_minimum_required(VERSION 2.8)
project(multirover)

include_directories(include)

set (LIB_NAME aadil)
set (TEST_EXEC runTests)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -I ~/resources/eigen")
add_subdirectory(include)

set( LIBS_TO_LINK Utilities Learning Domains Agents POMDPs)

## Create executable
add_executable(generateMultiRoverExperts generateMultiRoverExperts.cpp)
target_link_libraries( generateMultiRoverExperts ${LIBS_TO_LINK} )

## Create library
add_library(${LIB_NAME} SHARED dummy.cpp)
target_link_libraries(${LIB_NAME} ${LIBS_TO_LINK})

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

file(GLOB_RECURSE TEST_SRC
  "test/*.cpp"
)

enable_testing()
add_executable(${TEST_EXEC} ${TEST_SRC})
target_link_libraries(${TEST_EXEC} gtest gtest_main ${LIB_NAME})
add_test(NAME gtest-lib_name COMMAND ${TEST_EXEC})
