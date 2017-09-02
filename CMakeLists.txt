cmake_minimum_required(VERSION 2.8)
project(multirover)

include_directories(include)

set (LIB_NAME aadil)
set (MAIN_EXEC generateMultiRoverExperts)
set (TEST_EXEC runTests)
set (YAML_INCLUDE ~/resources/yaml-cpp/include)
set (INCLUDE_DIRS include ${YAML_INCLUDE})

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -Wno-reorder -I ~/resources/eigen")
add_subdirectory(include)

set( LIBS_TO_LINK Utilities Learning Domains Agents POMDPs)

## Create executable
file(GLOB_RECURSE EXEC_SRC
  "src/*.cpp"
)

find_package(yaml-cpp)

add_executable(${MAIN_EXEC} ${EXEC_SRC})
target_link_libraries(${MAIN_EXEC} ${LIBS_TO_LINK} yaml-cpp)
target_include_directories(${MAIN_EXEC} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS})

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
add_executable(${TEST_EXEC} ${TEST_SRC})# test/Agents/agent_test.cpp)
target_link_libraries(${TEST_EXEC} gtest gtest_main ${LIB_NAME})
add_test(NAME gtest-lib_name COMMAND ${TEST_EXEC})
