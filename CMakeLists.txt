cmake_minimum_required(VERSION 2.8)
project(multirover)



SET(Boost_USE_STATIC_LIBS ON)
SET(Boost_USE_MULTITHREADED ON)

find_package(Threads REQUIRED)
find_package(Boost 1.65.1 COMPONENTS log log_setup thread date_time filesystem system REQUIRED)
if (Boost_FOUND)
  link_directories(${Boost_LIBRARY_DIRS})
endif ()

include_directories( include SYSTEM ${Boost_INCLUDE_DIR} pthread)

set (LIB_NAME aadil)
set (MAIN_EXEC generateMultiRoverExperts)
set (TEST_EXEC runTests)
set (YAML_INCLUDE ~/resources/yaml-cpp/include)
set (INCLUDE_DIRS include ${YAML_INCLUDE})

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -Wno-reorder -I ~/resources/eigen -lpthread")

add_subdirectory(include)

set( LIBS_TO_LINK Utilities Learning Domains Agents)

## Create library
add_library(${LIB_NAME} SHARED dummy.cpp)
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS})
target_link_libraries(${LIB_NAME} ${LIBS_TO_LINK})



## Create executable
file(GLOB_RECURSE EXEC_SRC
  "src/*.cpp"
)

find_package(yaml-cpp)


add_executable(${MAIN_EXEC} ${EXEC_SRC})
target_link_libraries(${MAIN_EXEC}  ${LIB_NAME} yaml-cpp ${Boost_LIBRARIES})
target_include_directories(${MAIN_EXEC} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS})


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


# set(BOOST_LOG_DYN_LINK ON)
add_executable(${TEST_EXEC} ${TEST_SRC})
target_link_libraries(${TEST_EXEC}
             ${Boost_LIBRARIES}
	     ${LIB_NAME}
	     gtest
	     gtest_main)
target_include_directories(${TEST_EXEC} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIR})
enable_testing()
add_test(NAME gtest-lib_name COMMAND ${TEST_EXEC})

