cmake_minimum_required(VERSION 3.7.1)
project (AADIL)

enable_language(CXX)

set (CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -I /usr/local/Cellar/eigen/3.3.4/include/eigen3")
set (CMAKE_MACOSX_RPATH 1)
#set (CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)

set (AADIL_LIB aadil)
set (INCLUDE_DIRS include)

file(GLOB LIB_SRC
     "src/*.cpp"
)

add_library(${AADIL_LIB} ${LIB_SRC})
target_include_directories (${AADIL_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIRS} ${INCLUDE_DIRS})
set_target_properties(${AADIL_LIB}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "../lib"
)
