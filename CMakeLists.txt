cmake_minimum_required(VERSION 3.8)



set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin.dbg)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin.rel)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin.relwithdbg)
set(CMAKE_BUILD_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

project(billNumber)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
string(TIMESTAMP time_stemp "%Y-%m-%dT%H:%M:%S")

include_directories(lib test)

add_subdirectory(lib)
add_subdirectory(test)

option(BUILD_DOC "Build documentation" ON)
if(BUILD_DOC)
include_directories(doc)
add_subdirectory(doc)
endif()
