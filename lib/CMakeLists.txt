project(billNumberSearching)
add_library(billNumberSearching billNumberSearching.cpp billNumberSearching.h)

set_property(TARGET billNumberSearching PROPERTY CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(billNumberSearching ${OpenCV_LIBS})



set_target_properties(${PROJECT_NAME}
        PROPERTIES
        PUBLIC_HEADER billNumberSearching.h
        OUTPUT_NAME "billNumberSearching"
        CXX_STANDARD 11
        )

target_include_directories(${PROJECT_NAME} PRIVATE .)

install(TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Debug
        billNumberSearching
        DESTINATION dbg/lib
        PUBLIC_HEADER
        DESTINATION dbg/include)