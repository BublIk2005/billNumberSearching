project(billNumberSearching)
add_executable(billNumberSearching billNumberSearching.cpp billNumberSearching.h)
target_link_libraries(billNumberSearching ${OpenCV_LIBS})

