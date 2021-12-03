project(billNumberSearching_test)

add_executable(billNumberSearching_test test.cpp )
file(GLOB MY_DATA "data/*") 
file(COPY ${MY_DATA} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)
file(GLOB MY_RESULTS "results/*") 
file(COPY ${MY_RESULTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/results)



target_link_libraries(${PROJECT_NAME} PUBLIC billNumberSearching)

set_target_properties(${PROJECT_NAME}
        PROPERTIES
        OUTPUT_NAME "billNumberSearching_test"
        )

install(TARGETS ${PROJECT_NAME}  
        CONFIGURATIONS Release
        DESTINATION rel/bin)
        
install(FILES ${MY_DATA} 
        CONFIGURATIONS Release
        DESTINATION dbg/bin/data )
     
install(FILES ${MY_RESULTS} 
        CONFIGURATIONS Release
        DESTINATION dbg/bin/results )

install(TARGETS ${PROJECT_NAME} 
        CONFIGURATIONS Debug
        DESTINATION dbg/bin)
        
install(FILES ${MY_DATA} 
        CONFIGURATIONS Debug
        DESTINATION dbg/bin/data )
     
install(FILES ${MY_RESULTS} 
        CONFIGURATIONS Debug
        DESTINATION dbg/bin/results )


add_custom_target(install_${PROJECT_NAME}
        "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target install
        DEPENDS ${PROJECT_NAME}
        COMMENT "Installing ${PROJECT_NAME}")  
        
        
message(STATUS "Installing ${CMAKE_BINARY_DIR}")              
