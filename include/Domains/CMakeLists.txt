set( SRCS SingleRover.cpp MAPElitesRover.cpp Target.cpp MultiRover.cpp Env.cpp)
add_library( Domains SHARED ${SRCS} )
target_link_libraries(Domains Learning Utilities)
