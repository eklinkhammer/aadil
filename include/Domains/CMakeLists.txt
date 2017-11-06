set( SRCS SingleRover.cpp MAPElitesRover.cpp Target.cpp MultiRover.cpp Env.cpp G.cpp TeamForming.cpp)
add_library( Domains SHARED ${SRCS} )
target_link_libraries(Domains Learning Utilities)
