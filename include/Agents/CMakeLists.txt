file (GLOB_RECURSE SRCS "*.cpp")
add_library( Agents SHARED ${SRCS} )
target_link_libraries(Agents Domains Learning Utilities)
