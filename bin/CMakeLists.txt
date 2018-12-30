set(SRC example.c)

add_executable(jnn_example ${SRC})
target_link_libraries(jnn_example jnn)
