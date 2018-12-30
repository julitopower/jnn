set(SRC jnn.c include/jnn.h)

add_library(jnn_obj OBJECT ${SRC})
add_library(jnn STATIC $<TARGET_OBJECTS:jnn_obj>)
